# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import re
import gc
from contextlib import contextmanager

import torch

import transformer_engine as te
from transformer_engine.pytorch.attention import RotaryPositionEmbedding

import transformers
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaForCausalLM,
    LlamaRMSNorm,
    LlamaConfig,
)
from transformers.modeling_utils import _add_variant, load_state_dict
from transformers.utils import WEIGHTS_INDEX_NAME
from transformers.utils.hub import get_checkpoint_shard_files
from transformer_engine.common.recipe import Format, DelayedScaling


@contextmanager
def replace_decoder(te_decoder_cls):
    """
    Replace `LlamaDecoderLayer` with custom `TELlamaDecoderLayer`.
    """
    original_llama_decoder_cls = transformers.models.llama.modeling_llama.LlamaDecoderLayer
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = te_decoder_cls
    try:
        yield
    finally:
        transformers.models.llama.modeling_llama.LlamaDecoderLayer = original_llama_decoder_cls


# FP8 recipes: attention uses HYBRID, MLP uses E4M3 as you had
attn_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
mlp_recipe  = DelayedScaling(fp8_format=Format.E4M3,   amax_history_len=16, amax_compute_algo="max")


class TELlamaDecoderLayer(torch.nn.Module):
    def __init__(self, config, *args, dropout_rate=0.0, **kwargs):
        super().__init__()
        # TransformerEngine attention layer
        self.self_attention = te.pytorch.MultiheadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            bias=False,
            layernorm_epsilon=config.rms_norm_eps,
            attention_dropout=dropout_rate,
            fuse_qkv_params=False,
            normalization="RMSNorm",
            num_gqa_groups=config.num_key_value_heads,
            qkv_format="bshd",
            input_layernorm=True,
        )

        # TransformerEngine MLP with RMSNorm + SwiGLU
        self.layernorm_mlp = te.pytorch.LayerNormMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            normalization="RMSNorm",
            activation="swiglu",
        )

        # Precompute TE rotary embedding table up to config.max_position_embeddings
        te_rope = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)
        # Note: we'll ensure device alignment at forward() in case module moved after init
        self.te_rope_emb = te_rope(max_seq_len=config.max_position_embeddings).cuda()

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        # type checks
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch.Tensor")

        # Make sure RoPE lives on the same device as hidden_states (no-op if already correct)
        if self.te_rope_emb.device != hidden_states.device:
            self.te_rope_emb = self.te_rope_emb.to(hidden_states.device)

        # Fix attention mask format for TransformerEngine
        te_attention_mask = None
        if attention_mask is not None:
            # Convert from HuggingFace format (batch_size, seq_len) to TE format
            # TE expects (batch_size, 1, seq_len, seq_len) or (batch_size, num_heads, seq_len, seq_len)
            batch_size, seq_len = attention_mask.shape
            
            # Create causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=attention_mask.device, dtype=torch.bool))
            
            # Expand attention_mask to match causal mask
            # attention_mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            
            # Apply padding mask to causal mask
            # causal_mask: (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            
            # Combine masks: keep causal structure but mask out padded positions
            te_attention_mask = causal_mask & expanded_mask
            
            # Convert to float and invert (TE uses 0 for masked positions, -inf for attention weights)
            te_attention_mask = te_attention_mask.float()
            te_attention_mask = te_attention_mask.masked_fill(te_attention_mask == 0, float('-inf'))
            te_attention_mask = te_attention_mask.masked_fill(te_attention_mask == 1, 0.0)

        # Attention in FP8 (HYBRID)
        with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=attn_recipe):
            attn_out = self.self_attention(
                hidden_states,
                attention_mask=te_attention_mask,
                rotary_pos_emb=self.te_rope_emb,
            )

        hidden_states = hidden_states + attn_out

        # MLP in FP8 (E4M3)
        with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=mlp_recipe):
            ffn_out = self.layernorm_mlp(hidden_states)

        hidden_states = hidden_states + ffn_out
        return (hidden_states,)  # Return tuple to match HuggingFace format


class TELlamaForCausalLM:
    """
    Causal LM created with `LlamaModel`. The underlying `LlamaDecoderLayer`
    class is monkey-patched with `TELlamaDecoderLayer` class before
    initializing the causal LM with `LlamaForCausalLM`.

    Args:
        config: LlamaConfig
    """

    def __new__(cls, config: LlamaConfig):
        with replace_decoder(te_decoder_cls=TELlamaDecoderLayer):
            llama_for_causal_lm = LlamaForCausalLM(config)
        return llama_for_causal_lm

    @classmethod
    def from_pretrained_local(cls, pretrained_model_name_or_path, *args, config, **kwargs):
        """
        Custom method adapted from `from_pretrained` method in HuggingFace
        Transformers:
        https://github.com/huggingface/transformers/blob/f497f56/src/transformers/modeling_utils.py#L2579
        """
        # Set default dtype before loading weights (matches your training dtype)
        torch.set_default_dtype(kwargs["torch_dtype"])

        # Instantiate the TE-patched model
        vanilla_model = cls(config)
        subfolder = ""
        variant = None

        # Determine checkpoint layout
        if os.path.isfile(
            os.path.join(pretrained_model_name_or_path, subfolder, _add_variant("model.safetensors.index.json", variant))
        ):
            archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant("model.safetensors.index.json", variant))
            is_sharded = True
        elif os.path.isfile(
            os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
        ):
            archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
            is_sharded = True
        elif os.path.isfile(
            os.path.join(pretrained_model_name_or_path, subfolder, _add_variant("model.safetensors", variant))
        ):
            archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant("model.safetensors", variant))
            is_sharded = False
        else:
            raise AssertionError("Only sharded or single-file .safetensors PyTorch checkpoints are supported.")

        # Resolve shard files
        if not is_sharded:
            resolved_archive_file = [archive_file]
            print(f"Resolved archive file: {archive_file}")
        else:
            resolved_archive_file, _ = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                archive_file,
            )

        # Load and map weights into TE modules
        for shard_file in resolved_archive_file:
            state_dict = load_state_dict(shard_file)
            # Copy TE-specific parameters
            replace_params(state_dict, vanilla_model.state_dict(), config)
            # Load remaining params with strict=False (HF-style)
            vanilla_model.load_state_dict(state_dict, strict=False)

            # Free memory
            del state_dict
            gc.collect()

        return vanilla_model


def replace_params(hf_state_dict, te_state_dict, config):
    """
    Copy selected HF weights into TransformerEngine layer parameter slots.
    """
    # collect all layer prefixes to update
    all_layer_prefixes = set()
    for param_key in hf_state_dict.keys():
        layer_prefix_pat = r"model\.layers\.\d+\."
        m = re.match(layer_prefix_pat, param_key)
        if m is not None:
            all_layer_prefixes.add(m.group())

    for layer_prefix in all_layer_prefixes:
        # skip if HF layer missing
        if layer_prefix + "input_layernorm.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.layernorm_qkv.layer_norm_weight"].data[:] = \
                hf_state_dict[layer_prefix + "input_layernorm.weight"].data[:]

        if layer_prefix + "self_attn.q_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.layernorm_qkv.query_weight"].data[:] = \
                hf_state_dict[layer_prefix + "self_attn.q_proj.weight"].data[:]

        if layer_prefix + "self_attn.k_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.layernorm_qkv.key_weight"].data[:] = \
                hf_state_dict[layer_prefix + "self_attn.k_proj.weight"].data[:]

        if layer_prefix + "self_attn.v_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.layernorm_qkv.value_weight"].data[:] = \
                hf_state_dict[layer_prefix + "self_attn.v_proj.weight"].data[:]

        if layer_prefix + "self_attn.o_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.proj.weight"].data[:] = \
                hf_state_dict[layer_prefix + "self_attn.o_proj.weight"].data[:]

        if layer_prefix + "post_attention_layernorm.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.layer_norm_weight"].data[:] = \
                hf_state_dict[layer_prefix + "post_attention_layernorm.weight"].data[:]

        # MLP: gate+up concat to fc1, down -> fc2
        if layer_prefix + "mlp.gate_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc1_weight"].data[: config.intermediate_size] = \
                hf_state_dict[layer_prefix + "mlp.gate_proj.weight"].data

        if layer_prefix + "mlp.up_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc1_weight"].data[config.intermediate_size :] = \
                hf_state_dict[layer_prefix + "mlp.up_proj.weight"].data

        if layer_prefix + "mlp.down_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc2_weight"].data[:] = \
                hf_state_dict[layer_prefix + "mlp.down_proj.weight"].data[:]

    return all_layer_prefixes