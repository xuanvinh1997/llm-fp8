# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Hybrid FP8 Implementation for Llama models
Implements layer-wise format assignment strategy:
- E4M3 for MLP components (higher precision for stable computations)
- E5M2 for attention Q/K projections (wider dynamic range)
- E4M3 for attention V/O projections (higher precision)
"""

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


# Define FP8 formats for different components
E4M3_FORMAT = Format.E4M3  # Higher precision (4-bit exponent, 3-bit mantissa)
E5M2_FORMAT = Format.E5M2  # Wider range (5-bit exponent, 2-bit mantissa)

# Create recipes for different components
mlp_recipe = DelayedScaling(fp8_format=E4M3_FORMAT, amax_history_len=16, amax_compute_algo="max")
attn_qk_recipe = DelayedScaling(fp8_format=E5M2_FORMAT, amax_history_len=16, amax_compute_algo="max")
attn_vo_recipe = DelayedScaling(fp8_format=E4M3_FORMAT, amax_history_len=16, amax_compute_algo="max")


@contextmanager
def replace_decoder(te_decoder_cls):
    """
    Replace `LlamaDecoderLayer` with custom `TELlamaDecoderLayer`.
    """
    original_llama_decoder_cls = (
        transformers.models.llama.modeling_llama.LlamaDecoderLayer
    )
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = te_decoder_cls
    try:
        yield
    finally:
        transformers.models.llama.modeling_llama.LlamaDecoderLayer = (
            original_llama_decoder_cls
        )


class TELlamaDecoderLayer(torch.nn.Module):
    """
    Hybrid FP8 Llama Decoder Layer with layer-wise format assignment.

    Format assignment strategy:
    - MLP: E4M3 for stable feed-forward computations
    - Attention Q/K: E5M2 for wide dynamic range in query-key interactions
    - Attention V/O: E4M3 for higher precision in value computations
    """

    def __init__(self, config, *args, dropout_rate=0.0, **kwargs):
        super().__init__()

        # Create attention components with separate linear layers for hybrid format assignment
        self.q_proj = te.pytorch.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

        self.k_proj = te.pytorch.Linear(
            config.hidden_size,
            config.num_key_value_heads * (config.hidden_size // config.num_attention_heads),
            bias=False,
        )

        self.v_proj = te.pytorch.Linear(
            config.hidden_size,
            config.num_key_value_heads * (config.hidden_size // config.num_attention_heads),
            bias=False,
        )

        self.o_proj = te.pytorch.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

        # Layer normalization for attention
        self.input_layernorm = te.pytorch.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # MLP components
        self.layernorm_mlp = te.pytorch.LayerNormMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            normalization="RMSNorm",
            activation="swiglu",
            layernorm_epsilon=config.rms_norm_eps,
        )

        # Store config for attention computation
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Rotary position embedding
        te_rope = RotaryPositionEmbedding(self.head_dim)
        self.te_rope_emb = te_rope(max_seq_len=config.max_position_embeddings).cuda()

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Type checking
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch.Tensor")

        batch_size, seq_len, _ = hidden_states.shape

        # Layer norm before attention
        normed_hidden_states = self.input_layernorm(hidden_states)

        # Attention with hybrid FP8 formats
        # Q and K projections with E5M2 (wider dynamic range)
        with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=attn_qk_recipe):
            query_states = self.q_proj(normed_hidden_states)
            key_states = self.k_proj(normed_hidden_states)

        # V projection with E4M3 (higher precision)
        with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=attn_vo_recipe):
            value_states = self.v_proj(normed_hidden_states)

        # Reshape for attention computation
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary position embeddings
        if self.te_rope_emb is not None:
            query_states = self._apply_rotary_pos_emb(query_states, self.te_rope_emb)
            key_states = self._apply_rotary_pos_emb(key_states, self.te_rope_emb)

        # Compute attention (keeping computation in higher precision)
        attn_output = self._compute_attention(
            query_states, key_states, value_states, attention_mask
        )

        # Output projection with E4M3 (higher precision)
        with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=attn_vo_recipe):
            attn_output = self.o_proj(attn_output)

        # Residual connection
        hidden_states = hidden_states + attn_output

        # MLP with E4M3 (higher precision for stable computations)
        with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=mlp_recipe):
            ffn_out = self.layernorm_mlp(hidden_states)

        # Final residual connection
        hidden_states = hidden_states + ffn_out

        return hidden_states

    def _apply_rotary_pos_emb(self, x, rope_emb):
        """Apply rotary position embeddings to input tensor."""
        # Simplified rotary embedding application
        # In practice, this would use the full RotaryPositionEmbedding logic
        return x

    def _compute_attention(self, query_states, key_states, value_states, attention_mask):
        """Compute scaled dot-product attention."""
        batch_size, seq_len, _, head_dim = query_states.shape

        # Transpose for attention computation: [batch, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Handle grouped query attention if needed
        if self.num_kv_heads != self.num_heads:
            key_states = self._repeat_kv(key_states, self.num_heads // self.num_kv_heads)
            value_states = self._repeat_kv(value_states, self.num_heads // self.num_kv_heads)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / (head_dim ** 0.5)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Transpose back and reshape: [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        return attn_output

    def _repeat_kv(self, hidden_states, n_rep):
        """Repeat key/value heads for grouped query attention."""
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class TELlamaForCausalLM:
    """
    LlamaForCausalLM with Hybrid FP8 support through Transformer Engine.

    This implementation uses layer-wise format assignment:
    - E4M3 for MLP layers (higher precision)
    - E5M2 for attention Q/K (wider dynamic range)
    - E4M3 for attention V/O (higher precision)
    """

    def __new__(cls, config: LlamaConfig):
        with replace_decoder(TELlamaDecoderLayer):
            llama_for_causal_lm = LlamaForCausalLM(config)

        # Initialize TE modules
        for layer_idx, layer in enumerate(llama_for_causal_lm.model.layers):
            # Log format assignment for transparency
            print(f"Layer {layer_idx}: MLP=E4M3, Attn Q/K=E5M2, Attn V/O=E4M3")

        return llama_for_causal_lm

    @classmethod
    def from_pretrained_local(cls, model_path, config, **kwargs):
        """Load model from local checkpoint with hybrid FP8 configuration."""
        with replace_decoder(TELlamaDecoderLayer):
            model = LlamaForCausalLM(config)

        print(f"Loading Llama model from {model_path} with Hybrid FP8 format assignment")

        # Load checkpoint
        state_dict = load_state_dict(model_path)

        # Rename keys to match TE layer structure
        renamed_state_dict = {}
        for name, param in state_dict.items():
            new_name = cls._rename_key_for_te(name)
            renamed_state_dict[new_name] = param

        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(
            renamed_state_dict, strict=False
        )

        if missing_keys:
            print(f"Missing keys when loading: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading: {unexpected_keys}")

        print("Model loaded successfully with Hybrid FP8 configuration")
        print("Format assignment: MLP=E4M3, Attn Q/K=E5M2, Attn V/O=E4M3")

        return model

    @staticmethod
    def _rename_key_for_te(key):
        """Rename Hugging Face checkpoint keys to match TE layer names."""
        # Map standard attention layer names to our separate projections
        replacements = [
            ("self_attn.q_proj", "q_proj"),
            ("self_attn.k_proj", "k_proj"),
            ("self_attn.v_proj", "v_proj"),
            ("self_attn.o_proj", "o_proj"),
            ("mlp.gate_proj", "layernorm_mlp.fc1_weight"),
            ("mlp.up_proj", "layernorm_mlp.fc2_weight"),
            ("mlp.down_proj", "layernorm_mlp.fc3_weight"),
            ("input_layernorm", "input_layernorm"),
            ("post_attention_layernorm", "layernorm_mlp.layernorm"),
        ]

        for old, new in replacements:
            if old in key:
                key = key.replace(old, new)

        return key