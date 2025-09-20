# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. See LICENSE for license information.

import os
import re
import gc
import inspect
from contextlib import contextmanager
from typing import Optional, Tuple, Any, Dict

import torch

import transformer_engine as te
from transformer_engine.pytorch.attention import RotaryPositionEmbedding
from transformer_engine.common.recipe import Format, DelayedScaling

import transformers
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaConfig,
)
from transformers.modeling_utils import _add_variant, load_state_dict
from transformers.utils import WEIGHTS_INDEX_NAME
from transformers.utils.hub import get_checkpoint_shard_files


# ----------------------------
# Helpers / context management
# ----------------------------

@contextmanager
def replace_decoder(te_decoder_cls):
    """
    Temporarily replace HF's LlamaDecoderLayer with our TE-backed implementation.
    """
    original_llama_decoder_cls = transformers.models.llama.modeling_llama.LlamaDecoderLayer
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = te_decoder_cls
    try:
        yield
    finally:
        transformers.models.llama.modeling_llama.LlamaDecoderLayer = original_llama_decoder_cls


def _filter_kwargs_by_sig(cls_or_fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs that the callable actually accepts (robust to TE API changes)."""
    try:
        sig = inspect.signature(cls_or_fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except (ValueError, TypeError):
        # If we cannot introspect, pass through and hope for the best
        return kwargs


def _is_fp8_enabled_for_infer() -> bool:
    """Enable FP8 in inference only if explicitly requested."""
    env = os.getenv("ENABLE_FP8_INFER", "0").lower()
    return env in ("1", "true", "yes", "on")


# ----------
# FP8 recipes
# ----------

attn_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
mlp_recipe = DelayedScaling(fp8_format=Format.E4M3,   amax_history_len=16, amax_compute_algo="max")


# -------------------------
# TE-backed Decoder Layer
# -------------------------

class TELlamaDecoderLayer(torch.nn.Module):
    """
    Drop-in replacement for HF's LlamaDecoderLayer using TransformerEngine.
    Implements:
      - MHA with GQA
      - RoPE (device-aware, position-aware)
      - FP8 autocast wrappers (attention / MLP)
      - Proper HF forward signature (past_key_value, use_cache, output_attentions)
      - Residual connections outside sublayers
    """

    def __init__(self, config: LlamaConfig, *args, dropout_rate: float = 0.0, **kwargs):
        super().__init__()

        # Build TE MultiheadAttention (robust to arg-name differences across TE versions)
        mha_kwargs = dict(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            bias=False,
            layernorm_epsilon=getattr(config, "rms_norm_eps", 1e-5),
            attention_dropout=dropout_rate,
            fuse_qkv_params=False,
            normalization="RMSNorm",
            qkv_format="bshd",
            input_layernorm=True,
        )

        # TE versions differ: num_gqa_groups vs num_kv_heads
        if hasattr(config, "num_key_value_heads") and config.num_key_value_heads is not None:
            mha_kwargs["num_gqa_groups"] = config.num_key_value_heads
            mha_kwargs["num_kv_heads"] = config.num_key_value_heads  # keep both; _filter_kwargs_by_sig will prune
        self.self_attention = te.pytorch.MultiheadAttention(**_filter_kwargs_by_sig(te.pytorch.MultiheadAttention, mha_kwargs))

        # TE LayerNorm-MLP (SiLU/Swish-Gated)
        mlp_kwargs = dict(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            normalization="RMSNorm",
            activation="swiglu",
        )
        self.layernorm_mlp = te.pytorch.LayerNormMLP(**_filter_kwargs_by_sig(te.pytorch.LayerNormMLP, mlp_kwargs))

        # RoPE support
        head_dim = config.hidden_size // config.num_attention_heads
        # Many Llama configs expose rope_theta; RotaryPositionEmbedding may accept it as kwarg in some TE versions.
        rope_kwargs = {}
        if hasattr(config, "rope_theta"):
            rope_kwargs["rope_theta"] = config.rope_theta
        self.rope = RotaryPositionEmbedding(head_dim, **_filter_kwargs_by_sig(RotaryPositionEmbedding, rope_kwargs))

        # cache for cos/sin (not persistent; rebuilt per device/length)
        self.register_buffer("rope_cache", None, persistent=False)
        self.register_buffer("rope_cache_len", torch.tensor(0, dtype=torch.long), persistent=False)

    # ---- RoPE helpers ----
    def _ensure_rope(self, q_len: int, device: torch.device):
        """Ensure rope cache is on the correct device and sufficiently long."""
        need_new = (
            self.rope_cache is None
            or self.rope_cache.device != device
            or int(self.rope_cache_len.item()) < q_len
        )
        if need_new:
            # TE RotaryPositionEmbedding often returns packed cos/sin or a struct understood by TE MHA.
            cache = self.rope(max_seq_len=q_len)
            if isinstance(cache, torch.Tensor):
                cache = cache.to(device)
            elif isinstance(cache, (tuple, list)):
                cache = tuple(c.to(device) if isinstance(c, torch.Tensor) else c for c in cache)
            self.rope_cache = cache
            self.rope_cache_len.fill_(q_len)

    def _get_rope_slice(self, q_len: int, device: torch.device, position_ids: Optional[torch.Tensor] = None):
        self._ensure_rope(q_len, device)
        rope_obj = self.rope_cache
        if position_ids is None:
            return rope_obj
        # Slice rope by position_ids if rope is tensor-like
        if isinstance(rope_obj, torch.Tensor):
            # expect shape like (1, max_len, ...) -> index at dim=1
            return rope_obj[:, position_ids, ...]
        if isinstance(rope_obj, (tuple, list)) and isinstance(rope_obj[0], torch.Tensor):
            # slice each component
            return tuple(comp[:, position_ids, ...] for comp in rope_obj)
        # otherwise, return as-is (TE may handle position_ids internally)
        return rope_obj

    # ---- Mask helpers ----
    @staticmethod
    def _build_causal_bias(q_len: int, k_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # (q, k) upper-triangular -inf to mask future
        causal = torch.full((q_len, k_len), float("-inf"), device=device, dtype=dtype)
        causal = torch.triu(causal, diagonal=1)  # keep zeros on and below diagonal
        return causal.unsqueeze(0).unsqueeze(0)  # (1,1,q,k)

    @staticmethod
    def _hf_to_te_mask(attention_mask: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Convert HF attention_mask formats to TE attention bias (additive, -inf on masked positions) shape (B,1,Q,K).
        """
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            # HF convention: 1=keep, 0=pad
            bsz, k_len = attention_mask.shape
            q_len = hidden_states.size(1)
            device = hidden_states.device
            dtype = hidden_states.dtype
            causal = TELlamaDecoderLayer._build_causal_bias(q_len, k_len, device, dtype)
            pad = (1 - attention_mask).to(dtype) * -1e9
            pad = pad[:, None, None, :]  # (B,1,1,K)
            return causal + pad
        if attention_mask.dim() == 4:
            # assume already (B,1,Q,K) or compatible
            return attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
        raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

    # ---- Forward ----
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch.Tensor")

        bsz, q_len, _ = hidden_states.shape
        device = hidden_states.device

        # Prepare RoPE (slice by position_ids if provided)
        rope = self._get_rope_slice(q_len, device, position_ids)

        # Convert mask to TE additive bias
        te_mask = self._hf_to_te_mask(attention_mask, hidden_states)

        # Decide FP8 autocast
        fp8_ok = getattr(te.pytorch, "is_fp8_available", lambda: True)()
        fp8_enabled = fp8_ok and (self.training or _is_fp8_enabled_for_infer())

        # ---- Attention ----
        attn_out = None
        present_kv = None
        attn_probs = None

        attn_call_kwargs = dict(
            attention_mask=te_mask,
            rotary_pos_emb=rope,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_call_kwargs = _filter_kwargs_by_sig(self.self_attention.forward, attn_call_kwargs)

        with te.pytorch.fp8_autocast(enabled=fp8_enabled, fp8_recipe=attn_recipe):
            try:
                out = self.self_attention(hidden_states, **attn_call_kwargs)
            except TypeError:
                # Fallback: minimal args (API variance)
                out = self.self_attention(hidden_states, attention_mask=te_mask, rotary_pos_emb=rope)

        # Normalize returns:
        # Expect one of:
        #  - attn_out
        #  - (attn_out, present_kv)
        #  - (attn_out, present_kv, attn_probs)
        if isinstance(out, torch.Tensor):
            attn_out = out
        elif isinstance(out, (tuple, list)):
            if len(out) >= 1:
                attn_out = out[0]
            if len(out) >= 2:
                present_kv = out[1]
            if len(out) >= 3:
                attn_probs = out[2]
        else:
            raise RuntimeError("Unexpected return from TE MultiheadAttention")

        hidden_states = hidden_states + attn_out

        # ---- MLP ----
        with te.pytorch.fp8_autocast(enabled=fp8_enabled, fp8_recipe=mlp_recipe):
            ffn_out = self.layernorm_mlp(hidden_states)
        hidden_states = hidden_states + ffn_out

        # ---- HF-compatible returns ----
        output = (hidden_states,)
        if output_attentions:
            output += (attn_probs,)
        if use_cache:
            output += (present_kv,)
        return output


# -----------------------------------
# LlamaForCausalLM wrapper/factory
# -----------------------------------

class TELlamaForCausalLM:
    """
    Factory for HF LlamaForCausalLM but with our TELlamaDecoderLayer monkey-patched in.
    Use:
        model = TELlamaForCausalLM(config)   # identical to LlamaForCausalLM(config) but TE layers inside
        model = TELlamaForCausalLM.from_pretrained_local(path, config=cfg, torch_dtype=torch.bfloat16)
    """

    def __new__(cls, config: LlamaConfig):
        with replace_decoder(te_decoder_cls=TELlamaDecoderLayer):
            llama_for_causal_lm = LlamaForCausalLM(config)
        return llama_for_causal_lm

    @classmethod
    def from_pretrained_local(cls, pretrained_model_name_or_path: str, *args, config: LlamaConfig, torch_dtype=torch.float16, **kwargs):
        """
        Load a local HF checkpoint into the TE-backed model.
        Accepts:
          - model.safetensors.index.json (sharded)
          - pytorch_model.bin.index.json (sharded)
          - model.safetensors (single)
          - pytorch_model.bin (single)
        """
        prev_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch_dtype)
            # Instantiate model (with TE layers)
            vanilla_model = cls(config)

            subfolder = ""
            variant = None

            def _exists(fname: str) -> bool:
                return os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(fname, variant)))

            # Determine archive file
            if _exists("model.safetensors.index.json"):
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant("model.safetensors.index.json", variant))
                is_sharded = True
            elif _exists(WEIGHTS_INDEX_NAME):
                # usually "pytorch_model.bin.index.json"
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
                is_sharded = True
            elif _exists("model.safetensors"):
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant("model.safetensors", variant))
                is_sharded = False
            elif _exists("pytorch_model.bin"):
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant("pytorch_model.bin", variant))
                is_sharded = False
            else:
                raise AssertionError("Checkpoint not found (supported: *safetensors(.index.json), pytorch_model.bin(.index.json))")

            # Resolve files
            if not is_sharded:
                resolved_archive_files = [archive_file]
                print(f"[TELlama] Resolved archive file: {archive_file}")
            else:
                resolved_archive_files, _ = get_checkpoint_shard_files(
                    pretrained_model_name_or_path,
                    archive_file,
                )

            # Load each shard and:
            #  1) copy TE-specific params (e.g., fused fc1 weight packing)
            #  2) let HF load_state_dict map the rest with strict=False
            for shard_file in resolved_archive_files:
                state_dict = load_state_dict(shard_file)

                # Replace TE params from HF weights (in-place on model tensors)
                replace_params(state_dict, vanilla_model.state_dict(), config)

                # Load other params by HF name
                missing, unexpected = vanilla_model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    # Not fatal; useful for debugging
                    print(f"[TELlama] load_state_dict shard='{os.path.basename(shard_file)}' missing={len(missing)} unexpected={len(unexpected)}")

                # Free memory
                del state_dict
                gc.collect()

            return vanilla_model
        finally:
            torch.set_default_dtype(prev_dtype)


# ------------------------
# Weight mapping utilities
# ------------------------

_LAYER_PREFIX_RE = re.compile(r"model\.layers\.\d+\.")

def replace_params(hf_state_dict: Dict[str, torch.Tensor], te_state_dict: Dict[str, torch.Tensor], config: LlamaConfig):
    """
    Copy/pack HF weights into TE module parameter tensors where names/shapes differ.
    This function mutates te_state_dict tensors in-place (no return).
    """
    # Collect layer prefixes present in this shard
    all_layer_prefixes = set()
    for param_key in hf_state_dict.keys():
        m = _LAYER_PREFIX_RE.match(param_key)
        if m is not None:
            all_layer_prefixes.add(m.group())

    for layer_prefix in all_layer_prefixes:
        # ---- Attention QKV + LN ----
        # input_layernorm.weight -> self_attention.layernorm_qkv.layer_norm_weight
        src = layer_prefix + "input_layernorm.weight"
        dst = layer_prefix + "self_attention.layernorm_qkv.layer_norm_weight"
        if src in hf_state_dict and dst in te_state_dict:
            te_state_dict[dst].data.copy_(hf_state_dict[src].data)

        # q_proj, k_proj, v_proj
        mapping_qkv = [
            ("self_attn.q_proj.weight", "self_attention.layernorm_qkv.query_weight"),
            ("self_attn.k_proj.weight", "self_attention.layernorm_qkv.key_weight"),
            ("self_attn.v_proj.weight", "self_attention.layernorm_qkv.value_weight"),
        ]
        for src_tail, dst_tail in mapping_qkv:
            src = layer_prefix + src_tail
            dst = layer_prefix + dst_tail
            if src in hf_state_dict and dst in te_state_dict:
                te_state_dict[dst].data.copy_(hf_state_dict[src].data)

        # o_proj
        src = layer_prefix + "self_attn.o_proj.weight"
        dst = layer_prefix + "self_attention.proj.weight"
        if src in hf_state_dict and dst in te_state_dict:
            te_state_dict[dst].data.copy_(hf_state_dict[src].data)

        # ---- Post-attn LayerNorm (for MLP block) ----
        src = layer_prefix + "post_attention_layernorm.weight"
        dst = layer_prefix + "layernorm_mlp.layer_norm_weight"
        if src in hf_state_dict and dst in te_state_dict:
            te_state_dict[dst].data.copy_(hf_state_dict[src].data)

        # ---- MLP packing (gate + up) -> fc1_weight (concat along dim 0) ----
        fc1 = layer_prefix + "layernorm_mlp.fc1_weight"
        gate = layer_prefix + "mlp.gate_proj.weight"
        up   = layer_prefix + "mlp.up_proj.weight"
        if fc1 in te_state_dict:
            fc1_w = te_state_dict[fc1]
            if gate in hf_state_dict:
                g = hf_state_dict[gate].data
                fc1_w.data[: config.intermediate_size].copy_(g)
            if up in hf_state_dict:
                u = hf_state_dict[up].data
                fc1_w.data[config.intermediate_size : config.intermediate_size * 2].copy_(u)

        # down_proj -> fc2_weight
        src = layer_prefix + "mlp.down_proj.weight"
        dst = layer_prefix + "layernorm_mlp.fc2_weight"
        if src in hf_state_dict and dst in te_state_dict:
            te_state_dict[dst].data.copy_(hf_state_dict[src].data)


# ---------------
# Quick sanity test
# ---------------
if __name__ == "__main__":
    # Minimal smoketest (creates a tiny model and runs a forward)
    cfg = LlamaConfig(
        vocab_size=32000,
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=2816,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
    )

    model = TELlamaForCausalLM(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    attn_mask = torch.ones(2, 16, device=device, dtype=torch.long)

    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True, output_attentions=True)
        logits = out.logits
        past = out.past_key_values
        print("logits:", tuple(logits.shape))
        # single-step with kv cache
        out2 = model(input_ids=input_ids[:, -1:], attention_mask=attn_mask[:, -1:], past_key_values=past, use_cache=True)
        print("next logits:", tuple(out2.logits.shape))
