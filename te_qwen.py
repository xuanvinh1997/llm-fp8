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

"""Flexible import strategy for a hypothetical Qwen3 lineage.

We alias the discovered Config/ForCausalLM classes to Qwen3Config / Qwen3ForCausalLM
so the rest of the code remains agnostic.
"""

qwen_mod = None
DecoderLayerName = None

import importlib

_IMPORT_CANDIDATES = [
  ("transformers.models.qwen3.modeling_qwen3",
   "Qwen3ForCausalLM", "Qwen3Config", "Qwen3DecoderLayer"),
  ("transformers.models.qwen2.modeling_qwen2",
   "Qwen2ForCausalLM", "Qwen2Config", "Qwen2DecoderLayer"),
]


for module_name, causal_cls, cfg_cls, decoder_cls in _IMPORT_CANDIDATES:
    try:
        mod = importlib.import_module(module_name)
        Qwen3ForCausalLM = getattr(mod, causal_cls)
        Qwen3Config = getattr(mod, cfg_cls)
        DecoderLayerName = decoder_cls
        qwen_mod = mod
        break
    except Exception:  # noqa: broad-except (robust to absence)
        continue

if qwen_mod is None:
    raise ImportError("Could not import any Qwen(3/2.5/2) implementation from transformers.")

import transformers
from transformers.modeling_utils import _add_variant, load_state_dict
from transformers.utils import WEIGHTS_INDEX_NAME
from transformers.utils.hub import get_checkpoint_shard_files

@contextmanager
def replace_decoder(te_decoder_cls):
    # Dynamically patch the discovered decoder layer class name
    original = getattr(qwen_mod, DecoderLayerName)
    setattr(qwen_mod, DecoderLayerName, te_decoder_cls)
    try:
        yield
    finally:
        setattr(qwen_mod, DecoderLayerName, original)

def _filter_kwargs_by_sig(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except (ValueError, TypeError):
        return kwargs

def _is_fp8_enabled_for_infer() -> bool:
    return os.getenv("ENABLE_FP8_INFER", "0").lower() in ("1", "true", "yes", "on")

attn_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
mlp_recipe = DelayedScaling(fp8_format=Format.E4M3,   amax_history_len=16, amax_compute_algo="max")

class TEQwen3DecoderLayer(torch.nn.Module):
    def __init__(self, config: Qwen3Config, *args, dropout_rate: float = 0.0, **kwargs):
        super().__init__()
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
        if hasattr(config, "num_key_value_heads") and config.num_key_value_heads is not None:
            mha_kwargs["num_gqa_groups"] = config.num_key_value_heads
            mha_kwargs["num_kv_heads"] = config.num_key_value_heads
        self.self_attention = te.pytorch.MultiheadAttention(**_filter_kwargs_by_sig(te.pytorch.MultiheadAttention, mha_kwargs))
        mlp_kwargs = dict(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            normalization="RMSNorm",
            activation="swiglu",
        )
        self.layernorm_mlp = te.pytorch.LayerNormMLP(**_filter_kwargs_by_sig(te.pytorch.LayerNormMLP, mlp_kwargs))
        head_dim = config.hidden_size // config.num_attention_heads
        rope_kwargs = {}
        if hasattr(config, "rope_theta"):
            rope_kwargs["rope_theta"] = config.rope_theta
        self.rope = RotaryPositionEmbedding(head_dim, **_filter_kwargs_by_sig(RotaryPositionEmbedding, rope_kwargs))
        self.register_buffer("rope_cache", None, persistent=False)
        self.register_buffer("rope_cache_len", torch.tensor(0, dtype=torch.long), persistent=False)

    def _ensure_rope(self, q_len: int, device: torch.device, position_ids: Optional[torch.Tensor] = None):
        # Calculate required cache length considering position_ids
        required_len = q_len
        if position_ids is not None:
            # Handle multi-dimensional position_ids
            if position_ids.dim() > 1:
                position_ids = position_ids.view(-1)
            max_pos = position_ids.max().item() + 1
            required_len = max(q_len, max_pos)
        
        need_new = (self.rope_cache is None or self.rope_cache.device != device or int(self.rope_cache_len.item()) < required_len)
        if need_new:
            cache = self.rope(max_seq_len=required_len)
            if isinstance(cache, torch.Tensor):
                cache = cache.to(device)
            elif isinstance(cache, (tuple, list)):
                cache = tuple(c.to(device) if isinstance(c, torch.Tensor) else c for c in cache)
            self.rope_cache = cache
            self.rope_cache_len.fill_(required_len)

    def _get_rope_slice(self, q_len: int, device: torch.device, position_ids: Optional[torch.Tensor] = None):
        self._ensure_rope(q_len, device, position_ids)
        rope_obj = self.rope_cache
        if position_ids is None:
            return rope_obj
        
        # Handle multi-dimensional position_ids
        original_shape = position_ids.shape
        if position_ids.dim() > 1:
            position_ids = position_ids.view(-1)
        
        # Clamp position_ids to prevent out of bounds access
        max_pos = self.rope_cache_len.item() - 1
        position_ids = torch.clamp(position_ids, 0, max_pos)
        
        if isinstance(rope_obj, torch.Tensor):
            sliced = rope_obj[:, position_ids, ...]
            # Restore original position_ids shape if needed
            if len(original_shape) > 1:
                new_shape = (sliced.shape[0],) + original_shape + sliced.shape[2:]
                sliced = sliced.view(new_shape)
            return sliced
        if isinstance(rope_obj, (tuple, list)) and isinstance(rope_obj[0], torch.Tensor):
            sliced_components = []
            for comp in rope_obj:
                sliced_comp = comp[:, position_ids, ...]
                if len(original_shape) > 1:
                    new_shape = (sliced_comp.shape[0],) + original_shape + sliced_comp.shape[2:]
                    sliced_comp = sliced_comp.view(new_shape)
                sliced_components.append(sliced_comp)
            return tuple(sliced_components)
        return rope_obj

    @staticmethod
    def _build_causal_bias(q_len: int, k_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Create causal mask: mask future positions (upper triangular)
        # For autoregressive generation with KV cache, q_len might be 1 but k_len > 1
        if q_len == 1:
            # During generation, we only attend to all previous tokens
            causal = torch.zeros((1, k_len), device=device, dtype=dtype)
        else:
            # Standard causal mask for training/prefill
            causal = torch.zeros((q_len, k_len), device=device, dtype=dtype)
            # Create upper triangular mask with -inf for future positions
            if q_len == k_len:
                # Standard case: mask upper triangle (future positions)
                causal = causal.masked_fill(torch.triu(torch.ones_like(causal, dtype=torch.bool), diagonal=1), float('-inf'))
            else:
                # Handle case where q_len != k_len (e.g., when using past_key_values)
                # We assume k_len >= q_len and that query positions are the last q_len positions
                for i in range(q_len):
                    # Position i in query corresponds to position (k_len - q_len + i) in key
                    start_mask = k_len - q_len + i + 1
                    if start_mask < k_len:
                        causal[i, start_mask:] = float('-inf')
        
        return causal.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _hf_to_te_mask(attention_mask: torch.Tensor, hidden_states: torch.Tensor, past_key_value: Optional[Tuple] = None) -> torch.Tensor:
        if attention_mask is None:
            return None
        
        device = hidden_states.device
        dtype = hidden_states.dtype
        bsz, q_len, _ = hidden_states.shape
        
        if attention_mask.dim() == 2:
            k_len = attention_mask.shape[1]
            
            # Adjust k_len if we have past_key_value
            if past_key_value is not None and len(past_key_value) >= 2:
                past_key = past_key_value[0]
                if past_key is not None:
                    past_seq_len = past_key.shape[-2]
                    k_len = past_seq_len + q_len
            
            # Build causal mask
            causal = TEQwen3DecoderLayer._build_causal_bias(q_len, k_len, device, dtype)
            
            # Add padding mask
            if attention_mask.shape[1] == k_len:
                pad = (1 - attention_mask).to(dtype) * float('-inf')
                pad = pad[:, None, None, :]
            else:
                # Handle case where attention_mask doesn't match k_len
                extended_mask = torch.ones(bsz, k_len, device=device, dtype=attention_mask.dtype)
                if attention_mask.shape[1] <= k_len:
                    extended_mask[:, -attention_mask.shape[1]:] = attention_mask
                else:
                    extended_mask = attention_mask[:, -k_len:]
                pad = (1 - extended_mask).to(dtype) * float('-inf')
                pad = pad[:, None, None, :]
            
            return causal + pad
            
        elif attention_mask.dim() == 4:
            return attention_mask.to(device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs):
        bsz, q_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Handle position_ids for KV cache scenarios
        if position_ids is not None:
            # Ensure position_ids has the right shape
            if position_ids.dim() == 1:
                # Single dimension - expand to match batch size
                position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
            elif position_ids.dim() == 2:
                # Two dimensions - check if we need to expand batch dimension
                if position_ids.shape[0] == 1 and bsz > 1:
                    # Single batch entry - expand to match batch size
                    position_ids = position_ids.expand(bsz, -1)
                elif position_ids.shape[0] != bsz:
                    # Different batch sizes that can't be broadcast
                    if position_ids.shape[0] == 1:
                        position_ids = position_ids.expand(bsz, -1)
                    else:
                        raise ValueError(f"position_ids batch size {position_ids.shape[0]} cannot be broadcast to hidden_states batch size {bsz}")
            
            # Ensure position_ids sequence length matches hidden_states
            if position_ids.shape[1] != q_len:
                if position_ids.shape[1] == 1 and q_len == 1:
                    # Single token generation case - this is fine
                    pass
                else:
                    raise ValueError(f"position_ids seq length {position_ids.shape[1]} != hidden_states seq length {q_len}")
        
        rope = self._get_rope_slice(q_len, device, position_ids)
        te_mask = self._hf_to_te_mask(attention_mask, hidden_states, past_key_value)
        fp8_ok = getattr(te.pytorch, "is_fp8_available", lambda: True)()
        fp8_enabled = fp8_ok and (self.training or _is_fp8_enabled_for_infer())
        attn_call_kwargs = dict(attention_mask=te_mask, rotary_pos_emb=rope, past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions)
        attn_call_kwargs = _filter_kwargs_by_sig(self.self_attention.forward, attn_call_kwargs)
        with te.pytorch.fp8_autocast(enabled=fp8_enabled, fp8_recipe=attn_recipe):
            try:
                out = self.self_attention(hidden_states, **attn_call_kwargs)
            except TypeError:
                out = self.self_attention(hidden_states, attention_mask=te_mask, rotary_pos_emb=rope)
        if isinstance(out, torch.Tensor):
            attn_out = out
            present_kv = None
            attn_probs = None
        elif isinstance(out, (tuple, list)):
            attn_out = out[0]
            present_kv = out[1] if len(out) > 1 else None
            attn_probs = out[2] if len(out) > 2 else None
        else:
            raise RuntimeError("Unexpected return from TE MultiheadAttention")
        hidden_states = hidden_states + attn_out
        with te.pytorch.fp8_autocast(enabled=fp8_enabled, fp8_recipe=mlp_recipe):
            ffn_out = self.layernorm_mlp(hidden_states)
        hidden_states = hidden_states + ffn_out
        output = (hidden_states,)
        if output_attentions:
            output += (attn_probs,)
        if use_cache:
            output += (present_kv,)
        return output

class TEQwen3ForCausalLM:
    def __new__(cls, config: Qwen3Config):
        with replace_decoder(te_decoder_cls=TEQwen3DecoderLayer):
            model = Qwen3ForCausalLM(config)
        return model

    @classmethod
    def from_pretrained_local(cls, pretrained_model_name_or_path: str, *args, config: Qwen3Config, torch_dtype=torch.float16, **kwargs):
        prev_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch_dtype)
            model = cls(config)
            subfolder = ""
            variant = None
            def _exists(fname: str) -> bool:
                return os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(fname, variant)))
            if _exists("model.safetensors.index.json"):
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant("model.safetensors.index.json", variant))
                is_sharded = True
            elif _exists(WEIGHTS_INDEX_NAME):
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
                is_sharded = True
            elif _exists("model.safetensors"):
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant("model.safetensors", variant))
                is_sharded = False
            elif _exists("pytorch_model.bin"):
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant("pytorch_model.bin", variant))
                is_sharded = False
            else:
                raise AssertionError("Checkpoint not found")
            if not is_sharded:
                resolved_archive_files = [archive_file]
            else:
                resolved_archive_files, _ = get_checkpoint_shard_files(pretrained_model_name_or_path, archive_file)
            for shard in resolved_archive_files:
                state_dict = load_state_dict(shard)
                replace_params(state_dict, model.state_dict(), config)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    print(f"[TEQwen3] load_state_dict shard='{os.path.basename(shard)}' missing={len(missing)} unexpected={len(unexpected)}")
                del state_dict
                gc.collect()
            return model
        finally:
            torch.set_default_dtype(prev_dtype)

_LAYER_PREFIX_RE = re.compile(r"model\.layers\.\d+\.")

def replace_params(hf_state_dict: Dict[str, torch.Tensor], te_state_dict: Dict[str, torch.Tensor], config: Qwen3Config):
    all_layer_prefixes = set()
    for k in hf_state_dict.keys():
        m = _LAYER_PREFIX_RE.match(k)
        if m is not None:
            all_layer_prefixes.add(m.group())
    for layer_prefix in all_layer_prefixes:
        src = layer_prefix + "input_layernorm.weight"
        dst = layer_prefix + "self_attention.layernorm_qkv.layer_norm_weight"
        if src in hf_state_dict and dst in te_state_dict:
            te_state_dict[dst].data.copy_(hf_state_dict[src].data)
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
        src = layer_prefix + "self_attn.o_proj.weight"
        dst = layer_prefix + "self_attention.proj.weight"
        if src in hf_state_dict and dst in te_state_dict:
            te_state_dict[dst].data.copy_(hf_state_dict[src].data)
        src = layer_prefix + "post_attention_layernorm.weight"
        dst = layer_prefix + "layernorm_mlp.layer_norm_weight"
        if src in hf_state_dict and dst in te_state_dict:
            te_state_dict[dst].data.copy_(hf_state_dict[src].data)
        fc1 = layer_prefix + "layernorm_mlp.fc1_weight"
        gate = layer_prefix + "mlp.gate_proj.weight"
        up = layer_prefix + "mlp.up_proj.weight"
        if fc1 in te_state_dict:
            fc1_w = te_state_dict[fc1]
            if gate in hf_state_dict:
                g = hf_state_dict[gate].data
                fc1_w.data[: config.intermediate_size].copy_(g)
            if up in hf_state_dict:
                u = hf_state_dict[up].data
                fc1_w.data[config.intermediate_size: config.intermediate_size * 2].copy_(u)
        src = layer_prefix + "mlp.down_proj.weight"
        dst = layer_prefix + "layernorm_mlp.fc2_weight"
        if src in hf_state_dict and dst in te_state_dict:
            te_state_dict[dst].data.copy_(hf_state_dict[src].data)

if __name__ == "__main__":
    cfg = Qwen3Config(
        vocab_size=32000,
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=2816,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
    )
    model = TEQwen3ForCausalLM(cfg)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    attn_mask = torch.ones(2, 16, device=device, dtype=torch.long)
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True, output_attentions=True)
        logits = out.logits
        past = out.past_key_values
        print('logits:', tuple(logits.shape))
        out2 = model(input_ids=input_ids[:, -1:], attention_mask=attn_mask[:, -1:], past_key_values=past, use_cache=True)
        print('next logits:', tuple(out2.logits.shape))
