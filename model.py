import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Dict, Any
import warnings

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

try:
    from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("transformers library not available. HuggingFace model loading will not work.")

from kernel import FP8_E4M3, FP8_E5M2, act_quant, weight_dequant, fp8_gemm


world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.
    This configuration is set up for dense (non-MoE) transformer models.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers (unused in dense model).
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model (all layers in dense model).
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers (unused in dense model).
        n_shared_experts (int): Number of shared experts for MoE layers (unused in dense model).
        n_activated_experts (int): Number of activated experts in MoE layers (unused in dense model).
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 27  # All layers are dense in non-MoE model
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and 
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve 
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version 
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    # model.py – inside linear()
    elif gemm_impl == "fp8":
        # Forward path (Fprop) → E4M3 ----------------
        x_q , x_s  = act_quant(x, block_size, dtype=FP8_E4M3)
        y    = fp8_gemm(x_q, x_s, weight, weight.scale)

        if torch.is_grad_enabled():
            # Register a backward hook that re-quantises the saved activations to E5M2
            ctx = torch.autograd.graph.save_on_cpu
            saved = x.detach()        # keep FP32 master copy
            def bw_hook(grad):
                # 1️⃣ quantise incoming gradient (dY) to E5M2
                grad_q, grad_s = act_quant(grad, block_size, dtype=FP8_E5M2)
                # Dgrad / Wgrad expect E5M2 on A or B inputs
                x_qg , x_sg = act_quant(saved, block_size, dtype=FP8_E5M2)
                
                return fp8_gemm(grad_q, grad_s, x_qg, x_sg)
            y.register_hook(bw_hook)

        if bias is not None:
            y += bias
        return y
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.
    This implementation uses dense MLP layers for all transformer blocks.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MLA(args)
        # Use MLP for all layers in dense model
        self.ffn = MLP(args.dim, args.inter_dim)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits


def load_hf_model_and_convert(
    model_name_or_path: str,
    target_args: Optional[ModelArgs] = None,
    convert_to_fp8: bool = True,
    trust_remote_code: bool = False
) -> Transformer:
    """
    Load a Hugging Face model and convert it to our custom Transformer with FP8 support.
    
    Args:
        model_name_or_path (str): Path or name of the Hugging Face model
        target_args (Optional[ModelArgs]): Target model configuration. If None, will be inferred from HF config
        convert_to_fp8 (bool): Whether to convert weights to FP8 format
        trust_remote_code (bool): Whether to trust remote code when loading the model
        
    Returns:
        Transformer: Our custom transformer model with converted weights
        
    Raises:
        ImportError: If transformers library is not available
        ValueError: If model architecture is not supported
    """
    if not HF_AVAILABLE:
        raise ImportError("transformers library is required for loading HuggingFace models. Install with: pip install transformers")
    
    print(f"Loading model from {model_name_or_path}...")
    
    # Load HuggingFace model and config
    hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    
    # Try to load a causal LM model first (includes lm_head), fallback to base model
    try:
        print("Attempting to load as AutoModelForCausalLM (includes lm_head)...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            config=hf_config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16
        )
        hf_model.to("cuda")
        print("Successfully loaded CausalLM model with language modeling head")
    except Exception as e:
        print(f"Failed to load as CausalLM: {e}")
        print("Falling back to AutoModel (base model without lm_head)...")
        hf_model = AutoModel.from_pretrained(
            model_name_or_path, 
            config=hf_config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16
        )
        print("Loaded base model - note: this may not include lm_head")
    
    # Create target model args based on HF config if not provided
    if target_args is None:
        target_args = _create_model_args_from_hf_config(hf_config)
    
    # Set dtype based on conversion preference
    if convert_to_fp8:
        target_args.dtype = "fp8"
    else:
        target_args.dtype = "bf16"
    
    print(f"Creating target model with config: {target_args}")
    
    # Create our custom model
    custom_model = Transformer(target_args)
    
    # Convert and load weights
    _convert_and_load_weights(hf_model, custom_model, hf_config, convert_to_fp8)
    
    print("Model conversion completed successfully!")
    return custom_model


def _create_model_args_from_hf_config(hf_config: Any) -> ModelArgs:
    """
    Create ModelArgs from Hugging Face model configuration.
    
    Args:
        hf_config: Hugging Face model configuration
        
    Returns:
        ModelArgs: Converted model arguments
    """
    # Common mappings for different model architectures
    config_mappings = {
        'vocab_size': getattr(hf_config, 'vocab_size', 102400),
        'dim': getattr(hf_config, 'hidden_size', 2048),
        'n_layers': getattr(hf_config, 'num_hidden_layers', 27),
        'n_heads': getattr(hf_config, 'num_attention_heads', 16),
        'max_seq_len': getattr(hf_config, 'max_position_embeddings', 4096 * 4),
    }
    
    # Try to map intermediate dimension
    inter_dim = getattr(hf_config, 'intermediate_size', None)
    if inter_dim is None:
        # Common pattern: intermediate_size = 4 * hidden_size for many models
        inter_dim = config_mappings['dim'] * 4
    config_mappings['inter_dim'] = inter_dim
    
    # Create ModelArgs with mapped values
    args = ModelArgs()
    for key, value in config_mappings.items():
        if hasattr(args, key):
            setattr(args, key, value)
    
    # Ensure all layers are dense
    args.n_dense_layers = args.n_layers
    
    # Adjust MLA parameters based on HF config for better compatibility
    hf_dim = config_mappings['dim']
    hf_n_heads = config_mappings['n_heads']
    
    # Calculate standard head dimension
    standard_head_dim = hf_dim // hf_n_heads if hf_n_heads > 0 else 128
    
    # Adjust MLA dimensions to be more compatible with standard attention
    # Keep the total dimension close to what standard attention would expect
    total_qk_dim = standard_head_dim * hf_n_heads
    
    # Split this between nope and rope components
    args.qk_nope_head_dim = int(standard_head_dim * 0.7)  # 70% for non-positional
    args.qk_rope_head_dim = standard_head_dim - args.qk_nope_head_dim  # Rest for rotary
    args.v_head_dim = standard_head_dim  # Keep value head same as standard
    
    # Adjust kv_lora_rank to be proportional to dimension
    args.kv_lora_rank = min(512, hf_dim // 4)  # Cap at 512 but scale with model size
    
    print(f"Inferred config from HF model: vocab_size={args.vocab_size}, dim={args.dim}, "
          f"n_layers={args.n_layers}, n_heads={args.n_heads}, inter_dim={args.inter_dim}")
    print(f"MLA config: qk_nope_head_dim={args.qk_nope_head_dim}, qk_rope_head_dim={args.qk_rope_head_dim}, "
          f"v_head_dim={args.v_head_dim}, kv_lora_rank={args.kv_lora_rank}")
    
    return args


def _convert_and_load_weights(
    hf_model: nn.Module, 
    custom_model: Transformer, 
    hf_config: Any,
    convert_to_fp8: bool = True
) -> None:
    """
    Convert and load weights from HuggingFace model to our custom model.
    
    Args:
        hf_model: Source HuggingFace model
        custom_model: Target custom model
        hf_config: HuggingFace model configuration
        convert_to_fp8: Whether to convert weights to FP8
    """
    print("Converting and loading weights...")
    
    hf_state_dict = hf_model.state_dict()
    custom_state_dict = custom_model.state_dict()
    
    # Debug: Print all HuggingFace model keys to understand the structure
    print("HuggingFace model keys:")
    hf_keys = list(hf_state_dict.keys())
    for i, key in enumerate(sorted(hf_keys)):
        if i < 20:  # Print first 20 keys
            print(f"  {key}")
        elif i == 20:
            print(f"  ... and {len(hf_keys) - 20} more keys")
            break
    
    # Debug: Print some custom model keys for comparison
    print("\nCustom model keys (first 10):")
    custom_keys = list(custom_state_dict.keys())
    for key in sorted(custom_keys)[:10]:
        print(f"  {key}")
    
    # Create weight mapping
    weight_mapping = _create_weight_mapping(hf_config, hf_state_dict)
    
    # Extract layer pattern for MLA construction
    layer_pattern = None
    for pattern in ['layers.{}.{}', 'model.layers.{}.{}', 'transformer.h.{}.{}']:
        test_key = pattern.format(0, 'self_attn.q_proj.weight')
        if test_key in hf_state_dict:
            layer_pattern = pattern
            break
    if not layer_pattern:
        layer_pattern = 'layers.{}.{}'  # fallback
    
    converted_weights = {}
    missing_mappings = []
    shape_mismatches = []
    
    for custom_key, custom_param in custom_state_dict.items():
        # Skip scale parameters for now - they will be handled when converting weights to FP8
        if custom_key.endswith('.scale'):
            continue
        
        # Check if this is an MLA weight that needs special construction
        if _needs_mla_construction(custom_key):
            # Extract layer index for MLA construction
            layer_idx = None
            if 'layers.' in custom_key:
                try:
                    layer_idx = int(custom_key.split('.')[1])
                except (IndexError, ValueError):
                    pass
            
            if layer_idx is not None:
                try:
                    constructed_weight = _construct_mla_weight(custom_key, custom_param, hf_state_dict, 
                                                             layer_pattern, layer_idx, hf_config)
                    if convert_to_fp8 and 'weight' in custom_key and custom_param.element_size() == 1:
                        converted_weight, scale = _convert_to_fp8(constructed_weight, custom_param.shape)
                        converted_weights[custom_key] = converted_weight
                        scale_key = custom_key.replace('.weight', '.scale')
                        if scale_key in custom_state_dict:
                            converted_weights[scale_key] = scale
                    else:
                        converted_weights[custom_key] = constructed_weight.to(custom_param.dtype).to(custom_param.device)
                    continue
                except Exception as e:
                    print(f"Error constructing MLA weight {custom_key}: {e}")
                    missing_mappings.append(custom_key)
                    continue
            
        if custom_key in weight_mapping:
            hf_key = weight_mapping[custom_key]
            if hf_key in hf_state_dict:
                hf_weight = hf_state_dict[hf_key]
                
                # Special handling for attention weights that need adaptation
                if _needs_attention_adaptation(custom_key, hf_weight, custom_param):
                    print(f"Adapting attention weight {custom_key}")
                    try:
                        adapted_weight = _adapt_attention_weight(custom_key, hf_weight, custom_param, hf_config)
                        if convert_to_fp8 and 'weight' in custom_key and custom_param.element_size() == 1:
                            converted_weight, scale = _convert_to_fp8(adapted_weight, custom_param.shape)
                            converted_weights[custom_key] = converted_weight
                            scale_key = custom_key.replace('.weight', '.scale')
                            if scale_key in custom_state_dict:
                                converted_weights[scale_key] = scale
                        else:
                            converted_weights[custom_key] = adapted_weight.to(custom_param.dtype)
                    except Exception as e:
                        print(f"Error adapting {custom_key}: {e}")
                        shape_mismatches.append((custom_key, hf_weight.shape, custom_param.shape))
                else:
                    # Convert weight to appropriate format
                    if convert_to_fp8 and 'weight' in custom_key and custom_param.element_size() == 1:
                        # Convert to FP8 if target parameter is quantized
                        print(f"Converting {custom_key}: HF shape {hf_weight.shape} -> Custom shape {custom_param.shape}")
                        try:
                            # Ensure weight is on the correct device before conversion
                            if hf_weight.device != custom_param.device:
                                hf_weight = hf_weight.to(custom_param.device)
                            
                            converted_weight, scale = _convert_to_fp8(hf_weight, custom_param.shape)
                            converted_weights[custom_key] = converted_weight
                            
                            # Also set the corresponding scale parameter
                            scale_key = custom_key.replace('.weight', '.scale')
                            if scale_key in custom_state_dict:
                                converted_weights[scale_key] = scale
                        except ValueError as e:
                            print(f"Error converting {custom_key}: {e}")
                            shape_mismatches.append((custom_key, hf_weight.shape, custom_param.shape))
                    else:
                        # Direct copy with shape matching
                        if hf_weight.shape == custom_param.shape:
                            converted_weights[custom_key] = hf_weight.to(custom_param.dtype).to(custom_param.device)
                        else:
                            print(f"Shape mismatch for {custom_key}: HF {hf_weight.shape} vs Custom {custom_param.shape}")
                            shape_mismatches.append((custom_key, hf_weight.shape, custom_param.shape))
            else:
                print(f"Warning: HF key {hf_key} not found for custom key {custom_key}")
        else:
            missing_mappings.append(custom_key)
    
    # Print summary of what couldn't be loaded
    if missing_mappings:
        print(f"\nWarning: {len(missing_mappings)} parameters have no mapping:")
        for key in missing_mappings[:10]:  # Show first 10
            print(f"  {key}")
        if len(missing_mappings) > 10:
            print(f"  ... and {len(missing_mappings) - 10} more")
    
    if shape_mismatches:
        print(f"\nWarning: {len(shape_mismatches)} parameters have shape mismatches:")
        for custom_key, hf_shape, custom_shape in shape_mismatches[:5]:  # Show first 5
            print(f"  {custom_key}: HF shape {hf_shape} vs Custom shape {custom_shape}")
        if len(shape_mismatches) > 5:
            print(f"  ... and {len(shape_mismatches) - 5} more")
    
    # Load converted weights
    custom_model.load_state_dict(converted_weights, strict=False)
    
    print(f"Successfully loaded {len(converted_weights)} weight tensors")


def _needs_attention_adaptation(custom_key: str, hf_weight: torch.Tensor, custom_param: torch.Tensor) -> bool:
    """
    Check if an attention weight needs adaptation due to architectural differences.
    """
    # Check if it's an attention weight with shape mismatch
    if '.attn.' in custom_key and 'weight' in custom_key:
        return hf_weight.shape != custom_param.shape
    return False


def _needs_mla_construction(custom_key: str) -> bool:
    """
    Check if this is an MLA weight that needs to be constructed from HF weights.
    """
    mla_components = ['.attn.wkv_a.weight', '.attn.wkv_b.weight', '.attn.kv_norm.weight', '.attn.q_norm.weight']
    return any(component in custom_key for component in mla_components)


def _construct_mla_weight(custom_key: str, custom_param: torch.Tensor, hf_state_dict: Dict[str, torch.Tensor], 
                         layer_pattern: str, layer_idx: int, hf_config: Any) -> torch.Tensor:
    """
    Construct MLA weights (wkv_a, wkv_b) from standard attention weights (k_proj, v_proj).
    """
    print(f"Constructing MLA weight {custom_key}")
    
    target_device = custom_param.device
    
    # Get HF key/value projection weights
    k_proj_key = layer_pattern.format(layer_idx, 'self_attn.k_proj.weight')
    v_proj_key = layer_pattern.format(layer_idx, 'self_attn.v_proj.weight')
    
    if '.attn.wkv_a.weight' in custom_key:
        # wkv_a should have shape [kv_lora_rank + qk_rope_head_dim, dim]
        # For now, initialize randomly since HF doesn't have an equivalent
        print(f"Initializing wkv_a weight with random values (shape: {custom_param.shape})")
        # Use small random initialization similar to standard linear layers
        std = (2.0 / (custom_param.shape[0] + custom_param.shape[1])) ** 0.5
        return torch.randn_like(custom_param) * std
        
    elif '.attn.wkv_b.weight' in custom_key:
        # wkv_b should have shape [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
        # We can try to adapt from v_proj, but dimensions likely won't match
        if v_proj_key in hf_state_dict:
            v_proj = hf_state_dict[v_proj_key]
            print(f"Adapting wkv_b from v_proj: {v_proj.shape} -> {custom_param.shape}")
            
            # If we can reshape, do it
            if v_proj.numel() == custom_param.numel():
                return v_proj.view(custom_param.shape).to(target_device)
            else:
                # Initialize randomly since dimensions don't match
                print(f"Cannot adapt v_proj to wkv_b due to size mismatch, using random initialization")
                std = (2.0 / (custom_param.shape[0] + custom_param.shape[1])) ** 0.5
                return torch.randn_like(custom_param) * std
        else:
            print(f"v_proj not found, initializing wkv_b with random values")
            std = (2.0 / (custom_param.shape[0] + custom_param.shape[1])) ** 0.5
            return torch.randn_like(custom_param) * std
    
    elif '.attn.kv_norm.weight' in custom_key:
        # Initialize layer norm weights to ones (standard initialization)
        print(f"Initializing kv_norm weight with ones (shape: {custom_param.shape})")
        return torch.ones_like(custom_param)
    
    elif '.attn.q_norm.weight' in custom_key:
        # Initialize layer norm weights to ones (standard initialization) 
        print(f"Initializing q_norm weight with ones (shape: {custom_param.shape})")
        return torch.ones_like(custom_param)
    
    # Fallback
    return torch.zeros_like(custom_param)


def _adapt_attention_weight(custom_key: str, hf_weight: torch.Tensor, custom_param: torch.Tensor, hf_config: Any) -> torch.Tensor:
    """
    Adapt attention weights from HuggingFace format to custom MLA format.
    """
    print(f"Adapting {custom_key}: {hf_weight.shape} -> {custom_param.shape}")
    
    # Ensure we're working with the right device
    target_device = custom_param.device
    
    # For query weights that need expansion or contraction
    if '.attn.wq.weight' in custom_key:
        # HF format: [out_features, in_features] 
        # Custom format might expect different out_features due to MLA head dimensions
        custom_out, custom_in = custom_param.shape
        hf_out, hf_in = hf_weight.shape
        
        if custom_in != hf_in:
            raise ValueError(f"Input dimension mismatch: HF {hf_in} vs Custom {custom_in}")
        
        if custom_out > hf_out:
            # Need to expand - pad with zeros or repeat
            print(f"Expanding query weight from {hf_out} to {custom_out}")
            expanded = torch.zeros(custom_out, custom_in, dtype=hf_weight.dtype, device=target_device)
            expanded[:hf_out, :] = hf_weight.to(target_device)
            return expanded
        elif custom_out < hf_out:
            # Need to contract - truncate
            print(f"Contracting query weight from {hf_out} to {custom_out}")
            return hf_weight[:custom_out, :].to(target_device)
        else:
            return hf_weight.to(target_device)
    
    # For other attention weights, try simple reshaping if elements match
    if hf_weight.numel() == custom_param.numel():
        print(f"Reshaping {custom_key} from {hf_weight.shape} to {custom_param.shape}")
        return hf_weight.view(custom_param.shape).to(target_device)
    
    # If we can't adapt, zero-initialize with a warning
    print(f"Warning: Cannot adapt {custom_key}, initializing with zeros")
    return torch.zeros_like(custom_param)


def _create_weight_mapping(hf_config: Any, hf_state_dict: Dict[str, torch.Tensor] = None) -> Dict[str, str]:
    """
    Create mapping between our custom model keys and HuggingFace model keys.
    Supports multiple architectures and automatically detects the correct pattern.
    
    Args:
        hf_config: HuggingFace model configuration
        hf_state_dict: Optional HuggingFace state dict to check available keys
        
    Returns:
        Dict[str, str]: Mapping from custom keys to HF keys
    """
    mapping = {}
    
    # Get available keys to determine the actual structure
    available_keys = set(hf_state_dict.keys()) if hf_state_dict else set()
    
    print(f"Detecting model architecture from {len(available_keys)} available keys...")
    
    # Try different possible embedding patterns
    embed_patterns = [
        'embed_tokens.weight',
        'model.embed_tokens.weight',
        'embeddings.word_embeddings.weight',
        'transformer.wte.weight'
    ]
    
    for pattern in embed_patterns:
        if pattern in available_keys:
            mapping['embed.weight'] = pattern
            print(f"Found embedding pattern: {pattern}")
            break
    
    # Try different possible layer norm patterns for final norm
    final_norm_patterns = [
        'norm.weight',
        'model.norm.weight', 
        'layer_norm.weight',
        'transformer.ln_f.weight'
    ]
    
    final_norm_found = False
    for pattern in final_norm_patterns:
        if pattern in available_keys:
            mapping['norm.weight'] = pattern
            print(f"Found final norm pattern: {pattern}")
            final_norm_found = True
            break
    
    if not final_norm_found:
        print("Note: No final norm found in HF model - this is expected for some model architectures")
    
    # Try different possible output head patterns
    head_patterns = [
        'lm_head.weight',
        'output.weight',
        'classifier.weight',
        'head.weight'
    ]
    
    head_found = False
    for pattern in head_patterns:
        if pattern in available_keys:
            mapping['head.weight'] = pattern
            print(f"Found head pattern: {pattern}")
            head_found = True
            break
    
    if not head_found:
        print("Note: No output head found in HF model")
        print("  This is normal if you loaded a base model without language modeling head")
        print("  The custom model's head will be randomly initialized")
    
    # Layer mapping for transformer blocks
    num_layers = getattr(hf_config, 'num_hidden_layers', 27)
    
    # Try different layer patterns
    layer_patterns = [
        'layers.{}.{}',  # Direct layers pattern (what we see in the output)
        'model.layers.{}.{}',
        'transformer.h.{}.{}',
        'encoder.layer.{}.{}'
    ]
    
    # Attention sub-patterns
    attn_patterns = {
        'q_proj': ['self_attn.q_proj.weight', 'attention.self.query.weight', 'attn.q_proj.weight'],
        'k_proj': ['self_attn.k_proj.weight', 'attention.self.key.weight', 'attn.k_proj.weight'],
        'v_proj': ['self_attn.v_proj.weight', 'attention.self.value.weight', 'attn.v_proj.weight'],
        'o_proj': ['self_attn.o_proj.weight', 'attention.output.dense.weight', 'attn.o_proj.weight']
    }
    
    # MLP sub-patterns
    mlp_patterns = {
        'gate_proj': ['mlp.gate_proj.weight', 'mlp.dense_h_to_4h.weight', 'feed_forward.w1.weight'],
        'up_proj': ['mlp.up_proj.weight', 'mlp.dense_h_to_4h_2.weight', 'feed_forward.w3.weight'],
        'down_proj': ['mlp.down_proj.weight', 'mlp.dense_4h_to_h.weight', 'feed_forward.w2.weight']
    }
    
    # Layer norm sub-patterns
    norm_patterns = {
        'input_layernorm': ['input_layernorm.weight', 'attention.output.LayerNorm.weight', 'ln_1.weight'],
        'post_attention_layernorm': ['post_attention_layernorm.weight', 'output.LayerNorm.weight', 'ln_2.weight']
    }
    
    # Find the correct layer pattern
    found_layer_pattern = None
    for layer_pattern in layer_patterns:
        test_key = layer_pattern.format(0, 'self_attn.q_proj.weight')
        if test_key in available_keys:
            found_layer_pattern = layer_pattern
            print(f"Found layer pattern: {layer_pattern}")
            break
        # Try alternative patterns
        test_key = layer_pattern.format(0, 'attention.self.query.weight')
        if test_key in available_keys:
            found_layer_pattern = layer_pattern
            print(f"Found layer pattern: {layer_pattern}")
            break
    
    # If we didn't find a pattern from available keys, default to the simple pattern
    if not found_layer_pattern:
        found_layer_pattern = 'layers.{}.{}'
        print(f"Using default layer pattern: {found_layer_pattern}")
    
    # Map layer weights
    for layer_idx in range(num_layers):
        layer_prefix = f'layers.{layer_idx}'
        
        # Map attention weights - handle MLA architecture carefully
        for custom_attn_key, hf_patterns in attn_patterns.items():
            for hf_pattern in hf_patterns:
                full_hf_key = found_layer_pattern.format(layer_idx, hf_pattern)
                if full_hf_key in available_keys:
                    # Map based on custom attention structure
                    if custom_attn_key == 'q_proj':
                        # For MLA, we use wq for direct query projection when q_lora_rank == 0
                        mapping[f'{layer_prefix}.attn.wq.weight'] = full_hf_key
                    elif custom_attn_key == 'o_proj':
                        mapping[f'{layer_prefix}.attn.wo.weight'] = full_hf_key
                    # Note: k_proj and v_proj will be handled separately as they need special adaptation
                    # for MLA's wkv_a and wkv_b structure
                    break
        
        # Map MLP weights
        for custom_mlp_key, hf_patterns in mlp_patterns.items():
            for hf_pattern in hf_patterns:
                full_hf_key = found_layer_pattern.format(layer_idx, hf_pattern)
                if full_hf_key in available_keys:
                    if custom_mlp_key == 'gate_proj':
                        mapping[f'{layer_prefix}.ffn.w1.weight'] = full_hf_key
                    elif custom_mlp_key == 'down_proj':
                        mapping[f'{layer_prefix}.ffn.w2.weight'] = full_hf_key
                    elif custom_mlp_key == 'up_proj':
                        mapping[f'{layer_prefix}.ffn.w3.weight'] = full_hf_key
                    break
        
        # Map layer norms
        for custom_norm_key, hf_patterns in norm_patterns.items():
            for hf_pattern in hf_patterns:
                full_hf_key = found_layer_pattern.format(layer_idx, hf_pattern)
                if full_hf_key in available_keys:
                    if custom_norm_key == 'input_layernorm':
                        mapping[f'{layer_prefix}.attn_norm.weight'] = full_hf_key
                    elif custom_norm_key == 'post_attention_layernorm':
                        mapping[f'{layer_prefix}.ffn_norm.weight'] = full_hf_key
                    break
    
    print(f"Created {len(mapping)} weight mappings")
    return mapping


def _convert_to_fp8(weight: torch.Tensor, target_shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a weight tensor to FP8 format.
    
    Args:
        weight: Input weight tensor
        target_shape: Target shape for the converted weight
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: FP8 quantized weight tensor and its scale
    """
    # Check if shapes are compatible
    if weight.numel() != target_shape.numel():
        raise ValueError(f"Cannot reshape weight: source has {weight.numel()} elements "
                        f"but target shape {target_shape} requires {target_shape.numel()} elements. "
                        f"Source shape: {weight.shape}, Target shape: {target_shape}")
    
    # Ensure weight is in the right shape
    if weight.shape != target_shape:
        print(f"Reshaping weight from {weight.shape} to {target_shape}")
        weight = weight.view(target_shape)
    
    # Ensure weight is on GPU for Triton operations
    if weight.device.type == 'cpu':
        print(f"Moving weight to GPU for FP8 conversion (was on {weight.device})")
        weight = weight.cuda()
    
    # Quantize to FP8
    weight_q, weight_scale = act_quant(weight, block_size, dtype=FP8_E4M3)
    
    return weight_q, weight_scale


def load_tokenizer_from_hf(model_name_or_path: str, **kwargs):
    """
    Load tokenizer from Hugging Face model.
    
    Args:
        model_name_or_path (str): Path or name of the Hugging Face model
        **kwargs: Additional arguments for tokenizer loading
        
    Returns:
        Tokenizer from Hugging Face
    """
    if not HF_AVAILABLE:
        raise ImportError("transformers library is required. Install with: pip install transformers")
    
    return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)


# if __name__ == "__main__":
#     torch.set_default_dtype(torch.bfloat16)
#     torch.set_default_device("cuda")
#     torch.manual_seed(0)
#     args = ModelArgs()
#     x = torch.randint(0, args.vocab_size, (2, 128))
#     model = Transformer(args)
#     print(model(x).size())

