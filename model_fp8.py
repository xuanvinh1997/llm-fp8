import math
from typing import Optional
from typing_extensions import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

FP8_E4M3 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2

def apply_rope(x, seq_len):
    # x: (B, T, D)
    B, T, D = x.shape
    assert D % 2 == 0, "Embedding dim must be even for RoPE."
    half_dim = D // 2
    freq = torch.arange(half_dim, device=x.device) / half_dim
    freq = 10000 ** (-freq)
    pos = torch.arange(seq_len, device=x.device)
    angles = pos[:, None] * freq[None, :]
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    sin = sin[None, :, :].repeat(B, 1, 1)
    cos = cos[None, :, :].repeat(B, 1, 1)
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated

def get_maxval(fmt):
    return 448. if fmt == FP8_E4M3 else 57344.

@triton.jit
def quantize_kernel(x_ptr, y_ptr, s_ptr, N, BLOCK_SIZE: tl.constexpr, MAXVAL: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.max(tl.abs(x)) / MAXVAL
    y = (x / s).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid, s)

@triton.jit
def dequantize_kernel(x_ptr, y_ptr, s_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)

def quantize_fp8_triton(x: torch.Tensor, fmt=torch.float8_e4m3fn):
    assert x.is_contiguous() and x.dim() == 1
    N = x.shape[0]
    MAXVAL = get_maxval(fmt)
    y = torch.empty_like(x, dtype=fmt)
    s = torch.empty((N // BLOCK_SIZE,), dtype=torch.float32, device=x.device)
    grid = lambda META: (N // BLOCK_SIZE,)
    quantize_kernel[grid](x, y, s, N, BLOCK_SIZE=BLOCK_SIZE, MAXVAL=MAXVAL)
    return y, s

def dequantize_fp8_triton(x_fp8: torch.Tensor, s: torch.Tensor):
    N = x_fp8.shape[0]
    y = torch.empty_like(x_fp8, dtype=torch.float32)
    grid = lambda META: (N // BLOCK_SIZE,)
    dequantize_kernel[grid](x_fp8, y, s, N, BLOCK_SIZE=BLOCK_SIZE)
    return y



fp8_gemm_configs = [
    triton.Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] for block_n in [32, 64, 128] for num_stages in [3, 4, 5, 6]
]

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), 'Input tensors must be contiguous'
    assert a_s.is_contiguous() and b_s.is_contiguous(), 'Scaling factor tensors must be contiguous'
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c
world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "fp8"
attn_impl: Literal["naive", "absorb"] = "absorb"
import torch
import torch.nn as nn
from kernel import act_quant, fp8_gemm


class FP8Linear(nn.Module):
    """
    FP8Linear layer with FP8-weight storage and FP8-based GEMM for both training and inference.

    Attributes:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If True, adds a learnable bias to the output.
        block_size (int): Block size for FP8 quantization (default: 128).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, block_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        # FP8 weight tensor
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn)
        )
        # block-wise scale parameters
        scale_rows = (out_features + block_size - 1) // block_size
        scale_cols = (in_features + block_size - 1) // block_size
        self.weight_scale = nn.Parameter(
            torch.empty(scale_rows, scale_cols, dtype=torch.float32)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters: Xavier uniform for weights, ones for scales, zeros for bias.
        """
        nn.init.xavier_uniform_(self.weight.float())
        nn.init.ones_(self.weight_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using FP8 quantization for activations and weights.

        Steps:
        1. Quantize input activations to FP8 with block-wise scales.
        2. Perform FP8 GEMM (x_q * weight) with appropriate scaling.
        3. Add bias if present.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, *, out_features).
        """
        # Quantize input activations
        x_q, x_scale = act_quant(x, self.block_size)
        # FP8 GEMM: (x_q @ weight_fp8) -> high-precision output
        y = fp8_gemm(x_q, x_scale, self.weight, self.weight_scale)
        # Add bias if required
        if self.bias is not None:
            y = y + self.bias
        return y

    
class FP8LayerNorm(nn.Module):
    """
    LayerNorm variant with FP8-stored gamma/beta parameters and block-wise scale factors.

    Only supports 1D normalized_shape.
    """
    def __init__(self, normalized_shape, eps: float = 1e-5, block_size: int = 128):
        super().__init__()
        # Normalize shape must be 1D
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        if len(self.normalized_shape) != 1:
            raise ValueError("FP8LayerNorm only supports 1D normalized_shape")
        self.dim = self.normalized_shape[0]
        self.eps = eps
        self.block_size = block_size

        # FP8 weight (gamma) and bias (beta)
        self.weight = nn.Parameter(
            torch.empty(self.dim, dtype=torch.float8_e4m3fn)
        )
        self.bias = nn.Parameter(
            torch.empty(self.dim, dtype=torch.float8_e4m3fn)
        )
        # Block-wise scale parameters for gamma and beta
        num_blocks = (self.dim + block_size - 1) // block_size
        self.weight_scale = nn.Parameter(
            torch.ones(num_blocks, dtype=torch.float32)
        )
        self.bias_scale = nn.Parameter(
            torch.ones(num_blocks, dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize gamma to 1 and beta to 0, then quantize to FP8
        w = torch.ones(self.dim, dtype=torch.float32)
        b = torch.zeros(self.dim, dtype=torch.float32)
        self.weight.data = w.to(torch.float8_e4m3fn)
        self.bias.data = b.to(torch.float8_e4m3fn)
        # Scales default to 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        # Compute mean and variance in float32
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # Build expanded scale vectors
        scale_vec = self.weight_scale.repeat_interleave(self.block_size)[: self.dim]
        bias_vec = self.bias_scale.repeat_interleave(self.block_size)[: self.dim]
        # Dequantize gamma and beta
        gamma = self.weight.to(torch.float32) * scale_vec
        beta = self.bias.to(torch.float32) * bias_vec
        # Apply affine transform
        return x_norm * gamma + beta



class FP8MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        block_size: int = 128,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        # FP8 linear projections for Q, K, V, and output
        self.q_proj = FP8Linear(embed_dim, embed_dim, bias=False, block_size=block_size)
        self.k_proj = FP8Linear(embed_dim, embed_dim, bias=False, block_size=block_size)
        self.v_proj = FP8Linear(embed_dim, embed_dim, bias=False, block_size=block_size)
        self.out_proj = FP8Linear(embed_dim, embed_dim, bias=False, block_size=block_size)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x: (batch_size, seq_length, embed_dim)
        attn_mask: (batch_size, seq_length) or (batch_size, seq_length, seq_length)
        """
        B, T, C = x.size()
        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            # assume mask==0 means masked
            scores = scores.masked_fill(attn_mask.unsqueeze(1) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        # Final projection
        out = self.out_proj(context)
        return out


class FP8TransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        block_size: int = 128,
    ):
        super().__init__()
        self.self_attn = FP8MultiHeadAttention(
            embed_dim, num_heads, dropout=dropout, block_size=block_size
        )
        self.norm1 = FP8LayerNorm(embed_dim, block_size=block_size)
        self.norm2 = FP8LayerNorm(embed_dim, block_size=block_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # Feed-forward network
        self.fc1 = FP8Linear(embed_dim, mlp_dim, bias=True, block_size=block_size)
        self.fc2 = FP8Linear(mlp_dim, embed_dim, bias=True, block_size=block_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention block
        attn_out = self.self_attn(x, attn_mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        # Feed-forward block
        ffn_out = self.fc2(self.act(self.fc1(x)))
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)
        return x


class FP8Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        block_size: int = 128,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            FP8TransformerLayer(
                embed_dim, num_heads, mlp_dim, dropout=dropout, block_size=block_size
            )
            for _ in range(num_layers)
        ])
        self.norm = FP8LayerNorm(embed_dim, block_size=block_size)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.norm(x)



class MiniLM(nn.Module):
    def __init__(self, vocab_size=5000, d_model=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.transformer = FP8Transformer(
            num_layers=6,
            embed_dim=d_model,
            num_heads=8,
            mlp_dim=256,
            dropout=0.1,
            block_size=128
        )
        self.norm = FP8LayerNorm(d_model)
        self.lm_head = FP8Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.transformer(x)
        x = self.norm(x)
        return self.lm_head(x)

if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    model = MiniLM()
    model = model.to('cuda')
    input_ids = torch.randint(0, 5000, (2, 16)).to('cuda')
    logits = model(input_ids)
    print("Logits:", logits.shape)
    print("Logits dtype:", logits.dtype)
