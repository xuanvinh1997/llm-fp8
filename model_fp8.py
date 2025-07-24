import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

FP8_E4M3 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2
BLOCK_SIZE = 128

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
gemm_impl = 'fp16'  # Set to 'fp8' to use FP8 GEMM, or 'linear' for standard linear operation
class FP8Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        
        # x_deq = dequantize_fp8_triton(x_fp8, x_scale)
        # w_deq = dequantize_fp8_triton(w_fp8, w_scale)
        if gemm_impl == 'fp8':
            # Use FP8 GEMM for matrix multiplication
            x_fp8, x_scale = quantize_fp8_triton(x_flat.contiguous().flatten(), fmt=FP8_E4M3)
            w_fp8, w_scale = quantize_fp8_triton(self.weight.contiguous().flatten(), fmt=FP8_E4M3)
            out = fp8_gemm(x_fp8, x_scale, w_fp8, w_scale)
        else:
            # Fallback to standard linear operation if not using FP8
            out = torch.matmul(x_flat, self.weight.t())
        return out.view(B, T, -1)

class FP8LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x_fp8, scale = quantize_fp8_triton(x.contiguous().flatten(), fmt=FP8_E4M3)
        x_deq = dequantize_fp8_triton(x_fp8, scale).view_as(x)
        mean = x_deq.mean(-1, keepdim=True)
        var = x_deq.var(-1, unbiased=False, keepdim=True)
        normed = (x_deq - mean) / torch.sqrt(var + self.eps)
        return normed * self.weight + self.bias

class MiniFP8Transformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.q_proj = FP8Linear(d_model, d_model)
        self.k_proj = FP8Linear(d_model, d_model)
        self.v_proj = FP8Linear(d_model, d_model)
        self.out_proj = FP8Linear(d_model, d_model)
        self.norm = FP8LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.size()
        x = self.norm(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Apply RoPE to q and k
        q = apply_rope(q, T)
        k = apply_rope(k, T)

        
        # q_deq = dequantize_fp8_triton(q_fp8, q_scale).view_as(q)
        # k_deq = dequantize_fp8_triton(k_fp8, k_scale).view_as(k)
        if gemm_impl == 'fp8':
            q_fp8, q_scale = quantize_fp8_triton(q.contiguous().flatten(), fmt=FP8_E5M2)
            k_fp8, k_scale = quantize_fp8_triton(k.contiguous().flatten(), fmt=FP8_E5M2)
            attn_scores = fp8_gemm(q_fp8, q_scale, k_fp8, k_scale) / (D ** 0.5)
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        # if gemm_impl == 'fp8':
        #     v_fp8, v_scale = quantize_fp8_triton(v.contiguous().flatten(), fmt=FP8_E4M3)
        #     attn_x, attn_scale = quantize_fp8_triton(attn_probs.contiguous().flatten(), fmt=FP8_E4M3)
        #     attn_out = fp8_gemm(attn_x, attn_scale, v_fp8, v_scale)
        # else:
        print(attn_probs.shape)
        attn_out = torch.matmul(attn_probs, v)
        return self.out_proj(attn_out)

class MiniLM(nn.Module):
    def __init__(self, vocab_size=5000, d_model=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.transformer = MiniFP8Transformer(d_model=d_model)
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
