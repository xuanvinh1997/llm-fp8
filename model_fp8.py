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

class FP8Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        x_fp8, x_scale = quantize_fp8_triton(x_flat.contiguous().flatten(), fmt=FP8_E4M3)
        w_fp8, w_scale = quantize_fp8_triton(self.weight.contiguous().flatten(), fmt=FP8_E4M3)
        x_deq = dequantize_fp8_triton(x_fp8, x_scale)
        w_deq = dequantize_fp8_triton(w_fp8, w_scale)
        out = F.linear(x_deq.view(B*T, D), w_deq.view_as(self.weight))
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

        q_fp8, q_scale = quantize_fp8_triton(q.contiguous().flatten(), fmt=FP8_E5M2)
        k_fp8, k_scale = quantize_fp8_triton(k.contiguous().flatten(), fmt=FP8_E5M2)
        q_deq = dequantize_fp8_triton(q_fp8, q_scale).view_as(q)
        k_deq = dequantize_fp8_triton(k_fp8, k_scale).view_as(k)

        attn_scores = torch.matmul(q_deq, k_deq.transpose(-2, -1)) / (D ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)
        return self.out_proj(out)

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
