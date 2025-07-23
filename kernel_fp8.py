import torch
import triton
import triton.language as tl

@triton.jit
def quantize_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr, MAXVAL: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs, mask=offs < x_ptr.shape[0]).to(tl.float32)
    s = tl.max(tl.abs(x)) / MAXVAL
    y = (x / s).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=offs < x_ptr.shape[0])
    tl.store(s_ptr + pid, s)

@triton.jit
def dequantize_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs, mask=offs < x_ptr.shape[0]).to(tl.float32)
    s = tl.load(s_ptr + pid)
    y = x * s
    tl.store(y_ptr + offs, y, mask=offs < x_ptr.shape[0])

def quantize_fp8(x: torch.Tensor, fmt: torch.dtype, block_size=128):
    assert x.is_contiguous()
    assert x.dim() == 1
    MAXVAL = 448.0 if fmt == torch.float8_e4m3fn else 57344.0
    y = torch.empty_like(x, dtype=fmt)
    s = torch.empty((x.shape[0] // block_size,), dtype=torch.float32, device=x.device)
    grid = lambda META: (x.shape[0] // block_size,)
    quantize_kernel[grid](x, y, s, BLOCK_SIZE=block_size, MAXVAL=MAXVAL)
    return y, s

def dequantize_fp8(x_fp8: torch.Tensor, s: torch.Tensor, block_size=128):
    y = torch.empty_like(x_fp8, dtype=torch.float32)
    grid = lambda META: (x_fp8.shape[0] // block_size,)
    dequantize_kernel[grid](x_fp8, y, s, BLOCK_SIZE=block_size)
    return y


