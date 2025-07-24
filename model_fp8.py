import math
from typing import Optional
from typing_extensions import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import Qwen2Config
from kernel import act_quant, fp8_gemm

FP8_E4M3 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2
world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "fp8"
attn_impl: Literal["naive", "absorb"] = "absorb"



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class FP8Linear(nn.Module):
    def __init__(
        self, in_f, out_f, bias=True, block_size=128, fmt=FP8_E4M3
    ):  # <── pick E4M3 or E5M2 here
        super().__init__()
        self.fmt = fmt
        self.in_f, self.out_f, self.block_size = in_f, out_f, block_size

        self.weight = nn.Parameter(torch.empty(out_f, in_f, dtype=fmt))
        # per-block scale (1-D because you step over K)
        num_blocks = math.ceil(out_f / block_size)
        self.weight_scale = nn.Parameter(torch.ones(num_blocks, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_f, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        """
        Initialize parameters: Xavier uniform for weights, ones for scales, zeros for bias.
        """
        nn.init.xavier_uniform_(self.weight.float())
        nn.init.ones_(self.weight_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # 1️⃣  quantise activations (always E4M3 for stability)
        x_q, x_s = act_quant(x, self.block_size, dtype=self.fmt)

        # 2️⃣  FP8 GEMM (kernel is format-agnostic)
        y = fp8_gemm(x_q, x_s, self.weight, self.weight_scale)
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
        self.weight = nn.Parameter(torch.empty(self.dim, dtype=torch.float8_e4m3fn))
        self.bias = nn.Parameter(torch.empty(self.dim, dtype=torch.float8_e4m3fn))
        # Block-wise scale parameters for gamma and beta
        num_blocks = (self.dim + block_size - 1) // block_size
        self.weight_scale = nn.Parameter(torch.ones(num_blocks, dtype=torch.float32))
        self.bias_scale = nn.Parameter(torch.ones(num_blocks, dtype=torch.float32))
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
        layer_idx: int = 0,
        config=None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = FP8Linear(
            embed_dim, embed_dim, bias=False, block_size=block_size, fmt=FP8_E5M2
        )  # ⬅️ E5M2
        self.k_proj = FP8Linear(
            embed_dim, embed_dim, bias=False, block_size=block_size, fmt=FP8_E5M2
        )  # ⬅️ E5M2
        self.v_proj = FP8Linear(
            embed_dim,
            embed_dim,  # value can stay FP8_E5M2
            bias=False,
            block_size=block_size,
            fmt=FP8_E5M2,
        )
        self.out_proj = FP8Linear(
            embed_dim, embed_dim, bias=False, block_size=block_size, fmt=FP8_E5M2
        )
        self.dropout = nn.Dropout(dropout)
        self.sliding_window = config.sliding_window \
        if config.layer_types[layer_idx] == "sliding_attention" else None

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
            scores = scores.masked_fill(attn_mask.unsqueeze(1) == 0, float("-inf"))
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
        self.layers = nn.ModuleList(
            [
                FP8TransformerLayer(
                    embed_dim,
                    num_heads,
                    mlp_dim,
                    dropout=dropout,
                    block_size=block_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = FP8LayerNorm(embed_dim, block_size=block_size)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.norm(x)




class MiniLM(nn.Module):
    def __init__(self, vocab_size=5000, d_model=128, block_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.transformer = FP8Transformer(
            num_layers=6,
            embed_dim=d_model,
            num_heads=8,
            mlp_dim=256,
            dropout=0.1,
            block_size=128,
        )
        self.norm = FP8LayerNorm(d_model)
        self.lm_head = FP8Linear(d_model, vocab_size)
        self.block_size = block_size
        

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.transformer(x)
        x = self.norm(x)
        return self.lm_head(x)


# if __name__ == "__main__":
#     torch.set_default_dtype(torch.bfloat16)
#     model = MiniLM()
#     model = model.to("cuda")
#     input_ids = torch.randint(0, 5000, (2, 16)).to("cuda")
#     logits = model(input_ids)
#     print("Logits:", logits.shape)
#     print("Logits dtype:", logits.dtype)
