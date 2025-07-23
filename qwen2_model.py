"""
Qwen2-compatible GQA implementation with flexible head dimensions.
"""
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
import math

from model import ModelArgs, apply_rotary_emb, ColumnParallelLinear, RowParallelLinear

class Qwen2GQA(nn.Module):
    """
    Qwen2-compatible Grouped Query Attention (GQA) Layer.
    
    This implementation handles the case where Q, K, V projections 
    have different output dimensions, as in Qwen2.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        
        assert args.n_heads % args.n_kv_heads == 0, f"n_heads ({args.n_heads}) must be divisible by n_kv_heads ({args.n_kv_heads})"
        
        # For single process execution, set world_size = 1
        world_size = 1  # Will be set properly in distributed setup
        rank = 0
        
        self.n_rep = args.n_heads // args.n_kv_heads  # Number of query heads per kv head
        
        # For Qwen2 compatibility, calculate head dimensions from actual projection sizes
        # Standard calculation would be dim // n_heads, but Qwen2 may differ
        self.head_dim = args.dim // args.n_heads
        
        # For Qwen2: q_proj outputs 896 (14 heads * 64), k/v_proj outputs 128 (2 heads * 64)
        # So head_dim should be consistent across all projections
        self.kv_head_dim = self.head_dim  # Assume same head dim for now
        
        # Qwen2 uses bias=True for q_proj, k_proj, v_proj, but bias=False for o_proj
        self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.head_dim, bias=True)
        self.wk = ColumnParallelLinear(self.dim, self.n_kv_heads * self.kv_head_dim, bias=True)
        self.wv = ColumnParallelLinear(self.dim, self.n_kv_heads * self.kv_head_dim, bias=True)
        self.wo = RowParallelLinear(self.n_heads * self.head_dim, self.dim, bias=False)
        
        self.softmax_scale = self.head_dim ** -0.5
        
        # No mscale adjustment for standard Qwen2 (no YaRN by default)
        # if args.max_seq_len > args.original_seq_len:
        #     mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
        #     self.softmax_scale = self.softmax_scale * mscale * mscale

        # For distributed training
        self.n_local_heads = self.n_heads // world_size
        self.n_local_kv_heads = self.n_kv_heads // world_size

        # KV Cache for inference
        self.register_buffer("k_cache", torch.zeros(
            args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.kv_head_dim, dtype=torch.bfloat16
        ), persistent=False)
        self.register_buffer("v_cache", torch.zeros(
            args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.kv_head_dim, dtype=torch.bfloat16
        ), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for Qwen2-compatible GQA.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Query, Key, Value projections
        q = self.wq(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_local_kv_heads, self.kv_head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_local_kv_heads, self.kv_head_dim)
        
        # Apply rotary embeddings (only to the RoPE portion of the head dimension)
        rope_dim = min(self.head_dim, freqs_cis.size(-1) * 2)  # freqs_cis covers half dimensions
        
        if rope_dim > 0:
            q_rope = q[..., :rope_dim]
            k_rope = k[..., :rope_dim]
            
            q_rope = apply_rotary_emb(q_rope, freqs_cis)
            k_rope = apply_rotary_emb(k_rope, freqs_cis)
            
            # Reconstruct q and k with rotary embeddings applied
            if rope_dim < self.head_dim:
                q = torch.cat([q_rope, q[..., rope_dim:]], dim=-1)
                k = torch.cat([k_rope, k[..., rope_dim:]], dim=-1)
            else:
                q = q_rope
                k = k_rope
        
        # Update KV cache
        self.k_cache[:bsz, start_pos:end_pos] = k
        self.v_cache[:bsz, start_pos:end_pos] = v
        
        # Get keys and values from cache
        keys = self.k_cache[:bsz, :end_pos]  # (bsz, seq_len, n_local_kv_heads, kv_head_dim)
        values = self.v_cache[:bsz, :end_pos]  # (bsz, seq_len, n_local_kv_heads, kv_head_dim)
        
        # Repeat keys and values for each query head group
        keys = keys.repeat_interleave(self.n_rep, dim=2)
        values = values.repeat_interleave(self.n_rep, dim=2)
        
        # Compute attention scores
        scores = torch.einsum("bshd,bthd->bsht", q, keys) * self.softmax_scale
        
        # Apply causal mask
        if mask is not None:
            scores += mask.unsqueeze(1)
            
        # Apply softmax
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        
        # Apply attention to values
        output = torch.einsum("bsht,bthd->bshd", scores, values)
        
        # Reshape and apply output projection
        output = output.flatten(2)  # (bsz, seqlen, n_local_heads * head_dim)
        output = self.wo(output)
        
        return output


def create_qwen2_transformer(config_name: str = "qwen2_exact") -> nn.Module:
    """
    Create a transformer model configured for Qwen2 architecture.
    
    Args:
        config_name: Either "qwen2_exact" or "qwen2" for different configurations
    
    Returns:
        Transformer model with Qwen2 configuration
    """
    from qwen2_config import get_qwen2_config, get_qwen2_exact_config
    from model import Transformer, Block, MLP, RMSNorm
    
    if config_name == "qwen2_exact":
        args = get_qwen2_exact_config()
    else:
        args = get_qwen2_config()
    
    # Create a modified transformer class for Qwen2
    class Qwen2Transformer(Transformer):
        def __init__(self, args):
            # Set consistent dtype before initialization
            import torch
            global world_size, rank
            from model import Linear
            world_size = 1  # Will be set properly in distributed setup
            rank = 0
            Linear.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float8_e4m3fn
            
            # Initialize parent class but override the blocks
            super().__init__(args)
            
            # Replace the layers with Qwen2-compatible blocks
            self.layers = nn.ModuleList()
            for layer_id in range(args.n_layers):
                self.layers.append(Qwen2Block(layer_id, args))
    
    class Qwen2Block(nn.Module):
        """Qwen2-compatible transformer block."""
        def __init__(self, layer_id: int, args: ModelArgs):
            super().__init__()
            self.attn = Qwen2GQA(args)
            self.ffn = MLP(args.dim, args.inter_dim)
            self.attn_norm = RMSNorm(args.dim)
            self.ffn_norm = RMSNorm(args.dim)
        
        def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
            # Pre-norm architecture (like Qwen2)
            x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
            x = x + self.ffn(self.ffn_norm(x))
            return x
    
    return Qwen2Transformer(args)


if __name__ == "__main__":
    # Test the Qwen2 configuration
    print("Testing Qwen2 configuration...")
    
    model = create_qwen2_transformer("qwen2_exact")
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    tokens = torch.randint(0, 151936, (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(tokens)
    
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {logits.shape}")
    print("✓ Qwen2 model test passed!")
