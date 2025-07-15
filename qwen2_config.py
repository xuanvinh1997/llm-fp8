"""
Qwen2 model configuration for FP8 transformer implementation.
"""
from model import ModelArgs

def get_qwen2_config() -> ModelArgs:
    """
    Returns ModelArgs configured for Qwen2-0.5B model architecture.
    
    Based on the provided architecture:
    - embed_tokens: 151936 vocab, 896 dim
    - 24 layers
    - GQA with q_proj: 896->896, k_proj: 896->128, v_proj: 896->128
    - MLP: gate_proj & up_proj: 896->4864, down_proj: 4864->896
    """
    
    # Calculate attention heads from dimensions
    dim = 896
    # From q_proj: 896 -> 896, so 896 / head_dim = n_heads
    # From k_proj: 896 -> 128, so 128 / head_dim = n_kv_heads
    # If we assume head_dim = 64 (common choice):
    # n_heads = 896 / 64 = 14 (but this doesn't divide evenly)
    # Let's try head_dim = 56: n_heads = 896 / 56 = 16, n_kv_heads = 128 / 56 ≈ 2.3 (not integer)
    # Let's try head_dim = 64: n_heads = 896 / 64 = 14, n_kv_heads = 128 / 64 = 2
    
    # Actually, let's calculate properly:
    # q_proj output = n_heads * head_dim = 896
    # k_proj output = n_kv_heads * head_dim = 128
    # Assuming head_dim = 64: n_heads = 14, n_kv_heads = 2
    # But 14 is not divisible by 2, let's try head_dim = 112
    # head_dim = 112: n_heads = 8, n_kv_heads = 128/112 ≈ 1.14 (not integer)
    
    # Let's work backwards: if n_kv_heads divides n_heads evenly
    # Common ratios are 2:1, 4:1, 8:1
    # For GQA, let's try n_kv_heads = 2, n_heads = 8 (4:1 ratio)
    # Then head_dim = 896 / 8 = 112 for queries
    # And head_dim = 128 / 2 = 64 for keys/values
    
    # This suggests different head dimensions for Q vs KV, but our implementation assumes same head_dim
    # Let's adjust to make it work with our current architecture
    
    # Alternative approach: adjust to fit our architecture constraints
    n_heads = 14  # 896 / 64 = 14
    n_kv_heads = 2  # 128 / 64 = 2
    head_dim = 64
    
    # Note: This requires n_heads to be divisible by n_kv_heads
    # 14 is not divisible by 2 evenly (7 query heads per kv head)
    # Let's adjust to n_heads = 16, n_kv_heads = 2 (8:1 ratio)
    # But then q_proj would be 16 * 64 = 1024, not 896
    
    # Best fit approach: use the dimensions that work with our architecture
    n_heads = 16  # Closest even number that works with GQA
    n_kv_heads = 2  # From the k_proj/v_proj dimensions
    head_dim = 896 // n_heads  # 56
    
    return ModelArgs(
        # Model dimensions
        vocab_size=151936,
        dim=896,
        inter_dim=4864,  # From gate_proj/up_proj dimensions
        n_layers=24,
        n_dense_layers=24,  # All layers are dense (no MoE)
        
        # Attention configuration
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        
        # Rotary embedding configuration (using standard RoPE, not MLA)
        qk_nope_head_dim=0,  # No non-positional embedding part
        qk_rope_head_dim=head_dim,  # Full head dimension for RoPE
        v_head_dim=head_dim,
        
        # Sequence length and RoPE parameters
        max_seq_len=32768,  # Common for Qwen2
        original_seq_len=32768,
        rope_theta=1000000.0,  # Qwen2 uses 1M base frequency
        rope_factor=1.0,  # No YaRN scaling by default
        
        # Training parameters
        max_batch_size=8,
        dtype="bf16",  # or "fp8" for FP8 training
        
        # MoE parameters (unused for Qwen2)
        n_routed_experts=0,
        n_shared_experts=0,
        n_activated_experts=0,
        
        # MLA parameters (unused for standard attention)
        q_lora_rank=0,
        kv_lora_rank=0,
        
        # YaRN parameters
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
    )

def get_qwen2_exact_config() -> ModelArgs:
    """
    Returns ModelArgs configured to exactly match Qwen2 dimensions,
    with adjustments for our architecture constraints.
    """
    # To exactly match Qwen2, we need to handle the dimension mismatch
    # Qwen2 has different effective head dimensions for Q vs KV
    
    # Qwen2 dimensions:
    # - q_proj: 896 -> 896 (suggests 14 heads of 64 dim each, or other combinations)
    # - k_proj, v_proj: 896 -> 128 (suggests 2 heads of 64 dim each)
    
    # Calculate exact dimensions from Qwen2 architecture:
    # - q_proj: 896 -> 896 (14 heads * 64 dim each)  
    # - k_proj, v_proj: 896 -> 128 (2 heads * 64 dim each)
    
    # Qwen2 actually uses 14 heads total with 2 KV heads
    # Let's match this exactly  
    n_heads = 14
    n_kv_heads = 2
    head_dim = 896 // n_heads  # 64
    
    # Verify this matches the expected dimensions
    assert n_heads * head_dim == 896, f"Q projection dimension mismatch: {n_heads} * {head_dim} != 896"
    assert n_kv_heads * head_dim == 128, f"KV projection dimension mismatch: {n_kv_heads} * {head_dim} != 128"
    
    return ModelArgs(
        # Core model dimensions
        vocab_size=151936,
        dim=896,
        inter_dim=4864,
        n_layers=24,
        n_dense_layers=24,
        
        # Attention configuration
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        
        # Use standard RoPE (not MLA)
        qk_nope_head_dim=0,
        qk_rope_head_dim=head_dim,
        v_head_dim=head_dim,
        
        # Qwen2-specific parameters
        max_seq_len=32768,
        original_seq_len=32768,
        rope_theta=1000000.0,  # Qwen2's rope_theta
        rope_factor=1.0,
        
        # Training setup
        max_batch_size=8,
        dtype="bf16",
        
        # Unused parameters for dense model
        n_routed_experts=0,
        n_shared_experts=0,
        n_activated_experts=0,
        q_lora_rank=0,
        kv_lora_rank=0,
        
        # YaRN parameters
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
    )
