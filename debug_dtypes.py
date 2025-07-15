"""
Debug script to identify dtype mismatches in the model.
"""
import torch
from qwen2_model import create_qwen2_transformer

def debug_model_dtypes():
    """Debug the data types of model components."""
    print("Debugging Model Data Types...")
    print("=" * 50)
    
    model = create_qwen2_transformer("qwen2_exact")
    
    # Check all parameter dtypes
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")
    
    # Check all buffer dtypes
    for name, buffer in model.named_buffers():
        print(f"Buffer {name}: {buffer.dtype}")
    
    print("\n" + "="*50)
    
    # Test with specific input
    batch_size = 1
    seq_len = 4
    tokens = torch.randint(0, 151936, (batch_size, seq_len))
    print(f"Input tokens dtype: {tokens.dtype}")
    
    # Trace through the model step by step
    print("\nTracing forward pass...")
    
    # Embedding
    h = model.embed(tokens)
    print(f"After embedding: {h.dtype}, shape: {h.shape}")
    
    # Get freqs_cis
    freqs_cis = model.freqs_cis[:seq_len]
    print(f"freqs_cis dtype: {freqs_cis.dtype}")
    
    # Try first layer
    try:
        first_layer = model.layers[0]
        print(f"First layer type: {type(first_layer)}")
        
        # Check norm
        norm_out = first_layer.attn_norm(h)
        print(f"After attn_norm: {norm_out.dtype}")
        
        # Try attention - this is where it fails
        print("Attempting attention forward...")
        attn_out = first_layer.attn(norm_out, start_pos=0, freqs_cis=freqs_cis, mask=None)
        print(f"After attention: {attn_out.dtype}")
        
    except Exception as e:
        print(f"Error in forward pass: {e}")
        
        # Check attention module weights
        print("\nChecking attention weights:")
        attn = first_layer.attn
        print(f"wq weight dtype: {attn.wq.weight.dtype}")
        print(f"wk weight dtype: {attn.wk.weight.dtype}")
        print(f"wv weight dtype: {attn.wv.weight.dtype}")
        print(f"wo weight dtype: {attn.wo.weight.dtype}")
        
        if hasattr(attn.wq, 'bias') and attn.wq.bias is not None:
            print(f"wq bias dtype: {attn.wq.bias.dtype}")
        if hasattr(attn.wk, 'bias') and attn.wk.bias is not None:
            print(f"wk bias dtype: {attn.wk.bias.dtype}")
        if hasattr(attn.wv, 'bias') and attn.wv.bias is not None:
            print(f"wv bias dtype: {attn.wv.bias.dtype}")

if __name__ == "__main__":
    debug_model_dtypes()
