"""
Utility to convert Qwen2 HuggingFace model weights to FP8 transformer format.
"""
import torch
from typing import Dict, Any
import warnings

try:
    from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("transformers library not available. Cannot convert from HuggingFace.")

from qwen2_config import get_qwen2_exact_config
from qwen2_model import create_qwen2_transformer


def load_qwen2_hf_model(model_name: str = "Qwen/Qwen2.5-0.5B"):
    """Load Qwen2 model from HuggingFace."""
    if not HF_AVAILABLE:
        raise ImportError("transformers library required for HuggingFace model loading")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    
    return model, tokenizer, config


def print_hf_model_structure(model):
    """Print the structure of HuggingFace Qwen2 model for analysis."""
    print("HuggingFace Qwen2 Model Structure:")
    print("=" * 50)
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            if hasattr(module, 'bias') and module.bias is not None:
                print(f"{name}: {module.weight.shape} (bias: {module.bias.shape})")
            else:
                print(f"{name}: {module.weight.shape}")
    
    print("\nConfig parameters:")
    print(f"vocab_size: {model.config.vocab_size}")
    print(f"hidden_size: {model.config.hidden_size}")
    print(f"intermediate_size: {model.config.intermediate_size}")
    print(f"num_hidden_layers: {model.config.num_hidden_layers}")
    print(f"num_attention_heads: {model.config.num_attention_heads}")
    print(f"num_key_value_heads: {model.config.num_key_value_heads}")
    print(f"max_position_embeddings: {model.config.max_position_embeddings}")
    print(f"rope_theta: {model.config.rope_theta}")


def convert_qwen2_weights(hf_model, our_model) -> Dict[str, torch.Tensor]:
    """
    Convert weights from HuggingFace Qwen2 model to our FP8 transformer format.
    
    Args:
        hf_model: HuggingFace Qwen2ForCausalLM model
        our_model: Our transformer model with Qwen2 config
        
    Returns:
        Dictionary mapping our model's parameter names to converted weights
    """
    weight_mapping = {}
    
    # Embedding layer
    weight_mapping['embed.weight'] = hf_model.model.embed_tokens.weight
    
    # Final layer norm
    weight_mapping['norm.weight'] = hf_model.model.norm.weight
    
    # Language modeling head
    weight_mapping['head.weight'] = hf_model.lm_head.weight
    
    # Convert transformer layers
    for layer_idx in range(len(hf_model.model.layers)):
        hf_layer = hf_model.model.layers[layer_idx]
        layer_prefix = f'layers.{layer_idx}'
        
        # Attention layer norms
        weight_mapping[f'{layer_prefix}.attn_norm.weight'] = hf_layer.input_layernorm.weight
        weight_mapping[f'{layer_prefix}.ffn_norm.weight'] = hf_layer.post_attention_layernorm.weight
        
        # Attention projections
        weight_mapping[f'{layer_prefix}.attn.wq.weight'] = hf_layer.self_attn.q_proj.weight
        weight_mapping[f'{layer_prefix}.attn.wk.weight'] = hf_layer.self_attn.k_proj.weight
        weight_mapping[f'{layer_prefix}.attn.wv.weight'] = hf_layer.self_attn.v_proj.weight
        weight_mapping[f'{layer_prefix}.attn.wo.weight'] = hf_layer.self_attn.o_proj.weight
        
        # Attention biases (Qwen2 has biases for q, k, v projections)
        if hasattr(hf_layer.self_attn.q_proj, 'bias') and hf_layer.self_attn.q_proj.bias is not None:
            weight_mapping[f'{layer_prefix}.attn.wq.bias'] = hf_layer.self_attn.q_proj.bias
        if hasattr(hf_layer.self_attn.k_proj, 'bias') and hf_layer.self_attn.k_proj.bias is not None:
            weight_mapping[f'{layer_prefix}.attn.wk.bias'] = hf_layer.self_attn.k_proj.bias
        if hasattr(hf_layer.self_attn.v_proj, 'bias') and hf_layer.self_attn.v_proj.bias is not None:
            weight_mapping[f'{layer_prefix}.attn.wv.bias'] = hf_layer.self_attn.v_proj.bias
        
        # MLP projections
        weight_mapping[f'{layer_prefix}.ffn.w1.weight'] = hf_layer.mlp.gate_proj.weight
        weight_mapping[f'{layer_prefix}.ffn.w3.weight'] = hf_layer.mlp.up_proj.weight
        weight_mapping[f'{layer_prefix}.ffn.w2.weight'] = hf_layer.mlp.down_proj.weight
    
    return weight_mapping


def verify_weight_compatibility(hf_model, our_model):
    """
    Verify that weight shapes are compatible between HF and our model.
    """
    print("\nWeight shape compatibility check:")
    print("=" * 50)
    
    our_state_dict = {k: v for k, v in our_model.named_parameters()}
    converted_weights = convert_qwen2_weights(hf_model, our_model)
    
    compatible = True
    for our_name, our_param in our_state_dict.items():
        if our_name in converted_weights:
            hf_weight = converted_weights[our_name]
            if our_param.shape == hf_weight.shape:
                print(f"✓ {our_name}: {our_param.shape}")
            else:
                print(f"✗ {our_name}: our={our_param.shape} vs hf={hf_weight.shape}")
                compatible = False
        else:
            print(f"? {our_name}: {our_param.shape} (no HF equivalent found)")
    
    # Check for unused HF weights
    used_hf_weights = set(converted_weights.keys())
    all_our_weights = set(our_state_dict.keys())
    unused_hf = used_hf_weights - all_our_weights
    if unused_hf:
        print(f"\nUnused HF weights: {unused_hf}")
    
    return compatible


def load_converted_model(model_name: str = "Qwen/Qwen2.5-0.5B"):
    """
    Load a Qwen2 model and convert it to our format.
    
    Returns:
        our_model: Our transformer model with converted weights
        tokenizer: HuggingFace tokenizer
        hf_model: Original HuggingFace model for comparison
    """
    print(f"Loading Qwen2 model: {model_name}")
    
    # Load HuggingFace model
    hf_model, tokenizer, config = load_qwen2_hf_model(model_name)
    print(f"✓ Loaded HuggingFace model")
    
    # Print model structure for analysis
    print_hf_model_structure(hf_model)
    
    # Create our model
    our_model = create_qwen2_transformer("qwen2_exact")
    print(f"✓ Created our model")
    
    # Verify compatibility
    compatible = verify_weight_compatibility(hf_model, our_model)
    
    if compatible:
        # Convert and load weights
        converted_weights = convert_qwen2_weights(hf_model, our_model)
        our_model.load_state_dict(converted_weights, strict=False)
        print("✓ Weights converted and loaded successfully")
    else:
        print("⚠ Weight shapes are not compatible. Manual adjustment needed.")
    
    return our_model, tokenizer, hf_model


def compare_model_outputs(our_model, hf_model, tokenizer, text: str = "Hello, how are you?"):
    """
    Compare outputs between our model and HuggingFace model.
    """
    print(f"\nComparing model outputs for text: '{text}'")
    print("=" * 50)
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    print(f"Input tokens: {input_ids.shape}")
    
    # Get HF model output
    with torch.no_grad():
        hf_outputs = hf_model(input_ids)
        hf_logits = hf_outputs.logits
    
    # Get our model output
    with torch.no_grad():
        our_logits = our_model(input_ids)
    
    print(f"HF logits shape: {hf_logits.shape}")
    print(f"Our logits shape: {our_logits.shape}")
    
    # Compare logits
    if hf_logits.shape == our_logits.shape:
        diff = torch.abs(hf_logits - our_logits).max().item()
        mean_diff = torch.abs(hf_logits - our_logits).mean().item()
        print(f"Max difference: {diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        if diff < 1e-3:
            print("✓ Outputs are very similar!")
        elif diff < 1e-1:
            print("~ Outputs are reasonably similar")
        else:
            print("✗ Outputs differ significantly")
    else:
        print("✗ Output shapes don't match")


def convert_qwen2_to_fp8(hf_model, our_model, use_fp8: bool = True):
    """
    Convert HuggingFace Qwen2 model to our FP8 transformer format.
    
    Args:
        hf_model: HuggingFace Qwen2ForCausalLM model
        our_model: Our transformer model 
        use_fp8: Whether to apply FP8 quantization during conversion
    """
    from kernel import act_quant, FP8_E4M3
    
    print(f"Converting HuggingFace Qwen2 to FP8 format (fp8={use_fp8})...")
    
    # Convert weights
    converted_weights = convert_qwen2_weights(hf_model, our_model)
    
    # Apply FP8 quantization if requested
    if use_fp8:
        print("Applying FP8 quantization to weights...")
        fp8_weights = {}
        for name, weight in converted_weights.items():
            if 'weight' in name and weight.dim() >= 2:  # Only quantize matrix weights
                try:
                    quantized_weight, scale = act_quant(weight, dtype=FP8_E4M3)
                    fp8_weights[name] = quantized_weight
                    fp8_weights[name.replace('weight', 'scale')] = scale
                except Exception as e:
                    print(f"Warning: Could not quantize {name}: {e}")
                    fp8_weights[name] = weight
            else:
                fp8_weights[name] = weight
        converted_weights = fp8_weights
    
    # Load weights into our model
    missing_keys, unexpected_keys = our_model.load_state_dict(converted_weights, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    print("✓ Model conversion completed")


if __name__ == "__main__":
    # Test the conversion
    try:
        our_model, tokenizer, hf_model = load_converted_model("Qwen/Qwen2.5-0.5B")
        compare_model_outputs(our_model, hf_model, tokenizer)
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("This is expected if transformers library is not available or model is not accessible.")
