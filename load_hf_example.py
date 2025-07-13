#!/usr/bin/env python3
"""
Example script demonstrating how to load a Hugging Face model and convert it 
to our custom FP8-aware dense transformer implementation.

Usage:
    python load_hf_example.py --model_name "bert-base-uncased" --fp8
    python load_hf_example.py --model_name "microsoft/DialoGPT-medium" --no-fp8
"""

import argparse
import torch
from model import load_hf_model_and_convert, load_tokenizer_from_hf, ModelArgs


def main():
    parser = argparse.ArgumentParser(description="Load and convert HuggingFace model to FP8 dense transformer")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="HuggingFace model name or path")
    parser.add_argument("--fp8", action="store_true", default=True,
                        help="Convert to FP8 format (default: True)")
    parser.add_argument("--no-fp8", action="store_false", dest="fp8",
                        help="Keep in bfloat16 format")
    parser.add_argument("--trust_remote_code", action="store_true", default=False,
                        help="Trust remote code when loading model")
    parser.add_argument("--custom_config", type=str, default=None,
                        help="Path to custom ModelArgs config (optional)")
    parser.add_argument("--test_inference", action="store_true", default=True,
                        help="Run a simple inference test")
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_name}")
    print(f"Convert to FP8: {args.fp8}")
    print(f"Trust remote code: {args.trust_remote_code}")
    
    # Load custom config if provided
    target_args = None
    if args.custom_config:
        # You can implement custom config loading here
        print(f"Loading custom config from: {args.custom_config}")
        # target_args = load_custom_config(args.custom_config)
    
    try:
        # Load and convert the model
        model = load_hf_model_and_convert(
            model_name_or_path=args.model_name,
            target_args=target_args,
            convert_to_fp8=args.fp8,
            trust_remote_code=args.trust_remote_code
        )
        
        print("Model loaded successfully!")
        print(f"Model dtype: {'FP8' if args.fp8 else 'BF16'}")
        
        # Load tokenizer for testing
        if args.test_inference:
            print("\nLoading tokenizer...")
            try:
                tokenizer = load_tokenizer_from_hf(args.model_name)
                print("Tokenizer loaded successfully!")
                
                # Run a simple inference test
                print("\nRunning inference test...")
                test_inference(model, tokenizer)
                
            except Exception as e:
                print(f"Could not load tokenizer or run inference: {e}")
                print("This is normal for some model architectures.")
        
        # Print model info
        print("\nModel Information:")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Number of layers: {len(model.layers)}")
        print(f"Hidden dimension: {model.embed.dim}")
        print(f"Vocabulary size: {model.embed.vocab_size}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    return 0


def test_inference(model, tokenizer, test_text="Hello, how are you today?"):
    """
    Run a simple inference test with the converted model.
    
    Args:
        model: Converted model
        tokenizer: HuggingFace tokenizer
        test_text: Text to use for testing
    """
    try:
        # Tokenize input
        inputs = tokenizer(test_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        print(f"Input text: {test_text}")
        print(f"Input tokens shape: {input_ids.shape}")
        
        # Move to appropriate device
        if torch.cuda.is_available():
            model = model.cuda()
            input_ids = input_ids.cuda()
        
        # Run inference
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)
        
        print(f"Output logits shape: {outputs.shape}")
        print(f"Output logits dtype: {outputs.dtype}")
        
        # Get top predicted tokens
        predicted_token_ids = outputs.argmax(dim=-1)
        print(f"Predicted token IDs: {predicted_token_ids}")
        
        # Decode if possible
        try:
            predicted_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
            print(f"Predicted text: {predicted_text}")
        except:
            print("Could not decode predicted tokens (this is normal for some architectures)")
            
    except Exception as e:
        print(f"Inference test failed: {e}")


if __name__ == "__main__":
    exit(main())
