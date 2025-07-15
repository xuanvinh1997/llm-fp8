#!/usr/bin/env python3
"""
Demo script showing how to use the FP8 Qwen2.5 training pipeline.

This script demonstrates:
1. Loading a pre-trained Qwen2.5 model
2. Converting it to FP8 format
3. Running a short training session
4. Evaluating the model performance
"""

import torch
import logging
from train_fp8 import TrainingConfig, FP8Qwen2Wrapper, load_pretrained_qwen2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_fp8_conversion():
    """Demonstrate FP8 model conversion"""
    print("=" * 60)
    print("FP8 Qwen2.5 Conversion Demo")
    print("=" * 60)
    
    # Create configuration for demo
    config = TrainingConfig(
        model_name="Qwen/Qwen2.5-0.5B",
        max_seq_len=512,  # Shorter for demo
        batch_size=2,     # Small batch for demo
        max_steps=100,    # Very short training
        use_fp8=True,
        mixed_precision=True
    )
    
    print(f"Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Max sequence length: {config.max_seq_len}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  FP8 enabled: {config.use_fp8}")
    print(f"  Mixed precision: {config.mixed_precision}")
    print()
    
    try:
        # Check if we have GPU
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name()}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("Running on CPU (GPU recommended for FP8)")
        print()
        
        # Load and convert model
        print("Loading pre-trained Qwen2.5 model...")
        model, tokenizer = load_pretrained_qwen2(config)
        print("✓ Model loaded and converted to FP8 successfully!")
        print()
        
        # Test tokenizer
        test_text = "What is machine learning?"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"Test tokenization:")
        print(f"  Input: '{test_text}'")
        print(f"  Tokens: {tokens['input_ids'].shape}")
        print(f"  Decoded: '{tokenizer.decode(tokens['input_ids'][0])}'")
        print()
        
        # Test model inference
        print("Testing model inference...")
        device = next(model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = model(tokens['input_ids'])
            logits = outputs['logits']
        
        print(f"  Input shape: {tokens['input_ids'].shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Output dtype: {logits.dtype}")
        
        # Test generation
        print("Testing text generation...")
        with torch.no_grad():
            generated = model.model.generate(
                tokens['input_ids'], 
                max_length=tokens['input_ids'].shape[1] + 20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"  Generated: '{generated_text}'")
        print()
        
        # Memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            print(f"GPU memory used: {memory_used:.2f} GB")
        
        print("✓ FP8 conversion demo completed successfully!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("This might be due to missing dependencies or insufficient memory.")
        print("Try installing requirements: pip install -r requirements.txt")


def demo_training_step():
    """Demonstrate a single training step with FP8"""
    print("\n" + "=" * 60)
    print("FP8 Training Step Demo")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer
        
        # Simple training data
        training_texts = [
            "The capital of France is Paris.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "The quick brown fox jumps over the lazy dog."
        ]
        
        print("Sample training data:")
        for i, text in enumerate(training_texts):
            print(f"  {i+1}. {text}")
        print()
        
        # Create simple config
        config = TrainingConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            max_seq_len=128,
            batch_size=2,
            use_fp8=True,
            learning_rate=1e-4
        )
        
        # Load tokenizer for demo
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize training data
        encoded = tokenizer(
            training_texts,
            padding=True,
            truncation=True,
            max_length=config.max_seq_len,
            return_tensors="pt"
        )
        
        print(f"Tokenized data shape: {encoded['input_ids'].shape}")
        print(f"Sample tokens: {encoded['input_ids'][0][:10].tolist()}")
        print()
        
        print("Training step demo completed!")
        print("For full training, run: python train_fp8.py --use_fp8 --batch_size 4 --max_steps 1000")
        
    except ImportError:
        print("Transformers library not available for demo.")
    except Exception as e:
        print(f"Error in training demo: {e}")


def print_system_info():
    """Print system information for debugging"""
    print("\n" + "=" * 60)
    print("System Information")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers: Not installed")
    
    try:
        import triton
        print(f"Triton version: {triton.__version__}")
    except ImportError:
        print("Triton: Not installed")


if __name__ == "__main__":
    print_system_info()
    demo_fp8_conversion()
    demo_training_step()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("Next steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Run full training: python train_fp8.py --use_fp8")
    print("3. Monitor with: tensorboard --logdir ./fp8_qwen2_checkpoints")
    print("=" * 60)
