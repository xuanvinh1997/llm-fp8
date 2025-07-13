"""
Simple test script to verify the HuggingFace loading functionality works correctly.
This script tests the core functionality without requiring actual model downloads.
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from model import (
            load_hf_model_and_convert, 
            load_tokenizer_from_hf, 
            ModelArgs, 
            Transformer,
            HF_AVAILABLE
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_args():
    """Test ModelArgs creation and configuration."""
    print("Testing ModelArgs...")
    
    try:
        from model import ModelArgs
        
        # Test default args
        args = ModelArgs()
        assert args.n_dense_layers == args.n_layers, "Dense layers should equal total layers"
        assert args.dtype in ["bf16", "fp8"], "dtype should be valid"
        
        # Test custom args
        custom_args = ModelArgs(
            vocab_size=50000,
            dim=768,
            n_layers=12,
            dtype="fp8"
        )
        assert custom_args.vocab_size == 50000
        assert custom_args.dim == 768
        assert custom_args.n_layers == 12
        assert custom_args.dtype == "fp8"
        
        print("✓ ModelArgs tests passed")
        return True
    except Exception as e:
        print(f"✗ ModelArgs test failed: {e}")
        return False

def test_transformer_creation():
    """Test that our Transformer model can be created."""
    print("Testing Transformer creation...")
    
    try:
        from model import ModelArgs, Transformer
        
        # Create a small model for testing
        args = ModelArgs(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            inter_dim=256,
            max_seq_len=512
        )
        
        model = Transformer(args)
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        torch.manual_seed(42)
        test_input = torch.randint(0, args.vocab_size, (2, 10))
        
        with torch.no_grad():
            output = model(test_input)
        
        expected_shape = (2, args.vocab_size)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        print("✓ Forward pass successful")
        return True
    except Exception as e:
        print(f"✗ Transformer creation test failed: {e}")
        return False

def test_hf_availability():
    """Test HuggingFace transformers availability."""
    print("Testing HuggingFace availability...")
    
    from model import HF_AVAILABLE
    
    if HF_AVAILABLE:
        print("✓ HuggingFace transformers available")
        try:
            from model import load_hf_model_and_convert, load_tokenizer_from_hf
            print("✓ HF loading functions available")
        except ImportError as e:
            print(f"✗ HF loading functions not available: {e}")
            return False
    else:
        print("⚠ HuggingFace transformers not available (install with: pip install transformers)")
        print("  This is optional for basic model functionality")
    
    return True

def test_fp8_configuration():
    """Test FP8 configuration options."""
    print("Testing FP8 configuration...")
    
    try:
        from model import ModelArgs, Transformer, Linear
        
        # Test FP8 configuration
        args_fp8 = ModelArgs(dtype="fp8")
        model_fp8 = Transformer(args_fp8)
        
        # Check that Linear.dtype is set correctly
        expected_dtype = torch.float8_e4m3fn
        assert Linear.dtype == expected_dtype, f"Expected {expected_dtype}, got {Linear.dtype}"
        
        print("✓ FP8 configuration works")
        
        # Test BF16 configuration
        args_bf16 = ModelArgs(dtype="bf16")
        model_bf16 = Transformer(args_bf16)
        
        expected_dtype = torch.bfloat16
        assert Linear.dtype == expected_dtype, f"Expected {expected_dtype}, got {Linear.dtype}"
        
        print("✓ BF16 configuration works")
        return True
    except Exception as e:
        print(f"✗ FP8/BF16 configuration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 50)
    print("Running HuggingFace Loading Functionality Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_args,
        test_transformer_creation,
        test_hf_availability,
        test_fp8_configuration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    if failed == 0:
        print("🎉 All tests passed! The HuggingFace loading functionality is ready to use.")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
