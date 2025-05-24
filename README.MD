# Qwen-FP8: Language Model Training with FP8 Precision

This repository contains a production-ready training script for fine-tuning Qwen language models with FP8 precision support, powered by the official Accelerate API. It enables efficient instruction tuning with multiple precision options including BF16, FP16, and cutting-edge FP8 training.

## Features

- **Advanced FP8 Training Support**: Official integration with multiple FP8 backends:
  - **TransformersEngine (TE)**: NVIDIA's optimized FP8 training engine
  - **MS-AMP**: Microsoft's Automatic Mixed Precision for FP8
  - **TorchAO**: Experimental Adaptive Optimization with FP8 (early access)

- **Precision Options**:
  - Full support for `bf16`, `fp16`, and `fp8` training modes
  - Configurable FP8 formats: `E4M3`, `E5M2`, and `HYBRID`

- **Efficient Training Optimizations**:
  - Gradient checkpointing for memory efficiency
  - Gradient accumulation for larger effective batch sizes
  - Optimized data loading and preprocessing

- **Advanced Learning Rate Scheduling**:
  - Combined linear warmup + cosine annealing scheduler
  - Configurable warmup ratio and learning rate decay

- **Monitoring and Logging**:
  - Integrated Weights & Biases support for experiment tracking
  - Detailed logging of training metrics and precision information

- **Dataset Flexibility**:
  - Support for both local JSON datasets and HuggingFace datasets
  - Optimized chat template formatting for instruction tuning

## Requirements

```
torch>=2.1.0
transformers>=4.35.0
accelerate>=1.7.0  # Required for official FP8 support
datasets
wandb
```

For FP8 support, you'll need the backend-specific dependencies:
- **TransformersEngine**: `transformer-engine>=1.2.0` (NVIDIA GPUs only)
- **MS-AMP**: `msamp>=0.1.0`
- **TorchAO**: Early access version of `torch` with AO support

## Quick Start

### Data Preparation

Prepare your instruction dataset in JSON format:

```json
[
  {
    "problem": "What is the capital of France?",
    "generated_solution": "The capital of France is Paris..."
  },
  ...
]
```

### Running Training

```bash
# Install dependencies
pip install -r requirements.txt

# Run training with default parameters (FP8 with TransformersEngine)
python train.py
```

## Configuration Options

The script includes several pre-configured training options:

### 1. BF16 Training (Default for most GPUs)

```python
config = TrainingConfig(
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    dataset_path="data/instruction_dataset.json",
    output_dir="outputs/qwen-math-bf16",
    mixed_precision="bf16"
)
```

### 2. FP8 Training with TransformersEngine

```python
config = TrainingConfig(
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    dataset_path="data/instruction_dataset.json",
    output_dir="outputs/qwen-math-fp8-te",
    mixed_precision="fp8",  # Official FP8 support!
    fp8_backend="te",
    te_fp8_format="HYBRID",
    batch_size=8,  # Can use larger batch size with FP8
    gradient_accumulation_steps=4
)
```

### 3. FP8 Training with MS-AMP

```python
config = TrainingConfig(
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    dataset_path="data/instruction_dataset.json",
    output_dir="outputs/qwen-math-fp8-msamp",
    mixed_precision="fp8", 
    fp8_backend="msamp",
    msamp_opt_level="O2"
)
```

## Advanced Configuration

You can modify all training parameters by editing the `TrainingConfig` class in the script. Key parameters include:

- `model_name`: HuggingFace model ID or local path
- `dataset_path`: Path to instruction dataset (JSON file or HuggingFace dataset ID)
- `num_epochs`, `batch_size`, `learning_rate`: Basic training hyperparameters
- `mixed_precision`: Precision mode (`"no"`, `"fp16"`, `"bf16"`, or `"fp8"`)
- `fp8_backend`: FP8 backend when using FP8 (`"te"`, `"msamp"`, or `"ao"`)
- FP8 backend-specific configuration parameters

## Performance Comparison

FP8 training typically provides:
- **1.5x - 2x** training speedup compared to BF16
- **Up to 50%** memory reduction, enabling larger batch sizes
- Comparable model quality to BF16 training

## Citation & References

This implementation is based on Hugging Face's Accelerate library with official FP8 support:
- [Low Precision Training Guide](https://huggingface.co/docs/accelerate/usage_guides/low_precision_training)

FP8 training draws from the following research:
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

## License

This project is licensed under the MIT License - see the LICENSE file for details.