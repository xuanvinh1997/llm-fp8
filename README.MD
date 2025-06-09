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

## training
```bash
accelerate launch     --config_file ds_config.yaml     --num_processes 2     --num_machines 1     --machine_rank 0     --main_process_port 29500     train.py     --model_name "Qwen/Qwen2.5-7B"     --dataset_name "nvidia/OpenMathInstruct-2"     --max_length 2048     --max_samples 100000 
    --batch_size 1     --gradient_accumulation_steps 16     --learning_rate 1e-4     --num_epochs 2     --warmup_steps 500     --weight_decay 0.01     --max_grad_norm 1.0     --zero_stage 3     --zero_offload_optimizer     --zero_offload_param     --zero3_init_flag     --fp8_backend "te"     --te_fp8_format "HYBR
ID"     --output_dir "./qwen_7b_math_fp8_model"     --save_steps 10000     --eval_steps 10000     --seed 42
```

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

## Troubleshooting

```bash
sudo apt-get update && sudo apt-get install --reinstall -y   libmpich-dev   hwloc-nox libmpich12 mpich
```