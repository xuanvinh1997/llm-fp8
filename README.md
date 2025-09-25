# LLM FP8 Training

## Overview

This repository provides a clean interface for fine-tuning large language models with FP8 mixed precision training using NVIDIA's Transformer Engine. The refactored training script supports multiple Llama model variants with optimized memory usage and training speed.

## Supported Models

- **meta-llama/Llama-3.2-1B** - Llama 3.2 1B parameters (with Transformer Engine support)
- **meta-llama/Llama-3.2-3B** - Llama 3.2 3B parameters (with Transformer Engine support) 
- **meta-llama/Meta-Llama-3.1-8B** - Llama 3.1 8B parameters (with Transformer Engine support)
- **Qwen/Qwen2.5-14B** - Qwen 2.5 14B parameters (standard HuggingFace implementation)

## FP8 benchmarking with Weights & Biases

Authenticate with Weights & Biases before launching a run so that every experiment is captured for benchmarking:

```bash
wandb login
```

The commands below fine-tune using the `nvidia/OpenMathInstruct-2` dataset with the `train_1M` split at a sequence length of 1024. They enable logging to the default `llm-fp8` project so you can compare runs side by side—feel free to adjust `--wandb_run_name` to keep your dashboard organized.

### Meta-Llama 3.2 1B

```bash
python train_fp8.py \
  --model_name meta-llama/Llama-3.2-1B \
  --dataset_name nvidia/OpenMathInstruct-2 \
  --batch_size 12 \
  --mixed_precision fp8 \
  --max_seq_length 1024 \
  --split_name train_1M \
  --use_te \
  --use_wandb \
  --wandb_project llm-fp8 \
  --wandb_run_name llama32-1b-fp8
```

### Meta-Llama 3.2 3B

```bash
python train_fp8.py \
  --model_name meta-llama/Llama-3.2-3B \
  --dataset_name nvidia/OpenMathInstruct-2 \
  --batch_size 12 \
  --mixed_precision fp8 \
  --max_seq_length 1024 \
  --split_name train_1M \
  --use_te \
  --use_wandb \
  --wandb_project llm-fp8 \
  --wandb_run_name llama32-3b-fp8
```

### Meta-Llama 3.1 8B

```bash
python train_fp8.py \
  --model_name meta-llama/Meta-Llama-3.1-8B \
  --dataset_name nvidia/OpenMathInstruct-2 \
  --batch_size 8 \
  --mixed_precision fp8 \
  --max_seq_length 1024 \
  --split_name train_1M \
  --use_te \
  --use_wandb \
  --wandb_project llm-fp8 \
  --wandb_run_name llama31-8b-fp8
```

### Qwen2.5 14B

```bash
python train_fp8.py \
  --model_name Qwen/Qwen2.5-14B \
  --dataset_name nvidia/OpenMathInstruct-2 \
  --batch_size 4 \
  --mixed_precision fp8 \
  --max_seq_length 1024 \
  --split_name train_1M \
  --use_wandb \
  --wandb_project llm-fp8 \
  --wandb_run_name qwen25-14b-fp8
```

> **Note:** Llama models support Transformer Engine (`--use_te`) for optimized FP8 training with flash attention. Qwen models use the standard HuggingFace implementation (do not use `--use_te`).

## Dataset Split Options

You can specify different dataset splits using the `--split_name` parameter:

- `train_1M` - 1 million training samples (default)
- `train` - Full training set  
- `test` - Test set
- `validation` - Validation set (if available)

Example with different splits:

```bash
# Use full training set
python train_fp8.py \
  --model_name meta-llama/Llama-3.2-1B \
  --dataset_name nvidia/OpenMathInstruct-2 \
  --split_name train \
  --batch_size 12 \
  --mixed_precision fp8 \
  --use_te

# Use test set for evaluation
python train_fp8.py \
  --model_name meta-llama/Llama-3.2-1B \
  --dataset_name nvidia/OpenMathInstruct-2 \
  --split_name test \
  --batch_size 12 \
  --mixed_precision fp8 \
  --use_te
```

## Advanced FP8 Configurations

### MXFP8 scenario (Transformer Engine only)

For benchmarks that compare Transformer Engine's MXFP8 recipe against the default FP8 configuration, add the `--fp8_scenario mxfp8` flag alongside `--mixed_precision fp8` and `--use_te`:

```bash
python train_fp8.py \
  --model_name meta-llama/Llama-3.2-3B \
  --dataset_name nvidia/OpenMathInstruct-2 \
  --batch_size 12 \
  --mixed_precision fp8 \
  --fp8_scenario mxfp8 \
  --max_seq_length 1024 \
  --split_name train_1M \
  --use_te \
  --use_wandb \
  --wandb_project llm-fp8 \
  --wandb_run_name llama32-3b-mxfp8
```

The MXFP8 option falls back to the standard FP8 recipe if the installed `transformer_engine` package does not expose an MXFP8 format.

## BF16 baseline (optional)

Switch the precision flag to generate a BF16 benchmark run with W&B logging:

```bash
python train_fp8.py \
  --model_name meta-llama/Llama-3.2-3B \
  --dataset_name nvidia/OpenMathInstruct-2 \
  --batch_size 12 \
  --mixed_precision bf16 \
  --max_seq_length 1024 \
  --split_name train_1M \
  --use_wandb \
  --wandb_project llm-fp8 \
  --wandb_run_name llama32-3b-bf16
```

## Quick Start without W&B

For quick testing without Weights & Biases logging:

```bash
python train_fp8.py \
  --model_name meta-llama/Llama-3.2-1B \
  --dataset_name nvidia/OpenMathInstruct-2 \
  --batch_size 8 \
  --mixed_precision fp8 \
  --max_seq_length 512 \
  --split_name train_1M \
  --num_epochs 1 \
  --use_te
```

## Multi-GPU Training with Deepspeed
```bash
torchrun --nproc_per_node=2         train_multi_gpu.py         --model_name meta-llama/Llama-3.2-3B         --dataset_name nvidia/OpenMathInstruct-2         --batch_size 1         --sharding_mode fsdp_full         --mixed_precision fp8         --use_te         --gradient_checkpointing         --max_seq_length 512         --learning_rate 1e-5         --gradient_accumulation_steps 4         --empty_cache_freq 25
```