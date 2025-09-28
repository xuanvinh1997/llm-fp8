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

### Meta-Llama 3.2 3B
3b - our fp8 method
```bash
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
python train_fp8.py   --model_name meta-llama/Llama-3.2-3B   --dataset_name nvidia/OpenMathInstruct-2   --batch_size 16   --mixed_precision fp8   --max_seq_length 512   --split_name train_1M   --use_te   --use_wandb   --wandb_project llm-fp8   --wandb_run_name llama32-3b-fp8 --num_of_samples 100000
```
3b - mxfp8 method
```bash
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
python train_fp8.py   --model_name meta-llama/Llama-3.2-3B   --dataset_name nvidia/OpenMathInstruct-2   --batch_size 16   --mixed_precision fp8   --fp8_scenario mxfp8   --max_seq_length 512   --split_name train_1M   --use_te   --use_wandb   --wandb_project llm-fp8   --wandb_run_name llama32-3b-mxfp8 --num_of_samples 100000
```

3b - fp8 hybrid method (instead of mxfp8)
```bash
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
python train_fp8.py   --model_name meta-llama/Llama-3.2-3B   --dataset_name nvidia/OpenMathInstruct-2   --batch_size 16   --mixed_precision fp8   --fp8_scenario hybrid   --max_seq_length 512   --split_name train_1M   --use_te   --use_wandb   --wandb_project llm-fp8   --wandb_run_name llama32-3b-hybrid --num_of_samples 100000
```

3b - bf16 method
```bash
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
python train_fp8.py   --model_name meta-llama/Llama-3.2-3B   --dataset_name nvidia/OpenMathInstruct-2   --batch_size 16   --mixed_precision bf16   --max_seq_length 512   --split_name train_1M   --use_te   --use_wandb   --wandb_project llm-fp8   --wandb_run_name llama32-3b-bf16 --num_of_samples 100000
```

8b - our fp8 method
```bash
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
python train_fp8.py   --model_name meta-llama/Llama-3.1-8B   --dataset_name nvidia/OpenMathInstruct-2   --batch_size 12   --mixed_precision fp8   --max_seq_length 512   --split_name train_1M   --use_te   --use_wandb   --wandb_project llm-fp8   --wandb_run_name llama31-8b-fp8 --num_of_samples 100000
```
8b - mxfp8 method
```bash
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
python train_fp8.py   --model_name meta-llama/Llama-3.1-8B   --dataset_name nvidia/OpenMathInstruct-2   --batch_size 12   --mixed_precision fp8   --fp8_scenario mxfp8   --max_seq_length 512   --split_name train_1M   --use_te   --use_wandb   --wandb_project llm-fp8   --wandb_run_name llama31-8b-mxfp8 --num_of_samples 100000
```
8b - bf16 method
```bash
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
python train_fp8.py   --model_name meta-llama/Llama-3.1-8B   --dataset_name nvidia/OpenMathInstruct-2   --batch_size 12   --mixed_precision bf16   --max_seq_length 512   --split_name train_1M   --use_te   --use_wandb   --wandb_project llm-fp8   --wandb_run_name llama31-8b-bf16 --num_of_samples 100000
```

## Multi-GPU Training with Deepspeed
```bash
torchrun --nproc_per_node=4         train_multi_gpu.py         --model_name meta-llama/Llama-3.2-3B         --dataset_name nvidia/OpenMathInstruct-2   --split_name train_1M         --batch_size 1         --sharding_mode fsdp_full         --mixed_precision fp8         --use_te         --gradient_checkpointing         --max_seq_length 512         --learning_rate 1e-5         --gradient_accumulation_steps 4         --empty_cache_freq 25
```