# Training

## FP8 benchmarking with Weights & Biases

Authenticate with Weights & Biases before launching a run so that every experiment is captured for benchmarking:

```bash
wandb login
```

The commands below fine-tune `nvidia/OpenMathInstruct-2` for 100k samples at a sequence length of 1024. They enable logging to the default `llm-fp8` project so you can compare runs side by side—feel free to adjust `--wandb_run_name` to keep your dashboard organized.

### Meta-Llama 3.2 1B

```bash
python train_fp8.py \
  --model_name meta-llama/Llama-3.2-1B \
  --dataset_name nvidia/OpenMathInstruct-2 \
  --batch_size 12 \
  --mixed_precision fp8 \
  --max_seq_length 1024 \
  --num_of_samples 100000 \
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
  --num_of_samples 100000 \
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
  --num_of_samples 100000 \
  --use_te \
  --use_wandb \
  --wandb_project llm-fp8 \
  --wandb_run_name llama31-8b-fp8
```

### Qwen2.5 7B

```bash
python train_fp8.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset_name nvidia/OpenMathInstruct-2 \
  --batch_size 8 \
  --mixed_precision fp8 \
  --max_seq_length 1024 \
  --num_of_samples 100000 \
  --use_wandb \
  --wandb_project llm-fp8 \
  --wandb_run_name qwen25-7b-fp8
```

> **Tip:** Transformer Engine support (`--use_te`) is available for the Meta Llama models. For Qwen2.5 we rely on the standard Hugging Face implementation, so leave `--use_te` unset.

## BF16 baseline (optional)

Switch the precision flag to generate a BF16 benchmark run with W&B logging:

```bash
python train_fp8.py \
  --model_name meta-llama/Llama-3.2-3B \
  --dataset_name nvidia/OpenMathInstruct-2 \
  --batch_size 12 \
  --mixed_precision bf16 \
  --max_seq_length 1024 \
  --num_of_samples 100000 \
  --use_wandb \
  --wandb_project llm-fp8 \
  --wandb_run_name llama32-3b-bf16
```
