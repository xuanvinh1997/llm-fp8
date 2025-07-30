

# training

## fp8
```bash
python train_fp8.py  --model_name meta-llama/Llama-3.2-3B --dataset_name nvidia/OpenMathInstruct-2 --batch_size 12 --mixed_precision fp8 --max_seq_length 1024 --num_of_samples 100000 --use_te --use_wandb --wandb_run_name fp8
```

## bf16
```bash
python train_bf16.py  --model_name meta-llama/Llama-3.2-3B --dataset_name nvidia/OpenMathInstruct-2 --batch_size 12 --mixed_precision bf16 --max_seq_length 1024 --num_of_samples 100000 --use_wandb --wandb_run_name bf16
```