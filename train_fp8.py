#!/usr/bin/env python3
"""
FP8 Training Script for Qwen2.5 Model

This script demonstrates how to convert a Qwen2.5 model to FP8 format and train it
with FP8 quantization for improved performance and memory efficiency.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        AutoConfig,
        get_linear_schedule_with_warmup
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers not available")

# Import our custom modules
from model import ModelArgs, Transformer
from qwen2_model import Qwen2GQA
from qwen2_config import get_qwen2_config
from qwen2_converter import convert_qwen2_to_fp8
from kernel import FP8_E4M3, FP8_E5M2, act_quant, weight_dequant, fp8_gemm
from custom_datasets import load_and_prepare_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_fp8.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for FP8 training"""
    # Model settings
    model_name: str = "Qwen/Qwen2.5-0.5B"
    max_seq_len: int = 2048
    vocab_size: int = 151936
    
    # Training settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 10000
    eval_steps: int = 500
    save_steps: int = 1000
    
    # FP8 settings
    use_fp8: bool = True
    fp8_block_size: int = 128
    fp8_dtype: torch.dtype = FP8_E4M3
    
    # Data settings
    dataset_name: str = "nvidia/OpenMathInstruct-2"
    max_samples: int = 50000  # Limit for demo
    
    # Infrastructure
    output_dir: str = "./fp8_qwen2_checkpoints"
    log_interval: int = 10
    seed: int = 42
    mixed_precision: bool = True
    compile_model: bool = False  # torch.compile for speedup


class FP8Qwen2Wrapper(nn.Module):
    """
    Wrapper around Qwen2 model with FP8 quantization support
    """
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.model_args = get_qwen2_config()
        
        # Update model args with training config
        self.model_args.max_seq_len = config.max_seq_len
        self.model_args.vocab_size = config.vocab_size
        self.model_args.dtype = "fp8" if config.use_fp8 else "bf16"
        
        # Initialize the custom FP8-capable model
        self.model = Transformer(self.model_args)
        
        # FP8 quantization states
        self.fp8_enabled = config.use_fp8
        self.block_size = config.fp8_block_size
        self.fp8_dtype = config.fp8_dtype
        
        # Store original weights for conversion
        self._original_weights = {}
        self._fp8_scales = {}
        
    def enable_fp8(self):
        """Enable FP8 quantization for all linear layers"""
        if not self.fp8_enabled:
            return
            
        logger.info("Converting model to FP8...")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Store original weights
                self._original_weights[name] = module.weight.data.clone()
                
                # Quantize weights to FP8
                quantized_weight, scale = act_quant(
                    module.weight.data, 
                    block_size=self.block_size,
                    dtype=self.fp8_dtype
                )
                
                # Replace weight with quantized version
                module.weight.data = quantized_weight
                self._fp8_scales[name] = scale
                
        logger.info(f"Converted {len(self._original_weights)} linear layers to FP8")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass with optional FP8 quantization"""
        if self.fp8_enabled:
            # Apply FP8 quantization to inputs
            input_embeds = self.model.tok_embeddings(input_ids)
            quantized_inputs, input_scale = act_quant(
                input_embeds, 
                block_size=self.block_size,
                dtype=self.fp8_dtype
            )
            
            # Forward through quantized model
            outputs = self.model(input_ids, start_pos=0)
        else:
            outputs = self.model(input_ids, start_pos=0)
        
        if labels is not None:
            # Calculate loss
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            return {"loss": loss, "logits": outputs}
        
        return {"logits": outputs}


def load_pretrained_qwen2(config: TrainingConfig) -> Tuple[FP8Qwen2Wrapper, AutoTokenizer]:
    """Load pre-trained Qwen2.5 model and convert to FP8 format"""
    if not HF_AVAILABLE:
        raise ImportError("transformers library required for loading pre-trained models")
    
    logger.info(f"Loading pre-trained model: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load HuggingFace model for weight initialization
    hf_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu"  # Load to CPU first
    )
    
    # Create our FP8 wrapper
    fp8_model = FP8Qwen2Wrapper(config)
    
    # Convert weights from HuggingFace format
    logger.info("Converting HuggingFace weights to FP8 format...")
    convert_qwen2_to_fp8(hf_model, fp8_model.model, config.use_fp8)
    
    # Enable FP8 quantization
    if config.use_fp8:
        fp8_model.enable_fp8()
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp8_model = fp8_model.to(device)
    
    logger.info(f"Model loaded and converted to {device}")
    return fp8_model, tokenizer


def setup_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Setup optimizer with proper parameter grouping"""
    # Separate decay and no-decay parameters
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(nd in name for nd in ["bias", "norm", "embedding"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": config.weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    return optimizer


def collate_fn(batch, tokenizer, max_length: int = 2048):
    """Custom collate function for training data"""
    texts = [item["text"] for item in batch]
    
    # Tokenize
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    # Use input_ids as labels (causal LM)
    labels = input_ids.clone()
    # Mask padding tokens in labels
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def evaluate_model(model: nn.Module, eval_dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs["loss"]
            
            # Calculate number of valid tokens
            valid_tokens = (batch["labels"] != -100).sum().item()
            
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    model.train()
    return {"eval_loss": avg_loss, "perplexity": perplexity}


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   step: int, config: TrainingConfig, metrics: Dict[str, float]):
    """Save model checkpoint"""
    os.makedirs(config.output_dir, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": asdict(config),
        "metrics": metrics
    }
    
    checkpoint_path = os.path.join(config.output_dir, f"checkpoint_step_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Save config for easy loading
    config_path = os.path.join(config.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def train_step(model: nn.Module, batch: Dict[str, torch.Tensor], 
               optimizer: torch.optim.Optimizer, scheduler: Any,
               config: TrainingConfig, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> float:
    """Single training step"""
    if config.mixed_precision and scaler is not None:
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs["loss"] / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        return loss.item() * config.gradient_accumulation_steps
    else:
        outputs = model(**batch)
        loss = outputs["loss"] / config.gradient_accumulation_steps
        loss.backward()
        return loss.item() * config.gradient_accumulation_steps


def main():
    parser = argparse.ArgumentParser(description="Train Qwen2.5 with FP8 quantization")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B", help="Model name")
    parser.add_argument("--output_dir", default="./fp8_qwen2_checkpoints", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=10000, help="Max training steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--use_fp8", action="store_true", help="Enable FP8 quantization")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        use_fp8=args.use_fp8,
        mixed_precision=args.mixed_precision
    )
    
    # Set random seed
    torch.manual_seed(config.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_pretrained_qwen2(config)
    
    # Setup data
    logger.info("Loading dataset...")
    if not HF_AVAILABLE:
        logger.error("Cannot load dataset without transformers library")
        return
    
    train_dataloader, eval_dataloader = load_and_prepare_dataset(
        dataset_name=config.dataset_name,
        tokenizer=tokenizer,
        train_batch_size=config.batch_size,
        val_batch_size=config.batch_size
    )
    
    # Setup optimizer and scheduler
    optimizer = setup_optimizer(model, config)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )
    
    # Setup mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
    
    # Compile model for speedup (if supported)
    if config.compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model)
        logger.info("Model compiled for faster training")
    
    # Training loop
    logger.info("Starting training...")
    model.train()
    
    step = 0
    running_loss = 0.0
    best_eval_loss = float('inf')
    
    # Custom collate function with tokenizer
    def custom_collate(batch):
        return collate_fn(batch, tokenizer, config.max_seq_len)
    
    # Update dataloader collate function
    train_dataloader.collate_fn = custom_collate
    eval_dataloader.collate_fn = custom_collate
    
    start_time = time.time()
    
    try:
        while step < config.max_steps:
            for batch in train_dataloader:
                if step >= config.max_steps:
                    break
                
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Training step
                loss = train_step(model, batch, optimizer, scheduler, config, scaler)
                running_loss += loss
                
                # Gradient accumulation
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                
                step += 1
                
                # Logging
                if step % config.log_interval == 0:
                    avg_loss = running_loss / config.log_interval
                    elapsed_time = time.time() - start_time
                    tokens_per_sec = (step * config.batch_size * config.max_seq_len) / elapsed_time
                    
                    logger.info(
                        f"Step {step}/{config.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"Tokens/sec: {tokens_per_sec:.0f}"
                    )
                    running_loss = 0.0
                
                # Evaluation
                if step % config.eval_steps == 0:
                    logger.info("Running evaluation...")
                    eval_metrics = evaluate_model(model, eval_dataloader, device)
                    logger.info(
                        f"Eval Loss: {eval_metrics['eval_loss']:.4f} | "
                        f"Perplexity: {eval_metrics['perplexity']:.2f}"
                    )
                    
                    # Save best model
                    if eval_metrics['eval_loss'] < best_eval_loss:
                        best_eval_loss = eval_metrics['eval_loss']
                        save_checkpoint(model, optimizer, step, config, eval_metrics)
                
                # Regular checkpointing
                if step % config.save_steps == 0:
                    eval_metrics = evaluate_model(model, eval_dataloader, device)
                    save_checkpoint(model, optimizer, step, config, eval_metrics)
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    # Final evaluation and save
    logger.info("Training completed. Running final evaluation...")
    final_metrics = evaluate_model(model, eval_dataloader, device)
    save_checkpoint(model, optimizer, step, config, final_metrics)
    
    logger.info(
        f"Final Results - Loss: {final_metrics['eval_loss']:.4f} | "
        f"Perplexity: {final_metrics['perplexity']:.2f}"
    )
    
    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()