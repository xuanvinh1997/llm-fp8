#!/usr/bin/env python3
"""
Fine-tuning script for LLM models with FP8 precision using Transformer Engine.

This script provides a clean interface for fine-tuning large language models
with optional FP8 mixed precision training using NVIDIA's Transformer Engine.
"""

import argparse
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import tqdm
import wandb
from accelerate import Accelerator
try:
    from accelerate.utils.dataclasses import TERecipeKwargs as FP8RecipeKwargs
except ImportError:
    from accelerate.utils.dataclasses import FP8RecipeKwargs
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from config import TrainingConfig
from data import DataManager
from utils import GPUMonitor


class ModelManager:
    """Handles model downloading, initialization, and configuration."""
    
    SUPPORTED_MODELS = {
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "Qwen/Qwen2.5-14B",
        "Qwen/Qwen2.5-1.5B",
    }
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def ensure_model_downloaded(self) -> str:
        """Download model if needed and return cache directory."""
        if self.config.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {self.config.model_name}")
        
        from huggingface_hub import login, snapshot_download
        
        # Try to login if token is provided
        if self.config.hf_access_token:
            try:
                login(self.config.hf_access_token)
            except Exception as e:
                print(f"HuggingFace login issue: {e}")
        
        cache_dir = snapshot_download(
            repo_id=self.config.model_name,
            cache_dir=self.config.weights_cache_dir or None,
        )
        
        print(f"Model cached at: {cache_dir}")
        return cache_dir
    
    def create_model(self) -> torch.nn.Module:
        """Initialize and return the model."""
        cache_dir = self.ensure_model_downloaded()
        
        # Load configuration with flash attention
        config = AutoConfig.from_pretrained(cache_dir)
        config._attn_implementation = "flash_attention_2"
        
        # Create model based on whether to use Transformer Engine
        if self.config.use_te:
            model = self._create_te_model(cache_dir, config)
        else:
            model = self._create_standard_model(cache_dir, config)
        
        # Configure model
        model = model.cuda()
        model.config.use_cache = False
        
        return model
    
    def _create_te_model(self, cache_dir: str, config) -> torch.nn.Module:
        """Create model with Transformer Engine."""
        if self.config.fp8_scenario == "mxfp8":
            from te_llama_mxfp8 import TELlamaForCausalLM
        else:
            from te_llama import TELlamaForCausalLM

        return TELlamaForCausalLM.from_pretrained_local(
            cache_dir,
            config=config,
            torch_dtype=torch.bfloat16,
        )
    
    def _create_standard_model(self, cache_dir: str, config) -> torch.nn.Module:
        """Create standard model."""
        return AutoModelForCausalLM.from_pretrained(
            cache_dir,
            config=config,
            torch_dtype=torch.bfloat16,
        )

class FP8Handler:
    """Handles FP8 configuration and recipe creation."""

    @staticmethod
    def create_fp8_kwargs(config: TrainingConfig) -> Optional[list]:
        """Create FP8 kwargs handlers based on configuration."""
        if config.mixed_precision != "fp8":
            return None

        # Import TERecipeKwargs (the new name) or fall back to FP8RecipeKwargs
        try:
            from accelerate.utils.dataclasses import TERecipeKwargs
        except ImportError:
            from accelerate.utils.dataclasses import FP8RecipeKwargs as TERecipeKwargs

        if config.fp8_scenario == "default":
            # Use default FP8 settings
            return [TERecipeKwargs()]

        elif config.fp8_scenario == "mxfp8":
            return FP8Handler._create_mxfp8_kwargs(TERecipeKwargs)

        else:
            raise ValueError(f"Unsupported FP8 scenario: {config.fp8_scenario}")

    @staticmethod
    def _create_mxfp8_kwargs(TERecipeKwargs):
        """Create MXFP8 specific kwargs."""
        # MXFP8 configuration is handled directly in te_llama_mxfp8.py
        # We just need to return appropriate accelerate kwargs
        print("Using MXFP8 configuration (handled in te_llama_mxfp8.py)")

        # Return TERecipeKwargs with E4M3 format for MXFP8
        return [TERecipeKwargs(
            fp8_format="E4M3",  # Use E4M3 for MXFP8
            amax_history_len=16,  # Shorter history for MXFP8
            amax_compute_algo="max",
            margin=0,
            interval=1,
        )]


class Trainer:
    """Main training orchestrator."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.gpu_monitor = GPUMonitor()
    
    def setup_training(self, model: torch.nn.Module) -> Tuple:
        """Set up accelerator, optimizers, and dataloaders."""
        # Create FP8 kwargs if needed
        fp8_kwargs = None
        if self.config.mixed_precision == "fp8" and self.config.use_te:
            # Only use FP8 kwargs with Accelerate when using TE models
            fp8_kwargs = FP8Handler.create_fp8_kwargs(self.config)
        print(self.config.mixed_precision)
        # Initialize accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            kwargs_handlers=fp8_kwargs,
        )
        
        # Create dataloaders
        data_manager = DataManager(self.config)
        train_loader, eval_loader = data_manager.create_dataloaders(accelerator)
        
        # Calculate training steps
        num_batches = len(train_loader)
        total_steps = (num_batches * self.config.num_epochs // 
                      self.config.gradient_accumulation_steps)
        
        # Create optimizer and scheduler
        optimizer = AdamW(
            params=model.parameters(),
            lr=self.config.learning_rate,
            fused=True,
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Prepare everything with accelerator
        model, optimizer, train_loader, scheduler, eval_loader = accelerator.prepare(
            model, optimizer, train_loader, scheduler, eval_loader
        )
        
        return accelerator, model, optimizer, train_loader, scheduler, eval_loader
    
    def train(self, model, accelerator, train_loader, eval_loader, 
              optimizer, scheduler, writer: SummaryWriter, wandb_run=None):
        """Execute the training loop."""
        model.train()
        step_count = 0
        
        # Setup timing
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_time.record()
        
        total_steps = (len(train_loader) * self.config.num_epochs // 
                      self.config.gradient_accumulation_steps)
        
        # Training loop with progress bar
        with tqdm.tqdm(total=total_steps, desc="Training", unit="step") as pbar:
            for epoch in range(1, self.config.num_epochs + 1):
                accelerator.print(f"Epoch {epoch}/{self.config.num_epochs}")
                
                # Training epoch
                step_count = self._train_epoch(
                    model, accelerator, train_loader, optimizer, scheduler,
                    writer, wandb_run, step_count, total_steps, pbar
                )
                
                # Evaluation
                self._evaluate(model, accelerator, eval_loader, writer, 
                              wandb_run, epoch, step_count)
        
        # Finalize training
        self._finalize_training(accelerator, start_time, end_time, step_count,
                               writer, wandb_run)
    
    def _train_epoch(self, model, accelerator, train_loader, optimizer, scheduler,
                    writer, wandb_run, step_count, total_steps, pbar):
        """Train for one epoch."""
        for batch in train_loader:
            step_start = time.perf_counter()
            
            with accelerator.accumulate(model):
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Check for non-finite loss
                if not torch.isfinite(loss):
                    accelerator.print("Non-finite loss detected, stopping training.")
                    return step_count
                
                # Backward pass
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                step_count += 1
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
            
            # Log metrics
            step_duration = time.perf_counter() - step_start
            if step_count % 50 == 0:
                self._log_training_metrics(
                    loss.item(), step_duration, step_count, total_steps,
                    writer, wandb_run, accelerator
                )
        
        return step_count
    
    def _evaluate(self, model, accelerator, eval_loader, writer, wandb_run,
                 epoch, step_count):
        """Run evaluation with perplexity and other metrics."""
        model.eval()
        eval_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in eval_loader:
                outputs = model(**batch)
                loss = outputs.loss.detach().float()
                eval_loss += loss

                # Count tokens for accurate perplexity calculation
                # Exclude padding tokens if they exist
                if 'attention_mask' in batch:
                    total_tokens += batch['attention_mask'].sum().item()
                else:
                    total_tokens += batch['input_ids'].numel()

        # Calculate metrics
        avg_eval_loss = eval_loss / len(eval_loader)
        perplexity = torch.exp(avg_eval_loss).item()

        # Log evaluation metrics
        writer.add_scalar("Eval/Loss", avg_eval_loss, epoch)
        writer.add_scalar("Eval/Perplexity", perplexity, epoch)
        writer.add_scalar("Eval/Tokens", total_tokens, epoch)

        if wandb_run is not None:
            wandb_run.log({
                "Eval/Loss": avg_eval_loss,
                "Eval/Perplexity": perplexity,
                "Eval/Tokens": total_tokens
            }, step=step_count)

        accelerator.print(
            f"Eval metrics - Loss: {avg_eval_loss:.4f}, "
            f"Perplexity: {perplexity:.2f}, Tokens: {total_tokens}"
        )
        model.train()
    
    def _log_training_metrics(self, loss, step_duration, step_count, total_steps,
                             writer, wandb_run, accelerator):
        """Log training metrics."""
        mem_mb = self.gpu_monitor.get_memory_usage()[0]
        train_perplexity = torch.exp(torch.tensor(loss)).item()

        # TensorBoard logging
        writer.add_scalar("Train/Loss", loss, step_count)
        writer.add_scalar("Train/Perplexity", train_perplexity, step_count)
        writer.add_scalar("Train/StepTime_s", step_duration, step_count)
        writer.add_scalar("Train/GPU_Memory_MB", mem_mb, step_count)

        # Weights & Biases logging
        if wandb_run is not None:
            wandb_run.log({
                "Train/Loss": loss,
                "Train/Perplexity": train_perplexity,
                "Train/StepTime_s": step_duration,
                "Train/GPU_Memory_MB": mem_mb,
                "Train/Progress": step_count / total_steps,
            }, step=step_count)

        # Console logging
        accelerator.print(
            f"Step {step_count}: loss={loss:.4f}, ppl={train_perplexity:.2f}, "
            f"step_time={step_duration:.2f}s, gpu_mem={mem_mb:.0f}MB"
        )
    
    def _finalize_training(self, accelerator, start_time, end_time, step_count,
                          writer, wandb_run):
        """Finalize training and log summary metrics."""
        torch.cuda.synchronize()
        end_time.record()
        accelerator.end_training()
        
        # Calculate and log timing metrics
        ms_per_step = start_time.elapsed_time(end_time) / step_count
        writer.add_scalar("Train/TimePerStep_ms", ms_per_step, 0)
        writer.flush()
        writer.close()
        
        if wandb_run is not None:
            wandb_run.log({"Train/TimePerStep_ms": ms_per_step})
            wandb_run.finish()
        
        accelerator.print(
            f"Training completed: {step_count} steps, {ms_per_step:.0f} ms/step"
        )


class ModelSaver:
    """Handles model saving functionality."""
    
    @staticmethod
    def save_model(model, config: TrainingConfig, accelerator: Accelerator):
        """Save the trained model and tokenizer."""
        # Create output directory
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_to_save = accelerator.unwrap_model(model)
        model_to_save.save_pretrained(str(output_path))
        
        # Save tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            if getattr(tokenizer, "pad_token", None) is None:
                if getattr(tokenizer, "eos_token", None) is not None:
                    tokenizer.pad_token = tokenizer.eos_token
            tokenizer.save_pretrained(str(output_path))
        except Exception as e:
            print(f"Warning: tokenizer not saved: {e}")
        
        print(f"Model saved to: {output_path}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama with TE and FP8 using Accelerate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and data arguments
    model_group = parser.add_argument_group("Model and Data")
    model_group.add_argument(
        "--model_name", type=str, required=True,
        help="HuggingFace model identifier (e.g. meta-llama/Llama-3.2-1B)"
    )
    model_group.add_argument(
        "--dataset_name", type=str, required=True,
        help="Dataset identifier (e.g. nvidia/OpenMathInstruct-2)"
    )
    model_group.add_argument(
        "--dataset_text_field", type=str, default="text",
        help="Field name in dataset for raw text"
    )
    model_group.add_argument(
        "--use_te", action="store_true",
        help="Enable Transformer Engine (TELlama) model instantiation"
    )
    
    # Training hyperparameters
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument(
        "--mixed_precision", type=str, default="bf16",
        choices=["bf16", "fp8"], help="Mixed precision mode"
    )
    train_group.add_argument(
        "--fp8_scenario", type=str, default="default",
        choices=["default", "mxfp8"],
        help="FP8 recipe to use when mixed_precision is fp8"
    )
    train_group.add_argument(
        "--batch_size", type=int, default=8,
        help="Per-device batch size for training"
    )
    train_group.add_argument(
        "--eval_batch_size", type=int,
        help="Per-device batch size for evaluation (defaults to train batch_size)"
    )
    train_group.add_argument(
        "--max_seq_length", type=int, default=256,
        help="Maximum sequence length for input text"
    )
    train_group.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Gradient accumulation steps"
    )
    train_group.add_argument(
        "--num_epochs", type=int, default=3,
        help="Number of epochs to train"
    )
    train_group.add_argument(
        "--learning_rate", type=float, default=1.41e-5,
        help="Initial learning rate"
    )
    train_group.add_argument(
        "--num_warmup_steps", type=int, default=100,
        help="Number of warmup steps for scheduler"
    )
    train_group.add_argument(
        "--split_name", type=str, default="train_1M",
        help="Dataset split to use (e.g., train, train_1M, test)"
    )
    train_group.add_argument(
        "--num_of_samples", type=int,
        help="Number of samples to use from the dataset (defaults to all)"
    )
    
    # Logging and saving
    log_group = parser.add_argument_group("Logging and Saving")
    log_group.add_argument(
        "--log_dir", type=str, default="./runs",
        help="Directory for TensorBoard logs"
    )
    log_group.add_argument(
        "--output_dir", type=str, default="./saved_model",
        help="Where to save the fine-tuned model"
    )
    log_group.add_argument(
        "--use_wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )
    log_group.add_argument(
        "--wandb_project", type=str, default="llm-fp8",
        help="Weights & Biases project name"
    )
    log_group.add_argument(
        "--wandb_run_name", type=str,
        help="Weights & Biases run name"
    )
    
    # Technical settings
    technical_group = parser.add_argument_group("Technical Settings")
    technical_group.add_argument(
        "--weights_cache_dir", type=str, default="",
        help="Cache directory for model weights"
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = TrainingConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_text_field=args.dataset_text_field,
        mixed_precision=args.mixed_precision,
        fp8_scenario=args.fp8_scenario,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.num_warmup_steps,
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        weights_cache_dir=args.weights_cache_dir,
        max_seq_length=args.max_seq_length,
        split_name=args.split_name,
        num_of_samples=args.num_of_samples,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_te=args.use_te,
    )
    
    # Initialize components
    model_manager = ModelManager(config)
    trainer = Trainer(config)
    
    # Create model
    print("Initializing model...")
    model = model_manager.create_model()
    
    # Setup training
    print("Setting up training...")
    accelerator, model, optimizer, train_loader, scheduler, eval_loader = (
        trainer.setup_training(model)
    )
    
    # Setup logging
    writer = SummaryWriter(log_dir=config.log_dir)
    wandb_run = None
    if config.use_wandb:
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            dir=config.log_dir,
            config=vars(config)
        )
    
    # Train model
    print("Starting training...")
    trainer.train(
        model, accelerator, train_loader, eval_loader,
        optimizer, scheduler, writer, wandb_run
    )
    
    # Save model
    print("Saving model...")
    ModelSaver.save_model(model, config, accelerator)
    
    print("Training complete!")


if __name__ == "__main__":
    main()