#!/usr/bin/env python3
"""
Production-ready language model training script using Accelerate
Optimized for instruction tuning with FP8 support via official Accelerate API
Based on: https://huggingface.co/docs/accelerate/usage_guides/low_precision_training
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from transformers import (
    DataCollatorForSeq2Seq,
    AutoModelForCausalLM,
    AutoTokenizer
)

from config import TrainingConfig
from dataset import InstructionDataset
from loss import LossManager
from model import ModelManager
from optimizer import OptimizerManager

# Configure standard logging for pre-accelerator initialization
logging.basicConfig(level=logging.INFO)
std_logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with comprehensive options"""
    parser = argparse.ArgumentParser(
        description="Fine-tune language models with FP8 precision support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and data arguments
    model_group = parser.add_argument_group('Model and Data')
    model_group.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct",
                            help='HuggingFace model ID or local path')
    model_group.add_argument('--tokenizer_name', type=str, default=None,
                            help='Tokenizer name (defaults to model_name)')
    model_group.add_argument('--dataset_path', type=str, required=True,
                            help='Path to instruction dataset (JSON file or HF dataset ID)')
    model_group.add_argument('--output_dir', type=str, default="outputs/trained_model",
                            help='Output directory for trained model')
    
    # Training hyperparameters
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--num_epochs', type=int, default=3,
                            help='Number of training epochs')
    train_group.add_argument('--batch_size', type=int, default=4,
                            help='Training batch size per device')
    train_group.add_argument('--gradient_accumulation_steps', type=int, default=8,
                            help='Gradient accumulation steps')
    train_group.add_argument('--learning_rate', type=float, default=2e-5,
                            help='Peak learning rate')
    train_group.add_argument('--weight_decay', type=float, default=0.01,
                            help='Weight decay coefficient')
    train_group.add_argument('--max_length', type=int, default=2048,
                            help='Maximum sequence length')
    train_group.add_argument('--warmup_ratio', type=float, default=0.1,
                            help='Warmup ratio for learning rate scheduler')
    
    # Precision and optimization
    precision_group = parser.add_argument_group('Precision and Optimization')
    precision_group.add_argument('--mixed_precision', type=str, default="bf16",
                               choices=["no", "fp16", "bf16", "fp8"],
                               help='Mixed precision training mode')
    precision_group.add_argument('--fp8_backend', type=str, default="te",
                               choices=["te", "msamp", "ao"],
                               help='FP8 backend (only used with --mixed_precision=fp8)')
    precision_group.add_argument('--te_fp8_format', type=str, default="HYBRID",
                               choices=["E4M3", "E5M2", "HYBRID"],
                               help='TransformersEngine FP8 format')
    precision_group.add_argument('--msamp_opt_level', type=str, default="O2",
                               choices=["O1", "O2"],
                               help='MS-AMP optimization level')
    precision_group.add_argument('--gradient_checkpointing', action='store_true', default=True,
                               help='Enable gradient checkpointing')
    
    # System and performance
    system_group = parser.add_argument_group('System and Performance')
    system_group.add_argument('--dataloader_num_workers', type=int, default=4,
                            help='Number of dataloader workers')
    system_group.add_argument('--seed', type=int, default=42,
                            help='Random seed for reproducibility')
    system_group.add_argument('--resume_from_checkpoint', type=str, default=None,
                            help='Resume training from checkpoint path')
    
    # Logging and monitoring
    logging_group = parser.add_argument_group('Logging and Monitoring')
    logging_group.add_argument('--logging_steps', type=int, default=10,
                             help='Log every N training steps')
    logging_group.add_argument('--save_steps', type=int, default=500,
                             help='Save checkpoint every N steps')
    logging_group.add_argument('--eval_steps', type=int, default=500,
                             help='Evaluate every N steps')
    logging_group.add_argument('--save_total_limit', type=int, default=3,
                             help='Maximum number of checkpoints to keep')
    logging_group.add_argument('--use_wandb', action='store_true',
                             help='Enable Weights & Biases logging')
    logging_group.add_argument('--wandb_project', type=str, default="llm-instruction-tuning",
                             help='W&B project name')
    logging_group.add_argument('--wandb_run_name', type=str, default=None,
                             help='W&B run name')
    
    # Presets for common configurations
    preset_group = parser.add_argument_group('Configuration Presets')
    preset_group.add_argument('--preset', type=str, choices=['bf16', 'fp8_te', 'fp8_msamp', 'fp8_ao'],
                            help='Use predefined configuration preset')
    
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> None:
    """Apply configuration presets to override default values"""
    if args.preset == 'bf16':
        args.mixed_precision = "bf16"
        args.batch_size = 4
        args.gradient_accumulation_steps = 8
    elif args.preset == 'fp8_te':
        args.mixed_precision = "fp8"
        args.fp8_backend = "te"
        args.te_fp8_format = "HYBRID"
        args.batch_size = 8
        args.gradient_accumulation_steps = 4
    elif args.preset == 'fp8_msamp':
        args.mixed_precision = "fp8"
        args.fp8_backend = "msamp"
        args.msamp_opt_level = "O2"
        args.batch_size = 8
        args.gradient_accumulation_steps = 4
    elif args.preset == 'fp8_ao':
        args.mixed_precision = "fp8"
        args.fp8_backend = "ao"
        args.batch_size = 8
        args.gradient_accumulation_steps = 4


def args_to_config(args: argparse.Namespace) -> TrainingConfig:
    """Convert command line arguments to TrainingConfig"""
    return TrainingConfig(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name or args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        mixed_precision=args.mixed_precision,
        fp8_backend=args.fp8_backend,
        te_fp8_format=args.te_fp8_format,
        msamp_opt_level=args.msamp_opt_level,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint
    )


class Trainer:
    """Main training class with official FP8 support"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._setup_accelerator()
        self._setup_logging()
        self.logger = get_logger(__name__)
        self._log_config()
        
    def _setup_accelerator(self):
        """Initialize Accelerator with official FP8 support"""
        project_config = ProjectConfiguration(
            project_dir=self.config.output_dir,
            logging_dir=str(Path(self.config.output_dir) / "logs")
        )
        
        kwarg_handlers = []
        if self.config.mixed_precision == "fp8":
            fp8_kwargs = self.config.get_fp8_kwargs()
            if fp8_kwargs:
                kwarg_handlers.append(fp8_kwargs)
                std_logger.info(f"FP8 enabled with {self.config.fp8_backend} backend")
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            project_config=project_config,
            log_with="wandb" if self.config.use_wandb else None,
            kwargs_handlers=kwarg_handlers
        )
        
    def _setup_logging(self):
        """Setup logging and monitoring"""
        if self.config.use_wandb and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.config.wandb_project,
                config=self.config.__dict__,
                init_kwargs={"wandb": {"name": self.config.wandb_run_name}}
            )
    
    def _log_config(self):
        """Log training configuration"""
        self.logger.info("=== Training Configuration ===")
        self.logger.info(f"Model: {self.config.model_name}")
        self.logger.info(f"Dataset: {self.config.dataset_path}")
        self.logger.info(f"Output: {self.config.output_dir}")
        self.logger.info(f"Precision: {self.config.mixed_precision}")
        if self.config.mixed_precision == "fp8":
            self.logger.info(f"FP8 Backend: {self.config.fp8_backend}")
        self.logger.info(f"Batch Size: {self.config.batch_size}")
        self.logger.info(f"Learning Rate: {self.config.learning_rate}")
        self.logger.info("=" * 30)
            
    def train(self):
        """Main training loop with automatic FP8 handling"""
        set_seed(self.config.seed)
        
        # Setup components
        model, tokenizer = ModelManager.setup_model_and_tokenizer(self.config)
        train_dataset = InstructionDataset(self.config.dataset_path, tokenizer, self.config.max_length)
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding=True, return_tensors="pt"
        )
        
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True,
            collate_fn=data_collator, num_workers=self.config.dataloader_num_workers, pin_memory=True
        )
        
        optimizer = OptimizerManager.setup_optimizer(model, self.config)
        num_training_steps = len(train_dataloader) * self.config.num_epochs
        scheduler = OptimizerManager.setup_scheduler(optimizer, num_training_steps, self.config)
        
        # Prepare for training
        model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
        
        # Training info
        self.logger.info(f"Starting training: {len(train_dataset)} examples, {self.config.num_epochs} epochs")
        if self.config.mixed_precision == "fp8":
            self.logger.info("FP8 training active - Accelerate handles precision automatically")
        
        # Training loop
        global_step = 0
        for epoch in range(self.config.num_epochs):
            model.train()
            total_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = LossManager.compute_loss(outputs.logits, batch["labels"], batch["attention_mask"])
                    
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.detach().float()
                
                # Logging and checkpointing
                if global_step % self.config.logging_steps == 0:
                    self._log_metrics(epoch, global_step, total_loss / (step + 1), scheduler.get_last_lr()[0])
                
                if global_step % self.config.save_steps == 0 and global_step > 0:
                    self._save_checkpoint(model, tokenizer, global_step)
                
                global_step += 1
            
            avg_epoch_loss = total_loss / len(train_dataloader)
            self.logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
        
        self._save_final_model(model, tokenizer)
        if self.config.use_wandb:
            self.accelerator.end_training()
    
    def _log_metrics(self, epoch: int, step: int, loss: float, lr: float):
        """Log training metrics"""
        self.logger.info(f"Epoch {epoch}, Step {step}: Loss = {loss:.4f}, LR = {lr:.2e}")
        
        if self.config.use_wandb:
            logs = {
                "train/loss": loss, "train/learning_rate": lr, "train/epoch": epoch,
                "train/precision": self.config.mixed_precision
            }
            if self.config.mixed_precision == "fp8":
                logs[f"train/fp8_backend"] = self.config.fp8_backend
            self.accelerator.log(logs, step=step)
    
    def _save_checkpoint(self, model, tokenizer, step: int):
        """Save training checkpoint"""
        output_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        self.accelerator.save_state(output_dir)
        if self.accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            self.logger.info(f"Checkpoint saved to {output_dir}")
    
    def _save_final_model(self, model, tokenizer):
        """Save final trained model"""
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(model)
        
        if self.accelerator.is_main_process:
            unwrapped_model.save_pretrained(
                self.config.output_dir, save_function=self.accelerator.save,
                is_main_process=self.accelerator.is_main_process
            )
            tokenizer.save_pretrained(self.config.output_dir)
            self.logger.info(f"Final model saved to {self.config.output_dir}")


def main():
    """Main entry point with argument parsing"""
    args = parse_args()
    
    # Apply preset configurations if specified
    if args.preset:
        apply_preset(args)
        std_logger.info(f"Applied preset: {args.preset}")
    
    # Convert arguments to config
    config = args_to_config(args)
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Log startup info
    std_logger.info(f"Initializing training with {config.mixed_precision} precision")
    std_logger.info(f"Output directory: {config.output_dir}")
    
    # Initialize and run trainer
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()