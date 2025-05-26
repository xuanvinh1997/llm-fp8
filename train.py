#!/usr/bin/env python3
"""
Production-ready language model training script using Accelerate
Optimized for instruction tuning with FP8 support via official Accelerate API
Based on: https://huggingface.co/docs/accelerate/usage_guides/low_precision_training
"""

import os
import logging


from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    set_seed, ProjectConfiguration,
)

from transformers import (
    DataCollatorForSeq2Seq
)
from config import TrainingConfig
from dataset import InstructionDataset
from loss import LossManager
from model import ModelManager
from optimizer import OptimizerManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class Trainer:
    """Main training class with official FP8 support"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._setup_accelerator()
        self._setup_logging()
        
    def _setup_accelerator(self):
        """Initialize Accelerator with official FP8 support"""
        project_config = ProjectConfiguration(
            project_dir=self.config.output_dir,
            logging_dir=os.path.join(self.config.output_dir, "logs")
        )
        
        # Setup FP8 kwargs if using FP8
        kwarg_handlers = []
        if self.config.mixed_precision == "fp8":
            fp8_kwargs = self.config.get_fp8_kwargs()
            if fp8_kwargs:
                kwarg_handlers.append(fp8_kwargs)
                logger.info(f"FP8 enabled with {self.config.fp8_backend} backend")
        
        # Log precision configuration
        logger.info(f"Mixed precision: {self.config.mixed_precision}")
        if self.config.mixed_precision == "fp8":
            logger.info(f"FP8 backend: {self.config.fp8_backend}")
            if self.config.fp8_backend == "te":
                logger.info(f"FP8 format: {self.config.te_fp8_format}")
            elif self.config.fp8_backend == "msamp":
                logger.info(f"MS-AMP opt level: {self.config.msamp_opt_level}")
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,  # Can be "fp8"!
            project_config=project_config,
            log_with="wandb" if self.config.use_wandb else None,
            kwargs_handlers=kwarg_handlers  # Official FP8 configuration
        )
        
    def _setup_logging(self):
        """Setup logging and monitoring"""
        if self.config.use_wandb and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.config.wandb_project,
                config=self.config.__dict__,
                init_kwargs={"wandb": {"name": self.config.wandb_run_name}}
            )
            
    def train(self):
        """Main training loop with automatic FP8 handling"""
        # Set seed for reproducibility
        set_seed(self.config.seed)
        
        # Setup model and tokenizer (no manual FP8 handling needed)
        model, tokenizer = ModelManager.setup_model_and_tokenizer(self.config)
        
        # Setup dataset and dataloader
        train_dataset = InstructionDataset(
            self.config.dataset_path, 
            tokenizer, 
            self.config.max_length
        )
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            return_tensors="pt"
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True
        )
        
        # Setup optimizer and scheduler
        optimizer = OptimizerManager.setup_optimizer(model, self.config)
        
        num_training_steps = len(train_dataloader) * self.config.num_epochs
        scheduler = OptimizerManager.setup_scheduler(
            optimizer, num_training_steps, self.config
        )
        
        # Accelerate handles ALL precision including FP8 automatically!
        model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
        
        # Training loop - no manual FP8 context management needed
        logger.info("Starting training...")
        logger.info(f"Num examples: {len(train_dataset)}")
        logger.info(f"Num epochs: {self.config.num_epochs}")
        logger.info(f"Total steps: {num_training_steps}")
        
        if self.config.mixed_precision == "fp8":
            logger.info("FP8 training active - Accelerate handles precision automatically")
        
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            model.train()
            total_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(model):
                    # Standard forward pass - Accelerate handles FP8 automatically
                    outputs = model(**batch)
                    
                    # Compute loss - precision handled by Accelerate
                    loss = LossManager.compute_loss(
                        outputs.logits,
                        batch["labels"],
                        batch["attention_mask"]
                    )
                    
                    # Standard backward pass - Accelerate handles FP8 scaling
                    self.accelerator.backward(loss)
                    
                    # Standard gradient clipping
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # Standard optimizer step - Accelerate handles precision
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.detach().float()
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / (step + 1)
                    lr = scheduler.get_last_lr()[0]
                    
                    logger.info(
                        f"Epoch {epoch}, Step {global_step}: "
                        f"Loss = {avg_loss:.4f}, LR = {lr:.2e}"
                    )
                    
                    wandb_logs = {
                        "train/loss": avg_loss,
                        "train/learning_rate": lr,
                        "train/epoch": epoch,
                        "train/precision": self.config.mixed_precision
                    }
                    
                    if self.config.mixed_precision == "fp8":
                        wandb_logs["train/fp8_backend"] = self.config.fp8_backend
                        if self.config.fp8_backend == "te":
                            wandb_logs["train/fp8_format"] = self.config.te_fp8_format
                        elif self.config.fp8_backend == "msamp":
                            wandb_logs["train/msamp_opt_level"] = self.config.msamp_opt_level
                    
                    if self.config.use_wandb:
                        self.accelerator.log(wandb_logs, step=global_step)
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0 and global_step > 0:
                    self._save_checkpoint(model, tokenizer, global_step)
                
                global_step += 1
            
            # End of epoch logging
            avg_epoch_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save final model
        self._save_final_model(model, tokenizer)
        
        if self.config.use_wandb:
            self.accelerator.end_training()
    
    def _save_checkpoint(self, model, tokenizer, step):
        """Save training checkpoint"""
        output_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        self.accelerator.save_state(output_dir)
        
        if self.accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Checkpoint saved to {output_dir}")
    
    def _save_final_model(self, model, tokenizer):
        """Save final trained model"""
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(model)
        
        if self.accelerator.is_main_process:
            unwrapped_model.save_pretrained(
                self.config.output_dir,
                save_function=self.accelerator.save,
                is_main_process=self.accelerator.is_main_process
            )
            tokenizer.save_pretrained(self.config.output_dir)
            logger.info(f"Final model saved to {self.config.output_dir}")


def main():
    """Main entry point with official FP8 examples"""
    
    # Example 1: Standard BF16 training
    config_bf16 = TrainingConfig(
        model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
        dataset_path="data/instruction_dataset.json",
        output_dir="outputs/qwen-math-bf16",
        mixed_precision="bf16"
    )
    
    # Example 2: FP8 training with TransformersEngine (official API)
    config_fp8_te = TrainingConfig(
        model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
        dataset_path="data/instruction_dataset.json",
        output_dir="outputs/qwen-math-fp8-te",
        mixed_precision="fp8",  # Official FP8 support!
        fp8_backend="te",
        te_fp8_format="HYBRID",
        te_amax_history_len=1024,
        batch_size=8,  # Can use larger batch size with FP8
        gradient_accumulation_steps=4
    )
    
    # Example 3: FP8 training with MS-AMP (official API)
    config_fp8_msamp = TrainingConfig(
        model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
        dataset_path="data/instruction_dataset.json",
        output_dir="outputs/qwen-math-fp8-msamp",
        mixed_precision="fp8",  # Official FP8 support!
        fp8_backend="msamp",
        msamp_opt_level="O2",
        batch_size=8,  # Can use larger batch size with FP8
        gradient_accumulation_steps=4
    )
    
    # Example 4: FP8 training with TorchAO (experimental)
    config_fp8_ao = TrainingConfig(
        model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
        dataset_path="data/instruction_dataset.json",
        output_dir="outputs/qwen-math-fp8-ao",
        mixed_precision="fp8",  # Official FP8 support!
        fp8_backend="ao",
        batch_size=8,
        gradient_accumulation_steps=4
    )
    
    # Choose configuration
    config = config_fp8_te  # Change this to test different configurations
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Log configuration
    logger.info("Training Configuration:")
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize and run trainer
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()