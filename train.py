#!/usr/bin/env python3
"""
Production-ready language model training script using Accelerate
Optimized for instruction tuning with FP8 support via official Accelerate API
Based on: https://huggingface.co/docs/accelerate/usage_guides/low_precision_training
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    set_seed, ProjectConfiguration,
    FP8RecipeKwargs  # Official FP8 support in Accelerate 1.7.0+
)

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration parameters with official FP8 support"""
    # Model and data
    model_name: str = "Qwen/Qwen2.5-3B"
    tokenizer_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    dataset_path: str = "data/instruction_dataset.json"
    output_dir: str = "outputs/trained_model"
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_length: int = 2048
    warmup_ratio: float = 0.1
    
    # Precision configuration (official Accelerate API)
    mixed_precision: str = "bf16"  # Options: "no", "fp16", "bf16", "fp8"
    
    # FP8 backend configuration (only used when mixed_precision="fp8")
    fp8_backend: str = "te"  # Options: "te", "msamp", "ao"
    
    # TransformersEngine (TE) specific config
    te_fp8_format: str = "HYBRID"  # Options: "E4M3", "E5M2", "HYBRID"
    te_amax_history_len: int = 1024
    te_amax_compute_algo: str = "max"
    te_margin: int = 0
    te_interval: int = 1
    te_override_linear_precision: Tuple[bool, bool, bool] = (False, False, False)
    te_use_autocast_during_eval: bool = False
    
    # MS-AMP specific config
    msamp_opt_level: str = "O2"  # Options: "O1", "O2"
    
    # TorchAO specific config (experimental)
    # ao_* parameters can be added here as the API stabilizes
    
    # Other optimization settings
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Monitoring
    use_wandb: bool = True
    wandb_project: str = "llm-instruction-tuning"
    wandb_run_name: Optional[str] = None
    
    # Advanced options
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    
    def get_fp8_kwargs(self) -> Optional[FP8RecipeKwargs]:
        """Get FP8 recipe kwargs based on backend configuration"""
        if self.mixed_precision != "fp8":
            return None
            
        if self.fp8_backend == "te":
            return FP8RecipeKwargs(
                backend="te",
                fp8_format=self.te_fp8_format,
                amax_history_len=self.te_amax_history_len,
                amax_compute_algo=self.te_amax_compute_algo,
                margin=self.te_margin,
                interval=self.te_interval,
                override_linear_precision=self.te_override_linear_precision,
                use_autocast_during_eval=self.te_use_autocast_during_eval
            )
        elif self.fp8_backend == "msamp":
            return FP8RecipeKwargs(
                backend="msamp",
                opt_level=self.msamp_opt_level
            )
        elif self.fp8_backend == "ao":
            return FP8RecipeKwargs(
                backend="ao"
                # Add ao-specific parameters as they become available
            )
        else:
            raise ValueError(f"Unsupported FP8 backend: {self.fp8_backend}")


class InstructionDataset(Dataset):
    """Optimized dataset class for instruction tuning"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_and_process_data(data_path)
        
    def _load_and_process_data(self, data_path: str) -> List[Dict]:
        """Load and preprocess data efficiently"""
        logger.info(f"Loading dataset from {data_path}")
        
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # Support for HuggingFace datasets
            dataset = load_dataset(data_path)
            data = dataset['train'] if 'train' in dataset else dataset
            
        logger.info(f"Loaded {len(data)} examples")
        return data
    
    def _format_chat_template(self, problem: str, solution: str) -> str:
        """Format using Qwen chat template"""
        messages = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution}
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Combine problem and solution using chat template
        formatted_text = self._format_chat_template(
            item['problem'], item['generated_solution']
        )
        
        # Tokenize efficiently
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class ModelManager:
    """Modular model management class"""
    
    @staticmethod
    def setup_model_and_tokenizer(config: TrainingConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Initialize model and tokenizer with optimizations"""
        logger.info(f"Loading model: {config.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Determine dtype - for FP8, use BF16 as base precision
        if config.mixed_precision == "fp8":
            torch_dtype = torch.bfloat16
        elif config.mixed_precision == "bf16":
            torch_dtype = torch.bfloat16
        elif config.mixed_precision == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
            
        # Load model - Accelerate will handle all precision wrapping
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            use_cache=False if config.gradient_checkpointing else True
        )
        
        # Enable gradient checkpointing for memory efficiency
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            
        return model, tokenizer


class OptimizerManager:
    """Modular optimizer and scheduler management"""
    
    @staticmethod
    def setup_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
        """Setup AdamW optimizer with parameter grouping"""
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    @staticmethod
    def setup_scheduler(optimizer: torch.optim.Optimizer, 
                       num_training_steps: int, 
                       config: TrainingConfig):
        """Setup learning rate scheduler"""
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
        
        # Combined warmup + cosine scheduler
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            total_iters=num_warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=num_training_steps - num_warmup_steps,
            eta_min=config.learning_rate * 0.1
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps]
        )
        
        return scheduler


class LossManager:
    """Modular loss computation with optimizations"""
    
    @staticmethod
    def compute_loss(logits: torch.Tensor, 
                    labels: torch.Tensor, 
                    attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss with optimizations"""
        # Shift labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        
        # Flatten tokens
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_attention_mask = shift_attention_mask.view(-1)
        
        # Compute loss only on valid tokens
        # Accelerate handles FP8 precision automatically
        loss = F.cross_entropy(
            shift_logits, 
            shift_labels, 
            reduction='none'
        )
        
        # Mask out padding tokens
        loss = loss * shift_attention_mask
        
        # Return mean loss over valid tokens
        return loss.sum() / shift_attention_mask.sum().clamp(min=1)


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