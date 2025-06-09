import argparse
import json
import logging
import os
import re
from fp8_utils import MathTrainingArguments, clean_math_text, load_and_process_math_dataset
from typing import Dict, List, Optional

import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq,
)
from tqdm import tqdm

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import (
    FP8RecipeKwargs,
    set_seed,
)

# Wandb import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_qwen_model_and_tokenizer(model_name: str):
    """Setup Qwen model and tokenizer with proper chat formatting"""
    logger.info(f"Loading Qwen model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Qwen models have special tokens, ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,  # Disable for training
    )
    
    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    logger.info(f"Model loaded with {model.num_parameters():,} parameters")
    
    return model, tokenizer


def evaluate_math_model(model, eval_dataloader, accelerator):
    """Evaluate the model on math problems"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
            outputs = model(**batch)
            loss = outputs.loss
            
            # Gather losses from all processes
            all_losses = accelerator.gather(loss.repeat(batch["input_ids"].shape[0]))
            total_loss += all_losses.mean().item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {"eval_loss": avg_loss, "eval_perplexity": perplexity}


def save_model_and_config(model, tokenizer, accelerator: Accelerator, output_dir, args):
    """Save model, tokenizer, and training config (DeepSpeed-compatible)"""
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        
        # For DeepSpeed, use accelerator.save_model which handles Zero-3 properly
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model)
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        # Save training arguments
        with open(os.path.join(output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        
        logger.info(f"Model, tokenizer and config saved to {output_dir}")


def initialize_wandb(args, accelerator, model, num_training_steps):
    """Initialize Weights & Biases logging"""
    if not WANDB_AVAILABLE:
        logger.warning("wandb not available. Install with: pip install wandb")
        return False
    
    if not args.use_wandb:
        return False
    
    # Only initialize on main process
    if accelerator.is_main_process:
        # Create wandb config
        wandb_config = {
            # Model config
            "model_name": args.model_name,
            "model_parameters": model.num_parameters(),
            
            # Training config
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "num_training_steps": num_training_steps,
            
            # DeepSpeed config
            "zero_stage": args.zero_stage,
            "zero_offload_optimizer": args.zero_offload_optimizer,
            "zero_offload_param": args.zero_offload_param,
            
            # FP8 config
            "fp8_backend": args.fp8_backend,
            "te_fp8_format": args.te_fp8_format,
            
            # Dataset config
            "dataset_name": args.dataset_name,
            "max_samples": args.max_samples,
            "solution_field": args.solution_field,
            
            # System config
            "num_processes": accelerator.num_processes,
            "mixed_precision": str(accelerator.mixed_precision),
            "seed": args.seed,
        }
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=wandb_config,
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            resume="allow" if args.wandb_resume else None,
        )
        
        # Watch model (optional, can be memory intensive)
        if args.wandb_watch_model:
            wandb.watch(model, log="all", log_freq=args.wandb_watch_freq)
        
        logger.info(f"Initialized wandb project: {args.wandb_project}")
        return True
    
    return False


def log_metrics(metrics: Dict, step: int, prefix: str = "train", use_wandb: bool = False):
    """Log metrics to wandb if available"""
    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        wandb_metrics["step"] = step
        wandb.log(wandb_metrics, step=step)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-7B on math dataset with FP8 + DeepSpeed")
    
    # Model and data
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B", help="Qwen model name")
    parser.add_argument("--dataset_name", default="nvidia/OpenMathInstruct-2", help="Math dataset")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--max_samples", type=int, default=100000, help="Limit dataset size")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    
    # DeepSpeed Configuration
    parser.add_argument("--zero_stage", type=int, choices=[1, 2, 3], default=3, 
                       help="DeepSpeed ZeRO stage")
    parser.add_argument("--zero_offload_optimizer", action="store_true", default=True,
                       help="Offload optimizer states to CPU")
    parser.add_argument("--zero_offload_param", action="store_true", default=True,
                       help="Offload parameters to CPU (ZeRO-3 only)")
    parser.add_argument("--zero3_init_flag", action="store_true", default=True,
                       help="Enable ZeRO-3 initialization")
    
    # FP8 Backend
    parser.add_argument("--fp8_backend", choices=["te"], default="te", 
                       help="FP8 backend to use (TransformerEngine)")
    
    # TransformerEngine specific
    parser.add_argument("--te_fp8_format", choices=["HYBRID", "E4M3", "E5M2"], default="HYBRID",
                       help="TransformerEngine FP8 format")
    parser.add_argument("--te_amax_history_len", type=int, default=32,
                       help="TransformerEngine amax history length")
    parser.add_argument("--te_amax_compute_algo", choices=["max", "most_recent"], default="max",
                       help="TransformerEngine amax compute algorithm")
    
    # Math-specific
    parser.add_argument("--use_generated_solution", action="store_true", default=True,
                       help="Use generated_solution field instead of solution")
    parser.add_argument("--solution_field", default="generated_solution",
                       help="Which solution field to use")
    
    # I/O
    parser.add_argument("--output_dir", default="./qwen_7b_math_fp8_model", help="Output directory")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluate every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true", default=False,
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="qwen-7b-math-fp8",
                       help="Wandb project name")
    parser.add_argument("--wandb_entity", default=None,
                       help="Wandb entity (username or team)")
    parser.add_argument("--wandb_run_name", default=None,
                       help="Wandb run name (auto-generated if None)")
    parser.add_argument("--wandb_tags", nargs="*", default=["fp8", "qwen-7b", "math", "deepspeed"],
                       help="Wandb tags")
    parser.add_argument("--wandb_notes", default="",
                       help="Wandb run notes")
    parser.add_argument("--wandb_resume", action="store_true", default=False,
                       help="Resume wandb run if exists")
    parser.add_argument("--wandb_watch_model", action="store_true", default=False,
                       help="Watch model gradients/parameters (memory intensive)")
    parser.add_argument("--wandb_watch_freq", type=int, default=1000,
                       help="Frequency for watching model")
    parser.add_argument("--wandb_log_freq", type=int, default=10,
                       help="Frequency for logging training metrics")
    
    args = parser.parse_args()
    training_args = MathTrainingArguments(**vars(args))
    
    # Set seed
    set_seed(training_args.seed)
    
    # Setup DeepSpeed plugin
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=training_args.zero_stage,
        zero3_init_flag=training_args.zero3_init_flag,
        # zero_force_ds_cpu_optimizer=training_args.zero_offload_optimizer,
        zero3_save_16bit_model=True,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    )
    
    # Create custom DeepSpeed config
    if training_args.fp8_backend == "te":

        deepspeed_config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": training_args.batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "gradient_clipping": training_args.max_grad_norm,
            "zero_optimization": {
                "stage": training_args.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if training_args.zero_offload_optimizer else "none",
                    "pin_memory": True if training_args.zero_offload_optimizer else False
                },
                "offload_param": {
                    "device": "cpu" if training_args.zero_offload_param and training_args.zero_stage == 3 else "none",
                    "pin_memory": True if training_args.zero_offload_param and training_args.zero_stage == 3 else False
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e5,
                "stage3_gather_16bit_weights_on_model_save": True
            },
            "bf16": {
                "enabled": True,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "steps_per_print": float("inf"),
            "wall_clock_breakdown": False,
            "zero_allow_untested_optimizer": True,
        }
    elif training_args.fp8_backend == "msamp":
        import numpy as np
        config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": training_args.batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "gradient_clipping": training_args.max_grad_norm,
            "zero_optimization": {
                "stage": training_args.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if training_args.zero_offload_optimizer else "none",
                    "pin_memory": True if training_args.zero_offload_optimizer else False
                },
                "offload_param": {
                    "device": "cpu" if training_args.zero_offload_param and training_args.zero_stage == 3 else "none",
                    "pin_memory": True if training_args.zero_offload_param and training_args.zero_stage == 3 else False
                },
            },
            "gradient_clipping": 1.0,
            "steps_per_print": np.inf,
            "bf16": {"enabled": True},
            "fp16": {"enabled": False},
            "zero_allow_untested_optimizer": True,
            "msamp": {
                "enabled": True,
                "opt_level": training_args.msamp_opt_level,
            },
        }
    
    # Set custom config
    deepspeed_plugin.deepspeed_config = deepspeed_config
    
    # Setup FP8 kwargs
    fp8_kwargs = FP8RecipeKwargs(
        backend=training_args.fp8_backend,
        fp8_format=training_args.te_fp8_format,
        amax_history_len=training_args.te_amax_history_len,
        amax_compute_algo=training_args.te_amax_compute_algo,
    )
    
    # Setup accelerator with DeepSpeed and FP8
    accelerator = Accelerator(
        mixed_precision="fp8",
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=[fp8_kwargs],
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with="wandb" if training_args.use_wandb else None,
        project_dir=training_args.output_dir if training_args.use_wandb else None,
    )
    
    logger.info(f"Using DeepSpeed ZeRO stage {training_args.zero_stage}")
    logger.info(f"Mixed precision: {accelerator.mixed_precision}")
    logger.info(f"Number of processes: {accelerator.num_processes}")
    
    # Load model and tokenizer
    model, tokenizer = setup_qwen_model_and_tokenizer(training_args.model_name)
    
    # Load and process math dataset
    train_dataset, eval_dataset = load_and_process_math_dataset(
        training_args.dataset_name,
        tokenizer,
        training_args.max_length,
        training_args.max_samples
    )
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=16,
        return_tensors="pt",
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        drop_last=False,
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        eps=1e-8,
        betas=(0.9, 0.95),
    )
    
    num_training_steps = len(train_dataloader) * training_args.num_epochs // training_args.gradient_accumulation_steps
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Initialize wandb
    wandb_enabled = initialize_wandb(training_args, accelerator, model, num_training_steps)
    
    # Prepare with accelerator (DeepSpeed)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # Training loop
    logger.info("Starting training...")
    logger.info(f"Total training steps: {num_training_steps}")
    logger.info(f"Effective batch size: {training_args.batch_size * training_args.gradient_accumulation_steps * accelerator.num_processes}")
    logger.info(f"Max samples: {training_args.max_samples}")
    
    model.train()
    global_step = 0
    total_loss = 0
    
    for epoch in range(training_args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{training_args.num_epochs}")
        
        progress_bar = tqdm(
            train_dataloader, 
            desc=f"Training Epoch {epoch + 1}",
            disable=not accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().item()
                
                # Backward pass
                accelerator.backward(loss)
                
                # Optimizer step (DeepSpeed handles gradient clipping internally)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                global_step += 1
                
                # Calculate metrics
                avg_loss = total_loss / global_step
                current_lr = lr_scheduler.get_last_lr()[0]
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": global_step,
                })
                
                # Log training metrics to wandb
                if wandb_enabled and accelerator.is_main_process and global_step % training_args.wandb_log_freq == 0:
                    log_metrics({
                        "loss": loss.item(),
                        "avg_loss": avg_loss,
                        "learning_rate": current_lr,
                        "epoch": epoch + 1,
                    }, global_step, "train", wandb_enabled)
                
                # Evaluation
                if global_step % training_args.eval_steps == 0:
                    eval_metrics = evaluate_math_model(model, eval_dataloader, accelerator)
                    if accelerator.is_main_process:
                        logger.info(f"Step {global_step}: {eval_metrics}")
                        
                        # Log evaluation metrics to wandb
                        if wandb_enabled:
                            log_metrics(eval_metrics, global_step, "eval", wandb_enabled)
                    
                    model.train()
                
                # Save checkpoint
                if global_step % training_args.save_steps == 0:
                    checkpoint_dir = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
                    save_model_and_config(model, tokenizer, accelerator, checkpoint_dir, training_args)
                    
                    # Log checkpoint save to wandb
                    if wandb_enabled and accelerator.is_main_process:
                        log_metrics({"checkpoint_saved": 1}, global_step, "system", wandb_enabled)
    
    # Final evaluation
    logger.info("Final evaluation...")
    final_eval_metrics = evaluate_math_model(model, eval_dataloader, accelerator)
    if accelerator.is_main_process:
        logger.info(f"Final evaluation: {final_eval_metrics}")
        
        # Log final metrics to wandb
        if wandb_enabled:
            log_metrics(final_eval_metrics, global_step, "final", wandb_enabled)
    
    # Save final model
    save_model_and_config(model, tokenizer, accelerator, training_args.output_dir, training_args)
    
    # Finish wandb run
    if wandb_enabled and accelerator.is_main_process:
        # Log final model artifact (optional)
        if hasattr(training_args, 'wandb_log_model') and training_args.wandb_log_model:
            artifact = wandb.Artifact(
                name=f"qwen-7b-math-model-{wandb.run.id}",
                type="model",
                description=f"Fine-tuned Qwen 7B model on {training_args.dataset_name}"
            )
            artifact.add_dir(training_args.output_dir)
            wandb.log_artifact(artifact)
        
        wandb.finish()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()