from dataset import collate_fn, tokenize_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import json, datasets
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import mlflow
import mlflow.pytorch
import time
from datetime import datetime
import os
import tempfile
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FP8TrainingConfig:
    """Configuration for FP8 training."""

    def __init__(
        self,
        enabled: bool = True,
        margin: int = 0,
        interval: int = 1,
        fp8_format_forward: str = "HYBRID",
        amax_history_len: int = 1024,
        amax_compute_algo: str = "most_recent",
    ):
        self.enabled = enabled
        self.margin = margin
        self.interval = interval
        self.fp8_format_forward = getattr(recipe.Format, fp8_format_forward)
        self.amax_history_len = amax_history_len
        self.amax_compute_algo = amax_compute_algo


def create_fp8_recipe(config: FP8TrainingConfig):
    """Create FP8 recipe with specified formats."""
    return recipe.DelayedScaling(
        margin=config.margin,
        interval=config.interval,
        fp8_format=config.fp8_format_forward,
        amax_history_len=config.amax_history_len,
        amax_compute_algo=config.amax_compute_algo,
    )


def setup_mlflow(args):
    """Setup MLflow experiment tracking."""
    # Set MLflow tracking URI if provided
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    # Set or create experiment
    experiment_name = (
        args.experiment_name
        or f"language_model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(
                f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})"
            )
        else:
            experiment_id = experiment.experiment_id
            logger.info(
                f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})"
            )

        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning(f"Failed to setup MLflow experiment: {e}")
        logger.info("Continuing with default experiment")


def log_hyperparameters(args):
    """Log all hyperparameters to MLflow."""
    params = {
        # Model parameters
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "max_length": args.max_length,
        # Dataset parameters
        "dataset_name": args.dataset_name,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        # Training parameters
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        # Precision parameters
        "use_fp8": args.use_fp8,
        "use_bf16": args.use_bf16,
        "precision": "FP8" if args.use_fp8 else "BF16" if args.use_bf16 else "FP32",
        # Logging parameters
        "log_interval": args.log_interval,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
    }
    print(f"Logging hyperparameters: {params}")
    # Log parameters to MLflow
    mlflow.log_params(params)
    logger.info("Logged hyperparameters to MLflow")


def train_step(model, batch, optimizer, fp8_recipe, use_fp8=True, use_bf16=False, scaler=None):
    model.train()
    optimizer.zero_grad()

    # 1. Check input data validity
    input_ids = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    # Forward pass
    if use_fp8:
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    elif use_bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    else:
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # print(outputs)
    
    loss = outputs.loss
    
    # Backward pass with optional gradient scaling
    if scaler is not None:
        scaler.scale(loss).backward()
        # Gradient clipping with scaled gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        # Gradient clipping (recommended for training stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return loss.item()


def save_model_to_mlflow(
    model, tokenizer, step, eval_loss, eval_perplexity, model_name="best_model"
):
    """Save model to MLflow model registry."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model"
            model_path.mkdir(exist_ok=True)

            # Save model and tokenizer
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

            # Log model to MLflow
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_name,
                registered_model_name=f"{model_name}_{step}",
                extra_files=[str(model_path)],
            )

            # Log model metrics as tags
            mlflow.set_tag(f"{model_name}_step", step)
            mlflow.set_tag(f"{model_name}_eval_loss", f"{eval_loss:.4f}")
            mlflow.set_tag(f"{model_name}_eval_perplexity", f"{eval_perplexity:.2f}")

            logger.info(f"Model {model_name} logged to MLflow at step {step}")

    except Exception as e:
        logger.warning(f"Failed to save model to MLflow: {e}")


def train_model(args):
    """Main training function with enhanced speed tracking."""
    # Setup MLflow
    setup_mlflow(args)

    # Start MLflow run
    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log hyperparameters
        log_hyperparameters(args)

        # Log system information
        mlflow.log_param("cuda_available", torch.cuda.is_available())
        if torch.cuda.is_available():
            mlflow.log_param("cuda_device_count", torch.cuda.device_count())
            mlflow.log_param("cuda_device_name", torch.cuda.get_device_name(0))

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load and tokenize dataset
        logger.info(f"Loading dataset: {args.dataset_name}")
        dataset = datasets.load_dataset(args.dataset_name, split="train")

        # Log dataset information
        mlflow.log_metric("dataset_size", len(dataset))

        # Tokenize dataset with progress bar
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
            num_proc=args.num_workers if args.num_workers > 0 else None,
        )

        # Split dataset into train/eval
        print("Splitting dataset...")
        dataset_dict = tokenized_dataset.train_test_split(
            test_size=1 - args.train_ratio, seed=args.seed
        )
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]

        # Log dataset split information
        mlflow.log_metric("train_dataset_size", len(train_dataset))
        mlflow.log_metric("eval_dataset_size", len(eval_dataset))

        # Create dataloaders
        print("Creating data loaders...")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # Initialize model
        logger.info(f"Loading model: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if args.use_bf16 else torch.float32,
        )
        model = model.cuda()

        # Log model information
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        mlflow.log_param("total_parameters", num_params)
        mlflow.log_param("trainable_parameters", num_trainable_params)

        # Initialize optimizer
        optimizer = optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        # Initialize gradient scaler for mixed precision training
        scaler = None
        if (args.use_bf16 or hasattr(args, 'use_fp16') and args.use_fp16) and not args.use_fp8:
            scaler = torch.amp.GradScaler('cuda')
            logger.info("Initialized gradient scaler for mixed precision training")

        # Create FP8 recipe
        fp8_config = FP8TrainingConfig(
            enabled=args.use_fp8,
        )
        fp8_recipe = create_fp8_recipe(fp8_config)

        # Training metrics tracking
        train_losses = []
        eval_losses = []
        best_eval_loss = float("inf")

        # Speed tracking variables
        speed_window_size = 10  # Number of batches to average speed over
        batch_times = []
        
        # Training loop
        model.train()
        total_steps = 0
        start_time = time.time()

        precision_str = "FP8" if args.use_fp8 else "BF16" if args.use_bf16 else "FP32"
        logger.info(f"Starting training with {precision_str} precision")
        logger.info(f"Total training steps per epoch: {len(train_dataloader)}")

        # Log training start
        mlflow.log_metric("training_start_time", start_time)

        # Create main epoch progress bar
        epoch_pbar = tqdm(
            range(args.num_epochs),
            desc="Training Progress",
            ncols=140,  # Increased width for speed metrics
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}, {postfix}]",
        )

        for epoch in epoch_pbar:
            epoch_loss = 0.0
            num_batches = 0
            epoch_start_time = time.time()
            
            # Reset batch times for this epoch
            batch_times = []

            # Update epoch progress bar description
            epoch_pbar.set_description(f"Epoch {epoch+1}/{args.num_epochs}")

            # Create batch progress bar for current epoch
            batch_pbar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch+1} Batches",
                leave=False,
                ncols=160,  # Increased width for speed metrics
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
            )

            for batch_idx, batch in enumerate(batch_pbar):
                batch_start_time = time.time()
                
                loss = train_step(
                    model,
                    batch,
                    optimizer,
                    fp8_recipe,
                    use_fp8=args.use_fp8,
                    use_bf16=args.use_bf16,
                    scaler=scaler,
                )
                
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                batch_times.append(batch_time)
                
                # Keep only recent batch times for speed calculation
                if len(batch_times) > speed_window_size:
                    batch_times.pop(0)
                
                epoch_loss += loss
                num_batches += 1
                total_steps += 1

                # Calculate speed metrics
                if len(batch_times) >= 2:  # Need at least 2 samples for meaningful average
                    avg_batch_time = sum(batch_times) / len(batch_times)
                    batches_per_sec = 1.0 / avg_batch_time if avg_batch_time > 0 else 0
                    samples_per_sec = batches_per_sec * args.batch_size
                    
                    # Estimate tokens per second (assuming average sequence length)
                    # You can make this more accurate by tracking actual sequence lengths
                    avg_seq_len = args.max_length * 0.8  # Rough estimate
                    tokens_per_sec = samples_per_sec * avg_seq_len
                else:
                    batches_per_sec = 0
                    samples_per_sec = 0
                    tokens_per_sec = 0

                # Update batch progress bar with current loss, learning rate, and speed
                current_lr = optimizer.param_groups[0]["lr"]
                postfix_dict = {
                    "loss": f"{loss:.4f}",
                    "avg_loss": f"{epoch_loss/num_batches:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": total_steps,
                    "batch/s": f"{batches_per_sec:.2f}",
                    "samp/s": f"{samples_per_sec:.1f}",
                }
                
                # Add tokens/s only if it's a reasonable number (not too large)
                if tokens_per_sec > 0 and tokens_per_sec < 1e6:
                    postfix_dict["tok/s"] = f"{tokens_per_sec:.0f}"
                
                batch_pbar.set_postfix(postfix_dict)

                # Log training metrics to MLflow
                mlflow.log_metric("train_loss", loss, step=total_steps)
                mlflow.log_metric("batches_per_second", batches_per_sec, step=total_steps)
                mlflow.log_metric("samples_per_second", samples_per_sec, step=total_steps)
                mlflow.log_metric("tokens_per_second", tokens_per_sec, step=total_steps)

                if batch_idx % args.log_interval == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{args.num_epochs}, "
                        f"Step {batch_idx+1}/{len(train_dataloader)}, "
                        f"Loss: {loss:.4f}, "
                        f"Speed: {batches_per_sec:.2f} batch/s, {samples_per_sec:.1f} samp/s"
                    )

                    # Log learning rate
                    mlflow.log_metric("learning_rate", current_lr, step=total_steps)

                # Evaluation during training
                if args.eval_steps > 0 and total_steps % args.eval_steps == 0:
                    logger.info(f"Running evaluation at step {total_steps}...")
                    eval_loss, eval_perplexity = evaluate_model(
                        model,
                        eval_dataloader,
                        fp8_recipe,
                        use_fp8=args.use_fp8,
                        use_bf16=args.use_bf16,
                    )
                    eval_losses.append((total_steps, eval_loss))

                    # Log evaluation metrics to MLflow
                    mlflow.log_metric("eval_loss", eval_loss, step=total_steps)
                    mlflow.log_metric(
                        "eval_perplexity", eval_perplexity, step=total_steps
                    )

                    logger.info(
                        f"Evaluation - Loss: {eval_loss:.4f}, Perplexity: {eval_perplexity:.2f}"
                    )

                    # Save best model
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        mlflow.log_metric("best_eval_loss", best_eval_loss)

                        # Save to local directory
                        best_model_path = Path(args.output_dir) / "best_model"
                        best_model_path.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "step": total_steps,
                                "epoch": epoch,
                                "eval_loss": eval_loss,
                                "eval_perplexity": eval_perplexity,
                            },
                            best_model_path / "pytorch_model.bin",
                        )

                        # Save to MLflow
                        if args.log_models_to_mlflow:
                            save_model_to_mlflow(
                                model,
                                tokenizer,
                                total_steps,
                                eval_loss,
                                eval_perplexity,
                                "best_model",
                            )

                        logger.info(
                            f"New best model saved (eval_loss: {eval_loss:.4f})"
                        )

                    model.train()  # Switch back to training mode

                # Save checkpoint
                if args.save_steps > 0 and total_steps % args.save_steps == 0:
                    checkpoint_path = (
                        Path(args.output_dir) / f"checkpoint-{total_steps}"
                    )
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "step": total_steps,
                            "epoch": epoch,
                            "loss": loss,
                        },
                        checkpoint_path / "pytorch_model.bin",
                    )
                    logger.info(f"Saved checkpoint at step {total_steps}")

            # Close batch progress bar
            batch_pbar.close()

            # End-of-epoch evaluation and metrics
            avg_epoch_loss = epoch_loss / num_batches
            train_losses.append((epoch + 1, avg_epoch_loss))
            epoch_duration = time.time() - epoch_start_time
            
            # Calculate epoch-level speed metrics
            epoch_batches_per_sec = num_batches / epoch_duration if epoch_duration > 0 else 0
            epoch_samples_per_sec = epoch_batches_per_sec * args.batch_size

            # Log epoch metrics
            mlflow.log_metric("epoch_avg_loss", avg_epoch_loss, step=epoch + 1)
            mlflow.log_metric("epoch_duration", epoch_duration, step=epoch + 1)
            mlflow.log_metric("epoch_batches_per_sec", epoch_batches_per_sec, step=epoch + 1)
            mlflow.log_metric("epoch_samples_per_sec", epoch_samples_per_sec, step=epoch + 1)

            logger.info(
                f"Epoch {epoch+1} completed. Avg loss: {avg_epoch_loss:.4f}, "
                f"Duration: {epoch_duration:.2f}s, Speed: {epoch_batches_per_sec:.2f} batch/s"
            )

            # Run evaluation at end of epoch
            logger.info("Running end-of-epoch evaluation...")
            eval_loss, eval_perplexity = evaluate_model(
                model,
                eval_dataloader,
                fp8_recipe,
                use_fp8=args.use_fp8,
                use_bf16=args.use_bf16,
            )
            eval_losses.append((f"epoch_{epoch+1}", eval_loss))

            # Log end-of-epoch evaluation metrics
            mlflow.log_metric("epoch_eval_loss", eval_loss, step=epoch + 1)
            mlflow.log_metric("epoch_eval_perplexity", eval_perplexity, step=epoch + 1)

            logger.info(
                f"End-of-epoch evaluation - Loss: {eval_loss:.4f}, Perplexity: {eval_perplexity:.2f}"
            )

            # Update epoch progress bar with current metrics including speed
            epoch_pbar.set_postfix(
                {
                    "train_loss": f"{avg_epoch_loss:.4f}",
                    "eval_loss": f"{eval_loss:.4f}",
                    "best_eval": f"{best_eval_loss:.4f}",
                    "perplexity": f"{eval_perplexity:.2f}",
                    "batch/s": f"{epoch_batches_per_sec:.2f}",
                }
            )

            # Save best model if this is the best so far
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                mlflow.log_metric("best_eval_loss", best_eval_loss)

                best_model_path = Path(args.output_dir) / "best_model"
                best_model_path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": total_steps,
                        "epoch": epoch + 1,
                        "eval_loss": eval_loss,
                        "eval_perplexity": eval_perplexity,
                    },
                    best_model_path / "pytorch_model.bin",
                )

                # Save to MLflow
                if args.log_models_to_mlflow:
                    save_model_to_mlflow(
                        model,
                        tokenizer,
                        total_steps,
                        eval_loss,
                        eval_perplexity,
                        "best_model",
                    )

                logger.info(f"New best model saved at end of epoch {epoch+1}")

        # Close epoch progress bar
        epoch_pbar.close()

        # Training completed - calculate overall statistics
        total_training_time = time.time() - start_time
        overall_batches_per_sec = total_steps / total_training_time if total_training_time > 0 else 0
        overall_samples_per_sec = overall_batches_per_sec * args.batch_size
        
        mlflow.log_metric("total_training_time", total_training_time)
        mlflow.log_metric("final_best_eval_loss", best_eval_loss)
        mlflow.log_metric("overall_batches_per_sec", overall_batches_per_sec)
        mlflow.log_metric("overall_samples_per_sec", overall_samples_per_sec)

        # Save final model
        final_path = Path(args.output_dir) / "final_model"
        final_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": total_steps,
                "epoch": args.num_epochs,
                "config": vars(args),
                "train_losses": train_losses,
                "eval_losses": eval_losses,
                "best_eval_loss": best_eval_loss,
                "training_speed": {
                    "total_training_time": total_training_time,
                    "overall_batches_per_sec": overall_batches_per_sec,
                    "overall_samples_per_sec": overall_samples_per_sec,
                }
            },
            final_path / "pytorch_model.bin",
        )

        # Save final model to MLflow
        if args.log_models_to_mlflow:
            save_model_to_mlflow(
                model,
                tokenizer,
                total_steps,
                best_eval_loss,
                math.exp(best_eval_loss) if best_eval_loss < 100 else float("inf"),
                "final_model",
            )

        # Save training metrics
        metrics_path = Path(args.output_dir) / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "train_losses": train_losses,
                    "eval_losses": eval_losses,
                    "best_eval_loss": best_eval_loss,
                    "total_training_time": total_training_time,
                    "overall_batches_per_sec": overall_batches_per_sec,
                    "overall_samples_per_sec": overall_samples_per_sec,
                    "config": vars(args),
                },
                f,
                indent=2,
            )

        # Log metrics file as artifact
        mlflow.log_artifact(str(metrics_path))

        # Log additional artifacts
        if args.output_dir and Path(args.output_dir).exists():
            mlflow.log_artifacts(args.output_dir, artifact_path="training_outputs")

        logger.info(f"\nTraining completed!")
        logger.info(f"Total training time: {total_training_time:.2f} seconds")
        logger.info(f"Overall training speed: {overall_batches_per_sec:.2f} batches/sec, {overall_samples_per_sec:.1f} samples/sec")
        logger.info(f"Final model saved to {final_path}")
        logger.info(f"Best model saved to {Path(args.output_dir) / 'best_model'}")
        logger.info(f"Best evaluation loss: {best_eval_loss:.4f}")
        logger.info(f"Training metrics saved to {metrics_path}")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")


def evaluate_model(model, eval_dataloader, fp8_recipe, use_fp8=True, use_bf16=False):
    """Evaluate model on validation set with speed tracking."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Speed tracking for evaluation
    eval_start_time = time.time()

    # Create progress bar for evaluation with speed metrics
    eval_pbar = tqdm(
        eval_dataloader,
        desc="Evaluating",
        leave=False,
        ncols=120,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_pbar):
            # Move batch to GPU
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            attention_mask = batch["attention_mask"].cuda()

            # Forward pass with appropriate precision
            if use_fp8:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    outputs = model(input_ids, attention_mask, labels)
            elif use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids, attention_mask, labels)
            else:
                outputs = model(input_ids, attention_mask, labels)

            loss = outputs["loss"]
            total_loss += loss.item()
            num_batches += 1

            # Calculate evaluation speed
            elapsed_time = time.time() - eval_start_time
            if elapsed_time > 0 and num_batches > 0:
                eval_batches_per_sec = num_batches / elapsed_time
                eval_samples_per_sec = eval_batches_per_sec * eval_dataloader.batch_size
            else:
                eval_batches_per_sec = 0
                eval_samples_per_sec = 0

            # Update progress bar with current average loss and speed
            avg_loss_so_far = total_loss / num_batches
            eval_pbar.set_postfix({
                "avg_loss": f"{avg_loss_so_far:.4f}",
                "batch/s": f"{eval_batches_per_sec:.2f}",
                "samp/s": f"{eval_samples_per_sec:.1f}"
            })

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    
    # Log final evaluation speed
    total_eval_time = time.time() - eval_start_time
    final_eval_batches_per_sec = num_batches / total_eval_time if total_eval_time > 0 else 0
    
    logger.info(f"Evaluation speed: {final_eval_batches_per_sec:.2f} batches/sec")

    return avg_loss, perplexity

def main():
    parser = argparse.ArgumentParser(
        description="Train Language Model with FP8 precision and MLflow logging"
    )

    # Data arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Tokenizer to use",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nvidia/OpenMathInstruct-2",
        help="Name of the dataset",
    )
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of data to use for training (rest for evaluation)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for dataset splitting"
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    # Precision arguments
    parser.add_argument(
        "--use_fp8", action="store_true", help="Use FP8 precision training"
    )
    parser.add_argument(
        "--use_bf16", action="store_true", help="Use BF16 precision training"
    )

    # Logging and saving
    parser.add_argument(
        "--log_interval", type=int, default=100, help="Logging interval"
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Run evaluation every N steps (0 to disable)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for checkpoints",
    )

    # MLflow arguments
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default=None,
        help="MLflow tracking URI (e.g., http://localhost:5000)",
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="MLflow experiment name"
    )
    parser.add_argument(
        "--log_models_to_mlflow",
        action="store_true",
        help="Log models to MLflow model registry",
    )

    args = parser.parse_args()

    # Validate precision arguments
    precision_count = sum([args.use_fp8, args.use_bf16, getattr(args, 'use_fp16', False)])
    if precision_count > 1:
        raise ValueError("Cannot use multiple precision modes at the same time")

    # Validate train ratio
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    train_model(args)


if __name__ == "__main__":
    main()
