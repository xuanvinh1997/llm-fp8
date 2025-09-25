#!/usr/bin/env python3
"""
Multi-GPU FP8 Training Script with Transformer Engine
Supports both DDP and FSDP strategies with manual FP8 control
"""

import os
import gc
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import transformers
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from datasets import load_dataset
import wandb

# Import project modules
from config import TrainingConfig
from data import DataManager
from utils import GPUMonitor


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DistributedConfig(TrainingConfig):
    """Extended config for distributed training"""
    # Distributed settings
    use_fsdp: bool = False
    fsdp_min_num_params: int = 1e8
    master_addr: str = "localhost"
    master_port: str = "29500"
    backend: str = "nccl"
    
    # Performance settings
    find_unused_parameters: bool = False
    broadcast_buffers: bool = False
    gradient_checkpointing: bool = False
    
    # Debug settings
    debug_mode: bool = False
    log_interval: int = 10
    save_interval: int = 1000


# ============================================================================
# Model Creation
# ============================================================================

class ModelFactory:
    """Clean model creation with TE support"""
    
    @staticmethod
    def create_model(config: DistributedConfig, rank: int) -> torch.nn.Module:
        """Create model with optional TE wrapping"""
        
        # Download model if needed (only on rank 0)
        if rank == 0:
            from huggingface_hub import snapshot_download
            cache_dir = snapshot_download(
                repo_id=config.model_name,
                cache_dir=config.weights_cache_dir
            )
            print(f"Model cached at: {cache_dir}")
        
        # Wait for rank 0 to finish downloading
        if dist.is_initialized():
            dist.barrier()
        
        # Get cache directory
        cache_dir = Path(config.weights_cache_dir) / f"models--{config.model_name.replace('/', '--')}"
        cache_dir = str(next(cache_dir.glob("snapshots/*")))
        
        # Load config
        model_config = AutoConfig.from_pretrained(cache_dir)
        model_config._attn_implementation = "flash_attention_2"
        model_config.use_cache = False
        
        # Create model based on type
        if config.use_te:
            model = ModelFactory._create_te_model(config, cache_dir, model_config)
        else:
            model = ModelFactory._create_hf_model(cache_dir, model_config)
        
        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model
    
    @staticmethod
    def _create_te_model(config: DistributedConfig, cache_dir: str, model_config):
        """Create TE-wrapped model"""
        if "llama" in config.model_name.lower():
            from te_llama import TELlamaForCausalLM
            return TELlamaForCausalLM.from_pretrained_local(
                cache_dir,
                config=model_config,
                torch_dtype=torch.bfloat16
            )
        elif "qwen" in config.model_name.lower():
            from te_qwen import TEQwen3ForCausalLM
            return TEQwen3ForCausalLM.from_pretrained_local(
                cache_dir,
                config=model_config,
                torch_dtype=torch.bfloat16
            )
        else:
            raise ValueError(f"TE not supported for {config.model_name}")
    
    @staticmethod
    def _create_hf_model(cache_dir: str, model_config):
        """Create standard HuggingFace model"""
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            cache_dir,
            config=model_config,
            torch_dtype=torch.bfloat16
        )


# ============================================================================
# Distributed Setup
# ============================================================================

class DistributedSetup:
    """Handle distributed training setup"""
    
    @staticmethod
    def init_process_group(config: DistributedConfig):
        """Initialize distributed process group"""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # torchrun sets these
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            # Manual setup
            rank = 0
            world_size = 1
            local_rank = 0
            os.environ["MASTER_ADDR"] = config.master_addr
            os.environ["MASTER_PORT"] = config.master_port
        
        if world_size > 1:
            dist.init_process_group(backend=config.backend)
        
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        # Set seed for reproducibility
        torch.manual_seed(42 + rank)
        
        return rank, world_size, local_rank, device
    
    @staticmethod
    def wrap_model_ddp(
        model: torch.nn.Module, 
        config: DistributedConfig,
        device: torch.device,
        local_rank: int
    ) -> DDP:
        """Wrap model with DistributedDataParallel"""
        model = model.to(device)
        
        if dist.is_initialized():
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=config.find_unused_parameters,
                broadcast_buffers=config.broadcast_buffers
            )
        
        return model
    
    @staticmethod
    def wrap_model_fsdp(
        model: torch.nn.Module,
        config: DistributedConfig,
        device: torch.device
    ) -> FSDP:
        """Wrap model with FullyShardedDataParallel"""
        
        # Determine wrap class
        wrap_classes = set()
        if "llama" in config.model_name.lower():
            if config.use_te:
                from te_llama import TELlamaDecoderLayer
                wrap_classes.add(TELlamaDecoderLayer)
            else:
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                wrap_classes.add(LlamaDecoderLayer)
        elif "qwen" in config.model_name.lower():
            if config.use_te:
                from te_qwen import TEQwen3DecoderLayer
                wrap_classes.add(TEQwen3DecoderLayer)
            else:
                from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
                wrap_classes.add(Qwen2DecoderLayer)
        
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls=wrap_classes,
        )
        
        # Mixed precision for FSDP (BF16 for master weights, FP8 in TE layers)
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=False  # TE handles casting
        )
        
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=device,
            use_orig_params=True,  # Important for TE
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
            forward_prefetch=True,
        )
        
        return model


# ============================================================================
# Training Logic
# ============================================================================

class Trainer:
    """Clean trainer class for distributed FP8 training"""
    
    def __init__(
        self,
        config: DistributedConfig,
        model: torch.nn.Module,
        rank: int,
        world_size: int,
        device: torch.device
    ):
        self.config = config
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.gpu_monitor = GPUMonitor()
        
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with optional fused kernels"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
            fused=torch.cuda.is_available()
        )
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int):
        """Create learning rate scheduler"""
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=total_steps
        )
    
    def create_dataloader(self, dataset, is_train: bool = True) -> DataLoader:
        """Create distributed dataloader"""
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=is_train,
                seed=42
            )
        
        # Create tokenizer for collator
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        from transformers import DataCollatorForLanguageModeling
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=16
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=(sampler is None and is_train),
            num_workers=4,
            pin_memory=True,
            drop_last=is_train,
            collate_fn=collator
        )
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        accumulation_steps: int
    ) -> float:
        """Single training step"""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass (FP8 handled internally by TE layers)
        outputs = self.model(**batch)
        loss = outputs.loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return loss.item() * accumulation_steps
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        epoch: int
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_steps = 0
        
        # Progress tracking
        if self.rank == 0:
            import tqdm
            pbar = tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(train_loader):
            # Gradient accumulation
            is_accumulation = (step + 1) % self.config.gradient_accumulation_steps != 0
            
            # Forward/backward with optional gradient sync
            if isinstance(self.model, DDP) and is_accumulation:
                with self.model.no_sync():
                    loss = self.train_step(batch, step, self.config.gradient_accumulation_steps)
            else:
                loss = self.train_step(batch, step, self.config.gradient_accumulation_steps)
            
            total_loss += loss
            
            # Optimizer step
            if not is_accumulation:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                num_steps += 1
                
                # Logging
                if self.rank == 0:
                    if step % self.config.log_interval == 0:
                        avg_loss = total_loss / num_steps
                        lr = scheduler.get_last_lr()[0]
                        mem_gb = self.gpu_monitor.get_memory_usage()[0] / 1024
                        
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f'{lr:.2e}',
                            'mem': f'{mem_gb:.1f}GB'
                        })
                        
                        if wandb.run is not None:
                            wandb.log({
                                'train/loss': avg_loss,
                                'train/lr': lr,
                                'train/gpu_memory_gb': mem_gb,
                                'train/step': num_steps
                            })
            
            if self.rank == 0:
                pbar.update(1)
        
        if self.rank == 0:
            pbar.close()
        
        return total_loss / max(num_steps, 1)
    
    def evaluate(
        self,
        eval_loader: DataLoader,
        epoch: int
    ) -> float:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        # Gather losses from all ranks
        if dist.is_initialized():
            total_loss = torch.tensor([total_loss], device=self.device)
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item()
            num_batches *= self.world_size
        
        avg_loss = total_loss / max(num_batches, 1)
        
        if self.rank == 0:
            print(f"Epoch {epoch} - Eval Loss: {avg_loss:.4f}")
            if wandb.run is not None:
                wandb.log({'eval/loss': avg_loss, 'eval/epoch': epoch})
        
        return avg_loss
    
    def save_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        epoch: int,
        step: int
    ):
        """Save model checkpoint (rank 0 only)"""
        if self.rank != 0:
            return
        
        save_dir = Path(self.config.output_dir) / f"checkpoint-{epoch}-{step}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Unwrap model
        model_to_save = self.model
        if isinstance(self.model, (DDP, FSDP)):
            model_to_save = self.model.module
        
        # Save model
        model_to_save.save_pretrained(str(save_dir))
        
        # Save training state
        torch.save({
            'epoch': epoch,
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': asdict(self.config)
        }, save_dir / 'training_state.pt')
        
        print(f"Checkpoint saved to {save_dir}")

def wrap_model_fsdp_advanced(
    model: torch.nn.Module,
    config: DistributedConfig,
    device: torch.device
) -> FSDP:
    """Advanced FSDP configuration for very large models"""
    
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
        CPUOffload,
    )
    
    # Determine sharding strategy based on model size
    model_size = sum(p.numel() for p in model.parameters())
    
    if model_size > 30e9:  # > 30B parameters
        # FULL_SHARD: Shard everything (params, grads, optimizer states)
        sharding_strategy = ShardingStrategy.FULL_SHARD
        
        # Optional: CPU offloading for extremely large models
        cpu_offload = CPUOffload(offload_params=True)
    elif model_size > 10e9:  # > 10B parameters
        # SHARD_GRAD_OP: Shard gradients and optimizer states only
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        cpu_offload = None
    else:
        # HYBRID_SHARD: Shard within node, replicate across nodes
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
        cpu_offload = None
    
    # Auto wrap policy - shard at transformer layer level
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={get_layer_class(config)},
    )
    
    # Mixed precision
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,  # Use FP32 for reduction
        buffer_dtype=torch.bfloat16,
        cast_forward_inputs=False,
    )
    
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        device_id=device,
        use_orig_params=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        limit_all_gathers=True,
        sync_module_states=True,  # Sync params across GPUs at init
    )
    
    return model
# ============================================================================
# Main Training Function
# ============================================================================

def main(config: DistributedConfig):
    """Main training function"""
    
    # Initialize distributed training
    rank, world_size, local_rank, device = DistributedSetup.init_process_group(config)
    
    if rank == 0:
        print(f"Starting distributed training on {world_size} GPUs")
        print(f"Model: {config.model_name}")
        print(f"Use TE: {config.use_te}")
        print(f"Use FSDP: {config.use_fsdp}")
        print(f"Batch size per GPU: {config.batch_size}")
        print(f"Mixed precision: {config.mixed_precision}")
    
    # Initialize wandb (rank 0 only)
    if rank == 0 and config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"{config.model_name.split('/')[-1]}-fp8",
            config=asdict(config)
        )
    
    # Create model
    if rank == 0:
        print("Creating model...")
    model = ModelFactory.create_model(config, rank)
    
    # Wrap model for distributed training
    if config.use_fsdp:
        if rank == 0:
            print("Using FSDP...")
        model = DistributedSetup.wrap_model_fsdp(model, config, device)
    else:
        if rank == 0:
            print("Using DDP...")
        model = DistributedSetup.wrap_model_ddp(model, config, device, local_rank)
    
    # Create trainer
    trainer = Trainer(config, model, rank, world_size, device)
    
    # Create optimizer and scheduler
    optimizer = trainer.create_optimizer()
    
    # Load and prepare data
    if rank == 0:
        print("Loading dataset...")
    data_manager = DataManager(config)
    dataset = load_dataset(config.dataset_name, split=config.split_name)
    
    # Limit samples if specified
    if config.num_of_samples:
        dataset = dataset.select(range(min(config.num_of_samples, len(dataset))))
    
    # Process dataset
    from accelerate import Accelerator
    accelerator = Accelerator()  # Just for dataset processing
    with accelerator.main_process_first():
        dataset = dataset.map(
            data_manager._apply_template,
            remove_columns=dataset.column_names,
            num_proc=min(config.num_proc, 8)
        )
    
    # Split dataset
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    # Create dataloaders
    train_loader = trainer.create_dataloader(train_dataset, is_train=True)
    eval_loader = trainer.create_dataloader(eval_dataset, is_train=False)
    
    # Calculate total steps and create scheduler
    total_steps = (len(train_loader) * config.num_epochs) // config.gradient_accumulation_steps
    scheduler = trainer.create_scheduler(optimizer, total_steps)
    
    if rank == 0:
        print(f"Starting training for {config.num_epochs} epochs...")
        print(f"Total steps: {total_steps}")
    
    # Training loop
    for epoch in range(1, config.num_epochs + 1):
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer, scheduler, epoch)
        
        # Evaluate
        eval_loss = trainer.evaluate(eval_loader, epoch)
        
        # Save checkpoint
        if epoch % config.save_interval == 0 or epoch == config.num_epochs:
            trainer.save_checkpoint(optimizer, scheduler, epoch, total_steps)
        
        if rank == 0:
            print(f"Epoch {epoch}/{config.num_epochs} - Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    if rank == 0:
        print("Training completed!")
        if wandb.run is not None:
            wandb.finish()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-GPU FP8 Training")
    
    # Model and data
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_text_field", type=str, default="text")
    parser.add_argument("--split_name", type=str, default="train_1M")
    parser.add_argument("--num_of_samples", type=int)
    
    # Training settings
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    
    # Precision and optimization
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp8"])
    parser.add_argument("--fp8_scenario", type=str, default="default", choices=["default", "mxfp8"])
    parser.add_argument("--use_te", action="store_true")
    
    # Distributed settings
    parser.add_argument("--use_fsdp", action="store_true", help="Use FSDP instead of DDP")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="./runs")
    parser.add_argument("--output_dir", type=str, default="./saved_model")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="llm-fp8-multi")
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1)
    
    # Technical
    parser.add_argument("--weights_cache_dir", type=str, default="./model_cache")
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--debug_mode", action="store_true")
    
    args = parser.parse_args()
    
    # Create config
    config = DistributedConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_text_field=args.dataset_text_field,
        split_name=args.split_name,
        num_of_samples=args.num_of_samples,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size or args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.num_warmup_steps,
        max_seq_length=args.max_seq_length,
        mixed_precision=args.mixed_precision,
        fp8_scenario=args.fp8_scenario,
        use_te=args.use_te,
        use_fsdp=args.use_fsdp,
        gradient_checkpointing=args.gradient_checkpointing,
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        weights_cache_dir=args.weights_cache_dir,
        num_proc=args.num_proc,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        debug_mode=args.debug_mode,
    )
    
    # Run training
    main(config)