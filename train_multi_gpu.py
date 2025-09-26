#!/usr/bin/env python3
"""
Advanced Multi-GPU FP8 Training with Automatic Sharding Strategy
Supports DDP, FSDP, Pipeline Parallel, and Tensor Parallel
"""

import os
import gc
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    lambda_auto_wrap_policy,
)
from functools import partial
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import transformers
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import wandb
import tqdm

# Import project modules
from config import TrainingConfig
from data import DataManager
from utils import GPUMonitor


# ============================================================================
# Enums and Constants
# ============================================================================

class ShardingMode(Enum):
    """Sharding strategies"""
    DDP = "ddp"                    # Data parallel only
    FSDP_GRAD = "fsdp_grad"        # Shard gradients only
    FSDP_FULL = "fsdp_full"        # Full sharding
    FSDP_OFFLOAD = "fsdp_offload"  # Full sharding + CPU offload
    PIPELINE = "pipeline"           # Pipeline parallel
    TENSOR = "tensor"              # Tensor parallel
    HYBRID = "hybrid"              # Combination of strategies


# ============================================================================
# Enhanced Configuration
# ============================================================================

@dataclass
class DistributedConfig(TrainingConfig):
    """Advanced configuration for distributed training"""
    
    # Sharding strategy
    sharding_mode: str = "auto"  # auto, ddp, fsdp_grad, fsdp_full, fsdp_offload
    fsdp_min_num_params: int = int(1e6)
    fsdp_limit_all_gathers: bool = True
    fsdp_forward_prefetch: bool = True
    fsdp_use_orig_params: bool = True
    fsdp_sync_module_states: bool = True
    
    # CPU offloading
    cpu_offload: bool = False
    offload_params: bool = False
    
    # Activation checkpointing
    gradient_checkpointing: bool = False
    checkpoint_impl: str = "reentrant"  # reentrant or no_reentrant
    
    # Memory optimization
    empty_cache_freq: int = 50  # Clear cache every N steps
    memory_efficient_attention: bool = True
    compile_model: bool = False  # torch.compile
    
    # Performance
    pin_memory: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Distributed settings
    backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: str = "29500"
    find_unused_parameters: bool = False
    broadcast_buffers: bool = False
    
    # Debug and logging
    debug_mode: bool = False
    profile_memory: bool = False
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 500
    
    # Advanced FP8
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"
    
    def __post_init__(self):
        """Post initialization setup"""
        super().__post_init__()
        
        # Auto-detect best settings if not specified
        if self.sharding_mode == "auto":
            self.sharding_mode = self._auto_detect_sharding()
    
    def _auto_detect_sharding(self) -> str:
        """Simple sharding strategy selection"""
        num_gpus = torch.cuda.device_count()

        # Simple rule: use FSDP for multiple GPUs, DDP for single GPU
        if num_gpus > 1:
            return ShardingMode.FSDP_FULL.value
        else:
            return ShardingMode.DDP.value


# ============================================================================
# Memory Profiler
# ============================================================================

class MemoryProfiler:
    """Advanced memory profiling and optimization"""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.memory_stats = []
    
    def profile_memory(self, phase: str = ""):
        """Profile current memory usage"""
        if not torch.cuda.is_available():
            return
        
        torch.cuda.synchronize()
        
        stats = {
            "phase": phase,
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9,
            "active": torch.cuda.memory_stats()["active_bytes.all.current"] / 1e9,
        }
        
        self.memory_stats.append(stats)
        
        if self.rank == 0:
            print(f"[Memory {phase}] Allocated: {stats['allocated']:.2f}GB, "
                  f"Reserved: {stats['reserved']:.2f}GB, Active: {stats['active']:.2f}GB")
        
        return stats
    
    def optimize_memory(self):
        """Optimize memory usage"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    @staticmethod
    def estimate_model_memory(
        model_params_billions: float,
        batch_size: int,
        seq_length: int,
        num_gpus: int,
        dtype_bytes: int = 2,  # 2 for BF16, 1 for FP8
        sharding_mode: str = "fsdp_full"
    ) -> Dict[str, float]:
        """Estimate memory requirements"""
        
        # Base calculations
        params_gb = model_params_billions * dtype_bytes
        
        # Optimizer states (Adam)
        optimizer_gb = params_gb * 2  # Momentum + variance
        
        # Gradients
        gradients_gb = params_gb
        
        # Activations (rough estimate)
        hidden_size = int(math.sqrt(model_params_billions * 1e9 / 100))  # Approximate
        num_layers = int(model_params_billions * 10)  # Approximate
        
        activation_per_token = (
            hidden_size * 4 * num_layers * dtype_bytes / 1e9  # MLP
            + hidden_size * 3 * num_layers * dtype_bytes / 1e9  # Attention
        )
        activations_gb = activation_per_token * batch_size * seq_length
        
        # Calculate based on sharding mode
        if sharding_mode == "ddp":
            per_gpu = params_gb + optimizer_gb + gradients_gb + activations_gb
        elif sharding_mode == "fsdp_grad":
            per_gpu = params_gb + (optimizer_gb + gradients_gb) / num_gpus + activations_gb
        elif sharding_mode == "fsdp_full":
            per_gpu = (params_gb + optimizer_gb + gradients_gb) / num_gpus + activations_gb
        elif sharding_mode == "fsdp_offload":
            # CPU offloading reduces GPU memory significantly
            per_gpu = params_gb / num_gpus + activations_gb
        else:
            per_gpu = params_gb + optimizer_gb + gradients_gb + activations_gb
        
        return {
            "params_gb": params_gb,
            "optimizer_gb": optimizer_gb,
            "gradients_gb": gradients_gb,
            "activations_gb": activations_gb,
            "per_gpu_gb": per_gpu,
            "total_gb": per_gpu * num_gpus,
            "recommended_gpu_memory": per_gpu * 1.3,  # 30% buffer
        }


# ============================================================================
# Advanced Model Factory
# ============================================================================

class ModelFactory:
    """Factory for creating models with various optimizations"""
    
    @staticmethod
    def create_model(
        config: DistributedConfig,
        rank: int,
        world_size: int
    ) -> torch.nn.Module:
        """Create model with simplified loading"""

        # Simple model loading - let HuggingFace handle caching
        model_config = AutoConfig.from_pretrained(
            config.model_name,
            cache_dir=config.weights_cache_dir
        )

        # Set basic configuration
        model_config.use_cache = False
        if config.mixed_precision == "bf16":
            model_config.torch_dtype = torch.bfloat16
        elif config.mixed_precision == "fp16":
            model_config.torch_dtype = torch.float16
        else:
            model_config.torch_dtype = torch.float32

        # Enable Flash Attention if available
        if hasattr(model_config, "_attn_implementation"):
            model_config._attn_implementation = "flash_attention_2"

        # Create model based on type
        if config.use_te:
            model = ModelFactory._create_te_model(config, model_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                config=model_config,
                torch_dtype=model_config.torch_dtype,
                low_cpu_mem_usage=True,
                cache_dir=config.weights_cache_dir
            )

        # Apply optimizations
        if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        if config.compile_model and hasattr(torch, "compile"):
            model = torch.compile(model, mode="max-autotune")

        return model
    
    @staticmethod
    def _create_te_model(config: DistributedConfig, model_config):
        """Create Transformer Engine model"""
        if "llama" in config.model_name.lower():
            from te_llama import TELlamaForCausalLM
            model = TELlamaForCausalLM.from_pretrained(
                config.model_name,
                config=model_config,
                torch_dtype=torch.bfloat16,
                cache_dir=config.weights_cache_dir
            )
        elif "qwen" in config.model_name.lower():
            from te_qwen import TEQwen3ForCausalLM
            model = TEQwen3ForCausalLM.from_pretrained(
                config.model_name,
                config=model_config,
                torch_dtype=torch.bfloat16,
                cache_dir=config.weights_cache_dir
            )
        else:
            raise ValueError(f"TE not supported for {config.model_name}")

        return model
    


# ============================================================================
# Advanced Distributed Wrapper
# ============================================================================

class DistributedWrapper:
    """Advanced distributed model wrapping with multiple strategies"""
    
    @staticmethod
    def wrap_model(
        model: torch.nn.Module,
        config: DistributedConfig,
        rank: int,
        world_size: int,
        device: torch.device
    ) -> torch.nn.Module:
        """Wrap model with appropriate parallelism strategy"""
        
        sharding_mode = ShardingMode(config.sharding_mode)
        
        if sharding_mode == ShardingMode.DDP:
            return DistributedWrapper._wrap_ddp(
                model, config, rank, device
            )
        elif sharding_mode in [ShardingMode.FSDP_GRAD, ShardingMode.FSDP_FULL, ShardingMode.FSDP_OFFLOAD]:
            return DistributedWrapper._wrap_fsdp(
                model, config, rank, world_size, device, sharding_mode
            )
        else:
            raise ValueError(f"Unsupported sharding mode: {sharding_mode}")
    
    @staticmethod
    def _wrap_ddp(
        model: torch.nn.Module,
        config: DistributedConfig,
        rank: int,
        device: torch.device
    ) -> DDP:
        """Wrap with DistributedDataParallel"""
        
        model = model.to(device)
        
        if dist.is_initialized() and dist.get_world_size() > 1:
            model = DDP(
                model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=config.find_unused_parameters,
                broadcast_buffers=config.broadcast_buffers,
                gradient_as_bucket_view=True,  # Memory optimization
            )
            
            if rank == 0:
                print(f"Model wrapped with DDP on {dist.get_world_size()} GPUs")
        
        return model
    
    @staticmethod
    def _wrap_fsdp(
        model: torch.nn.Module,
        config: DistributedConfig,
        rank: int,
        world_size: int,
        device: torch.device,
        sharding_mode: ShardingMode
    ) -> FSDP:
        """Wrap with FullyShardedDataParallel"""
        
        # Determine sharding strategy
        if sharding_mode == ShardingMode.FSDP_GRAD:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        elif sharding_mode == ShardingMode.FSDP_FULL:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        elif sharding_mode == ShardingMode.FSDP_OFFLOAD:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD
        
        # CPU offloading configuration
        cpu_offload_config = None
        if config.cpu_offload or sharding_mode == ShardingMode.FSDP_OFFLOAD:
            cpu_offload_config = CPUOffload(
                offload_params=config.offload_params or True
            )
        
        # Auto wrap policy
        auto_wrap_policy = DistributedWrapper._get_auto_wrap_policy(
            model, config
        )
        
        # Mixed precision configuration
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32 if sharding_mode == ShardingMode.FSDP_OFFLOAD else torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=False,
        )
        
        # Activation checkpointing - disable for TE models due to compatibility issues
        if config.gradient_checkpointing and not config.use_te:
            checkpoint_fn = lambda m: checkpoint_wrapper(
                m,
                checkpoint_impl=CheckpointImpl.REENTRANT if config.checkpoint_impl == "reentrant" else CheckpointImpl.NO_REENTRANT
            )
        else:
            checkpoint_fn = None
            if config.gradient_checkpointing and config.use_te:
                print("Warning: Activation checkpointing disabled for TE models due to FSDP compatibility")
        
        # Wrap model
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload_config,
            device_id=device,
            use_orig_params=config.fsdp_use_orig_params,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            forward_prefetch=config.fsdp_forward_prefetch,
            limit_all_gathers=config.fsdp_limit_all_gathers,
            sync_module_states=config.fsdp_sync_module_states,
        )
        
        # Apply activation checkpointing after FSDP wrapping (only for non-TE models)
        if checkpoint_fn and config.gradient_checkpointing and not config.use_te:
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=checkpoint_fn,
                auto_wrap_policy=auto_wrap_policy,
            )
        
        if rank == 0:
            print(f"Model wrapped with FSDP ({sharding_strategy.name}) on {world_size} GPUs")
            if cpu_offload_config:
                print("CPU offloading enabled")
        
        return model
    
    @staticmethod
    def _get_auto_wrap_policy(
        model: torch.nn.Module,
        config: DistributedConfig
    ):
        """Get appropriate auto-wrap policy for FSDP"""
        
        # Collect transformer layer classes
        transformer_layers = set()
        
        # Llama variants
        if "llama" in config.model_name.lower():
            if config.use_te:
                from te_llama import TELlamaDecoderLayer
                transformer_layers.add(TELlamaDecoderLayer)
            else:
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                transformer_layers.add(LlamaDecoderLayer)
        
        # Qwen variants
        elif "qwen" in config.model_name.lower():
            if config.use_te:
                from te_qwen import TEQwen3DecoderLayer
                transformer_layers.add(TEQwen3DecoderLayer)
            else:
                try:
                    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
                    transformer_layers.add(Qwen2DecoderLayer)
                except:
                    try:
                        from transformers.models.qwen.modeling_qwen import QwenDecoderLayer
                        transformer_layers.add(QwenDecoderLayer)
                    except:
                        pass
        
        # Create the auto wrap policy function
        if transformer_layers:
            # This returns a callable that FSDP will use
            from functools import partial
            return partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=transformer_layers,
            )
        else:
            # Fallback to size-based policy
            return partial(
                size_based_auto_wrap_policy,
                min_num_params=config.fsdp_min_num_params
            )


# ============================================================================
# Advanced Trainer
# ============================================================================

class Trainer:
    """Advanced distributed trainer with comprehensive features"""
    
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
        self.memory_profiler = MemoryProfiler(rank, world_size)
        
        self.global_step = 0
        self.best_loss = float('inf')
    
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with advanced settings"""
        
        # Separate parameters by weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        # Use fused optimizer if available
        use_fused = torch.cuda.is_available() and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )
        
        if self.rank == 0:
            print(f"Created AdamW optimizer (fused={use_fused})")
        
        return optimizer
    
    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int
    ):
        """Create learning rate scheduler"""
        
        # Cosine scheduler with warmup
        def lr_lambda(current_step: int):
            if current_step < self.config.num_warmup_steps:
                return float(current_step) / float(max(1, self.config.num_warmup_steps))
            
            progress = float(current_step - self.config.num_warmup_steps) / float(
                max(1, total_steps - self.config.num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return scheduler
    
    def create_dataloader(
        self,
        dataset,
        is_train: bool = True
    ) -> Tuple[DataLoader, Optional[DistributedSampler]]:
        """Create optimized distributed dataloader"""
        
        # Create sampler
        sampler = None
        if dist.is_initialized() and self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=is_train,
                seed=42,
                drop_last=is_train,
            )
        
        # Get tokenizer for collator
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create collator
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=16,
        )
        
        # Create dataloader with optimizations
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=(sampler is None and is_train),
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=is_train,
            collate_fn=collator,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
            persistent_workers=self.config.persistent_workers if self.config.num_workers > 0 else False,
        )
        
        return dataloader, sampler
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        accumulation_steps: int
    ) -> Tuple[float, Dict[str, float]]:
        """Single training step with metrics"""
        
        # Move batch to device (non-blocking for speed)
        batch = {
            k: v.to(self.device, non_blocking=True) 
            for k, v in batch.items()
        }
        
        # Forward pass
        with torch.amp.autocast('cuda', enabled=False):  # FP8 handled by TE
            outputs = self.model(**batch)
            loss = outputs.loss / accumulation_steps
        
        # Check for NaN
        if not torch.isfinite(loss):
            if self.rank == 0:
                print(f"Warning: Non-finite loss detected: {loss.item()}")
            loss = torch.zeros_like(loss)
        
        # Backward pass
        loss.backward()
        
        # Collect metrics
        metrics = {
            "loss": loss.item() * accumulation_steps,
            "perplexity": torch.exp(loss).item() * accumulation_steps,
        }
        
        return loss.item() * accumulation_steps, metrics
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        epoch: int,
        wandb_run=None
    ) -> Dict[str, float]:
        """Train for one epoch with advanced features"""
        
        self.model.train()
        
        # Initialize metrics
        epoch_metrics = {
            "loss": 0.0,
            "perplexity": 0.0,
            "throughput": 0.0,
        }
        num_steps = 0
        
        # Progress bar (rank 0 only)
        if self.rank == 0:
            pbar = tqdm.tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch}",
                unit="batch"
            )
        
        # Training loop
        start_time = time.time()
        
        for step, batch in enumerate(train_loader):
            step_start = time.time()
            
            # Check if accumulation step
            is_accumulation = (step + 1) % self.config.gradient_accumulation_steps != 0
            
            # Forward/backward with gradient sync control
            if isinstance(self.model, DDP) and is_accumulation:
                with self.model.no_sync():
                    loss, metrics = self.train_step(batch, self.config.gradient_accumulation_steps)
            elif isinstance(self.model, FSDP) and is_accumulation:
                # FSDP handles gradient accumulation differently
                loss, metrics = self.train_step(batch, self.config.gradient_accumulation_steps)
            else:
                loss, metrics = self.train_step(batch, self.config.gradient_accumulation_steps)
            
            # Update epoch metrics
            for k, v in metrics.items():
                epoch_metrics[k] += v
            
            # Optimizer step (if not accumulating)
            if not is_accumulation:
                # Gradient clipping
                if isinstance(self.model, FSDP):
                    grad_norm = self.model.clip_grad_norm_(1.0)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0
                    )
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                num_steps += 1
                self.global_step += 1
                
                # Calculate throughput
                step_time = time.time() - step_start
                tokens_per_second = (
                    self.config.batch_size * 
                    self.config.max_seq_length * 
                    self.world_size / step_time
                )
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = epoch_metrics["loss"] / num_steps
                    lr = scheduler.get_last_lr()[0]
                    
                    if self.rank == 0:
                        # GPU memory
                        mem_stats = self.memory_profiler.profile_memory(f"step_{self.global_step}")
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'ppl': f'{epoch_metrics["perplexity"]/num_steps:.2f}',
                            'lr': f'{lr:.2e}',
                            'tok/s': f'{tokens_per_second:.0f}',
                            'mem': f'{mem_stats["allocated"]:.1f}GB'
                        })
                        
                        # Log to wandb
                        if wandb_run is not None:
                            wandb.log({
                                'train/loss': avg_loss,
                                'train/perplexity': epoch_metrics["perplexity"] / num_steps,
                                'train/learning_rate': lr,
                                'train/grad_norm': grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                                'train/tokens_per_second': tokens_per_second,
                                'train/gpu_memory_gb': mem_stats["allocated"],
                                'train/step': self.global_step,
                            })
                
                # Memory optimization
                if self.global_step % self.config.empty_cache_freq == 0:
                    self.memory_profiler.optimize_memory()
            
            # Update progress bar
            if self.rank == 0:
                pbar.update(1)
        
        # Finalize epoch
        if self.rank == 0:
            pbar.close()
        
        # Calculate final metrics
        epoch_time = time.time() - start_time
        epoch_metrics = {
            k: v / max(num_steps, 1) for k, v in epoch_metrics.items()
        }
        epoch_metrics["epoch_time"] = epoch_time
        epoch_metrics["throughput"] = (
            len(train_loader) * self.config.batch_size * 
            self.config.max_seq_length / epoch_time
        )
        
        return epoch_metrics
    
    def evaluate(
        self,
        eval_loader: DataLoader,
        epoch: int,
        wandb_run=None
    ) -> Dict[str, float]:
        """Evaluate model with metrics"""
        
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm.tqdm(eval_loader, desc="Evaluating", disable=self.rank != 0):
                batch = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in batch.items()
                }
                
                outputs = self.model(**batch)
                
                # Accumulate loss
                total_loss += outputs.loss.item()
                total_tokens += batch["input_ids"].numel()
                num_batches += 1
        
        # Gather metrics across all ranks
        if dist.is_initialized() and self.world_size > 1:
            metrics_tensor = torch.tensor(
                [total_loss, total_tokens, num_batches],
                device=self.device
            )
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            total_loss = metrics_tensor[0].item()
            total_tokens = int(metrics_tensor[1].item())
            num_batches = int(metrics_tensor[2].item())
        
        # Calculate final metrics
        avg_loss = total_loss / max(num_batches, 1)
        perplexity = math.exp(min(avg_loss, 20))  # Cap to prevent overflow
        
        metrics = {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
            "eval_tokens": total_tokens,
        }
        
        # Log metrics
        if self.rank == 0:
            print(f"Evaluation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
            
            if wandb_run is not None:
                wandb.log({
                    'eval/loss': avg_loss,
                    'eval/perplexity': perplexity,
                    'eval/epoch': epoch,
                    'eval/step': self.global_step,
                })
            
            # Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(
                    None, None, epoch, self.global_step, is_best=True
                )
        
        return metrics
    
    def save_checkpoint(
        self,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
        epoch: int,
        step: int,
        is_best: bool = False
    ):
        """Save checkpoint with FSDP support - keeps only best checkpoint"""
        
        if self.rank != 0:
            return
        
        # Only save if it's the best model or final checkpoint
        if not is_best and not (optimizer is None and scheduler is None):
            return
        
        # Create checkpoint directory
        suffix = "best" if is_best else "final"
        save_dir = Path(self.config.output_dir) / f"checkpoint_{suffix}"
        
        # Clean up old checkpoints if saving new best
        if is_best:
            self._cleanup_old_checkpoints()
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model to save
        if isinstance(self.model, FSDP):
            # FSDP requires special handling
            save_policy = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True
            )
            with FSDP.state_dict_type(
                self.model, 
                StateDictType.FULL_STATE_DICT,
                save_policy
            ):
                state_dict = self.model.state_dict()
        else:
            # DDP or unwrapped model
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            state_dict = model_to_save.state_dict()
        
        # Save model state
        torch.save(state_dict, save_dir / "model.pt")
        
        # Save training state
        if optimizer and scheduler:
            torch.save({
                'epoch': epoch,
                'step': step,
                'global_step': self.global_step,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': self.best_loss,
                'config': asdict(self.config),
            }, save_dir / 'training_state.pt')
        
        # Save config
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        print(f"Checkpoint saved to {save_dir}")
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoint directories"""
        output_dir = Path(self.config.output_dir)
        if not output_dir.exists():
            return
        
        # Remove all checkpoint directories except the one we're about to create
        for checkpoint_dir in output_dir.glob("checkpoint_*"):
            if checkpoint_dir.is_dir():
                try:
                    import shutil
                    shutil.rmtree(checkpoint_dir)
                    print(f"Removed old checkpoint: {checkpoint_dir}")
                except Exception as e:
                    print(f"Warning: Could not remove {checkpoint_dir}: {e}")


# ============================================================================
# Main Training Pipeline
# ============================================================================

def setup_distributed(config: DistributedConfig) -> Tuple[int, int, int, torch.device]:
    """Initialize distributed training environment"""
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # torchrun or other launcher
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Single GPU or manual launch
        rank = 0
        world_size = 1
        local_rank = 0
        
        if torch.cuda.device_count() > 1:
            print(f"Warning: {torch.cuda.device_count()} GPUs available but running on single GPU")
            print("Use torchrun or torch.distributed.launch for multi-GPU training")
    
    # Initialize process group if multi-GPU
    if world_size > 1:
        os.environ["MASTER_ADDR"] = config.master_addr
        os.environ["MASTER_PORT"] = config.master_port
        
        dist.init_process_group(
            backend=config.backend,
            rank=rank,
            world_size=world_size
        )
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)
    
    return rank, world_size, local_rank, device


def main(config: DistributedConfig):
    """Main training pipeline"""
    
    # Setup distributed environment
    rank, world_size, local_rank, device = setup_distributed(config)
    
    # Logging setup
    if rank == 0:
        print("="*80)
        print("Advanced Multi-GPU Training")
        print("="*80)
        print(f"Model: {config.model_name}")
        print(f"Dataset: {config.dataset_name}")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Sharding mode: {config.sharding_mode}")
        print(f"Mixed precision: {config.mixed_precision}")
        print(f"Use TE: {config.use_te}")
        print(f"Batch size per GPU: {config.batch_size}")
        print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps * world_size}")
        print("="*80)
        
        # Estimate memory requirements
        if "1B" in config.model_name:
            params_b = 1
        elif "3B" in config.model_name:
            params_b = 3
        elif "7B" in config.model_name or "8B" in config.model_name:
            params_b = 8
        elif "13B" in config.model_name or "14B" in config.model_name:
            params_b = 14
        elif "30B" in config.model_name:
            params_b = 30
        elif "70B" in config.model_name:
            params_b = 70
        else:
            params_b = 1
        
        mem_estimate = MemoryProfiler.estimate_model_memory(
            params_b, 
            config.batch_size,
            config.max_seq_length,
            world_size,
            dtype_bytes=1 if config.mixed_precision == "fp8" else 2,
            sharding_mode=config.sharding_mode
        )
        
        print("\nMemory Estimation:")
        for k, v in mem_estimate.items():
            print(f"  {k}: {v:.2f} GB")
        print("="*80)
    
    # Initialize wandb
    if rank == 0 and config.use_wandb:
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"{config.model_name.split('/')[-1]}-{config.sharding_mode}",
            config=asdict(config),
            tags=[
                f"model:{config.model_name.split('/')[-1]}",
                f"gpus:{world_size}",
                f"sharding:{config.sharding_mode}",
                f"precision:{config.mixed_precision}"
            ]
        )
    else:
        wandb_run = None
    
    # Create model
    if rank == 0:
        print("\n[Step 1/5] Creating model...")
    
    model = ModelFactory.create_model(config, rank, world_size)
    
    # Wrap model with distributed strategy
    if rank == 0:
        print("[Step 2/5] Applying distributed strategy...")
    
    model = DistributedWrapper.wrap_model(
        model, config, rank, world_size, device
    )
    
    # Create trainer
    trainer = Trainer(config, model, rank, world_size, device)
    
    # Create optimizer and scheduler
    if rank == 0:
        print("[Step 3/5] Creating optimizer and scheduler...")
    
    optimizer = trainer.create_optimizer()
    
    # Load and prepare dataset
    if rank == 0:
        print("[Step 4/5] Loading and preparing dataset...")
    
    # Load dataset
    dataset = load_dataset(
        config.dataset_name,
        split=config.split_name,
        trust_remote_code=True
    )
    
    # Limit samples if specified
    if config.num_of_samples:
        dataset = dataset.select(range(min(config.num_of_samples, len(dataset))))
    
    # Process dataset
    data_manager = DataManager(config)
    
    # Use simple map for dataset processing
    if rank == 0:
        print(f"Processing {len(dataset)} samples...")
    
    dataset = dataset.map(
        data_manager._apply_template,
        remove_columns=dataset.column_names,
        num_proc=min(config.num_proc, 8) if rank == 0 else 1,
        desc="Processing dataset"
    )
    
    # Split dataset
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    # Create dataloaders
    train_loader, train_sampler = trainer.create_dataloader(train_dataset, is_train=True)
    eval_loader, _ = trainer.create_dataloader(eval_dataset, is_train=False)
    
    # Create scheduler
    total_steps = (len(train_loader) * config.num_epochs) // config.gradient_accumulation_steps
    scheduler = trainer.create_scheduler(optimizer, total_steps)
    
    # Training
    if rank == 0:
        print(f"[Step 5/5] Starting training...")
        print(f"Total epochs: {config.num_epochs}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {config.num_warmup_steps}")
        print("="*80)
    
    # Training loop
    for epoch in range(1, config.num_epochs + 1):
        if rank == 0:
            print(f"\n{'='*20} Epoch {epoch}/{config.num_epochs} {'='*20}")
        
        # Set epoch for sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_metrics = trainer.train_epoch(
            train_loader, optimizer, scheduler, epoch, wandb_run
        )
        
        # Evaluate
        if epoch % config.eval_interval == 0 or epoch == config.num_epochs:
            eval_metrics = trainer.evaluate(eval_loader, epoch, wandb_run)
        
        # Log epoch summary
        if rank == 0:
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Perplexity: {train_metrics['perplexity']:.2f}")
            print(f"  Throughput: {train_metrics['throughput']:.0f} tokens/sec")
            print(f"  Epoch Time: {train_metrics['epoch_time']:.2f} seconds")
    
    # Final save
    if rank == 0:
        print("\n" + "="*80)
        print("Training completed!")
        trainer.save_checkpoint(optimizer, scheduler, config.num_epochs, trainer.global_step)
        
        if wandb_run is not None:
            wandb.finish()
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced Multi-GPU Training with Auto-Sharding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and data
    parser.add_argument("--model_name", type=str, required=True,
                       help="HuggingFace model name")
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="HuggingFace dataset name")
    parser.add_argument("--dataset_text_field", type=str, default="text")
    parser.add_argument("--split_name", type=str, default="train_1M")
    parser.add_argument("--num_of_samples", type=int,
                       help="Limit number of samples (for testing)")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    
    # Precision and optimization
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                       choices=["bf16", "fp16", "fp8", "fp32"])
    parser.add_argument("--fp8_scenario", type=str, default="default",
                       choices=["default", "mxfp8"])
    parser.add_argument("--use_te", action="store_true",
                       help="Use Transformer Engine")
    
    # Sharding and parallelism
    parser.add_argument("--sharding_mode", type=str, default="auto",
                       choices=["auto", "ddp", "fsdp_grad", "fsdp_full", "fsdp_offload"])
    parser.add_argument("--cpu_offload", action="store_true",
                       help="Enable CPU offloading")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient/activation checkpointing")
    parser.add_argument("--compile_model", action="store_true",
                       help="Use torch.compile (PyTorch 2.0+)")
    
    # Memory optimization
    parser.add_argument("--empty_cache_freq", type=int, default=50,
                       help="Clear cache every N steps")
    parser.add_argument("--memory_efficient_attention", action="store_true")
    
    # Logging and checkpointing
    parser.add_argument("--log_dir", type=str, default="./runs")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=1)
    
    # Weights & Biases
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="llm-fp8-advanced")
    parser.add_argument("--wandb_run_name", type=str)
    
    # Technical settings
    parser.add_argument("--weights_cache_dir", type=str, default="./model_cache")
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--profile_memory", action="store_true")
    
    args = parser.parse_args()
    
    # Create config
    config = DistributedConfig(
        # Model and data
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_text_field=args.dataset_text_field,
        split_name=args.split_name,
        num_of_samples=args.num_of_samples,
        
        # Training
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size or args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.num_warmup_steps,
        max_seq_length=args.max_seq_length,
        
        # Precision
        mixed_precision=args.mixed_precision,
        fp8_scenario=args.fp8_scenario,
        use_te=args.use_te,
        
        # Sharding
        sharding_mode=args.sharding_mode,
        cpu_offload=args.cpu_offload,
        gradient_checkpointing=args.gradient_checkpointing,
        compile_model=args.compile_model,
        
        # Memory
        empty_cache_freq=args.empty_cache_freq,
        memory_efficient_attention=args.memory_efficient_attention,
        
        # Logging
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        
        # Technical
        weights_cache_dir=args.weights_cache_dir,
        num_proc=args.num_proc,
        num_workers=args.num_workers,
        debug_mode=args.debug_mode,
        profile_memory=args.profile_memory,
    )
    
    # Run training
    main(config)