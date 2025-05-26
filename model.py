import torch
import logging
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import TrainingConfig

logger = logging.getLogger(__name__)


class ModelManager:
    """Modular model management class"""
    
    @staticmethod
    def setup_model_and_tokenizer(config: TrainingConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Initialize model and tokenizer with optimizations"""
        logger.info(f"Loading model: {config.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name,  # Use tokenizer_name from config
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