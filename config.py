from accelerate.utils import FP8RecipeKwargs
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any


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
    
    # Precision configuration
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16", "fp8"
    fp8_backend: str = "te"  # "te", "msamp", "ao"
    
    # TransformersEngine (TE) specific config
    te_fp8_format: str = "HYBRID"  # "E4M3", "E5M2", "HYBRID"
    te_amax_history_len: int = 1024
    te_amax_compute_algo: str = "max"
    te_margin: int = 0
    te_interval: int = 1
    te_override_linear_precision: Tuple[bool, bool, bool] = (False, False, False)
    te_use_autocast_during_eval: bool = False
    
    # MS-AMP specific config
    msamp_opt_level: str = "O2"  # "O1", "O2"
    
    # Optimization settings
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
            
        backend_configs = {
            "te": lambda: FP8RecipeKwargs(
                backend="te",
                fp8_format=self.te_fp8_format,
                amax_history_len=self.te_amax_history_len,
                amax_compute_algo=self.te_amax_compute_algo,
                margin=self.te_margin,
                interval=self.te_interval,
                override_linear_precision=self.te_override_linear_precision,
                use_autocast_during_eval=self.te_use_autocast_during_eval
            ),
            "msamp": lambda: FP8RecipeKwargs(
                backend="msamp",
                opt_level=self.msamp_opt_level
            ),
            "ao": lambda: FP8RecipeKwargs(backend="ao")
        }
        
        if self.fp8_backend not in backend_configs:
            raise ValueError(f"Unsupported FP8 backend: {self.fp8_backend}")
        
        return backend_configs[self.fp8_backend]()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.max_length > 0, "max_length must be positive"
        assert 0 <= self.warmup_ratio <= 1, "warmup_ratio must be between 0 and 1"
        assert self.mixed_precision in ["no", "fp16", "bf16", "fp8"], f"Invalid mixed_precision: {self.mixed_precision}"
        
        if self.mixed_precision == "fp8":
            assert self.fp8_backend in ["te", "msamp", "ao"], f"Invalid fp8_backend: {self.fp8_backend}"