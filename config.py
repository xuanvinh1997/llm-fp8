
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    
    # Model and data configuration
    model_name: str = "meta-llama/Llama-2-7b-hf"
    dataset_name: str = "timdettmers/openassistant-guanaco"
    dataset_text_field: str = "text"
    max_seq_length: int = 256
    split_name: str = "train_1M"
    num_of_samples: Optional[int] = None
    
    # Training hyperparameters
    learning_rate: float = 1.41e-5
    batch_size: int = 8
    eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1
    num_epochs: int = 3
    num_warmup_steps: int = 100
    
    # Precision and optimization
    mixed_precision: str = "bf16"
    fp8_scenario: str = "default"
    use_te: bool = False
    
    # Evaluation settings
    eval_split: str = "validation"
    
    # Logging and saving
    log_dir: str = "./runs"
    output_dir: str = "./saved_model"
    use_wandb: bool = False
    wandb_project: str = "llm-fp8"
    wandb_run_name: Optional[str] = None
    
    # Technical settings
    weights_cache_dir: str = ""
    num_proc: int = 48
    hf_access_token: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization to set derived values."""
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size