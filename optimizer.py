from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch
from torch.optim import AdamW
from config import TrainingConfig

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