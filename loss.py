
import torch
import torch.nn.functional as F

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
