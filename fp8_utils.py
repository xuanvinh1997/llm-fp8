from dataclasses import dataclass
import re
import torch
from tqdm import tqdm
from datasets import Dataset, load_dataset
from typing import Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MathTrainingArguments:
    """Training configuration for math instruction tuning"""
    # Model and data
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    dataset_name: str = "nvidia/OpenMathInstruct-2"
    max_length: int = 2048  # Longer for math solutions
    max_samples: Optional[int] = None  # Limit dataset size for testing
    
    # Training hyperparameters
    batch_size: int = 2     # Smaller for 3B model
    gradient_accumulation_steps: int = 8  # Compensate for smaller batch
    learning_rate: float = 2e-5  # Conservative for math
    num_epochs: int = 2
    warmup_steps: int = 200
    weight_decay: float = 0.01
    seed: int = 42
    
    # Evaluation and saving
    save_steps: int = 1000
    eval_steps: int = 500
    output_dir: str = "./qwen_math_fp8_model"
    
    mixed_precision: str = "fp8"  # Use FP8 for training
    # FP8 configuration
    fp8_backend: str = "msamp"
    
    # MS-AMP specific
    msamp_opt_level: str = "O2"
    
    # TransformerEngine specific  
    te_fp8_format: str = "HYBRID"
    te_amax_history_len: int = 32
    te_amax_compute_algo: str = "max"
    
    # Math-specific settings
    use_generated_solution: bool = True  # Use generated_solution field
    solution_field: str = "generated_solution"  # or "solution"
    
    # Weights & Biases configuration
    use_wandb: bool = True  # Use Weights & Biases for logging
    wandb_project: str = "qwen-math-instruct"
    wandb_entity: str = "your_wandb_entity"  # Replace with your entity
    wandb_run_name: Optional[str] = None  # Auto-generated if None
    wandb_tags: Optional[List[str]] = None  # Will default to ["fp8", "qwen", "math"]
    wandb_notes: str = ""
    wandb_resume: bool = False
    wandb_watch_model: bool = False  # Memory intensive
    wandb_watch_freq: int = 1000
    wandb_log_freq: int = 10
    wandb_log_model: bool = False  # Log final model as artifact
    max_grad_norm: float = 1.0  # Gradient clipping for stability
    zero_stage: int = 0  # No ZeRO optimization for simplicity
    zero_offload_param: bool = False  # No offloading for simplicity
    zero_offload_optimizer: bool = False  # No offloading for simplicity
    zero3_init_flag: bool = False  # No ZeRO-3 initialization for simplicity


def clean_math_text(text):
    """Clean and normalize math text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Normalize LaTeX formatting
    text = re.sub(r'\$\$([^$]+)\$\$', r'$$\1$$', text)  # Block math
    text = re.sub(r'\$([^$]+)\$', r'$\1$', text)        # Inline math
    
    return text


def load_and_process_math_dataset(dataset_name: str, tokenizer, max_length: int, max_samples: Optional[int] = None):
    """Load and process the OpenMathInstruct-2 dataset"""
    logger.info(f"Loading math dataset: {dataset_name}")
    
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, split='train')
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            logger.info(f"Limited dataset to {len(dataset)} samples")
        
        logger.info(f"Dataset loaded with {len(dataset)} examples")
        logger.info(f"Dataset columns: {dataset.column_names}")
      
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
    # Process and format examples
    formatted_examples = []
    skipped = 0
    
    for i, example in enumerate(tqdm(dataset, desc="Processing examples")):
        formatted = format_math_problem(example)
        if formatted:
            formatted_examples.append({"text": formatted})
        else:
            skipped += 1
    
    logger.info(f"Processed {len(formatted_examples)} examples, skipped {skipped}")
    
    if not formatted_examples:
        raise ValueError("No valid examples found in dataset!")
    
    # Create dataset from formatted examples
    formatted_dataset = Dataset.from_list(formatted_examples)
    
    # Tokenize
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
        )
        # For causal LM, labels are the same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=formatted_dataset.column_names,
        desc="Tokenizing"
    )
    
    # Split dataset (90% train, 10% eval)
    train_size = int(0.9 * len(tokenized_dataset))
    eval_size = len(tokenized_dataset) - train_size
    
    train_dataset, eval_dataset = torch.utils.data.random_split(
        tokenized_dataset, [train_size, eval_size]
    )
    
    logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def format_math_problem(example):
    """Format a math problem for training"""
    problem = clean_math_text(example.get('problem', ''))
    
    # Choose which solution to use
    if 'generated_solution' in example and example['generated_solution']:
        solution = clean_math_text(example['generated_solution'])
    elif 'solution' in example and example['solution']:
        solution = clean_math_text(example['solution'])
    else:
        # Skip examples without solutions
        return None
    
    # Format as instruction-following return result in \boxed{}
    formatted_text = f"""<|im_start|>system
You are a helpful assistant that solves math problems step by step and returns the final answer in a \\boxed{{}} format.
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
{solution}<|im_end|>"""
    
    return formatted_text