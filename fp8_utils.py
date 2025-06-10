from dataclasses import dataclass
import re
import torch
from tqdm import tqdm
from datasets import Dataset, load_dataset
from typing import Optional
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MathTrainingArguments:
    """Training configuration for math instruction tuning"""
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    dataset_name: str = "nvidia/OpenMathInstruct-2"
    max_length: int = 2048  # Longer for math solutions
    batch_size: int = 2     # Smaller for 3B model
    gradient_accumulation_steps: int = 16  # Compensate for smaller batch
    learning_rate: float = 2e-5  # Conservative for math
    num_epochs: int = 2
    warmup_steps: int = 200
    save_steps: int = 1000
    eval_steps: int = 500
    output_dir: str = "./qwen_math_fp8_model"
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
    max_samples: Optional[int] = None  # Limit dataset size for testing
    
    # General training
    seed: int = 42
    weight_decay: float = 0.01
    


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
        
        # Show sample
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"Sample keys: {list(sample.keys())}")
            if 'problem' in sample:
                logger.info(f"Sample problem: {sample['problem'][:200]}...")
            if 'generated_solution' in sample:
                logger.info(f"Has generated_solution: {bool(sample['generated_solution'])}")
            if 'solution' in sample:
                logger.info(f"Has solution: {bool(sample['solution'])}")
        
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
    
    # Format as instruction-following
    formatted_text = f"""<|im_start|>system
You are a helpful assistant that solves math problems step by step.<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
{solution}<|im_end|>"""
    
    return formatted_text