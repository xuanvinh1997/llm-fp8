from typing import Tuple
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from accelerate import Accelerator
from config import TrainingConfig


class DataManager:
    """Handles data loading, preprocessing, and DataLoader creation."""
    
    LLAMA_CHAT_TEMPLATE = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful assistant that solves math problems step by step. "
        "Please reason step by step, and put your final answer within \\boxed{{}}."
        "\n<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n{problem}\n<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n{solution}<|eot_id|>"
    )

    QWEN_CHAT_TEMPLATE = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful assistant that solves math problems step by step. "
        "Please reason step by step, and put your final answer within \\boxed{{}}."
        "\n<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n{problem}\n<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n{solution}<|eot_id|>"
    )
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = self._setup_tokenizer()
        if "llama" in self.config.model_name.lower():
            self.CHAT_TEMPLATE = self.LLAMA_CHAT_TEMPLATE
        elif "qwen" in self.config.model_name.lower():
            self.CHAT_TEMPLATE = self.QWEN_CHAT_TEMPLATE
        else:
            raise ValueError("Unsupported model for chat template.")
    
    def _setup_tokenizer(self) -> AutoTokenizer:
        """Initialize and configure tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def create_dataloaders(self, accelerator: Accelerator) -> Tuple[DataLoader, DataLoader]:
        """Create training and evaluation dataloaders."""
        # Load and preprocess dataset
        dataset = self._load_and_process_dataset(accelerator)
        
        # Split dataset
        dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]
        
        # Create data collator
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=16,
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=collator,
            shuffle=True,
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.eval_batch_size,
            collate_fn=collator,
        )
        
        return train_loader, eval_loader
    
    def _load_and_process_dataset(self, accelerator: Accelerator):
        """Load dataset and apply preprocessing."""
        # Load dataset
        dataset = load_dataset(self.config.dataset_name, split=self.config.split_name)
        
        # Limit the number of samples if specified
        if self.config.num_of_samples is not None:
            dataset = dataset.select(range(min(self.config.num_of_samples, len(dataset))))
        
        # Apply chat template and tokenization
        with accelerator.main_process_first():
            dataset = dataset.map(
                self._apply_template,
                remove_columns=dataset.column_names,
                num_proc=self.config.num_proc,
            )
        
        return dataset
    
    def _apply_template(self, example):
        """Apply chat template and tokenize example."""
        text = self.CHAT_TEMPLATE.format(
            problem=example["problem"],
            solution=example["generated_solution"],
        )
        
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_seq_length,
        )
        
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }
