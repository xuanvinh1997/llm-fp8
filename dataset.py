from typing import Dict, List
from datasets import Dataset, load_dataset
import json
import logging
import torch
from torch.utils.data import Dataset as TorchDataset

logger = logging.getLogger(__name__)

class InstructionDataset(TorchDataset):  # Changed from Dataset to TorchDataset
    """Optimized dataset class for instruction tuning"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_and_process_data(data_path)  # Changed from 'data' to 'examples'
        
    def _load_and_process_data(self, data_path: str) -> List[Dict]:
        """Load and preprocess data efficiently"""
        logger.info(f"Loading dataset from {data_path}")
        
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # Support for HuggingFace datasets
            dataset = load_dataset(data_path)
            data = dataset['train'] if 'train' in dataset else dataset
            # Convert to list if it's a HF Dataset
            if hasattr(data, 'to_list'):
                data = data.to_list()
            elif hasattr(data, '__iter__'):
                data = list(data)
            
        logger.info(f"Loaded {len(data)} examples")
        return data
    
    def _format_chat_template(self, problem: str, solution: str) -> str:
        """Format using Qwen chat template"""
        try:
            messages = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution}
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception as e:
            logger.warning(f"Chat template failed: {e}, using fallback format")
            return f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n{solution}<|im_end|>"
    
    def __len__(self) -> int:
        return len(self.examples)  # Changed from self.data to self.examples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]  # Changed from self.data to self.examples
        
        # Handle different data formats
        if isinstance(item, dict):
            problem = item.get('problem', item.get('instruction', item.get('question', '')))
            solution = item.get('generated_solution', item.get('solution', item.get('answer', '')))
        else:
            # Fallback for other formats
            problem = str(item)
            solution = ""
        
        # Combine problem and solution using chat template
        formatted_text = self._format_chat_template(problem, solution)
        
        # Tokenize efficiently
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        
        # Handle single token case
        if len(input_ids.shape) == 0:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }