from typing import Dict, List
from datasets import Dataset, load_dataset
import json
import logging

import torch

logger = logging.getLogger(__name__)

class InstructionDataset(Dataset):
    """Optimized dataset class for instruction tuning"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_and_process_data(data_path)
        
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
            
        logger.info(f"Loaded {len(data)} examples")
        return data
    
    def _format_chat_template(self, problem: str, solution: str) -> str:
        """Format using Qwen chat template"""
        messages = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution}
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Combine problem and solution using chat template
        formatted_text = self._format_chat_template(
            item['problem'], item['generated_solution']
        )
        
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
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
