from typing import Dict, List
import json
import logging
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class InstructionDataset(Dataset):
    """Memory-efficient dataset with length filtering"""
    
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        self.examples = self._load_and_filter(data_path)
        
    def _load_and_filter(self, data_path: str) -> List[Dict]:
        """Load data and filter by tokenized length"""
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        filtered = []
        for item in raw_data:
            # Format text
            problem = item.get('problem', item.get('instruction', ''))
            solution = item.get('generated_solution', item.get('solution', ''))
            
            try:
                messages = [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution}
                ]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except:
                text = f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n{solution}<|im_end|>"
            
            # Check length and filter
            tokens = self.tokenizer(text, return_tensors="pt")
            if tokens.input_ids.shape[1] <= self.max_length:
                filtered.append(item)
        
        logger.info(f"Filtered {len(raw_data)} -> {len(filtered)} examples ({len(filtered)/len(raw_data)*100:.1f}% retained)")
        return filtered
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        
        problem = item.get('problem', item.get('instruction', ''))
        solution = item.get('generated_solution', item.get('solution', ''))
        
        try:
            messages = [{"role": "user", "content": problem}, {"role": "assistant", "content": solution}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except:
            text = f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n{solution}<|im_end|>"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        
        if input_ids.dim() == 0:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }