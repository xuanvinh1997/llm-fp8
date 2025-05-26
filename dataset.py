from typing import Dict, List, Optional
from datasets import Dataset, load_dataset
import json
import logging

import torch

logger = logging.getLogger(__name__)

class MathInstructionFormatter:
    """Custom formatter for math instruction datasets - Qwen and Llama 3.2 only"""
    
    @staticmethod
    def qwen_format(problem: str, solution: str) -> str:
        """Qwen-style conversation format"""
        return f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n{solution}<|im_end|>"
    
    @staticmethod
    def llama_format(problem: str, solution: str) -> str:
        """Llama 3.2-style conversation format"""
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{problem}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{solution}<|eot_id|>"
    
    @staticmethod
    def auto_detect_format(tokenizer, problem: str, solution: str) -> str:
        """Auto-detect the best format based on tokenizer type"""
        model_name = getattr(tokenizer, 'name_or_path', '').lower()
        
        if 'qwen' in model_name:
            return MathInstructionFormatter.qwen_format(problem, solution)
        elif 'llama' in model_name:
            return MathInstructionFormatter.llama_format(problem, solution)
        else:
            # Default to Qwen format for unknown models
            return MathInstructionFormatter.qwen_format(problem, solution)


class InstructionDataset(Dataset):
    """Enhanced dataset class for instruction tuning with robust formatting"""
    
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 max_length: int = 2048,
                 format_style: str = "auto",
                 mask_instruction: bool = False):
        """
        Initialize dataset with flexible formatting options
        
        Args:
            data_path: Path to dataset (JSON file or HuggingFace dataset name)
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            format_style: Formatting style - "auto", "qwen", "llama"
            mask_instruction: Whether to mask instruction tokens in loss computation
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_style = format_style
        self.mask_instruction = mask_instruction
        self.data = self._load_and_process_data(data_path)
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
    
    def _load_and_process_data(self, data_path: str) -> List[Dict]:
        """Load and preprocess data efficiently"""
        logger.info(f"Loading dataset from {data_path}")
        
        try:
            if data_path.endswith('.json'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                # Support for HuggingFace datasets
                dataset = load_dataset(data_path)
                data = dataset['train'] if 'train' in dataset else dataset
                
            logger.info(f"Loaded {len(data)} examples")
            return data
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def _format_conversation(self, problem: str, solution: str) -> str:
        """Format conversation using specified style with fallback"""
        try:
            # First try apply_chat_template if available
            if hasattr(self.tokenizer, 'apply_chat_template') and callable(self.tokenizer.apply_chat_template):
                try:
                    messages = [
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": solution}
                    ]
                    return self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                except Exception as e:
                    logger.warning(f"apply_chat_template failed: {e}, falling back to custom formatting")
            
            # Use custom formatting based on style
            if self.format_style == "auto":
                return MathInstructionFormatter.auto_detect_format(self.tokenizer, problem, solution)
            elif self.format_style == "qwen":
                return MathInstructionFormatter.qwen_format(problem, solution)
            elif self.format_style == "llama":
                return MathInstructionFormatter.llama_format(problem, solution)
            else:
                logger.warning(f"Unknown format style: {self.format_style}, using auto")
                return MathInstructionFormatter.auto_detect_format(self.tokenizer, problem, solution)
                
        except Exception as e:
            logger.error(f"Error in formatting: {e}")
            # Ultimate fallback
            return f"Problem: {problem}\n\nSolution: {solution}"
    
    def _create_instruction_mask(self, input_ids: torch.Tensor, formatted_text: str) -> torch.Tensor:
        """Create mask to exclude instruction tokens from loss computation"""
        if not self.mask_instruction:
            return torch.ones_like(input_ids)
        
        try:
            # Find where the assistant/response starts
            if self.format_style == "qwen" or "assistant" in formatted_text.lower():
                assistant_start = formatted_text.find("<|im_start|>assistant")
                if assistant_start == -1:
                    assistant_start = formatted_text.find("assistant")
            elif self.format_style == "llama":
                assistant_start = formatted_text.find("<|start_header_id|>assistant<|end_header_id|>")
                if assistant_start == -1:
                    assistant_start = formatted_text.find("assistant")
            else:
                # Simple fallback - look for common response indicators
                indicators = ["assistant", "<|im_start|>assistant", "<|start_header_id|>assistant"]
                assistant_start = -1
                for indicator in indicators:
                    pos = formatted_text.find(indicator)
                    if pos != -1:
                        assistant_start = pos
                        break
            
            if assistant_start == -1:
                logger.warning("Could not find response start, not masking instruction")
                return torch.ones_like(input_ids)
            
            # Tokenize the instruction part to find where to start unmasking
            instruction_part = formatted_text[:assistant_start]
            instruction_tokens = self.tokenizer(
                instruction_part,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids.squeeze()
            
            # Create mask (0 for instruction, 1 for response)
            mask = torch.zeros_like(input_ids)
            if len(instruction_tokens.shape) == 0:
                instruction_tokens = instruction_tokens.unsqueeze(0)
            
            start_idx = len(instruction_tokens)
            if start_idx < len(input_ids):
                mask[start_idx:] = 1
            
            return mask
            
        except Exception as e:
            logger.warning(f"Error creating instruction mask: {e}, using full sequence")
            return torch.ones_like(input_ids)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            item = self.data[idx]
            
            # Handle different data formats flexibly
            if isinstance(item, dict):
                # Standard format with 'problem' and 'generated_solution'
                if 'problem' in item and 'generated_solution' in item:
                    problem = item['problem']
                    solution = item['generated_solution']
                # Alternative formats
                elif 'instruction' in item and 'output' in item:
                    problem = item['instruction']
                    solution = item['output']
                elif 'question' in item and 'answer' in item:
                    problem = item['question']
                    solution = item['answer']
                elif 'input' in item and 'output' in item:
                    problem = item['input']
                    solution = item['output']
                else:
                    # Try to infer from available keys
                    keys = list(item.keys())
                    logger.warning(f"Unknown data format with keys: {keys}")
                    problem = str(item.get(keys[0], ""))
                    solution = str(item.get(keys[1], "")) if len(keys) > 1 else ""
            else:
                logger.error(f"Unexpected item type: {type(item)}")
                problem = str(item)
                solution = ""
            
            # Format the conversation
            formatted_text = self._format_conversation(problem, solution)
            
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
            
            # Create labels with optional instruction masking
            labels = input_ids.clone()
            
            if self.mask_instruction:
                instruction_mask = self._create_instruction_mask(input_ids, formatted_text)
                # Set instruction tokens to -100 (ignored in loss)
                labels = labels * instruction_mask + (-100) * (1 - instruction_mask)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            # Return a minimal valid sample as fallback
            dummy_text = "Error processing sample"
            encoding = self.tokenizer(
                dummy_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt"
            )
            input_ids = encoding.input_ids.squeeze()
            attention_mask = encoding.attention_mask.squeeze()
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone()
            }


class MathDatasetValidator:
    """Utility class to validate and analyze math instruction datasets"""
    
    @staticmethod
    def validate_dataset(dataset_path: str) -> Dict[str, any]:
        """Validate dataset format and return statistics"""
        try:
            if dataset_path.endswith('.json'):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                dataset = load_dataset(dataset_path)
                data = dataset['train'] if 'train' in dataset else dataset
            
            if not data:
                return {"valid": False, "error": "Empty dataset"}
            
            # Analyze data structure
            sample = data[0]
            available_keys = list(sample.keys()) if isinstance(sample, dict) else []
            
            # Check for common field patterns
            problem_fields = ['problem', 'instruction', 'question', 'input']
            solution_fields = ['generated_solution', 'solution', 'output', 'answer']
            
            found_problem = any(field in available_keys for field in problem_fields)
            found_solution = any(field in available_keys for field in solution_fields)
            
            stats = {
                "valid": found_problem and found_solution,
                "num_samples": len(data),
                "available_keys": available_keys,
                "problem_field_found": found_problem,
                "solution_field_found": found_solution,
                "sample_lengths": []
            }
            
            # Analyze sample lengths
            for i, item in enumerate(data[:100]):  # Check first 100 samples
                if isinstance(item, dict):
                    total_length = sum(len(str(v)) for v in item.values())
                    stats["sample_lengths"].append(total_length)
            
            if stats["sample_lengths"]:
                stats["avg_length"] = sum(stats["sample_lengths"]) / len(stats["sample_lengths"])
                stats["max_length"] = max(stats["sample_lengths"])
                stats["min_length"] = min(stats["sample_lengths"])
            
            return stats
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    @staticmethod
    def preview_formatting(dataset_path: str, tokenizer, num_samples: int = 3) -> List[str]:
        """Preview how samples will be formatted"""
        try:
            dataset = InstructionDataset(dataset_path, tokenizer)
            previews = []
            
            for i in range(min(num_samples, len(dataset))):
                item = dataset.data[i]
                if isinstance(item, dict):
                    problem = item.get('problem', item.get('instruction', item.get('question', '')))
                    solution = item.get('generated_solution', item.get('solution', item.get('answer', '')))
                    formatted = dataset._format_conversation(problem, solution)
                    previews.append(formatted)
            
            return previews
            
        except Exception as e:
            return [f"Error: {e}"]