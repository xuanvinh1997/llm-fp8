from typing import Dict, List, Tuple
import json
import logging
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import pickle
import hashlib

logger = logging.getLogger(__name__)

def _process_chunk_worker(args: Tuple) -> Tuple[List[Dict], int, int]:
    """Worker function for parallel processing of data chunks with robust error handling"""
    try:
        chunk_data, tokenizer_name, max_length = args
        
        # Recreate tokenizer in worker process with error handling
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, 
                trust_remote_code=True, 
                use_fast=True,
                local_files_only=False
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            # Return empty result if tokenizer fails
            return [], 0, 0
        
        def format_chat_template(problem: str, solution: str) -> str:
            """Format using chat template with fallback"""
            try:
                messages = [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution}
                ]
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except:
                return f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n{solution}<|im_end|>"
        
        filtered_chunk = []
        total_chars_before = 0
        total_chars_after = 0
        
        # Process chunk with memory limits
        for i, item in enumerate(chunk_data):
            try:
                # Skip processing if chunk is getting too large (memory protection)
                if len(filtered_chunk) > 10000:  # Limit chunk results
                    break
                    
                # Handle different data formats
                if isinstance(item, dict):
                    problem = item.get('problem', item.get('instruction', item.get('question', '')))
                    solution = item.get('generated_solution', item.get('solution', item.get('answer', '')))
                else:
                    problem = str(item)
                    solution = ""
                
                # Skip empty or overly long items
                if not problem.strip() or not solution.strip():
                    continue
                if len(problem) > 50000 or len(solution) > 50000:  # Skip very long examples
                    continue
                
                # Format text
                text = format_chat_template(problem, solution)
                total_chars_before += len(text)
                
                # Check tokenized length with error handling
                try:
                    tokens = tokenizer(text, return_tensors="pt", max_length=max_length*2, truncation=True)
                    if tokens.input_ids.shape[1] <= max_length:
                        filtered_chunk.append(item)
                        total_chars_after += len(text)
                except:
                    # Skip item if tokenization fails
                    continue
                    
            except Exception:
                # Skip individual item if it causes issues
                continue
        
        return filtered_chunk, total_chars_before, total_chars_after
        
    except Exception as e:
        # Return empty result if entire chunk fails
        return [], 0, 0

class InstructionDataset(Dataset):
    """Memory-efficient dataset with HuggingFace support and multiprocessing optimization"""
    
    def __init__(self, 
                 data_path: str, 
                 tokenizer: AutoTokenizer, 
                 max_length: int = 2048,
                 num_workers: int = None,
                 chunk_size: int = 1000,
                 use_cache: bool = True,
                 split: str = "train",
                 streaming: bool = False,
                 max_workers: int = None):  # Alternative parameter name
        """
        Initialize dataset with HuggingFace support and robust parallel processing
        
        Args:
            data_path: HuggingFace dataset name or local JSON file path
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            num_workers: Number of parallel workers (auto-detect if None)
            chunk_size: Size of chunks for parallel processing
            use_cache: Whether to cache processed results
            split: Dataset split to use (for HF datasets)
            streaming: Whether to use streaming for large HF datasets
            max_workers: Alternative name for num_workers (for compatibility)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = data_path
        self.use_cache = use_cache
        self.split = split
        self.streaming = streaming
        
        # Auto-detect optimal workers with conservative defaults
        if num_workers is None and max_workers is None:
            # More conservative worker count to prevent memory issues
            cpu_count = mp.cpu_count()
            if cpu_count <= 4:
                self.num_workers = 2
            elif cpu_count <= 8:
                self.num_workers = 4
            else:
                self.num_workers = min(6, cpu_count - 2)  # Leave some CPUs free
        else:
            self.num_workers = num_workers or max_workers or 2
        
        # Adaptive chunk size based on available memory and workers
        self.chunk_size = min(chunk_size, 2000)  # Cap chunk size to prevent memory issues
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Initializing dataset with {self.num_workers} workers, chunk size {self.chunk_size}")
        
        # Try to load from cache first
        cache_file = self._get_cache_file()
        if self.use_cache and os.path.exists(cache_file):
            logger.info(f"Loading cached dataset from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    self.examples = pickle.load(f)
                logger.info(f"Loaded {len(self.examples)} cached examples")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, processing fresh dataset")
        
        # Process dataset with robust error handling
        try:
            start_time = time.time()
            self.examples = self._load_and_filter_parallel(data_path)
            processing_time = time.time() - start_time
            logger.info(f"Dataset processing completed in {processing_time:.2f} seconds")
            
            # Save to cache
            if self.use_cache and self.examples:
                self._save_to_cache(cache_file)
                
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            logger.info("Falling back to sequential processing...")
            try:
                raw_data = self._load_raw_data(data_path)
                self.examples = self._load_and_filter_sequential(raw_data)
                logger.info(f"Sequential fallback completed with {len(self.examples)} examples")
            except Exception as fallback_error:
                logger.error(f"Sequential fallback also failed: {fallback_error}")
                raise ValueError(f"Unable to process dataset: {fallback_error}")
    
    def _get_cache_file(self) -> str:
        """Generate cache file path"""
        # Create hash from parameters
        cache_key = f"{self.data_path}_{self.max_length}_{self.tokenizer.name_or_path}_{self.split}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        
        cache_dir = os.path.join(os.getcwd(), ".dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"dataset_{cache_hash}.pkl")
    
    def _save_to_cache(self, cache_file: str):
        """Save processed examples to cache"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.examples, f)
            logger.info(f"Saved {len(self.examples)} examples to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _is_local_file(self, data_path: str) -> bool:
        """Check if data_path is a local file"""
        return os.path.exists(data_path) and (data_path.endswith('.json') or data_path.endswith('.jsonl'))
    
    def _load_raw_data(self, data_path: str) -> List[Dict]:
        """Load raw data from either HuggingFace or local file"""
        if self._is_local_file(data_path):
            logger.info(f"Loading local dataset from {data_path}")
            if data_path.endswith('.jsonl'):
                # Handle JSONL files
                raw_data = []
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        raw_data.append(json.loads(line.strip()))
            else:
                # Handle JSON files
                with open(data_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
        else:
            logger.info(f"Loading HuggingFace dataset: {data_path}")
            try:
                if self.streaming:
                    # Use streaming for large datasets
                    dataset = load_dataset(data_path, split=self.split, streaming=True)
                    # Take a reasonable subset for streaming
                    raw_data = list(dataset.take(100000))  # Adjust as needed
                    logger.info(f"Loaded {len(raw_data)} examples from streaming dataset")
                else:
                    # Load full dataset
                    dataset = load_dataset(data_path)
                    
                    # Handle different dataset structures
                    if isinstance(dataset, dict):
                        if self.split in dataset:
                            raw_data = dataset[self.split]
                        else:
                            # Use first available split
                            available_splits = list(dataset.keys())
                            logger.warning(f"Split '{self.split}' not found. Available: {available_splits}")
                            self.split = available_splits[0]
                            raw_data = dataset[self.split]
                    else:
                        raw_data = dataset
                    
                    # Convert HF Dataset to list
                    if hasattr(raw_data, 'to_list'):
                        raw_data = raw_data.to_list()
                    elif hasattr(raw_data, '__iter__'):
                        raw_data = list(raw_data)
                        
            except Exception as e:
                logger.error(f"Failed to load HuggingFace dataset {data_path}: {e}")
                raise
        
        logger.info(f"Loaded {len(raw_data)} raw examples")
        return raw_data
    
    def _load_and_filter_parallel(self, data_path: str) -> List[Dict]:
        """Load data and filter by tokenized length using robust parallel processing"""
        # Load raw data
        raw_data = self._load_raw_data(data_path)
        
        if not raw_data:
            raise ValueError("No data loaded")
        
        # If dataset is small or multiprocessing disabled, use sequential
        if len(raw_data) < 1000 or self.num_workers == 1:
            return self._load_and_filter_sequential(raw_data)
        
        # Adaptive chunk size based on dataset size and available memory
        adaptive_chunk_size = min(self.chunk_size, max(100, len(raw_data) // (self.num_workers * 4)))
        
        # Split data into smaller chunks to prevent memory issues
        chunks = [raw_data[i:i + adaptive_chunk_size] for i in range(0, len(raw_data), adaptive_chunk_size)]
        logger.info(f"Processing {len(raw_data)} examples in {len(chunks)} chunks of size {adaptive_chunk_size} using {self.num_workers} workers")
        
        # Prepare arguments for workers
        worker_args = [(chunk, self.tokenizer.name_or_path, self.max_length) for chunk in chunks]
        
        filtered_examples = []
        total_chars_before = 0
        total_chars_after = 0
        failed_chunks = 0
        
        try:
            # Use multiprocessing with timeout and error recovery
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all chunks with timeout
                future_to_chunk = {}
                for i, args in enumerate(worker_args):
                    future = executor.submit(_process_chunk_worker, args)
                    future_to_chunk[future] = i
                
                # Collect results with robust error handling
                for future in as_completed(future_to_chunk, timeout=300):  # 5 minute timeout per chunk
                    chunk_idx = future_to_chunk[future]
                    try:
                        chunk_filtered, chars_before, chars_after = future.result(timeout=60)  # 1 minute result timeout
                        filtered_examples.extend(chunk_filtered)
                        total_chars_before += chars_before
                        total_chars_after += chars_after
                        logger.info(f"Processed chunk {chunk_idx + 1}/{len(chunks)} - {len(chunk_filtered)} examples")
                    except Exception as e:
                        failed_chunks += 1
                        logger.warning(f"Chunk {chunk_idx + 1} failed: {str(e)[:100]}... - Processing sequentially")
                        
                        # Process failed chunk sequentially as fallback
                        try:
                            chunk_data = worker_args[chunk_idx][0]  # Get original chunk data
                            fallback_result = self._process_chunk_sequential(chunk_data)
                            filtered_examples.extend(fallback_result)
                            logger.info(f"Recovered chunk {chunk_idx + 1} with sequential processing")
                        except Exception as fallback_error:
                            logger.error(f"Failed to recover chunk {chunk_idx + 1}: {fallback_error}")
                        
        except Exception as e:
            logger.error(f"Multiprocessing completely failed: {e}")
            if failed_chunks > len(chunks) // 2:  # If more than half failed
                logger.warning("Too many chunks failed, falling back to sequential processing")
                return self._load_and_filter_sequential(raw_data)
        
        # Report results
        if failed_chunks > 0:
            logger.warning(f"{failed_chunks}/{len(chunks)} chunks failed but were recovered")
        
        # Log statistics
        if len(raw_data) > 0:
            retention_rate = len(filtered_examples) / len(raw_data) * 100
            avg_length_before = total_chars_before / len(raw_data) if total_chars_before > 0 else 0
            avg_length_after = total_chars_after / len(filtered_examples) if filtered_examples else 0
            
            logger.info(f"Robust parallel filtering complete:")
            logger.info(f"  Original: {len(raw_data)} examples")
            logger.info(f"  Filtered: {len(filtered_examples)} examples")
            logger.info(f"  Retention rate: {retention_rate:.1f}%")
            logger.info(f"  Failed chunks: {failed_chunks}/{len(chunks)}")
            logger.info(f"  Avg length before: {avg_length_before:.0f} chars")
            logger.info(f"  Avg length after: {avg_length_after:.0f} chars")
        
        return filtered_examples
    
    def _process_chunk_sequential(self, chunk_data: List[Dict]) -> List[Dict]:
        """Process a single chunk sequentially (fallback for failed chunks)"""
        filtered_chunk = []
        
        for item in chunk_data:
            try:
                # Handle different data formats
                if isinstance(item, dict):
                    problem = item.get('problem', item.get('instruction', item.get('question', '')))
                    solution = item.get('generated_solution', item.get('solution', item.get('answer', '')))
                else:
                    problem = str(item)
                    solution = ""
                
                if not problem.strip() or not solution.strip():
                    continue
                
                # Skip overly long examples
                if len(problem) > 50000 or len(solution) > 50000:
                    continue
                
                try:
                    messages = [
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": solution}
                    ]
                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                except:
                    text = f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n{solution}<|im_end|>"
                
                # Check length and filter
                tokens = self.tokenizer(text, return_tensors="pt", max_length=self.max_length*2, truncation=True)
                if tokens.input_ids.shape[1] <= self.max_length:
                    filtered_chunk.append(item)
            except:
                continue
        
        return filtered_chunk
    
    def _load_and_filter_sequential(self, raw_data: List[Dict]) -> List[Dict]:
        """Fallback sequential processing (original method)"""
        logger.info("Using sequential processing")
        
        filtered = []
        for item in raw_data:
            # Handle different data formats
            if isinstance(item, dict):
                problem = item.get('problem', item.get('instruction', item.get('question', '')))
                solution = item.get('generated_solution', item.get('solution', item.get('answer', '')))
            else:
                problem = str(item)
                solution = ""
            
            if not problem.strip() or not solution.strip():
                continue
            
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
        
        logger.info(f"Sequential filtering: {len(raw_data)} -> {len(filtered)} examples ({len(filtered)/len(raw_data)*100:.1f}% retained)")
        return filtered
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item (keeping original structure exactly)"""
        item = self.examples[idx]
        
        # Handle different data formats (original logic)
        if isinstance(item, dict):
            problem = item.get('problem', item.get('instruction', item.get('question', '')))
            solution = item.get('generated_solution', item.get('solution', item.get('answer', '')))
        else:
            problem = str(item)
            solution = ""
        
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
    
    def clear_cache(self):
        """Clear cached data"""
        cache_file = self._get_cache_file()
        if os.path.exists(cache_file):
            os.remove(cache_file)
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            "total_examples": len(self.examples),
            "max_length": self.max_length,
            "num_workers": self.num_workers,
            "chunk_size": self.chunk_size,
            "cached": self.use_cache,
            "split": self.split,
            "streaming": self.streaming
        }


# Example usage with HuggingFace datasets
def load_math_dataset():
    """Example: Load math instruction dataset from HuggingFace"""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    
    # HuggingFace math dataset
    dataset = InstructionDataset(
        data_path="microsoft/orca-math-word-problems-200k",
        tokenizer=tokenizer,
        max_length=2048,
        split="train"
    )
    
    print(f"Loaded {len(dataset)} examples")
    print("Stats:", dataset.get_stats())
    return dataset

def load_coding_dataset():
    """Example: Load coding instruction dataset"""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    
    # HuggingFace coding dataset
    dataset = InstructionDataset(
        data_path="iamtarun/python_code_instructions_18k_alpaca",
        tokenizer=tokenizer,
        max_length=2048,
        split="train"
    )
    
    print(f"Loaded {len(dataset)} examples")
    return dataset

def load_large_streaming_dataset():
    """Example: Load large dataset with streaming"""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    
    # Large dataset with streaming
    dataset = InstructionDataset(
        data_path="HuggingFaceTB/cosmopedia",
        tokenizer=tokenizer,
        max_length=1024,  # Smaller for large datasets
        split="train",
        streaming=True,
        chunk_size=500
    )
    
    print(f"Loaded {len(dataset)} examples (streaming)")
    return dataset
