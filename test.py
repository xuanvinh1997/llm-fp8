# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script tests to ensure that `accelerate` performs at the same level as raw `MS-AMP`.

This particular script verifies this for single GPU training.
"""

# import evaluate
import msamp
import torch
from fp8_utils import evaluate_model, get_training_utilities
import gc
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import FP8RecipeKwargs, get_grad_scaler, set_seed, MSAMPRecipeKwargs


MODEL_NAME = "Qwen/Qwen2.5-3B"

def print_memory_usage():
    """
    Print the current memory usage of the GPU.
    """
    if torch.cuda.is_available():
        print(f"Current memory usage: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        print(f"Max memory usage: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
    else:
        print("CUDA is not available.")
def train_integration(
    mixed_precision="fp8",  # "fp8" or "fp16",
    opt_level="O2"):
    if mixed_precision == "fp8":
        kwargs_handlers = [MSAMPRecipeKwargs(opt_level=opt_level)]

        accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=kwargs_handlers)
    elif mixed_precision == "fp16":
        accelerator = Accelerator()
    # set_seed(42)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = get_training_utilities(
        MODEL_NAME, accelerator=accelerator, batch_size=1
    )

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    # base_model_results = evaluate_model(model, eval_dataloader, METRIC)
    model.train()
    step = 0
    accumulation_steps = 16  # Gradient accumulation steps
    for batch in tqdm(train_dataloader):
    # Clear cache before each batch
        torch.cuda.empty_cache()
        
        with accelerator.accumulate(model):  # Use accelerator's accumulation context
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps  # Scale loss by accumulation steps
            accelerator.backward(loss)
            
            # Only step when accumulation is complete
            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
        
        step += 1
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
            print_memory_usage()
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # for opt_level in ["O1", "O2"]:
        # baseline_not_trained, baseline_trained = train_baseline(opt_level)
    accelerator_not_trained, accelerator_trained = train_integration(
        mixed_precision="fp8",
        opt_level="O2")
