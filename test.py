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

from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import FP8RecipeKwargs, get_grad_scaler, set_seed, MSAMPRecipeKwargs


MODEL_NAME = "Qwen/Qwen2.5-1.5B"


def train_baseline(opt_level="O2"):
    # set_seed(42)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = get_training_utilities(MODEL_NAME, batch_size=1)

    model, optimizer = msamp.initialize(model, optimizer, opt_level=opt_level)
    model.to("cuda")

    # base_model_results = evaluate_model(model, eval_dataloader, METRIC)
    model.train()
    scaler = get_grad_scaler()

    for batch in train_dataloader:
        batch = batch.to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
        loss = outputs.loss
        loss = scaler.scale(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    # trained_model_results = evaluate_model(model, eval_dataloader, METRIC)

    # assert trained_model_results["accuracy"] > base_model_results["accuracy"], (
    #     f"Accuracy should be higher for the trained model: {trained_model_results['accuracy']} > {base_model_results['accuracy']}"
    # )
    # assert trained_model_results["f1"] > base_model_results["f1"], (
    #     f"F1 score should be higher for the trained model: {trained_model_results['f1']} > {base_model_results['f1']}"
    # )

    # return base_model_results, trained_model_results


def train_integration(opt_level="O2"):
    kwargs_handlers = [MSAMPRecipeKwargs(opt_level=opt_level)]

    accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=kwargs_handlers)

    # set_seed(42)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = get_training_utilities(
        MODEL_NAME, accelerator=accelerator, batch_size=1
    )

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    # base_model_results = evaluate_model(model, eval_dataloader, METRIC)
    model.train()
    step = 0
    accumulation_steps = 8  # Gradient accumulation steps
    for batch in train_dataloader:
        print(f"Processing batch {step + 1}")
        print(f"Batch size: {len(batch['input_ids'])}")
        print("Len input_ids:", batch['input_ids'].shape)
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        step += 1
        
        lr_scheduler.step()


if __name__ == "__main__":
    for opt_level in ["O1", "O2"]:
        # baseline_not_trained, baseline_trained = train_baseline(opt_level)
        accelerator_not_trained, accelerator_trained = train_integration(opt_level)
