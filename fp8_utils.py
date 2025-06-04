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
import torch


def get_dataloaders(model_name: str, batch_size: int = 16):
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    tokenized_datasets = load_dataset("nvidia/OpenMathInstruct-2", split="train")
    # select 1% of the dataset for testing
    tokenized_datasets = tokenized_datasets.select(range(0, int(len(tokenized_datasets) * 0.01)))
    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        # apply chat template to the examples
        inputs = []
        for problem, generated_solution in zip(
            examples["problem"], examples["generated_solution"]
        ):
            # Use the chat template
            inputs.append(
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": problem,
                        },
                        {
                            "role": "assistant",
                            "content": generated_solution,
                        },
                    ],
                    add_generation_prompt=False,
                    tokenize=False
            ))
        # Remove return_tensors="pt" to keep as lists for later padding
        outputs = tokenizer(inputs, padding="max_length", max_length=2048, truncation=True)
        return outputs

    # # Apply the method we just defined to all the examples in all the splits of the dataset
    # # starting with the main process first:
    tokenized_datasets = tokenized_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["problem_source", "expected_answer"],
        desc="Tokenizing dataset",
        num_proc=32,  # Adjust based on your CPU cores
    )
    # add labels to the dataset
    tokenized_datasets = tokenized_datasets.map(
        lambda x: {"labels": x["input_ids"]},
        remove_columns=["generated_solution"],
        desc="Adding labels",
    )
    # # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # # transformers library
    # tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        """
        Custom collate function to handle padding and conversion to tensors.
        """
        # Convert input_ids and attention_mask to tensors
        input_ids = torch.tensor([ex["input_ids"] for ex in examples], dtype=torch.int64)
        attention_mask = torch.tensor(
            [ex["attention_mask"] for ex in examples], dtype=torch.int64
        )
        labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.int64)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # split the dataset into train and validation sets
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2, seed=42)
    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["test"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=True,
    )

    return train_dataloader, eval_dataloader


def get_training_utilities(model_name: str, batch_size: int = 16, accelerator=None):
    """
    Returns a tuple of:
        - Model
        - Optimizer
        - Train dataloader (prepared)
        - Eval dataloader (prepared)
        - LR Scheduler
    Suitable for training on the MRPC dataset
    """
    from torch.optim import AdamW
    from transformers import (
        AutoModelForCausalLM,
        get_linear_schedule_with_warmup,
    )

    from accelerate import Accelerator

    if accelerator is None:
        accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(model_name)
    train_dataloader, eval_dataloader = get_dataloaders(model_name, batch_size)
    optimizer = AdamW(model.parameters(), lr=0.0001)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * 2,
    )
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )
    return model, optimizer, train_dataloader, eval_dataloader, lr_scheduler


def get_named_parameters(model):
    """
    Same thing as `Accelerator.get_named_parameters` Returns a list of the named parameters of the model (extracted
    from parallel)
    """
    from accelerate.utils import extract_model_from_parallel

    model = extract_model_from_parallel(model)
    return {n: p for n, p in model.named_parameters()}


def evaluate_model(model, dataloader, metric, accelerator=None):
    "Turns model to .eval(), runs dataloader, calculates metric, then turns eval back on"
    model.eval()
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            # W/ MS-AMP, we need to cast while evaluating
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        references = batch["labels"]
        if accelerator is not None and accelerator.num_processes > 1:
            predictions, references = accelerator.gather_for_metrics(
                (predictions, references)
            )
        metric.add_batch(predictions=predictions, references=references)
    return metric.compute()
