import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize the input examples."""
    texts = []
    for problem, solution in zip(examples["problem"], examples["generated_solution"]):
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(full_text)

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].squeeze()
    labels = input_ids.clone()
    # ignore compute loss for response tokens
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids),
    }


def collate_fn(batch):
    """Custom collate function for dynamic padding."""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    labels = []
    attention_masks = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])

        input_ids.append(F.pad(torch.tensor(item["input_ids"]), (0, pad_len), value=0))
        labels.append(F.pad(torch.tensor(item["labels"]), (0, pad_len), value=-100))
        attention_masks.append(F.pad(torch.tensor(item["attention_mask"]), (0, pad_len), value=0))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_masks),
    }
