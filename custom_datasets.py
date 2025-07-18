from datasets import load_dataset
import torch


def load_and_prepare_dataset(
    dataset_name: str = "nvidia/OpenMathInstruct-2",
    split: str = "train",
    tokenizer=None,
    train_batch_size: int = 8,
    val_batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load and prepare the dataset for training.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to use (e.g., 'train', 'test').

    Returns:
        tuple: A tuple containing the training and validation DataLoaders.
    """

    dataset = load_dataset(dataset_name, split=split)

    # check key existence
    if (
        "problem" in dataset.column_names
        and "generated_solution" in dataset.column_names
    ):
        # create a new column 'text' if it doesn't exist
        if "text" not in dataset.column_names:
            def format_conversation(example):
                # Create a simple conversation format
                conversation = [
                    {"role": "user", "content": example["problem"]},
                    {"role": "assistant", "content": example["generated_solution"]}
                ]
                try:
                    text = tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=False
                    )
                except:
                    # Fallback to simple concatenation if chat template fails
                    text = f"Problem: {example['problem']}\nSolution: {example['generated_solution']}"
                return {"text": text}
            
            dataset = dataset.map(
                format_conversation,
                batched=False,
                num_proc=4,
            )
    elif "text" not in dataset.column_names:
        raise ValueError(
            "Dataset must contain either 'text' column or both 'problem' and 'generated_solution' columns."
        )
    
    # split the dataset into training and validation sets
    train_size = int(0.9 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    # create DataLoaders for training and validation sets
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader

def collate_fn(batch):
    """
    Custom collate function to handle the batch data.

    Args:
        batch (list): A list of samples from the dataset.

    Returns:
        dict: A dictionary containing the batched data.
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }