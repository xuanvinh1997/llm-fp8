# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
#
# See LICENSE for license information.

import time
import sys
import IPython

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AutoConfig,
)
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils.dataclasses import FP8RecipeKwargs


class HyperParameters:
    def __init__(self):
        self.mixed_precision = "bf16"

        # Model & data
        self.model_name = "meta-llama/Llama-2-7b-hf"
        self.dataset_name = "timdettmers/openassistant-guanaco"
        self.dataset_text_field = "text"
        self.max_seq_length = 256

        # Training hyperparams
        self.learning_rate = 1.41e-5
        self.batch_size = 8
        self.gradient_accumulation_steps = 1
        self.num_epochs = 3
        self.num_warmup_steps = 100

        # Evaluation settings
        self.eval_split = "validation"
        self.eval_batch_size = self.batch_size

        # Logging & saving
        self.log_dir = "./runs"
        self.output_dir = "./saved_model"

        # This will be set by snapshot_download
        self.weights_cache_dir = ""

        self.number_samples = None  # Set to None to use the full dataset


hyperparams = HyperParameters()


def get_dataloader(
    accelerator: Accelerator, hp: HyperParameters
):
    dataset = load_dataset(hp.dataset_name, split="train")
    tokenizer = AutoTokenizer.from_pretrained(hp.model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    chat_template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful assistant that solves math problems step by step. "
        "Please reason step by step, and put your final answer within \\boxed{{}}."
        "\n<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n{problem}\n<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n{solution}<|eot_id|>"
    )

    with accelerator.main_process_first():

        def apply_template(ex):
            text = chat_template.format(
                problem=ex["problem"],
                solution=ex["generated_solution"],
            )
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=hp.max_seq_length,
            )
            return {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
            }

        dataset = dataset.map(
            apply_template,
            remove_columns=dataset.column_names,
            num_proc=12,
        )
    # split into train and eval sets
    if hyperparams.number_samples is not None:
        dataset = dataset.select(range(hyperparams.number_samples))

    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=16,
    )
    return DataLoader(
        train_dataset,
        batch_size=hp.batch_size,
        collate_fn=collator,
        drop_last=True,
    ), DataLoader(
        eval_dataset,
        batch_size=hp.eval_batch_size,
        collate_fn=collator,
    )


def ensure_model_is_downloaded(hp: HyperParameters):
    assert hp.model_name in [
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-2-7b-hf",
    ], "Unsupported model!"

    from huggingface_hub import login, snapshot_download

    try:
        login(hp.hf_access_token)
    except Exception as e:
        print(f"HF login issue: {e}")

    hp.weights_cache_dir = snapshot_download(
        repo_id=hp.model_name,
        cache_dir=(hp.weights_cache_dir or None),
    )
    print(f"Model cached at: {hp.weights_cache_dir}")


def init_model(hp: HyperParameters, use_te: bool = False):
    ensure_model_is_downloaded(hp)
    config = AutoConfig.from_pretrained(hp.weights_cache_dir)
    config._attn_implementation = "flash_attention_2"
    if use_te:
        from te_llama import TELlamaForCausalLM

        model = TELlamaForCausalLM.from_pretrained_local(
            hp.weights_cache_dir,
            config=config,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hp.weights_cache_dir,
            config=config,
            torch_dtype=torch.bfloat16,
        )
    model = model.cuda()
    model.config.use_cache = False
    return model


def wrap_with_accelerator(model, hp: HyperParameters):
    fp8_handler = (
        [FP8RecipeKwargs(backend="te")] if hp.mixed_precision == "fp8" else None
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        mixed_precision=hp.mixed_precision,
        kwargs_handlers=fp8_handler,
    )

    train_loader, eval_loader = get_dataloader(accelerator, hp)

    num_batches = len(train_loader)
    total_steps = num_batches * hp.num_epochs // hp.gradient_accumulation_steps

    optimizer = AdamW(
        params=model.parameters(),
        lr=hp.learning_rate,
        fused=True,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=hp.num_warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader, scheduler, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, scheduler, eval_loader
    )
    return accelerator, model, optimizer, train_loader, scheduler, eval_loader


def finetune_model(
    model,
    hp: HyperParameters,
    accelerator,
    train_loader,
    eval_loader,
    optimizer,
    scheduler,
    writer: SummaryWriter,
):
    model.train()
    step_count = 0
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_time.record()

    for epoch in range(1, hp.num_epochs + 1):
        accelerator.print(f"Epoch {epoch}/{hp.num_epochs}")
        epoch_loss = 0.0
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                epoch_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                # log training loss every 50 steps
                if step_count % 50 == 0:
                    writer.add_scalar("Train/Loss", loss.item(), step_count)

        # evaluation at end of epoch
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                outputs = model(**batch)
                eval_loss += outputs.loss.detach().float()
        avg_eval = eval_loss / len(eval_loader)
        writer.add_scalar("Eval/Loss", avg_eval, epoch)
        accelerator.print(f" Eval loss: {avg_eval:.4f}")
        model.train()

    torch.cuda.synchronize()
    end_time.record()
    accelerator.end_training()

    total_steps = step_count
    ms_per_step = start_time.elapsed_time(end_time) / total_steps
    writer.add_scalar("Train/TimePerStep_ms", ms_per_step, 0)
    writer.flush()
    writer.close()

    accelerator.print(f"Training done: {total_steps} steps, {ms_per_step:.0f} ms/step")


def save_model(model, hp: HyperParameters, accelerator):
    model_to_save = accelerator.unwrap_model(model)
    model_to_save.save_pretrained(hp.output_dir)
    print(f"Model saved to: {hp.output_dir}")


def restart_jupyter_notebook():
    IPython.Application.instance().kernel.do_shutdown(True)
    if torch.cuda.memory_allocated() != 0:
        import warnings

        warnings.warn("CUDA memory not freed; retrying via HTML reload...")
        from IPython.core.display import HTML

        HTML("<script>Jupyter.notebook.kernel.restart()</script>")
        if torch.cuda.memory_allocated() != 0:
            print("Please manually restart the Jupyter kernel!")

    if not sys.warnoptions:
        import warnings

        warnings.simplefilter("ignore")
        torch.set_warn_always(False)


# if __name__ == "__main__":
#     model = init_model(hyperparams, use_te=False)
#     accelerator, model, opt, train_loader, sched, eval_loader = wrap_with_accelerator(
#         model, hyperparams
#     )
#     writer = SummaryWriter(log_dir=hyperparams.log_dir)
#     finetune_model(
#         model, hyperparams, accelerator,
#         train_loader, eval_loader, opt, sched, writer
#     )
#     save_model(model, hyperparams, accelerator)
