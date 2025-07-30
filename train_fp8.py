# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
#
# See LICENSE for license information.

import time
import sys
import IPython
from click import Tuple
import wandb
import tqdm

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

import subprocess

def get_gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE, text=True
    )
    mem_list = result.stdout.strip().split('\n')
    # Returns a list of memory usage for each GPU
    return [int(x) for x in mem_list]
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
        self.use_wandb = False
        self.wandb_project = "llm-fp8"
        self.wandb_run_name = None

        # This will be set by snapshot_download
        self.weights_cache_dir = ""
        self.num_proc = 48
        self.num_of_samples = None


hyperparams = HyperParameters()


def get_dataloader(accelerator: Accelerator, hp: HyperParameters) -> tuple[DataLoader, DataLoader] | None:
    dataset = load_dataset(hp.dataset_name, split="train")
    if hp.num_of_samples is not None:
        dataset = dataset.select(range(hp.num_of_samples))
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
                problem=ex["problem"], solution=ex["generated_solution"],
            )
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=hp.max_seq_length,
            )
            return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}

        dataset = dataset.map(
            apply_template,
            remove_columns=dataset.column_names,
            num_proc=hp.num_proc,
        )
    # split into train and validation sets
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=16,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=hp.batch_size,
        collate_fn=collator,
        shuffle=True,)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=hp.eval_batch_size,
        collate_fn=collator,
    )
    return train_loader, eval_loader


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
    total_steps = (
        num_batches * hp.num_epochs // hp.gradient_accumulation_steps
    )

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
    model, hp: HyperParameters, accelerator,
    train_loader, eval_loader, optimizer, scheduler,
    writer: SummaryWriter,
    wandb_run=None
):
    model.train()
    step_count = 0
    start_time = torch.cuda.Event(enable_timing=True)
    end_time   = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_time.record()

    total_steps = len(train_loader) * hp.num_epochs // hp.gradient_accumulation_steps

    # Wrap the training steps with tqdm
    with tqdm.tqdm(total=total_steps, desc="Training", unit="step") as pbar:
        for epoch in range(1, hp.num_epochs + 1):
            accelerator.print(f"Epoch {epoch}/{hp.num_epochs}")
            epoch_loss = 0.0
            for batch in train_loader:
                step_start = time.perf_counter()
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    if not torch.isfinite(loss):
                        accelerator.print("Non-finite loss detected, stopping training.")
                        return
                    epoch_loss += loss.detach().float()
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step_count += 1
                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item())
                step_duration = time.perf_counter() - step_start
                mem_mb = get_gpu_memory()[0] # Convert to MB

                # log training metrics every 50 steps
                if step_count % 50 == 0:
                    writer.add_scalar("Train/Loss", loss.item(), step_count)
                    writer.add_scalar("Train/StepTime_s", step_duration, step_count)
                    writer.add_scalar("Train/GPU_Memory_MB", mem_mb, step_count)
                    if wandb_run is not None:
                        wandb_run.log({
                            "Train/Loss": loss.item(),
                            "Train/StepTime_s": step_duration,
                            "Train/GPU_Memory_MB": mem_mb,
                            "Train/Progress": step_count / total_steps,
                        }, step=step_count)
                    accelerator.print(
                        f"Step {step_count}: "
                        f"loss={loss.item():.4f}, "
                        f"step_time={step_duration:.2f}s, "
                        f"gpu_mem={mem_mb:.0f}MB"
                    )
            # evaluation at end of epoch
            model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for batch in eval_loader:
                    outputs = model(**batch)
                    eval_loss += outputs.loss.detach().float()
            avg_eval = eval_loss / len(eval_loader)
            writer.add_scalar("Eval/Loss", avg_eval, epoch)
            if wandb_run is not None:
                wandb_run.log({"Eval/Loss": avg_eval}, step=step_count)
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
    if wandb_run is not None:
        wandb_run.log({"Train/TimePerStep_ms": ms_per_step})
        wandb_run.finish()

    accelerator.print(
        f"Training done: {total_steps} steps, {ms_per_step:.0f} ms/step"
    )


def save_model(model, hp: HyperParameters, accelerator):
    model_to_save = accelerator.unwrap_model(model)
    model_to_save.save_pretrained(hp.output_dir)
    print(f"Model saved to: {hp.output_dir}")


def restart_jupyter_notebook():
    IPython.Application.instance().kernel.do_shutdown(True)
    if torch.cuda.memory_allocated() != 0:
        import warnings
        warnings.warn(
            "CUDA memory not freed; retrying via HTML reload..."
        )
        from IPython.core.display import HTML
        HTML("<script>Jupyter.notebook.kernel.restart()</script>")
        if torch.cuda.memory_allocated() != 0:
            print("Please manually restart the Jupyter kernel!")

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
        torch.set_warn_always(False)


# train.py: command-line entry point for fine-tuning
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Llama with TE and FP8 using Accelerate")
    # model & data
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model identifier (e.g. meta-llama/Llama-3.2-1B)")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset identifier (e.g. nvidia/OpenMathInstruct-2)")
    parser.add_argument("--dataset_text_field", type=str, default="text",
                        help="Field name in dataset for raw text")
    parser.add_argument("--use_te", action="store_true",
                        help="Enable Transformer Engine (TELlama) model instantiation")

    # training hyperparams
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["bf16", "fp8"], help="Mixed precision mode")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device batch size for training")
    parser.add_argument("--eval_batch_size", type=int,
                        help="Per-device batch size for evaluation (defaults to train batch_size)")

    # --max_length
    parser.add_argument("--max_seq_length", type=int, default=256,
                        help="Maximum sequence length for input text")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1.41e-5,
                        help="Initial learning rate")
    parser.add_argument("--num_warmup_steps", type=int, default=100,
                        help="Number of warmup steps for scheduler")
    parser.add_argument("--eval_split", type=str, default="validation",
                        help="Dataset split to use for evaluation")
    
    # num of samples
    parser.add_argument("--num_of_samples", type=int, default=None,
                        help="Number of samples to use from the dataset (for debugging)")

    # logging & saving
    parser.add_argument("--log_dir", type=str, default="./runs",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--output_dir", type=str, default="./saved_model",
                        help="Where to save the fine-tuned model")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="llm-fp8",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name")

    # weights cache
    parser.add_argument("--weights_cache_dir", type=str, default="",
                        help="Cache directory for model weights snapshot_download")

    args = parser.parse_args()

    # populate hyperparameters
    hp = HyperParameters()
    hp.model_name = args.model_name
    hp.dataset_name = args.dataset_name
    hp.dataset_text_field = args.dataset_text_field
    hp.mixed_precision = args.mixed_precision
    hp.batch_size = args.batch_size
    hp.eval_batch_size = args.eval_batch_size or args.batch_size
    hp.gradient_accumulation_steps = args.gradient_accumulation_steps
    hp.num_epochs = args.num_epochs
    hp.learning_rate = args.learning_rate
    hp.num_warmup_steps = args.num_warmup_steps
    hp.eval_split = args.eval_split
    hp.log_dir = args.log_dir
    hp.output_dir = args.output_dir
    hp.weights_cache_dir = args.weights_cache_dir
    hp.max_seq_length = args.max_seq_length
    hp.num_of_samples = args.num_of_samples
    hp.use_wandb = args.use_wandb
    hp.wandb_project = args.wandb_project
    hp.wandb_run_name = args.wandb_run_name

    # initialize model and accelerator
    model = init_model(hp, use_te=args.use_te)
    accelerator, model, optimizer, train_loader, scheduler, eval_loader = \
        wrap_with_accelerator(model, hp)

    # tensorboard writer
    writer = SummaryWriter(log_dir=hp.log_dir)
    wandb_run = None
    if hp.use_wandb:
        wandb_run = wandb.init(project=hp.wandb_project,
                               name=hp.wandb_run_name,
                               dir=hp.log_dir,
                               config=vars(hp))

    # fine-tune
    finetune_model(
        model,
        hp,
        accelerator,
        train_loader,
        eval_loader,
        optimizer,
        scheduler,
        writer,
        wandb_run,
    )

    # save final model
    save_model(model, hp, accelerator)

    print("Training complete.")
