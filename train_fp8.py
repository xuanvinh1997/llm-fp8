# train.py
import os, math, argparse, time, json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import load_dataset
from transformers import AutoTokenizer

from model import ModelArgs, Transformer     # ← canvas file

# ---------- helpers ----------------------------------------------------------
def ddp_setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def save_ckpt(model, opt, scheduler, step, path):
    if dist.get_rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step
        }, path)

def load_ckpt(model, opt, scheduler, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"]

# ---------- training ---------------------------------------------------------
def main(cfg):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    ddp_setup(rank, world)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(cfg.dataset, "wikitext-103-raw-v1", split="train")
    ds = ds.shuffle()

    def tok_fn(ex):
        ids = tokenizer(ex["text"], return_tensors="pt",
                        max_length=cfg.seq_len, truncation=True, padding="max_length")
        return {"ids": ids.input_ids[0]}
    ds = ds.map(tok_fn, num_proc=16).with_format("torch")

    sampler = DistributedSampler(ds, shuffle=True)
    dl = DataLoader(ds, batch_size=cfg.micro_bsz,
                    sampler=sampler, num_workers=2, pin_memory=True)
    gemm_impl = cfg.dtype
    margs = ModelArgs(dtype=cfg.dtype,
                      vocab_size=len(tokenizer),
                      dim=cfg.model_dim,
                      inter_dim=cfg.inter_dim,
                      n_layers=cfg.layers,
                      n_heads=cfg.heads,
                      n_kv_heads=cfg.kv_heads,
                      max_seq_len=cfg.seq_len)

    model = Transformer(margs).cuda(rank)
    model = DDP(model, device_ids=[rank])

    opt = AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = CosineAnnealingLR(opt, cfg.total_steps)

    start_step = 0
    if cfg.resume and Path(cfg.resume).exists():
        start_step = load_ckpt(model, opt, scheduler, Path(cfg.resume))

    accum_steps = cfg.batch_size // cfg.micro_bsz
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == "bf16"))

    step, t0 = start_step, time.time()
    dl_iter = iter(dl)
    while step < cfg.total_steps:
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(accum_steps):
            try:
                batch = next(dl_iter)
            except StopIteration:
                sampler.set_epoch(step)        # shuffle next pass
                dl_iter = iter(dl)
                batch = next(dl_iter)

            ids = batch["ids"].cuda(rank, non_blocking=True)
            labels = ids.clone()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.dtype=="bf16")):
                logits = model(ids, start_pos=0)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=tokenizer.pad_token_id
                ) / accum_steps
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if step % cfg.log_every == 0 and rank == 0:
            dt = time.time() - t0
            print(f"step {step:6d} | loss {loss_accum:.4f} | {dt*1000/accum_steps:.1f} ms/iter")
            t0 = time.time()

        if step % cfg.ckpt_every == 0 and step != start_step:
            save_ckpt(model, opt, scheduler, step,
                      Path(cfg.ckpt_dir) / f"ckpt_{step:06d}.pt")

        step += 1

# ---------- CLI --------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    p.add_argument("--dataset",   default="wikitext", help="HF dataset name")
    p.add_argument("--seq_len",   type=int, default=512)
    p.add_argument("--dtype",     choices=["bf16", "fp8"], default="fp8")
    p.add_argument("--model_dim", type=int, default=2048)
    p.add_argument("--inter_dim", type=int, default=10944)
    p.add_argument("--layers",    type=int, default=27)
    p.add_argument("--heads",     type=int, default=16)
    p.add_argument("--kv_heads",  type=int, default=4)

    p.add_argument("--batch_size", type=int, default=256, help="global tokens per step")
    p.add_argument("--micro_bsz",  type=int, default=8,   help="per‑GPU batch")
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--total_steps",type=int, default=30_000)
    p.add_argument("--grad_clip",  type=float, default=1.0)

    p.add_argument("--log_every",  type=int, default=20)
    p.add_argument("--ckpt_every", type=int, default=2_000)
    p.add_argument("--ckpt_dir",   default="checkpoints")
    p.add_argument("--resume",     default="")  # path to .pt

    cfg = p.parse_args()
    main(cfg)
