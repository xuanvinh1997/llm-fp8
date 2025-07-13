# train_fp8.py
"""
FP8 training loop for DeepSeek-style MoE/MLA Transformer
-------------------------------------------------------
Launch with:
  torchrun --standalone --nnodes 1 --nproc_per_node 8 \
      train_fp8.py \
      --config config_16B.json \
      --dataset /path/to/pile_tokens.bin \
      --ckpt-out /checkpoints/fp8_run
"""
import os, json, math, time, argparse, itertools, functools, random, pathlib
from pathlib import Path

import torch, torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from safetensors.torch import save_file

# --- project code -----------------------------------------------------------
from model import Transformer, ModelArgs          # :contentReference[oaicite:0]{index=0}
import kernel                                     # brings in act_quant, fp8_gemm …

# ---------- tiny streaming dataset ------------------------------------------
class PackedDataset(IterableDataset):
    """Streams an mmap'ed uint16/32 file of token-ids and yields fixed-length blocks."""
    def __init__(self, bin_path: str, seq_len: int, micro_bsz: int, seed=0):
        super().__init__()
        self.seq_len, self.micro_bsz = seq_len, micro_bsz
        arr = torch.fromfile(bin_path, dtype=torch.int32)   # assume 32-bit ids
        self.data = arr
        self.rng  = random.Random(seed)

    def __iter__(self):
        while True:                               # infinite stream
            idx = self.rng.randrange(0, len(self.data)-self.seq_len-1)
            block = self.data[idx: idx+self.seq_len+1]
            x = block[:-1].view(self.micro_bsz, -1).contiguous()
            y = block[1: ].view(self.micro_bsz, -1).contiguous()
            yield x, y

# ---------- training utilities ----------------------------------------------
def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK"      , 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return world_size, rank, local_rank

def save_ckpt(model, opt, step, out_dir, rank):
    if rank != 0: return
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fn_model = out/f"model{step:06d}-mp1.safetensors"
    save_file(model.state_dict(), str(fn_model))
    torch.save({"step":step, "opt":opt.state_dict()}, out/"optim.pt")
    print(f"[ckpt] step {step} saved to {fn_model}")

# ------------------- main ----------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   required=True)
    ap.add_argument("--dataset",  required=True)
    ap.add_argument("--tokenizer",default="deepseek-ai/deepseek-llm-7b-base")
    ap.add_argument("--micro-bsz",type=int, default=8)
    ap.add_argument("--global-bsz",type=int, default=1024)
    ap.add_argument("--seq-len",  type=int, default=2048)
    ap.add_argument("--lr",       type=float, default=2.0e-4)
    ap.add_argument("--warmup",   type=int, default=200)
    ap.add_argument("--steps",    type=int, default=20_000)
    ap.add_argument("--ckpt-out", required=True)
    args = ap.parse_args()

    world, rank, local_rank = setup_ddp()
    torch.manual_seed(1337 + rank)

    # ---- build model --------------------------------------------------------
    with open(args.config) as f:
        cfg = ModelArgs(**json.load(f))
    cfg.dtype    = "fp8"                 # <-- critical
    kernel.gemm_impl = "fp8"             # enables FP8 matmul in model.py :contentReference[oaicite:1]{index=1}
    model = Transformer(cfg).cuda()
    model.train()

    # ---- optimizer (FP32 master weights) -----------------------------------
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    # ---- data --------------------------------------------------------------
    ds  = PackedDataset(args.dataset, args.seq_len, args.micro_bsz, seed=rank)
    loader = DataLoader(ds, batch_size=None, num_workers=0, pin_memory=True)

    # ---- helpers -----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    grad_acc  = args.global_bsz // (args.micro_bsz*world)
    scaler    = torch.cuda.amp.GradScaler(enabled=False)    # FP8 path uses custom scaling

    # ---- training loop -----------------------------------------------------
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    step, tokens = 0, 0
    t0 = time.time()

    for x, y in itertools.islice(loader, args.steps*grad_acc):
        x = x.cuda(non_blocking=True);  y = y.cuda(non_blocking=True)
        logits = model(x, 0)            # model returns (B, S, V)
        loss   = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1)) / grad_acc
        loss.backward()
        if (step+1) % grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
            step += 1

            if rank==0 and step % 50 == 0:
                dtok = args.global_bsz*args.seq_len*50
                tokens += dtok
                speed = dtok/(time.time()-t0); t0=time.time()
                print(f"step {step:>6d}  loss {loss.item()*grad_acc:.4f}  {speed:,.0f} tok/s")

            if step in {500,1000,5000} or step % 2000 == 0:
                save_ckpt(model, opt, step, args.ckpt_out, rank)

        if step >= args.steps: break

    if world > 1: dist.destroy_process_group()

if __name__ == "__main__":
    main()
