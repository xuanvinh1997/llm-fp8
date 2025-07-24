# ---------- optim_fp8.py  (ultra-light wrapper around Adam) -------
import torch

from kernel import act_quant


class AdamFP8(torch.optim.Adam):
    """Adam that works on *master* FP32 weights yet stores FP8 copies in model."""

    def __init__(self, params, lr=1e-3, block_size=128, **kw):
        # feed the FP32 masters to Adam
        super().__init__(
            [p.master for p in params if hasattr(p, "master")], lr=lr, **kw
        )
        self.block_size = block_size

    @torch.no_grad()
    def step(self, *a, **k):
        super().step(*a, **k)  # Adam on FP32 masters
        # after update → re-quantise masters back into model params
        for group in self.param_groups:
            for m in group["params"]:  # 'm' is the master
                p = m.fp8_param_ref  # link to original
                max_val = 448.0 if p.data.dtype == torch.float8_e4m3fn else 57344.0
                m.data.clamp_(-max_val, max_val)        # keep FP32 master in-range
                q_flat, s = act_quant(
                    m.data,                     # keep original 2-D shape
                    block_size=self.block_size,
                    dtype=p.data.dtype,
                )
                p.data.copy_(q_flat)            # same (out_f, in_f) layout
                new_scale = s.mean()
                if hasattr(p, "weight_scale"):
                    # copy scale to match the original weight's scale
                    p.weight_scale.copy_(new_scale)  # one scale value (out_f=128 ⇒ 1 block)
                else:
                    # copy scale to match the original layernorm's scale
                    p.weight_scale = new_scale
