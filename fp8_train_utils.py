import torch
from kernel import FP8_E4M3, FP8_E5M2, act_quant


def attach_grad_hooks(model):
    def _make_hook(param):
        fmt = param.data.dtype
        def _hook(grad):
            # view(-1) to match Triton's 1-D kernel
            g_q, g_s = act_quant(grad.contiguous().view(-1),
                                           block_size=model.block_size, dtype=fmt)
            # stash scale so optimiser can see it
            param.grad_scale = g_s
            return g_q.view_as(grad)
        return _hook

    for p in model.parameters():
        if p.requires_grad and p.data.dtype in (FP8_E4M3, FP8_E5M2):
            p.register_hook(_make_hook(p))

# 2️⃣  create a FP32 master copy (once, at startup)
def make_master_params(model):
    for p in model.parameters():
        if p.requires_grad and p.data.dtype in (FP8_E4M3, FP8_E5M2):
            p.master = torch.nn.Parameter(p.data.to(torch.float32),
                                          requires_grad=False)
                
def link_master_to_param(model):
    for p in model.parameters():
        if hasattr(p, "master"):
            p.master.fp8_param_ref = p