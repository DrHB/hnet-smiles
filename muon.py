"""Muon (Jordan et al. 2024) for the Transformer block matrices, with Moonlight's update scale (Liu et al. 2025, arXiv
2502.16982): the orthogonalised update times 0.2 sqrt(max(rows, cols)) has AdamW's update RMS, so AdamW's learning rate
and decoupled weight decay carry over unchanged."""
import torch


def ns5(G, steps=5):
    """Newton-Schulz quintic on G [k, m, n] (bf16): each matrix -> about U V^T of its SVD (singular values in ~[0.7, 1.2])."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tr = X.shape[-2] > X.shape[-1]
    if tr:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if tr else X


class Muon(torch.optim.Optimizer):
    """Nesterov momentum, then ns5. Group key "split": dim 0 of each weight is that many stacked matrices (fused q/k/v/gate,
    SwiGLU gate/up), each orthogonalised on its own. Weights of one shape in a group go through ns5 as one batch."""

    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, split=1))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            by = {}
            for p in g["params"]:
                if p.grad is not None:
                    by.setdefault(p.shape, []).append(p)
            for (r, c), ps in by.items():
                us = []
                for p in ps:
                    buf = self.state[p].setdefault("buf", torch.zeros_like(p))
                    buf.lerp_(p.grad, 1 - g["momentum"])
                    us.append(p.grad.lerp(buf, g["momentum"]))
                k = g["split"]
                O = ns5(torch.stack(us).view(len(ps) * k, r // k, c)).view(len(ps), r, c)
                s = 0.2 * max(r // k, c) ** 0.5
                for p, o in zip(ps, O):
                    p.mul_(1 - g["lr"] * g["weight_decay"]).add_(o.to(p.dtype), alpha=-g["lr"] * s)


class Opts:
    """Several optimizers stepped as one; param_groups is all of theirs, state_dict the list of theirs."""

    def __init__(self, *opts):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return [o.state_dict() for o in self.opts]

    def load_state_dict(self, s):
        for o, x in zip(self.opts, s, strict=True):
            o.load_state_dict(x)
