"""Train the byte H-Net on a corpus directory (prep.py) with one of four objectives.

next_byte         causal stacks, cross-entropy on the next byte
fingerprint       bidirectional, clean input, BCE on the 2,048 Morgan bits
descriptors       bidirectional, clean input, MSE on 185 descriptor z-scores
masked_fragments  bidirectional, cross-entropy on masked bytes (masking.py)

Loss = objective + alpha * (ratio loss of both routers). Writes <out>/model.safetensors, config.json and metrics.jsonl,
and ckpt.pt at every evaluation (--resume 1 continues from it).
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from data import Corpus, loader, plan, unpack_fp
from evaluate import objective_metrics
from masking import apply_mask
from model import EOS, MASK, HNet, V, ratio_loss, save
from muon import Muon, Opts

FP_ON = 62.0  # mean on-bits per molecule: the fingerprint loss is summed over bits / FP_ON


def targets(x, n):
    """Next symbol at each position: bytes, then EOS at position n, -100 beyond."""
    y = torch.full_like(x, -100)
    y[:, :-1] = x[:, 1:]
    y[torch.arange(len(n), device=x.device), n] = EOS
    y[torch.arange(x.shape[1], device=x.device)[None] > n[:, None]] = -100
    return y


def optimizer(model, a):
    """AdamW (0.9, 0.95), weight decay 0.1 on matrices outside the embedding, routers and residual projections.
    --opt muon: Muon for the Transformer block matrices. Learning-rate multipliers 2 / 1.5 / 1 by stage."""
    mults, groups, mgroups = (a.lr_outer, a.lr_mid, 1.0), {}, {}
    for name, p in model.named_parameters():
        mult = mults[model.stage_of(name)]
        if a.opt == "muon" and ".blocks." in name and p.ndim == 2:
            split = 3 if name.endswith("qkv.weight") else 2 if name.endswith("fc1.weight") else 1
            mgroups.setdefault((split, mult), []).append(p)
            continue
        dec = p.ndim == 2 and not name.startswith(("emb", "q.", "k.", "res.", "q1.", "k1.", "res1."))
        groups.setdefault((dec, mult), []).append(p)
    opt = torch.optim.AdamW([{"params": v, "weight_decay": 0.1 if k[0] else 0.0, "mult": k[1]} for k, v in groups.items()],
                            lr=a.lr, betas=(0.9, 0.95))
    if mgroups:
        opt = Opts(opt, Muon([{"params": v, "split": k[0], "mult": k[1]} for k, v in mgroups.items()], lr=a.lr, weight_decay=0.1))
    return opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="corpus directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--objective", required=True, choices=("next_byte", "fingerprint", "descriptors", "masked_fragments"))
    ap.add_argument("--widths", default="256,384,512")
    ap.add_argument("--layers", default="2,2,6")
    ap.add_argument("--readout", default="mean", choices=("mean", "cls"), help="fingerprint, descriptors: see model.Head")
    ap.add_argument("--opt", default="adamw", choices=("adamw", "muon"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--budget", type=int, default=131072, help="padded byte positions per step")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr_outer", type=float, default=2.0)
    ap.add_argument("--lr_mid", type=float, default=1.5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--N", type=float, default=3.0, help="target bytes per chunk")
    ap.add_argument("--N1", type=float, default=2.0, help="target chunks per group")
    ap.add_argument("--alpha", type=float, default=0.03)
    ap.add_argument("--mask", default="0.25,0.45,0,1", help="masked_fragments: rate lo, hi; fragment share lo, hi")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--resume", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    c = Corpus(a.data)
    obj, mask = a.objective, tuple(float(v) for v in a.mask.split(","))
    train_rows = np.setdiff1d(np.arange(len(c.nb)), c.held)
    held = np.random.default_rng(0).permutation(c.held)
    rows, bounds = [], []
    for e in range(a.epochs):  # fresh shuffle and batch order per epoch; one learning-rate schedule over all epochs
        r_, b_ = plan(train_rows, c.nb, a.budget, a.seed + e)
        bounds.append(b_ + sum(len(x) for x in rows)); rows.append(r_)
    rows, bounds = np.concatenate(rows), np.concatenate(bounds)
    steps = len(bounds)
    end = min(steps, a.max_steps or steps)
    out = {"fingerprint": 2048, "descriptors": c.desc.shape[1] if c.desc is not None else 0}.get(obj, 0)
    model = HNet(tuple(int(w) for w in a.widths.split(",")), tuple(int(v) for v in a.layers.split(",")),
                 causal=obj == "next_byte", out=out, readout=a.readout).to(dev)
    opt = optimizer(model, a)
    d, start = Path(a.out), 0
    d.mkdir(parents=True, exist_ok=True)
    if a.resume and (d / "ckpt.pt").exists():
        ck = torch.load(d / "ckpt.pt", map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"]
    if a.compile and dev == "cuda":
        for s in (model.enc, model.dec, model.enc1, model.dec1, model.main):
            s.compile(dynamic=True)
    log = open(d / "metrics.jsonl", "a")

    def emit(r):
        print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()

    emit({"args": vars(a), "params": sum(p.numel() for p in model.parameters()), "steps": steps, "train_rows": len(train_rows)})
    t0, acc, k = time.time(), torch.zeros(3, device=dev), 0
    for step, (x, n, *ex) in enumerate(loader(a.data, rows, bounds[:end], obj, mask, a.seed, a.workers, start), start):
        lr = a.lr * min(1, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, step / steps))))
        for g in opt.param_groups:
            g["lr"] = lr * g["mult"]
        x, n, ex = x.to(dev), n.to(dev), [t.to(dev) for t in ex]
        if obj == "masked_fragments":
            x, y = apply_mask(x, ex[0])
        with torch.autocast(dev, torch.bfloat16, enabled=dev == "cuda"):
            o, r = model(x, n)
        o = o.float()
        if obj == "fingerprint":
            loss = F.binary_cross_entropy_with_logits(o, unpack_fp(ex[0]), reduction="sum") / (len(n) * FP_ON)
        elif obj == "descriptors":
            y, m = ex[0].float() * (6 / 127), ex[0] != -128
            loss = (((o - y) ** 2) * m).sum() / m.sum().clamp(min=1)
        else:
            loss = F.cross_entropy(o.view(-1, V), (targets(x, n) if obj == "next_byte" else y).view(-1), ignore_index=-100)
        rl = ratio_loss(r["p"], r["b"], r["valid"] & (x != MASK), a.N) + ratio_loss(r["p1"], r["b1"], r["valid1"], a.N1)
        opt.zero_grad(set_to_none=True)
        (loss + a.alpha * rl).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        acc += torch.stack([loss.detach(), rl.detach(), (n.sum() / (r["b"] & r["valid"])[:, 1:].sum().clamp(min=1)).float()]); k += 1
        if (step + 1) % 50 == 0 or step + 1 == end:
            v = (acc / k).tolist()
            emit({"step": step + 1, "lr": lr, "loss": v[0], "ratio_loss": v[1], "bytes_per_chunk": v[2], "wall_s": round(time.time() - t0)})
            acc.zero_(); k = 0
        if (step + 1) % a.eval_every == 0 or step + 1 == end:
            model.eval()
            ev = held if step + 1 == steps else held[:a.n_eval]  # all held-out rows at the end of the schedule
            emit({"step": step + 1, "eval_rows": len(ev), **objective_metrics(model, c.smiles(ev), obj, a.budget,
                                                                              fp=c.fp, desc=c.desc, rows=ev, mask_rate=sum(mask[:2]) / 2)})
            model.train()
            save(model, d, objective=obj, params=sum(p.numel() for p in model.parameters()), steps=step + 1)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1}, d / "ckpt.pt")


if __name__ == "__main__":
    main()
