"""Two-stage byte H-Net (arXiv 2507.07955) on right-padded [B, L] batches.

bytes -> encoder -> router 0 -> chunks -> encoder 1 -> router 1 -> groups -> main network, then back down twice:
EMA smoothing over the units, upsampling times a straight-through confidence, plus a residual from the encoder.
Router: p_t = (1 - cos(q h_{t-1}, k h_t)) / 2, p_0 = 1, boundary where p_t >= 0.5; each unit is the encoder state at
its first position. Stacks are causal (next-byte objective) or bidirectional with padding masked (the other three).
"""
import json
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

BOS, EOS, PAD, MASK, V = 256, 257, 258, 259, 260


def rope(L, dh, device):
    f = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=device).float() / dh))
    a = torch.outer(torch.arange(L, device=device).float(), f)
    return a.cos(), a.sin()


def rot(x, cs):
    c, s = cs
    x1, x2 = x.float().chunk(2, -1)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1).type_as(x)


class Block(nn.Module):
    """Pre-RMSNorm attention with RoPE, then SwiGLU."""

    def __init__(self, d, nh, hid):
        super().__init__()
        self.nh = nh
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv, self.out_proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.fc1, self.fc2 = nn.Linear(d, 2 * hid, bias=False), nn.Linear(hid, d, bias=False)

    def forward(self, x, cs, am=None):
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.nh, d // self.nh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rot(q, cs), rot(k, cs), v, attn_mask=am, is_causal=am is None)
        x = x + self.out_proj(a.transpose(1, 2).reshape(B, L, d))
        g, u = self.fc1(self.n2(x)).chunk(2, -1)
        return x + self.fc2(F.silu(g) * u)


class Stack(nn.Module):
    def __init__(self, n, d, nh, hid):
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, nh, hid) for _ in range(n))
        self.norm = nn.RMSNorm(d)
        self.dh = d // nh

    def forward(self, x, valid=None):
        """valid [B, L] bool: bidirectional over the valid positions; None: causal."""
        cs = rope(x.shape[1], self.dh, x.device)
        am = None if valid is None else valid[:, None, None, :]
        for b in self.blocks:
            x = b(x, cs, am)
        return self.norm(x)


def fp32():
    return torch.autocast("cuda", enabled=False)


def route(h, q, k, valid, vis=None):
    """Router on h [B, L, d]. Returns fp32 h, p, boundaries b, the unit index of each position, the unit vectors
    z [B, M, d] (right-padded) and their probabilities P [B, M].
    vis (masked input): only these positions can start a unit, each compared with the previous visible position."""
    with fp32():
        hf = h.float()
        if vis is None:
            cos = (F.normalize(q(hf[:, :-1]), dim=-1) * F.normalize(k(hf[:, 1:]), dim=-1)).sum(-1)
            p = F.pad(((1 - cos) / 2).clamp(0, 1), (1, 0), value=1.0)
        else:
            pos = torch.arange(h.shape[1], device=h.device)[None]
            prev = F.pad(torch.where(vis, pos, 0).cummax(1).values[:, :-1], (1, 0))
            hp = hf.gather(1, prev[..., None].expand(-1, -1, hf.shape[-1]))
            cos = (F.normalize(q(hp), dim=-1) * F.normalize(k(hf), dim=-1)).sum(-1)
            p = torch.where(vis, ((1 - cos) / 2).clamp(0, 1), 0.0)
            p[:, 0] = 1.0
        b = (p >= 0.5) & valid
        ci = b.cumsum(1) - 1
        M = int(b.sum(1).max())
        rr, tt = b.nonzero(as_tuple=True)
        z = h.new_zeros(h.shape[0], M, h.shape[-1])
        z[rr, ci[rr, tt]] = h[rr, tt]
        P = hf.new_zeros(h.shape[0], M)
        P[rr, ci[rr, tt]] = p[rr, tt]
    return hf, p, b, ci, z, P


def dechunk(z, P, p, b, ci):
    """EMA over units, zbar_j = P_j z_j + (1 - P_j) zbar_{j-1}, copied to every position of the unit, times a
    straight-through confidence: forward 1, gradient through p."""
    with fp32():
        M = z.shape[1]
        Pc = P.clamp(1e-4, 1 - 1e-4)
        S = torch.log1p(-Pc).cumsum(1)
        tril = torch.ones(M, M, dtype=torch.bool, device=z.device).tril()
        zb = ((S[:, :, None] - S[:, None, :]).masked_fill(~tril, -torch.inf).exp() * Pc[:, None, :]) @ z.float()
        up = zb.gather(1, ci.clamp(min=0)[..., None].expand(-1, -1, z.shape[-1]))
        c = torch.where(b, p, 1 - p)
        return up * (1 + c - c.detach())[..., None]


class Head(nn.Module):
    """One vector per molecule -> RMSNorm -> MLP -> `out` values (fingerprint logits or descriptor z-scores).
    readout "mean": mean and max of the byte decoder output over the valid positions.
    readout "cls": the byte decoder output at BOS and the main network output at position 0."""

    def __init__(self, d, dm, out, readout, hid=1024):
        super().__init__()
        self.readout = readout
        self.norm = nn.RMSNorm(2 * d if readout == "mean" else d + dm)
        self.fc1, self.fc2 = nn.Linear(self.norm.normalized_shape[0], hid), nn.Linear(hid, out)

    def forward(self, o, valid, m0):
        v, o = valid[..., None], o.float()
        if self.readout == "mean":
            h = torch.cat([(o * v).sum(1) / v.sum(1), o.masked_fill(~v, -torch.inf).amax(1)], -1)
        else:
            h = torch.cat([o[:, 0], m0.float()], -1)
        return self.fc2(F.gelu(self.fc1(self.norm(h))))


class HNet(nn.Module):
    """widths: byte stage, chunk stage, main network. layers: per byte-stage stack, per chunk-stage stack, main.
    out > 0: a molecule-level Head with that many outputs in place of the byte head."""

    def __init__(self, widths=(256, 384, 512), layers=(2, 2, 6), causal=True, out=0, readout="mean", dh=64):
        super().__init__()
        d, (n0, n1, nm) = widths, layers
        hid = [math.ceil(8 * w / 3 / 64) * 64 for w in d]
        self.d, self.causal, self.out = d, causal, out
        self.config = {"widths": list(widths), "layers": list(layers), "causal": causal, "out": out, "readout": readout}
        self.emb = nn.Embedding(V, d[0])
        self.enc, self.dec = Stack(n0, d[0], d[0] // dh, hid[0]), Stack(n0, d[0], d[0] // dh, hid[0])
        self.enc1, self.dec1 = Stack(n1, d[1], d[1] // dh, hid[1]), Stack(n1, d[1], d[1] // dh, hid[1])
        self.main = Stack(nm, d[2], d[2] // dh, hid[2])
        self.q, self.k = nn.Linear(d[0], d[0], bias=False), nn.Linear(d[0], d[0], bias=False)
        self.q1, self.k1 = nn.Linear(d[1], d[1], bias=False), nn.Linear(d[1], d[1], bias=False)
        self.res, self.res1 = nn.Linear(d[0], d[0]), nn.Linear(d[1], d[1])
        self.pad, self.pad1 = nn.Parameter(torch.zeros(d[1] - d[0])), nn.Parameter(torch.zeros(d[2] - d[1]))
        self.head = Head(d[0], d[2], out, readout) if out else nn.Linear(d[0], V, bias=False)
        depth = (2 * n0, 2 * n0 + 2 * n1, 2 * n0 + 2 * n1 + nm)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                std = 0.02 / math.sqrt(2 * depth[self.stage_of(name)]) if name.endswith(("out_proj", "fc2")) else 0.02
                nn.init.normal_(m.weight, std=std)
        nn.init.normal_(self.emb.weight, std=0.02)
        with torch.no_grad():
            for lin in (self.q, self.k, self.q1, self.k1):
                lin.weight.copy_(torch.eye(lin.in_features))
            for lin in (self.res, self.res1):
                lin.weight.zero_(); lin.bias.zero_()
            if out:
                self.head.fc1.bias.zero_()
                self.head.fc2.bias.fill_(math.log(62 / (2048 - 62)) if out == 2048 else 0.0)  # mean on-bit rate

    def stage_of(self, name):
        """0 byte stage, 1 chunk stage, 2 main network (learning-rate multiplier index)."""
        if name.startswith(("main", "pad1")):
            return 2
        return 1 if name.startswith(("enc1.", "dec1.", "q1.", "k1.", "res1.")) or name == "pad" else 0

    def forward(self, x, n):
        """x [B, L] long (BOS, bytes, PAD), n [B] byte counts.
        Returns the output (byte logits [B, L, V], or [B, out]) and a dict:
        p, b: chunk-start probability and flag per byte; valid: BOS and content positions;
        p1, b1, valid1: the same per chunk for group starts; bg: byte starts a group;
        pg: p1 of the chunk a byte starts, else 0; main: main network output [B, groups, d2]; dec: byte decoder output."""
        valid = torch.arange(x.shape[1], device=x.device)[None] <= n[:, None]
        km = (lambda v: None) if self.causal else (lambda v: v)
        h = self.enc(self.emb(x), km(valid))
        hf, p, b, ci, z, P = route(h, self.q, self.k, valid, None if self.causal else valid & (x != MASK))
        valid1 = torch.arange(z.shape[1], device=x.device)[None] < b.sum(1)[:, None]
        h1 = self.enc1(torch.cat([z, self.pad.to(z.dtype).expand(*z.shape[:2], -1)], -1), km(valid1))
        h1f, p1, b1, ci1, z1, P1 = route(h1, self.q1, self.k1, valid1)
        valid2 = torch.arange(z1.shape[1], device=x.device)[None] < b1.sum(1)[:, None]
        zm = self.main(torch.cat([z1, self.pad1.to(z1.dtype).expand(*z1.shape[:2], -1)], -1), km(valid2))
        with fp32():
            y1 = dechunk(zm[..., :self.d[1]], P1, p1, b1, ci1) + self.res1(h1f)
        z = self.dec1(y1.to(h.dtype), km(valid1))[..., :self.d[0]]
        with fp32():
            y = dechunk(z, P, p, b, ci) + self.res(hf)
        o = self.dec(y.to(h.dtype), km(valid))
        out = self.head(o, valid, zm[:, 0]) if self.out else self.head(o)
        g = ci.clamp(min=0)
        return out, {"p": p, "b": b, "valid": valid, "p1": p1, "b1": b1, "valid1": valid1, "main": zm, "dec": o,
                     "bg": b & b1.gather(1, g), "pg": torch.where(b, p1.gather(1, g), 0.0)}


def ratio_loss(p, b, valid, N):
    """H-Net eq. 10 on content positions (the forced boundary at BOS excluded): holds 1 / N of them as boundaries."""
    m = valid.clone()
    m[:, 0] = False
    k = m.sum().clamp(min=1)
    Fr, G = (b & m).sum() / k, (p * m).sum() / k
    return N / (N - 1) * ((N - 1) * Fr * G + (1 - Fr) * (1 - G))


def save(model, path, **extra):
    """Weights as model.safetensors and the constructor arguments (plus extra) as config.json in directory `path`."""
    from safetensors.torch import save_file
    Path(path).mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              str(Path(path) / "model.safetensors"))
    (Path(path) / "config.json").write_text(json.dumps(model.config | extra, indent=1))


def load(path, device="cpu"):
    """Model in eval mode from a directory holding config.json and model.safetensors, or from the Hub as
    "user/repo/folder" (a private repo needs `hf auth login`)."""
    from safetensors.torch import load_file
    path = str(path)
    if not Path(path).exists():
        from huggingface_hub import snapshot_download
        user, repo, *sub = path.split("/")
        path = Path(snapshot_download(f"{user}/{repo}", allow_patterns=["/".join(sub + ["*"])])).joinpath(*sub)
    cfg = json.loads((Path(path) / "config.json").read_text())
    model = HNet(tuple(cfg["widths"]), tuple(cfg["layers"]), cfg["causal"], cfg["out"], cfg["readout"])
    model.load_state_dict(load_file(str(Path(path) / "model.safetensors")))
    model.config = cfg
    return model.to(device).eval()
