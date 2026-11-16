"""SMILES as bytes. A corpus directory (prep.py) holds
bytes.bin  uint8, all rows concatenated      off.npy   int64 [rows + 1] row starts
held.npy   held-out rows                     fp.npy    uint8 [rows, 256] packed Morgan fingerprints (optional)
desc.npy   int8 [rows, 185] descriptor codes (optional)
"""
import json
from pathlib import Path
import numpy as np
import torch
from model import BOS, PAD

HERE = Path(__file__).parent
DESC = json.loads((HERE / "data/desc_meta.json").read_text())  # names and the fitted transform of the 185 descriptors


def encode(smiles):
    """List of SMILES -> x int64 [B, L] = BOS, bytes, PAD (L = longest + 1, rounded up to 8), n int64 [B] byte counts."""
    s = [t.encode() for t in smiles]
    n = np.array([len(t) for t in s])
    x = np.full((len(s), -(-(n.max() + 1) // 8) * 8), PAD, np.int64)
    x[:, 0] = BOS
    for i, t in enumerate(s):
        x[i, 1:1 + len(t)] = np.frombuffer(t, np.uint8)
    return torch.from_numpy(x), torch.from_numpy(n)


def decode(x, n):
    return [bytes(r[1:k + 1].tolist()).decode() for r, k in zip(x.cpu(), n.tolist())]


def split(smiles, starts):
    """One molecule's units as strings. starts: its row of the model's "b" (pieces) or "bg" (groups), BOS first."""
    st = [0] + [j for j in range(1, len(smiles)) if starts[j + 1]]
    return [smiles[a:b] for a, b in zip(st, st[1:] + [len(smiles)])]


class Corpus:
    def __init__(self, path):
        p = Path(path)
        self.by = np.memmap(p / "bytes.bin", np.uint8, "r")
        self.off = np.load(p / "off.npy")
        self.nb = np.diff(self.off)
        self.held = np.load(p / "held.npy")
        self.fp = np.load(p / "fp.npy", mmap_mode="r") if (p / "fp.npy").exists() else None
        self.desc = np.load(p / "desc.npy", mmap_mode="r") if (p / "desc.npy").exists() else None

    def smiles(self, rows):
        return [bytes(self.by[self.off[r]:self.off[r + 1]]).decode() for r in rows]

    def batch(self, rows):
        return encode(self.smiles(rows))


def fingerprint(mol):
    """Morgan radius 3, 2,048 bits, packed to uint8 [256] (bit order little)."""
    from rdkit.Chem import rdFingerprintGenerator as G
    return np.packbits(G.GetMorganGenerator(radius=3, fpSize=2048).GetFingerprintAsNumPy(mol), bitorder="little")


def unpack_fp(p):
    """uint8 [B, 256] -> float [B, 2048]."""
    return ((p.long()[..., None] >> torch.arange(8, device=p.device)) & 1).flatten(1).float()


def descriptors(mol):
    """The 185 RDKit descriptors as int8 codes: z = (transform(x) - mean) / sd clipped to +-6, code = round(z * 127 / 6),
    -128 where a value is missing. The transform (log1p, asinh or none per descriptor), mean and sd are those fitted on
    the training corpus (data/desc_meta.json). A descriptor model predicts z = code * 6 / 127."""
    from rdkit.Chem import Descriptors
    from rdkit.Chem.EState import EStateIndices
    fn = dict(Descriptors._descList)
    v = np.full(len(DESC["names"]), np.nan)
    for j, k in enumerate(DESC["names"]):
        if not k.endswith("EStateIndex"):
            try:
                v[j] = fn[k](mol)
            except Exception:
                pass
    if mol.GetNumAtoms():
        e = np.asarray(EStateIndices(mol))
        es = {"MaxEStateIndex": e.max(), "MinEStateIndex": e.min(), "MaxAbsEStateIndex": np.abs(e).max(),
              "MinAbsEStateIndex": np.abs(e).min()}
        for j, k in enumerate(DESC["names"]):
            if k in es:
                v[j] = es[k]
    lg, ah, mu, sd = (np.asarray(DESC[k]) for k in ("log1p", "asinh", "mean", "sd"))
    with np.errstate(all="ignore"):
        u = np.where(lg, np.log1p(np.maximum(v, 0)), np.where(ah, np.arcsinh(v), v))
        q = np.round(np.clip((u - mu) / sd, -6, 6) * 127 / 6)
    q[~np.isfinite(q)] = -128
    return q.astype(np.int8)


def plan(rows, nb, budget, seed):
    """One pass over rows: random order, bucketed by padded length L, budget // L rows per batch, batch order shuffled."""
    rng = np.random.default_rng(seed)
    rows = rng.permutation(rows)
    L = -(-(nb[rows].astype(np.int64) + 1) // 8) * 8
    o = np.argsort(L, kind="stable")
    rows, L = rows[o], L[o]
    bounds = []
    for l in np.unique(L):
        a, b = np.searchsorted(L, [l, l + 1])
        bounds += [(i, min(i + budget // l, b)) for i in range(a, b, budget // l)]
    return rows, np.array(bounds)[rng.permutation(len(bounds))]


class Batches(torch.utils.data.Dataset):
    """One item = one batch: x, n and the objective's extra tensor (fingerprints, descriptor codes or mask codes)."""

    def __init__(self, path, rows, bounds, objective, mask=None, seed=0, start=0):
        self.path, self.rows, self.bounds, self.objective, self.mask, self.seed, self.start, self.c = path, rows, bounds, objective, mask, seed, start, None

    def __len__(self):
        return len(self.bounds) - self.start

    def __getitem__(self, i):
        self.c = self.c or Corpus(self.path)
        i += self.start
        a, b = self.bounds[i]
        r = self.rows[a:b]
        x, n = self.c.batch(r)
        if self.objective in ("fingerprint", "descriptors"):
            t = self.c.fp if self.objective == "fingerprint" else self.c.desc
            o = np.argsort(r)
            y = np.empty((len(r), t.shape[1]), t.dtype)
            y[o] = t[r[o]]
            return x, n, torch.from_numpy(y)
        if self.objective == "masked_fragments":
            from masking import mask_batch
            return x, n, torch.from_numpy(mask_batch(x.numpy(), n.numpy(), np.random.default_rng([self.seed, i]), *self.mask))
        return x, n


def loader(path, rows, bounds, objective, mask=None, seed=0, workers=8, start=0):
    return torch.utils.data.DataLoader(Batches(path, rows, bounds, objective, mask, seed, start), batch_size=None, shuffle=False,
                                       num_workers=workers, prefetch_factor=4 if workers else None)
