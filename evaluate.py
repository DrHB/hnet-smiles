"""Where a model cuts, scored against chemistry, on held-out molecules.

python evaluate.py MODEL [MODEL ...] [--smiles data/heldout.smi] [--n 50000] [--out results]

MODEL is a directory with config.json and model.safetensors, or "user/repo/folder" on the Hub. Writes one JSON per model:
the objective's own metric, cut statistics for pieces (first router) and groups (second router), each next to a control
that keeps every molecule's number of cuts and shuffles their positions, and probes of a pooled molecule vector.
tables.py prints the post's tables from these files.
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import BRICS
from data import descriptors, encode, fingerprint, plan, unpack_fp
from masking import FG, FGP, PS, apply_mask, mask_batch
from model import EOS, load

SMI_RE = re.compile(r"(\[[^\]]+\]|Br|Cl|%\d{2}|[BCNOSPFI]|[bcnops]|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|\d)")
TH = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def batches(smiles, budget):
    """Index arrays of length-bucketed batches (data.plan, seed 0) with their encoded x, n on the CPU."""
    nb = np.array([len(s) for s in smiles])
    order, bounds = plan(np.arange(len(smiles)), nb, budget, 0)
    for a, b in bounds:
        yield order[a:b], *encode([smiles[i] for i in order[a:b]])


def forward(model, x, n):
    dev = next(model.parameters()).device
    with torch.no_grad(), torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
        return model(x.to(dev), n.to(dev))


def cuts(model, smiles, budget=131072):
    """Per molecule: SMILES, piece starts b and group starts bg over its bytes (bool), and the mean main-network state."""
    out = [None] * len(smiles)
    for idx, x, n in batches(smiles, budget):
        _, r = forward(model, x, n)
        nc = r["b1"].sum(1)  # positions the main network runs on
        mm = (torch.arange(r["main"].shape[1], device=nc.device)[None] < nc[:, None]).float()[..., None]
        pool = ((r["main"].float() * mm).sum(1) / nc[:, None]).cpu().numpy().astype(np.float16)
        b, bg = r["b"].cpu().numpy(), r["bg"].cpu().numpy()
        for j, i in enumerate(idx):
            k = len(smiles[i])
            out[i] = (smiles[i], b[j, 1:k + 1], bg[j, 1:k + 1], pool[j])
    return out


@torch.no_grad()
def objective_metrics(model, smiles, objective, budget=131072, fp=None, desc=None, rows=None, mask_rate=0.35):
    """The held-out metric of each objective. fp, desc: corpus target arrays indexed by rows; computed with RDKit if absent."""
    dev = next(model.parameters()).device
    if objective == "next_byte":
        bits = 0.0
        for idx, x, n in batches(smiles, budget):
            y = x.clone()
            y[:, :-1] = x[:, 1:]
            y[torch.arange(len(n)), n] = EOS
            y[torch.arange(x.shape[1])[None] > n[:, None]] = -100
            bits += F.cross_entropy(forward(model, x, n)[0].float().flatten(0, 1), y.flatten().to(dev), ignore_index=-100,
                                    reduction="sum").item() / math.log(2)
        return {"bits_per_byte": bits / sum(len(s) for s in smiles)}
    if objective == "masked_fragments":  # random masks, then fragments first (share 1), both at mask_rate
        res = {}
        for phi, pre in ((0.0, "random_mask_"), (1.0, "fragment_mask_")):
            rng, s = np.random.default_rng(0), np.zeros(6)
            for idx, x, n in batches(smiles, budget):
                if phi:
                    k = mask_batch(x.numpy(), n.numpy(), rng, mask_rate, mask_rate, phi, phi)
                else:  # every content byte masked independently, at least one per molecule
                    pos = np.arange(x.shape[1])[None]
                    k = (rng.random(x.shape) < mask_rate) & (pos >= 1) & (pos <= n.numpy()[:, None])
                    k |= (pos == 1 + (rng.random(len(n)) * n.numpy()).astype(int)[:, None]) & ~k.any(1, keepdims=True)
                k = torch.from_numpy(k.astype(np.int8)).to(dev)
                xin, y = apply_mask(x.to(dev), k)
                ok = forward(model, xin, n)[0].argmax(-1) == y
                m, f = y >= 0, k == 2
                s += [ok.sum().item(), m.sum().item(), (ok | ~m).all(1).sum().item(), len(n), ok[f].sum().item(), f.sum().item()]
            res |= {pre + "accuracy": s[0] / s[1], pre + "molecules_exact": s[2] / s[3]}
            if phi:
                res |= {"fragment_bytes_accuracy": s[4] / s[5], "other_bytes_accuracy": (s[0] - s[4]) / (s[1] - s[5])}
        return res
    P, Y = [], []
    for idx, x, n in batches(smiles, budget):
        P.append(forward(model, x, n)[0].float().cpu())
        if objective == "fingerprint":
            Y.append(np.stack([fp[r] for r in rows[idx]]) if fp is not None else
                     np.stack([fingerprint(Chem.MolFromSmiles(smiles[i])) for i in idx]))
        else:
            Y.append(np.stack([desc[r] for r in rows[idx]]) if desc is not None else
                     np.stack([descriptors(Chem.MolFromSmiles(smiles[i])) for i in idx]))
    p, y = torch.cat(P), torch.from_numpy(np.concatenate(Y))
    if objective == "fingerprint":
        s, y, tani = torch.sigmoid(p), unpack_fp(y), {}
        for t in TH:
            q = (s > t).float()
            i = (q * y).sum(1)
            tani[t] = (i / (q.sum(1) + y.sum(1) - i).clamp(min=1)).mean().item()
        return {"tanimoto": max(tani.values()), "threshold": max(tani, key=tani.get), "tanimoto_at_0.5": tani[0.5]}
    m, y, p = (y != -128).double(), y.double() * 6 / 127, p.double()
    mu = (y * m).sum(0) / m.sum(0).clamp(min=1)
    r2 = 1 - (((p - y) ** 2) * m).sum(0) / (((y - mu) ** 2) * m).sum(0).clamp(min=1e-9)
    return {"r2_mean": r2.mean().item(), "r2_median": r2.median().item()}


def units(s, cb):
    st = [0] + [j for j in range(1, len(s)) if cb[j]]
    return [(a, b) for a, b in zip(st, st[1:] + [len(s)])]


def string_stats(mols, rng):
    """Unit strings: mean length, distinct strings, share of multi-character atoms split. Learned and shuffled cuts."""
    res = {}
    for name in ("learned", "random"):
        cnt, n_units, n_bytes, multi, split = Counter(), 0, 0, 0, 0
        for s, cb in mols:
            if name == "random":
                cb = cb.copy()
                cb[1:] = rng.permutation(cb[1:])
            inner = [(t.start(), t.end()) for t in SMI_RE.finditer(s) if t.end() - t.start() > 1]
            multi += len(inner)
            split += sum(cb[a + 1:e].any() for a, e in inner)
            u = units(s, cb)
            cnt.update(s[a:b] for a, b in u)
            n_units += len(u); n_bytes += len(s)
        res[name] = {"bytes_per_unit": n_bytes / n_units, "distinct": len(cnt), "atoms_split": split / multi}
        if name == "learned":
            res["counts"] = cnt
    return res


def chem_stats(mols, rng):
    """Bonds and substructures: every atom token is mapped to its RDKit atom, and a bond is cut when its two atoms sit in
    different units. BRICS and non-BRICS cut rates are over bonds outside rings."""
    pats = {k: Chem.MolFromSmarts(v) for k, v in FG.items()}
    prep, skipped = [], 0
    for s, cb in mols:
        m = Chem.MolFromSmiles(s, PS)
        at = [t for t in SMI_RE.finditer(s) if t.group(0)[0] == "[" or t.group(0)[0].isalpha()]
        if m is None or len(at) != m.GetNumAtoms() or any(
                re.match(r"\[?\d*([A-Za-z][a-z]?)", t.group(0)).group(1).capitalize() != a.GetSymbol() and
                re.match(r"\[?\d*([A-Za-z])", t.group(0)).group(1).upper() != a.GetSymbol() for t, a in zip(at, m.GetAtoms())):
            skipped += 1
            continue
        brics = {frozenset(b[0]) for b in BRICS.FindBRICSBonds(m)}
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.IsInRing(),
                  frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) in brics) for b in m.GetBonds()]
        rings = [r for r in m.GetRingInfo().AtomRings() if len(r) in (5, 6)]
        prep.append((np.array([t.start() for t in at]), cb, bonds, rings, {k: m.GetSubstructMatches(q) for k, q in pats.items()}))
    res = {"molecules": len(prep), "skipped": skipped}
    for name in ("learned", "random"):
        c = Counter()
        for st, cb, bonds, rings, fgs in prep:
            cb = cb.copy()
            if name == "random":
                cb[1:] = rng.permutation(cb[1:])
            cb[0] = True
            ac = (np.cumsum(cb) - 1)[st]  # unit of each atom
            for i, j, ring, br in bonds:
                k = "ring" if ring else "brics" if br else "nonbrics"
                c[k] += 1; c[k + "_cut"] += ac[i] != ac[j]
            for r in rings:
                c["rings"] += 1; c["rings_whole"] += len({ac[a] for a in r}) == 1
            for k, ms in fgs.items():
                for mt in ms:
                    c["fg_" + k] += 1; c["fg_" + k + "_whole"] += len({ac[a] for a in mt}) == 1
        res[name] = {"cut_rate_brics": c["brics_cut"] / max(c["brics"], 1), "cut_rate_nonbrics": c["nonbrics_cut"] / max(c["nonbrics"], 1),
                     "cut_rate_ring": c["ring_cut"] / max(c["ring"], 1), "rings_whole": c["rings_whole"] / max(c["rings"], 1),
                     "fg_whole": {k: c[f"fg_{k}_whole"] / c["fg_" + k] for k in FG if c["fg_" + k]}}
    res["fg_count"] = {k: c["fg_" + k] for k in FG}
    return res


def context_stats(mols):
    """Shares of unit occurrences: no atom (only brackets, digits or bond symbols), a whole functional group inside,
    and a cut through an atom token at either end."""
    n, no_atom, k, fg, tok = 0, 0, 0, 0, 0
    for s, cb in mols:
        u = units(s, cb)
        n += len(u)
        for a0, a1 in u:
            body = re.sub(r"\[[^\]]*\]", "A", s[a0:a1])
            broken = "[" in body or "]" in body or body[:1] in ("l", "r") or re.search(r"%\d?$", body)
            no_atom += not broken and not re.search(r"[A-Za-z]", body)
        m = Chem.MolFromSmiles(s, PS)
        toks = list(SMI_RE.finditer(s))
        st = np.array([t.start() for t in toks if t.group(0)[0] == "[" or t.group(0)[0].isalpha()])
        if m is None or len(st) != m.GetNumAtoms():
            continue
        inside = np.zeros(len(s) + 1, bool)
        for t in toks:
            inside[t.start() + 1:t.end()] = True
        fgs = [(min(x), max(x)) for q in FGP for x in m.GetSubstructMatches(q)]
        for a0, a1 in u:
            i0, i1 = np.searchsorted(st, a0), np.searchsorted(st, a1)
            k += 1
            tok += bool(inside[a0] or inside[a1])
            fg += i1 > i0 and any(lo >= i0 and hi < i1 for lo, hi in fgs)
    return {"no_atom": no_atom / n, "whole_functional_group": fg / k, "cuts_atom": tok / k}


def probes(mols, rng):
    """From the mean main-network state: ridge R^2 for heavy atoms and rings (80/20 split), and the mean Tanimoto
    (Morgan radius 2) between 2,000 query molecules and their nearest neighbour by cosine, with the best possible and a
    random neighbour for scale."""
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
    ms = [Chem.MolFromSmiles(m[0]) for m in mols]
    Y = {"heavy_atoms": np.array([m.GetNumHeavyAtoms() for m in ms], float),
         "rings": np.array([rdMolDescriptors.CalcNumRings(m) for m in ms], float)}
    X = np.stack([m[3] for m in mols]).astype(float)
    idx = rng.permutation(np.arange(len(mols)))
    tr, te = idx[:int(.8 * len(idx))], idx[int(.8 * len(idx)):]
    Z = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)

    def fit(rows, y, lam):
        return np.linalg.solve(Z[rows].T @ Z[rows] + lam * np.eye(Z.shape[1]), Z[rows].T @ (y[rows] - y[rows].mean()))

    def ridge(y):
        a, b = tr[:len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]  # strength chosen on a quarter of the training rows
        lam = max((1 - ((Z[b] @ fit(a, y, l) + y[a].mean() - y[b]) ** 2).mean() / y[b].var(), l)
                  for l in (1e-2, 1e-1, 1, 10, 100, 1000))[1]
        return float(1 - ((Z[te] @ fit(tr, y, lam) + y[tr].mean() - y[te]) ** 2).mean() / y[te].var())

    out = {"r2_" + k: ridge(y) for k, y in Y.items()}
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(m) for m in ms]
    q = rng.choice(np.arange(len(mols)), min(2000, len(mols)), replace=False)
    U = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    nn, best, rand = [], [], []
    for i in q:
        sim = U @ U[i]
        sim[i] = -np.inf
        nn.append(DataStructs.TanimotoSimilarity(fps[i], fps[int(np.argmax(sim))]))
    for i in q:
        t = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
        t[i] = -1
        best.append(t.max()); rand.append(t[rng.integers(len(t))])
    return out | {"nn_tanimoto": float(np.mean(nn)), "nn_tanimoto_best": float(np.mean(best)), "nn_tanimoto_random": float(np.mean(rand))}


def level(mols, rng):
    s, c = string_stats(mols, rng), chem_stats(mols, rng)
    sel = lambda r: r["cut_rate_brics"] - r["cut_rate_nonbrics"]
    return s.pop("counts"), {"strings": s, "chem": c, "brics_preference": sel(c["learned"]) - sel(c["random"]), **context_stats(mols)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--smiles", default=str(Path(__file__).parent / "data/heldout.smi"))
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    smiles = [l.strip() for l in open(a.smiles) if l.strip()][:a.n]
    Path(a.out).mkdir(exist_ok=True)
    vocab = {}
    for path in a.models:
        name = Path(path).name
        model = load(path, dev)
        cfg = json.loads((Path(path) / "config.json").read_text()) if Path(path).exists() else {}
        mols = cuts(model, smiles)
        rng = np.random.default_rng(0)  # one stream, in this order, for every shuffled control and probe split
        vocab[name], pieces = level([(m[0], m[1]) for m in mols], rng)
        _, groups = level([(m[0], m[2]) for m in mols], rng)
        rep = {"model": name, "molecules": len(mols), "device": dev, "pieces": pieces, "groups": groups, "representation": probes(mols, rng)}
        if cfg.get("objective"):
            rep["objective"] = {"name": cfg["objective"]} | objective_metrics(model, smiles, cfg["objective"])
        (Path(a.out) / f"{name}.json").write_text(json.dumps(rep, indent=1))
        print(name, json.dumps({k: v for k, v in rep.items() if k in ("objective", "representation")}), flush=True)
    if len(vocab) > 1:
        shared = sum(all(s in v for v in vocab.values()) for s in next(iter(vocab.values())))
        (Path(a.out) / ("shared_" + "_".join(sorted(vocab)) + ".json")).write_text(json.dumps({"models": sorted(vocab), "pieces_in_all": shared}))
        print("pieces in all models:", shared)


if __name__ == "__main__":
    main()
