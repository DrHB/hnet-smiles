"""Build a corpus directory from a text file with one SMILES per line.

python prep.py smiles.txt corpus --held 50000 --targets fp,desc --procs 16

Each row is rewritten as canonical SMILES without stereo; rows RDKit cannot parse are dropped. --targets adds the
fingerprint and descriptor targets (descriptors take about 7 ms per molecule on one core).
"""
import argparse
import multiprocessing as mp
from pathlib import Path
import numpy as np
from rdkit import Chem, RDLogger


def work(smiles):
    from data import descriptors, fingerprint
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        can = Chem.MolToSmiles(m, isomericSmiles=False)
        m = Chem.MolFromSmiles(can)
        out.append((can.encode(), fingerprint(m) if "fp" in TARGETS else None, descriptors(m) if "desc" in TARGETS else None))
    return out


def init(targets):
    global TARGETS
    TARGETS = targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("smiles")
    ap.add_argument("out")
    ap.add_argument("--held", type=int, default=50000, help="held-out rows, drawn at random")
    ap.add_argument("--targets", default="", help="fp, desc or fp,desc")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [l.split()[0] for l in open(a.smiles) if l.strip()]
    parts = [lines[i:i + 4096] for i in range(0, len(lines), 4096)]
    with mp.get_context("spawn").Pool(a.procs, init, (a.targets.split(","),)) as pool:
        rows = [r for part in pool.imap(work, parts) for r in part]
    (out / "bytes.bin").write_bytes(b"".join(r[0] for r in rows))
    np.save(out / "off.npy", np.concatenate([[0], np.cumsum([len(r[0]) for r in rows])]).astype(np.int64))
    np.save(out / "held.npy", np.sort(np.random.default_rng(0).choice(len(rows), min(a.held, len(rows) // 10), replace=False)))
    if "fp" in a.targets:
        np.save(out / "fp.npy", np.stack([r[1] for r in rows]))
    if "desc" in a.targets:
        np.save(out / "desc.npy", np.stack([r[2] for r in rows]))
    print(f"{len(rows)} rows ({len(lines) - len(rows)} dropped), {sum(len(r[0]) for r in rows)} bytes -> {out}")


if __name__ == "__main__":
    main()
