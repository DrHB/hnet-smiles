"""Fragment masking for the masked-fragments objective (adapted from FACT: whole fragments, then random bytes).

Per molecule of n bytes: rate r ~ U(r_lo, r_hi), k = max(1, round(r n)) bytes masked; fragment share phi ~ U(f_lo, f_hi).
Whole fragments, in random order, are masked while the total stays <= k, until they cover >= round(phi k) bytes; random
bytes then bring the total to exactly k. Every masked byte becomes MASK.
Fragments are atom sets: functional-group matches (FG), rings, and BRICS blocks smaller than the molecule.
A fragment's bytes are the bytes its atoms own: the '(' or '.' and the bond written before an atom, the atom, its
ring-closure bonds and labels, and the ')' closing a branch it starts.
A molecule whose string and RDKit atoms cannot be aligned gets random bytes only.
"""
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
from model import MASK

RDLogger.DisableLog("rdApp.*")
FG = {"carboxylic_acid": "C(=O)[OX2H1]", "ester": "[#6]C(=O)O[#6]", "amide": "C(=O)N", "carbonyl": "[CX3]=O",
      "nitro": "[N+](=O)[O-]", "sulfonyl": "S(=O)=O", "nitrile": "C#N", "cf3": "C(F)(F)F", "methoxy": "[CH3]O",
      "phenyl": "c1ccccc1"}
FGP = [Chem.MolFromSmarts(v) for v in FG.values()]
PS = Chem.SmilesParserParams()
PS.removeHs = False  # explicit [H] atoms keep their own index, so RDKit atom i is the string's atom i
ORG, BONDS = set(b"BCNOPSFIbcnops"), set(b"=#$-:/\\")


def owners(s):
    """Owner atom (string order) of every byte of SMILES bytes s."""
    o = np.full(len(s), -1, np.int32)
    cur, pend, stack, i, n, a = -1, [], [], 0, len(s), -1
    while i < n:
        c = s[i]
        j = s.index(b"]", i) + 1 if c == 91 else (i + 2 if s[i:i + 2] in (b"Cl", b"Br") else i + 1) if c in ORG else 0
        if j:
            a += 1
            if pend[:1] == [40]:  # the first atom of a branch owns its '(' and the matching ')'
                stack[-1][1] = a
            o[i - len(pend):j] = a
            cur, pend, i = a, [], j
            continue
        if c == 40:  # (
            stack.append([cur, None]); pend = [40]
        elif c == 41:  # )
            cur, o[i] = stack.pop()
        elif c == 46:  # .
            pend = [46]
        elif c in BONDS:
            if i + 1 < n and (48 <= s[i + 1] <= 57 or s[i + 1] == 37):  # bond of a ring closure
                o[i] = cur
            else:
                pend.append(c)
        elif c == 37:  # %nn
            o[i:i + 3] = cur; i += 3; continue
        elif 48 <= c <= 57:
            o[i] = cur
        else:
            raise ValueError(f"unexpected byte {chr(c)!r}")
        i += 1
    if (o < 0).any():
        raise ValueError("unowned byte")
    return o


def atom_sets(m):
    sets = {frozenset(t) for q in FGP for t in m.GetSubstructMatches(q)}
    sets |= {frozenset(r) for r in m.GetRingInfo().AtomRings()}
    bb = [m.GetBondBetweenAtoms(i, j).GetIdx() for (i, j), _ in BRICS.FindBRICSBonds(m)]
    if bb:
        sets |= {frozenset(f) for f in Chem.GetMolFrags(Chem.FragmentOnBonds(m, bb, addDummies=False))}
    sets.discard(frozenset(range(m.GetNumAtoms())))
    return sets


def fragments(s):
    """Bool byte masks [F, len(s)] of the fragments of SMILES bytes s; None when string and RDKit cannot be aligned."""
    try:
        o = owners(s)
    except (ValueError, IndexError, TypeError):
        return None
    m = Chem.MolFromSmiles(s.decode(), PS)
    if m is None or m.GetNumAtoms() != int(o.max()) + 1:
        return None
    sets = atom_sets(m)
    if not sets:
        return np.zeros((0, len(s)), bool)
    A = np.zeros((m.GetNumAtoms(), len(s)), bool)
    A[o, np.arange(len(s))] = True
    return np.stack([A[list(f)].any(0) for f in sets])


def choose(fr, n, k, kf, rng):
    """int8 [n]: 2 fragment byte, 1 random byte, 0 kept."""
    m = np.zeros(n, np.int8)
    c = 0
    if kf and fr is not None:
        for j in rng.permutation(len(fr)):
            u = fr[j] & (m == 0)
            if c + int(u.sum()) <= k:
                m[u] = 2
                c += int(u.sum())
                if c >= kf:
                    break
    m[rng.choice(np.flatnonzero(m == 0), k - c, replace=False)] = 1
    return m


def mask_batch(x, n, rng, r_lo, r_hi, f_lo, f_hi):
    """x [B, L] (BOS, bytes, PAD), n [B] byte counts (numpy) -> int8 [B, L] codes as choose (0 at BOS and padding)."""
    out = np.zeros(x.shape, np.int8)
    r, phi = rng.uniform(r_lo, r_hi, len(n)), rng.uniform(f_lo, f_hi, len(n))
    for i, ni in enumerate(n.tolist()):
        k = max(1, round(r[i] * ni))
        kf = round(phi[i] * k)
        fr = fragments(x[i, 1:ni + 1].astype(np.uint8).tobytes()) if kf else None
        out[i, 1:ni + 1] = choose(fr, ni, k, kf, rng)
    return out


def apply_mask(x, codes):
    """Input with MASK at the coded positions, and the targets (the original byte there, -100 elsewhere)."""
    return torch.where(codes > 0, MASK, x), torch.where(codes > 0, x, -100)
