# hnet-smiles

One byte-level H-Net ([Hwang, Wang, Gu](https://arxiv.org/abs/2507.07955)) trained on SMILES under four objectives: next byte, fingerprint, descriptors and masked fragments. The network learns where to cut the string, and this repository holds the model, the training script and the evaluation that scores the learned cuts against chemical fragments.

[![Blog post](https://img.shields.io/badge/blog-post-1f6feb)](https://drhb.github.io/blog/posts/learned-cuts/)
[![Hugging Face models](https://img.shields.io/badge/Hugging%20Face-models-ffd21e?logo=huggingface&logoColor=000)](https://huggingface.co/Asimo/hnet-smiles)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Post: [How the training objective shapes learned tokens](https://drhb.github.io/blog/posts/learned-cuts/)

Weights: [Asimo/hnet-smiles](https://huggingface.co/Asimo/hnet-smiles)

## Install

```bash
pip install -r requirements.txt
```

Python 3.11, PyTorch 2.4 or newer. RDKit is pinned to 2026.03.3, the version behind the reported numbers; newer releases count rings differently.

## Models

Eight models, two per objective. The 30M round uses widths 256/384/512, AdamW and 3 epochs. The 20M round uses widths 256/320/384, Muon and 5 epochs, and its fingerprint and descriptor heads read one position instead of averaging over bytes.

| folder | objective | parameters | steps |
|---|---|---|--- |
| `next-byte-30m` | next byte | 30,342,784 | 101,289 |
| `fingerprint-30m` | fingerprint | 32,901,248 | 101,289 |
| `descriptors-30m` | descriptors | 30,991,673 | 101,289 |
| `masked-fragments-30m` | masked fragments | 30,342,784 | 101,289 |
| `next-byte-20m` | next byte | 19,555,520 | 168,815 |
| `fingerprint-20m` | fingerprint | 22,245,184 | 168,815 |
| `descriptors-20m` | descriptors | 20,335,609 | 168,815 |
| `masked-fragments-20m` | masked fragments | 19,555,520 | 168,815 |

```python
import torch
from data import encode, split
from model import load

model = load("Asimo/hnet-smiles/masked-fragments-30m")
smiles = "CC(C)S(=O)(=O)CCSc1cccc(C(=O)O)c1"
x, n = encode([smiles])
with torch.no_grad():
    out, r = model(x, n)
print(split(smiles, r["b"][0]))
# ['CC(C)', 'S(=O)(=O)', 'CC', 'Sc', '1cccc', '(C(=O)O', ')c1']
```

`r["b"]` marks the bytes that start a piece and `r["bg"]` the bytes that start a group. `out` is byte logits for the next-byte and masked-fragment models, 2,048 fingerprint logits for the fingerprint models, and 185 descriptor z-scores for the descriptor models (names and scaling in `data/desc_meta.json`).

## Train

Build a corpus from a text file with one SMILES per line, then train:

```bash
python prep.py smiles.txt corpus --held 50000 --targets fp,desc --procs 16
```

```bash
python train.py --data corpus --out runs/next-byte-30m --objective next_byte
python train.py --data corpus --out runs/fingerprint-30m --objective fingerprint
python train.py --data corpus --out runs/descriptors-30m --objective descriptors
python train.py --data corpus --out runs/masked-fragments-30m --objective masked_fragments

python train.py --data corpus --out runs/next-byte-20m --objective next_byte --widths 256,320,384 --opt muon --epochs 5
python train.py --data corpus --out runs/fingerprint-20m --objective fingerprint --readout cls --widths 256,320,384 --opt muon --epochs 5
python train.py --data corpus --out runs/descriptors-20m --objective descriptors --readout cls --widths 256,320,384 --opt muon --epochs 5
python train.py --data corpus --out runs/masked-fragments-20m --objective masked_fragments --widths 256,320,384 --opt muon --epochs 5
```

Every other setting is the default: 131,072 padded bytes per step (about 2,700 molecules), peak learning rate 5e-4 with multipliers 2 / 1.5 / 1 for the byte stage, chunk stage and main network, 1,000 warmup steps, cosine decay to 10%, ratio loss target 3 bytes per piece and 2 pieces per group with weight 0.03, bf16 autocast.

The training corpus is not in this repository. It was 90,955,563 canonical, non-stereo SMILES (PubChem, COCONUT, ChEBI, LIPID MAPS and public spectral libraries), of which 50,000 were held out. The held-out molecules are in `data/heldout.smi`.

## Reproduce the cut tables

```bash
python evaluate.py Asimo/hnet-smiles/next-byte-30m Asimo/hnet-smiles/fingerprint-30m Asimo/hnet-smiles/descriptors-30m Asimo/hnet-smiles/masked-fragments-30m
python evaluate.py Asimo/hnet-smiles/next-byte-20m Asimo/hnet-smiles/fingerprint-20m Asimo/hnet-smiles/descriptors-20m Asimo/hnet-smiles/masked-fragments-20m
python tables.py 30m
python tables.py 20m
```

`evaluate.py` runs each model over the 50,000 held-out molecules and writes `results/<model>.json`. `tables.py` prints the tables. The files in `results/` are the output of these commands.

## Results

Pieces of the 30M models on 50,000 held-out molecules. Random cuts keep each molecule's number of cuts and shuffle their positions.

| | next byte | fingerprint | descriptors | masked fragments |
|---|---|---|---|---|
| distinct pieces | 8,620 | 4,271 | 3,810 | 14,928 |
| bytes per piece | 2.75 | 2.59 | 2.82 | 4.33 |
| pieces with no atom | 21.7% | 17.2% | 7.7% | 3.7% |
| pieces holding a whole functional group | 2.9% | 2.4% | 3.9% | 14.7% |
| pieces that cut through an atom | 1.1% | 0.3% | 4.1% | 0.03% |
| multi-character atoms split | 18% | 5% | 65% | 0.4% |
| same, random cuts | 47% | 48% | 46% | 31% |
| preference for breakable (BRICS) bonds | -0.04 | +0.16 | +0.14 | +0.38 |
| carboxylic acids kept whole | 17% | 0% | 0% | 100% |
| sulfonyls kept whole | 13% | 59% | 96% | 98% |

191 pieces are used by all four models.

Readout and representation, both rounds:

| | fingerprint 30M | fingerprint 20M | descriptors 30M | descriptors 20M |
|---|---|---|---|---|
| readout | mean and max over bytes | one position | mean and max over bytes | one position |
| held-out accuracy | Tanimoto 0.831 | Tanimoto 0.832 | R² 0.998 | R² 0.998 |
| multi-character atoms split | 5% | 70% | 65% | 29% |

| 20M model | atoms split | nearest-neighbour Tanimoto | probe R², heavy atoms / rings |
|---|---|---|---|
| masked fragments | 2% | 0.282 | .77 / .78 |
| next byte | 6% | 0.278 | .78 / .81 |
| descriptors | 29% | 0.277 | .85 / .82 |
| fingerprint | 70% | 0.336 | .49 / .48 |

Nearest-neighbour Tanimoto is 0.103 for a random neighbour and 0.429 for the best possible one.

The numbers above are those in the post, computed on an A6000 with bf16 autocast. The files in `results/` were produced on an M2 Max in fp32, where a few boundaries fall on the other side of the 0.5 threshold. Shares agree with the server's to within 0.005. Counts of distinct pieces differ by up to 3 for the 30M models and up to 27 for the 20M models. The nearest-neighbour Tanimoto of `fingerprint-20m` is 0.337 here and 0.336 in the post.

## Not included

The downstream benchmarks (MoleculeNet, QM9, MoleculeACE, TDC ADMET), the control models (one-stage H-Net, BPE Transformer, random masking, SAFE input) and the corpus build are not in this repository.

## Licence

MIT.
