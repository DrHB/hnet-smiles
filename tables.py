"""Print the post's cut tables from results/*.json (evaluate.py).

python tables.py [30m|20m] [results]
"""
import json
import sys
from pathlib import Path

OBJ = ("next-byte", "fingerprint", "descriptors", "masked-fragments")
FGS = ("carboxylic_acid", "carbonyl", "cf3", "nitrile", "sulfonyl", "methoxy", "nitro", "ester", "amide", "phenyl")
size = sys.argv[1] if len(sys.argv) > 1 else "30m"
d = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent / "results")
R = [json.loads((d / f"{o}-{size}.json").read_text()) for o in OBJ]
pct = lambda v: f"{100 * v:.1f}%"


def table(title, rows):
    print(f"\n{title}\n\n| | " + " | ".join(o.replace("-", " ") for o in OBJ) + " |\n|---|" + "---|" * len(OBJ))
    for name, f in rows:
        print(f"| {name} | " + " | ".join(str(f(r)) for r in R) + " |")


def level(lv, unit):
    s, c = (lambda r: r[lv]["strings"]), (lambda r: r[lv]["chem"])
    table(f"{unit.capitalize()}s ({size}), learned cuts with random cuts in brackets", [
        (f"distinct {unit}s", lambda r: f'{s(r)["learned"]["distinct"]:,} ({s(r)["random"]["distinct"]:,})'),
        (f"bytes per {unit}", lambda r: f'{s(r)["learned"]["bytes_per_unit"]:.2f}'),
        (f"{unit}s with no atom", lambda r: pct(r[lv]["no_atom"])),
        (f"{unit}s holding a whole functional group", lambda r: pct(r[lv]["whole_functional_group"])),
        (f"{unit}s that cut through an atom", lambda r: f'{100 * r[lv]["cuts_atom"]:.2f}%'),
        ("multi-character atoms split", lambda r: f'{pct(s(r)["learned"]["atoms_split"])} ({pct(s(r)["random"]["atoms_split"])})'),
        ("preference for breakable bonds", lambda r: f'{r[lv]["brics_preference"]:+.2f}'),
        ("5- and 6-rings kept whole", lambda r: f'{pct(c(r)["learned"]["rings_whole"])} ({pct(c(r)["random"]["rings_whole"])})'),
    ])
    table(f"Functional groups kept whole in one {unit} ({size}), random cuts in brackets", [
        (g.replace("_", " "), lambda r, g=g: f'{pct(c(r)["learned"]["fg_whole"][g])} ({pct(c(r)["random"]["fg_whole"][g])})') for g in FGS])


level("pieces", "piece")
level("groups", "group")
table(f"Objective metric and representation ({size})", [
    ("held-out metric", lambda r: ", ".join(f"{k} {v:.4f}" for k, v in r["objective"].items() if isinstance(v, float))),
    ("nearest-neighbour Tanimoto", lambda r: f'{r["representation"]["nn_tanimoto"]:.3f}'),
    ("probe R2, heavy atoms / rings", lambda r: f'{r["representation"]["r2_heavy_atoms"]:.2f} / {r["representation"]["r2_rings"]:.2f}'),
])
rep = R[0]["representation"]
print(f'\nNearest-neighbour Tanimoto: random {rep["nn_tanimoto_random"]:.3f}, best possible {rep["nn_tanimoto_best"]:.3f}.')
for f in sorted(d.glob(f"shared_*{size}*.json")):
    print("Pieces used by all four models:", json.loads(f.read_text())["pieces_in_all"])
