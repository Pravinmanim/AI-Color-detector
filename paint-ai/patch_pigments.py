import sys; sys.path.insert(0, '.')
from simulator.dataset_generator import generate_dataset
import csv, os

path = generate_dataset(n_dirichlet=5000, n_repeats=1,
                        include_edges=True, include_tints=True,
                        n_lot_batches=3, seed=42)

with open(path) as f:
    rows = list(csv.DictReader(f))

print(f"Generated {len(rows)} rows")
print(f"File: {path}")
print(f"File size: {os.path.getsize(path)//1024} KB")

Ls = [float(r["L"]) for r in rows]
print(f"L* range: {min(Ls):.1f} - {max(Ls):.1f}")

import collections
lots = collections.Counter(r["paint_batch"] for r in rows)
print(f"Lot distribution: {dict(lots)}")
sources = set(r["source"] for r in rows)
print(f"Sources: {sources}")
