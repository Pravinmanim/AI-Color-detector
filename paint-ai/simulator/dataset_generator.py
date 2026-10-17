"""
Simulated dataset generator for R:B:Y:W paint recipes.

Generates CSV files with columns matching the real-data template,
plus noise (dispensing error, scan noise, lot-to-lot shift).

Every row is labelled  source = SIMULATED.

Sampling strategy:
  1. Uniform Dirichlet: random 4-component mixtures
  2. Edge recipes: single pigment, two-pigment blends
  3. Tint series: one colour + white at 10 %-steps
  4. Fine grid near interesting regions

DATA SOURCE: SIMULATED
"""

from __future__ import annotations
import csv
import math
import os
import random
from typing import List, Optional, Tuple

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    DISPENSE_ERROR_PCT, SCAN_NOISE_L, SCAN_NOISE_AB,
    LOT_SHIFT_SIGMA, SRC_SIMULATED,
)
from simulator.kubelka_munk import recipe_to_lab, recipe_to_lab_with_noise


# ──────────────────────────────────────────────
# Recipe sampling helpers
# ──────────────────────────────────────────────

def _dirichlet_sample(rng: random.Random, k: int = 4, alpha: float = 1.0) -> List[float]:
    """
    Sample from a symmetric Dirichlet(alpha) distribution.
    Returns k fractions summing to 1.0, then scaled to 100.
    """
    # Gamma samples
    samples = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    total = sum(samples)
    return [s / total * 100.0 for s in samples]


def _edge_recipes() -> List[Tuple[float, float, float, float]]:
    """Pure pigments and two-pigment blends at 10 % steps."""
    recipes = []
    # Pure pigments
    recipes.append((100, 0, 0, 0))
    recipes.append((0, 100, 0, 0))
    recipes.append((0, 0, 100, 0))
    recipes.append((0, 0, 0, 100))

    # Two-pigment blends
    pigments = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    for i in range(4):
        for j in range(i + 1, 4):
            for pct in range(10, 100, 10):  # 10, 20, ... 90
                recipe = [0.0, 0.0, 0.0, 0.0]
                recipe[i] = float(pct)
                recipe[j] = float(100 - pct)
                recipes.append(tuple(recipe))

    return recipes


def _tint_series() -> List[Tuple[float, float, float, float]]:
    """Each colour tinted with white: 5 %-steps from 5 % to 95 %."""
    recipes = []
    # Red + white, Blue + white, Yellow + white
    for colour_idx in range(3):
        for colour_pct in range(5, 100, 5):
            recipe = [0.0, 0.0, 0.0, 0.0]
            recipe[colour_idx] = float(colour_pct)
            recipe[3] = float(100 - colour_pct)
            recipes.append(tuple(recipe))
    return recipes


def generate_dataset(
    n_dirichlet: int = 5000,
    n_repeats: int = 1,
    include_edges: bool = True,
    include_tints: bool = True,
    n_lot_batches: int = 3,
    seed: int = 42,
    output_path: Optional[str] = None,
    add_noise: bool = True,
) -> str:
    """
    Generate a full simulated dataset and write to CSV.

    Parameters
    ----------
    n_dirichlet : int
        Number of random Dirichlet recipes.
    n_repeats : int
        Repeat measurements per recipe (for noise estimation).
    include_edges : bool
        Include pure-pigment and two-pigment edge recipes.
    include_tints : bool
        Include tint series.
    n_lot_batches : int
        Number of simulated paint lot batches (each with a random shift).
    seed : int
        Random seed for reproducibility.
    output_path : str or None
        Output CSV path. If None, uses data/simulated/simulated_dataset.csv.
    add_noise : bool
        Whether to add dispensing + scan + lot noise.

    Returns
    -------
    str : path to the generated CSV file.
    """
    rng = random.Random(seed)

    if output_path is None:
        base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "simulated")
        os.makedirs(base_dir, exist_ok=True)
        output_path = os.path.join(base_dir, "simulated_dataset.csv")

    # Collect all target recipes
    all_recipes: List[Tuple[float, float, float, float]] = []

    # Dirichlet samples (various alpha values for diversity)
    for alpha in [0.3, 0.5, 1.0, 2.0]:
        count = n_dirichlet // 4
        for _ in range(count):
            r, b, y, w = _dirichlet_sample(rng, alpha=alpha)
            all_recipes.append((r, b, y, w))

    if include_edges:
        all_recipes.extend(_edge_recipes())

    if include_tints:
        all_recipes.extend(_tint_series())

    # Generate lot shifts
    lot_shifts = []
    for _ in range(n_lot_batches):
        shift = (
            rng.gauss(0, LOT_SHIFT_SIGMA),
            rng.gauss(0, LOT_SHIFT_SIGMA * 0.7),
            rng.gauss(0, LOT_SHIFT_SIGMA * 0.7),
        )
        lot_shifts.append(shift)

    # Write CSV
    fieldnames = [
        "sample_id", "recipe_id", "repeat_id",
        "red_pct", "blue_pct", "yellow_pct", "white_pct",
        "L", "a", "b",
        "paint_batch", "temperature_c", "measurement_condition",
        "source",
    ]

    sample_id = 0
    rows_written = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for recipe_id, (r, b, y, w) in enumerate(all_recipes):
            # Assign to a random lot
            lot_idx = rng.randint(0, n_lot_batches - 1)
            lot_name = f"LOT-{lot_idx + 1:03d}"
            lot_shift = lot_shifts[lot_idx]

            for repeat in range(n_repeats):
                if add_noise:
                    (L, a_val, b_val), (ar, ab, ay, aw) = recipe_to_lab_with_noise(
                        r, b, y, w,
                        dispense_error=DISPENSE_ERROR_PCT,
                        scan_noise_L=SCAN_NOISE_L,
                        scan_noise_ab=SCAN_NOISE_AB,
                        lot_shift=lot_shift,
                        rng=rng,
                    )
                else:
                    L, a_val, b_val = recipe_to_lab(r, b, y, w)
                    ar, ab, ay, aw = r, b, y, w

                writer.writerow({
                    "sample_id": f"SIM-{sample_id:06d}",
                    "recipe_id": f"R-{recipe_id:05d}",
                    "repeat_id": repeat,
                    "red_pct": round(ar, 2),
                    "blue_pct": round(ab, 2),
                    "yellow_pct": round(ay, 2),
                    "white_pct": round(aw, 2),
                    "L": round(L, 4),
                    "a": round(a_val, 4),
                    "b": round(b_val, 4),
                    "paint_batch": lot_name,
                    "temperature_c": round(22.0 + rng.gauss(0, 1.5), 1),
                    "measurement_condition": "SIMULATED-KM",
                    "source": SRC_SIMULATED,
                })
                sample_id += 1
                rows_written += 1

    print(f"[SIMULATED] Generated {rows_written} samples -> {output_path}")
    return output_path


def generate_repeats_dataset(
    n_recipes: int = 200,
    repeats_per_recipe: int = 3,
    seed: int = 99,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a dataset with repeated measurements of the same recipes.
    Useful for estimating measurement noise.

    DATA SOURCE: SIMULATED
    """
    return generate_dataset(
        n_dirichlet=n_recipes,
        n_repeats=repeats_per_recipe,
        include_edges=False,
        include_tints=False,
        n_lot_batches=1,
        seed=seed,
        output_path=output_path,
        add_noise=True,
    )


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate simulated paint dataset")
    parser.add_argument("--n", type=int, default=5000, help="Number of Dirichlet recipes")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per recipe")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--no-noise", action="store_true", help="Disable noise")
    args = parser.parse_args()

    generate_dataset(
        n_dirichlet=args.n,
        n_repeats=args.repeats,
        seed=args.seed,
        output_path=args.output,
        add_noise=not args.no_noise,
    )
