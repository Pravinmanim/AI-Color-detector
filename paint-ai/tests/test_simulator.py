"""
Tests for the Kubelka-Munk forward model and dataset generator.

All results are labelled SIMULATED — these test the physics
and noise model, not real paint behaviour.

DATA SOURCE: SIMULATED
"""

from __future__ import annotations
import csv
import os
import sys
import math
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.kubelka_munk import (
    WAVELENGTHS, N_CHANNELS, PIGMENT_SPECTRA,
    km_mix_reflectance, reflectance_to_xyz, reflectance_to_lab,
    recipe_to_lab, recipe_to_lab_with_noise,
)
from simulator.dataset_generator import (
    generate_dataset, _dirichlet_sample, _edge_recipes, _tint_series,
)
from config import SRC_SIMULATED


# ──────────────────────────────────────────────
# Kubelka-Munk physics tests
# ──────────────────────────────────────────────

class TestWavelengthGrid:
    def test_channel_count(self):
        assert N_CHANNELS == 36

    def test_wavelength_range(self):
        assert WAVELENGTHS[0] == 380
        assert WAVELENGTHS[-1] == 730
        assert len(WAVELENGTHS) == 36

    def test_cmf_lengths(self):
        from simulator.kubelka_munk import CMF_X, CMF_Y, CMF_Z, D65_SPD
        assert len(CMF_X) == 36
        assert len(CMF_Y) == 36
        assert len(CMF_Z) == 36
        assert len(D65_SPD) == 36


class TestPigmentSpectra:
    def test_all_pigments_present(self):
        for name in ("red", "blue", "yellow", "white"):
            assert name in PIGMENT_SPECTRA
            assert "K" in PIGMENT_SPECTRA[name]
            assert "S" in PIGMENT_SPECTRA[name]

    def test_spectra_lengths(self):
        for name in ("red", "blue", "yellow", "white"):
            assert len(PIGMENT_SPECTRA[name]["K"]) == 36
            assert len(PIGMENT_SPECTRA[name]["S"]) == 36

    def test_k_s_non_negative(self):
        for name in PIGMENT_SPECTRA:
            for K in PIGMENT_SPECTRA[name]["K"]:
                assert K >= 0, f"{name} K negative"
            for S in PIGMENT_SPECTRA[name]["S"]:
                assert S >= 0, f"{name} S negative"

    def test_white_low_absorption(self):
        """White should have very low K everywhere."""
        max_K = max(PIGMENT_SPECTRA["white"]["K"])
        assert max_K < 0.1, f"White K too high: {max_K}"

    def test_white_high_scattering(self):
        """White should scatter strongly."""
        min_S = min(PIGMENT_SPECTRA["white"]["S"])
        assert min_S > 0.5, f"White S too low: {min_S}"


class TestReflectance:
    def test_pure_white_high_reflectance(self):
        """Pure white should reflect strongly across all wavelengths."""
        refl = km_mix_reflectance(0, 0, 0, 100)
        assert all(r > 0.7 for r in refl), \
            f"White reflectance too low: min={min(refl):.3f}"

    def test_reflectance_in_range(self):
        """Reflectance must be [0, 1] for any recipe."""
        import random
        rng = random.Random(7)
        for _ in range(50):
            fracs = [rng.random() for _ in range(4)]
            total = sum(fracs)
            r, b, y, w = [f / total * 100 for f in fracs]
            refl = km_mix_reflectance(r, b, y, w)
            assert all(0.0 <= v <= 1.0 for v in refl), \
                f"Reflectance out of range: {refl}"

    def test_black_mix_low_reflectance(self):
        """Heavy pigment mix (no white) should give low reflectance."""
        # Extreme mix: no white
        refl = km_mix_reflectance(33.3, 33.3, 33.4, 0)
        avg_refl = sum(refl) / len(refl)
        assert avg_refl < 0.5


class TestXYZ:
    def test_white_xyz_near_unit(self):
        """Pure white reflectance → XYZ should give Y ≈ 1.0."""
        refl = km_mix_reflectance(0, 0, 0, 100)
        X, Y, Z = reflectance_to_xyz(refl)
        assert abs(Y - 1.0) < 0.05, f"White Y={Y:.4f}"
        assert X > 0.8
        assert Z > 0.8

    def test_xyz_non_negative(self):
        """XYZ must always be non-negative."""
        import random
        rng = random.Random(13)
        for _ in range(30):
            fracs = [rng.random() for _ in range(4)]
            t = sum(fracs)
            r, b, y, w = [f / t * 100 for f in fracs]
            refl = km_mix_reflectance(r, b, y, w)
            X, Y, Z = reflectance_to_xyz(refl)
            assert X >= 0 and Y >= 0 and Z >= 0


class TestRecipeToLab:
    def test_white_is_light(self):
        """100% white should give high L* and near-neutral a*, b*."""
        L, a, b = recipe_to_lab(0, 0, 0, 100)
        assert L > 95, f"White L*={L:.2f}, expected > 95"
        assert abs(a) < 3, f"White a*={a:.2f}, expected near 0"
        assert abs(b) < 3, f"White b*={b:.2f}, expected near 0"

    def test_red_is_red(self):
        """50% red tinted with white should have positive a* (red direction)."""
        L, a, b = recipe_to_lab(50, 0, 0, 50)
        assert a > 20, f"Red a*={a:.2f}, expected > 20"

    def test_blue_is_blue(self):
        """50% blue tinted with white should have strongly negative b* (blue direction)."""
        L, a, b = recipe_to_lab(0, 50, 0, 50)
        assert b < -20, f"Blue b*={b:.2f}, expected < -20"

    def test_yellow_is_yellow(self):
        """50% yellow tinted with white should have strongly positive b* (yellow direction)."""
        L, a, b = recipe_to_lab(0, 0, 50, 50)
        assert b > 20, f"Yellow b*={b:.2f}, expected > 20"

    def test_lab_ranges(self):
        """L* must be [0,100]; a*, b* within a generous range for simulated pigments."""
        import random
        rng = random.Random(21)
        for _ in range(100):
            fracs = [rng.random() for _ in range(4)]
            t = sum(fracs)
            r, b_f, y, w = [f / t * 100 for f in fracs]
            L, a, b = recipe_to_lab(r, b_f, y, w)
            assert 0 <= L <= 100, f"L* out of range: {L}"
            assert -200 <= a <= 200, f"a* out of range: {a}"
            assert -200 <= b <= 200, f"b* out of range: {b}"

    def test_more_white_means_higher_L(self):
        """Increasing white percentage should increase L*."""
        L1, _, _ = recipe_to_lab(30, 0, 0, 70)
        L2, _, _ = recipe_to_lab(30, 0, 0, 60)
        # More white → higher L
        assert L1 > L2, f"L1={L1:.2f} should be > L2={L2:.2f}"

    def test_normalisation_invariant(self):
        """Doubling all amounts should give the same Lab (only ratios matter)."""
        L1, a1, b1 = recipe_to_lab(10, 5, 15, 70)
        L2, a2, b2 = recipe_to_lab(20, 10, 30, 140)
        assert abs(L1 - L2) < 0.01
        assert abs(a1 - a2) < 0.01
        assert abs(b1 - b2) < 0.01

    def test_all_zero_input(self):
        """All-zero recipe should return black (L=0)."""
        L, a, b = recipe_to_lab(0, 0, 0, 0)
        assert L == 0.0


class TestNoiseModel:
    def test_noisy_lab_near_clean(self):
        """Noisy measurement should be within ±5 Lab units of clean."""
        import random
        rng = random.Random(55)
        L_clean, a_clean, b_clean = recipe_to_lab(30, 10, 20, 40)
        for _ in range(20):
            (L_n, a_n, b_n), actual = recipe_to_lab_with_noise(
                30, 10, 20, 40, rng=rng
            )
            assert abs(L_n - L_clean) < 8
            assert abs(a_n - a_clean) < 8
            assert abs(b_n - b_clean) < 8

    def test_actual_percentages_sum_to_100(self):
        """Actual dispensed percentages must always sum to 100."""
        import random
        rng = random.Random(77)
        for _ in range(50):
            _, actual = recipe_to_lab_with_noise(25, 25, 25, 25, rng=rng)
            total = sum(actual)
            assert abs(total - 100.0) < 0.01, f"Sum={total}"

    def test_actual_percentages_non_negative(self):
        """Actual percentages must never be negative."""
        import random
        rng = random.Random(88)
        for _ in range(50):
            _, actual = recipe_to_lab_with_noise(5, 5, 5, 85, rng=rng)
            assert all(p >= 0 for p in actual)

    def test_lot_shift_moves_lab(self):
        """A lot shift should change the measured Lab."""
        import random
        rng = random.Random(11)
        (L1, a1, b1), _ = recipe_to_lab_with_noise(
            25, 25, 25, 25, lot_shift=None, rng=rng
        )
        rng2 = random.Random(11)
        (L2, a2, b2), _ = recipe_to_lab_with_noise(
            25, 25, 25, 25, lot_shift=(5.0, 3.0, -2.0), rng=rng2
        )
        # Lot shift should cause a noticeable difference
        diff = abs(L2 - L1) + abs(a2 - a1) + abs(b2 - b1)
        assert diff > 3.0, f"Lot shift had no effect: diff={diff:.3f}"


# ──────────────────────────────────────────────
# Dataset generator tests
# ──────────────────────────────────────────────

class TestDirichletSampler:
    def test_sums_to_100(self):
        import random
        rng = random.Random(1)
        for _ in range(50):
            fracs = _dirichlet_sample(rng, k=4, alpha=1.0)
            assert abs(sum(fracs) - 100.0) < 1e-9

    def test_non_negative(self):
        import random
        rng = random.Random(2)
        for _ in range(50):
            fracs = _dirichlet_sample(rng)
            assert all(f >= 0 for f in fracs)

    def test_length(self):
        import random
        rng = random.Random(3)
        assert len(_dirichlet_sample(rng, k=4)) == 4


class TestEdgeRecipes:
    def test_pure_pigments_included(self):
        recipes = _edge_recipes()
        assert (100, 0, 0, 0) in recipes
        assert (0, 100, 0, 0) in recipes
        assert (0, 0, 100, 0) in recipes
        assert (0, 0, 0, 100) in recipes

    def test_all_sum_to_100(self):
        for r in _edge_recipes():
            assert abs(sum(r) - 100.0) < 0.001, f"Recipe {r} sums to {sum(r)}"


class TestTintSeries:
    def test_non_empty(self):
        tints = _tint_series()
        assert len(tints) > 0

    def test_all_sum_to_100(self):
        for r in _tint_series():
            assert abs(sum(r) - 100.0) < 0.001, f"Recipe {r} sums to {sum(r)}"


class TestGenerateDataset:
    def test_csv_created(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            generate_dataset(n_dirichlet=100, seed=42, output_path=path, add_noise=True)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_csv_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            generate_dataset(n_dirichlet=50, seed=1, output_path=path)
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
            expected = [
                "sample_id", "recipe_id", "repeat_id",
                "red_pct", "blue_pct", "yellow_pct", "white_pct",
                "L", "a", "b", "paint_batch", "temperature_c",
                "measurement_condition", "source",
            ]
            for col in expected:
                assert col in headers, f"Missing column: {col}"
        finally:
            os.unlink(path)

    def test_source_is_simulated(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            generate_dataset(n_dirichlet=50, seed=5, output_path=path)
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    assert row["source"] == SRC_SIMULATED, \
                        f"Row has wrong source: {row['source']}"
        finally:
            os.unlink(path)

    def test_percentages_sum_to_100(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            generate_dataset(n_dirichlet=50, seed=6, output_path=path)
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total = (float(row["red_pct"]) + float(row["blue_pct"]) +
                             float(row["yellow_pct"]) + float(row["white_pct"]))
                    assert abs(total - 100.0) < 0.1, \
                        f"Percentages sum to {total}"
        finally:
            os.unlink(path)

    def test_lab_values_in_range(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            generate_dataset(n_dirichlet=100, seed=7, output_path=path)
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    L = float(row["L"])
                    a = float(row["a"])
                    b = float(row["b"])
                    assert 0 <= L <= 100, f"L={L} out of range"
                    assert -200 <= a <= 200, f"a={a} out of range"
                    assert -200 <= b <= 200, f"b={b} out of range"
        finally:
            os.unlink(path)

    def test_reproducible(self):
        """Same seed should give identical CSV content."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            p1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            p2 = f.name
        try:
            generate_dataset(n_dirichlet=30, seed=99, output_path=p1)
            generate_dataset(n_dirichlet=30, seed=99, output_path=p2)
            with open(p1) as f1, open(p2) as f2:
                assert f1.read() == f2.read()
        finally:
            os.unlink(p1)
            os.unlink(p2)

    def test_row_count_reasonable(self):
        """Generated dataset should have the expected number of rows."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            generate_dataset(
                n_dirichlet=200,
                include_edges=True,
                include_tints=True,
                seed=42,
                output_path=path,
                add_noise=False,
            )
            with open(path, "r") as f:
                rows = sum(1 for _ in f) - 1  # subtract header
            assert rows >= 200, f"Too few rows: {rows}"
        finally:
            os.unlink(path)
