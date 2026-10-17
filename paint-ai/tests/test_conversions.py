"""
Unit tests for colour-space conversions.

Reference values were computed with known published data:
  - sRGB → Lab for pure red (255,0,0):  L*≈53.23, a*≈80.11, b*≈67.22
  - sRGB → Lab for pure white (255,255,255): L*=100, a*≈0, b*≈0
  - sRGB → Lab for pure black (0,0,0): L*=0, a*=0, b*=0

Cross-checked against:
  - Lindbloom's colour calculator (brucelindbloom.com)
  - The colormath Python library (for verification only; not used in production)

DATA SOURCE: PUBLIC-MEASURED (published sRGB/Lab reference values)
"""

import pytest
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from color.conversions import (
    srgb_to_linear, linear_to_srgb,
    linear_rgb_to_xyz, xyz_to_linear_rgb,
    xyz_to_lab, lab_to_xyz,
    srgb255_to_lab, lab_to_srgb255, lab_to_srgb255_checked,
    hex_to_rgb, rgb_to_hex,
    rgb_to_hsl, hsl_to_rgb,
)


class TestSRGBLinear:
    """sRGB companding round-trip tests."""

    def test_black(self):
        assert srgb_to_linear(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)

    def test_white(self):
        r, g, b = srgb_to_linear(1.0, 1.0, 1.0)
        assert abs(r - 1.0) < 1e-10
        assert abs(g - 1.0) < 1e-10
        assert abs(b - 1.0) < 1e-10

    def test_round_trip(self):
        """sRGB → linear → sRGB must round-trip."""
        for val in [0.0, 0.01, 0.04045, 0.5, 0.735, 1.0]:
            lr, lg, lb = srgb_to_linear(val, val, val)
            sr, sg, sb = linear_to_srgb(lr, lg, lb)
            assert abs(sr - val) < 1e-6, f"Round trip failed for {val}"

    def test_mid_grey(self):
        """sRGB 0.5 → linear ≈ 0.2140."""
        r, _, _ = srgb_to_linear(0.5, 0.0, 0.0)
        assert abs(r - 0.214041) < 0.0001


class TestXYZ:
    """Linear RGB ↔ XYZ tests."""

    def test_white_xyz(self):
        """D65 white: linear (1,1,1) → XYZ ≈ (0.9505, 1.0, 1.0890)."""
        X, Y, Z = linear_rgb_to_xyz(1.0, 1.0, 1.0)
        assert abs(X - 0.95047) < 0.001
        assert abs(Y - 1.0) < 0.001
        assert abs(Z - 1.08883) < 0.001

    def test_black_xyz(self):
        X, Y, Z = linear_rgb_to_xyz(0.0, 0.0, 0.0)
        assert X == 0.0 and Y == 0.0 and Z == 0.0

    def test_round_trip(self):
        """Linear RGB → XYZ → linear RGB must round-trip for in-gamut."""
        for rgb in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.3, 0.7)]:
            X, Y, Z = linear_rgb_to_xyz(*rgb)
            r, g, b = xyz_to_linear_rgb(X, Y, Z)
            assert abs(r - rgb[0]) < 1e-6
            assert abs(g - rgb[1]) < 1e-6
            assert abs(b - rgb[2]) < 1e-6


class TestLab:
    """XYZ ↔ Lab tests."""

    def test_white_lab(self):
        """D65 white → L*=100, a*≈0, b*≈0."""
        L, a, b = xyz_to_lab(0.95047, 1.0, 1.08883)
        assert abs(L - 100.0) < 0.01
        assert abs(a) < 0.01
        assert abs(b) < 0.01

    def test_black_lab(self):
        """Black → L*=0, a*=0, b*=0."""
        L, a, b = xyz_to_lab(0.0, 0.0, 0.0)
        assert L == 0.0 and a == 0.0 and b == 0.0

    def test_round_trip(self):
        L, a, b = 53.23, 80.11, 67.22
        X, Y, Z = lab_to_xyz(L, a, b)
        L2, a2, b2 = xyz_to_lab(X, Y, Z)
        assert abs(L2 - L) < 0.01
        assert abs(a2 - a) < 0.01
        assert abs(b2 - b) < 0.01


class TestSRGB255Lab:
    """End-to-end sRGB [0-255] ↔ Lab tests."""

    def test_pure_red(self):
        """Red (255,0,0) → Lab ≈ (53.23, 80.11, 67.22)."""
        L, a, b = srgb255_to_lab(255, 0, 0)
        assert abs(L - 53.23) < 0.1
        assert abs(a - 80.11) < 0.1
        assert abs(b - 67.22) < 0.1

    def test_pure_green_css(self):
        """CSS green (0,128,0) → Lab ≈ (46.23, -51.70, 49.90)."""
        L, a, b = srgb255_to_lab(0, 128, 0)
        assert abs(L - 46.23) < 0.5
        assert abs(a - (-51.70)) < 0.5
        assert abs(b - 49.90) < 0.5

    def test_pure_blue(self):
        """Blue (0,0,255) → Lab ≈ (32.30, 79.20, -107.86)."""
        L, a, b = srgb255_to_lab(0, 0, 255)
        assert abs(L - 32.30) < 0.1
        assert abs(a - 79.20) < 0.1
        assert abs(b - (-107.86)) < 0.1

    def test_white(self):
        L, a, b = srgb255_to_lab(255, 255, 255)
        assert abs(L - 100.0) < 0.01
        assert abs(a) < 0.05
        assert abs(b) < 0.05

    def test_black(self):
        L, a, b = srgb255_to_lab(0, 0, 0)
        assert L == 0.0

    def test_round_trip_red(self):
        """sRGB(255,0,0) → Lab → sRGB must round-trip."""
        L, a, b = srgb255_to_lab(255, 0, 0)
        r, g, bl = lab_to_srgb255(L, a, b)
        assert (r, g, bl) == (255, 0, 0)

    def test_round_trip_many(self):
        """Round-trip for a grid of sRGB values."""
        for r in range(0, 256, 51):
            for g in range(0, 256, 51):
                for b in range(0, 256, 51):
                    L, a, bl = srgb255_to_lab(r, g, b)
                    r2, g2, b2 = lab_to_srgb255(L, a, bl)
                    assert abs(r2 - r) <= 1, f"R fail: {r} -> {r2}"
                    assert abs(g2 - g) <= 1, f"G fail: {g} -> {g2}"
                    assert abs(b2 - b) <= 1, f"B fail: {b} -> {b2}"


class TestGamutCheck:
    """Gamut checking tests."""

    def test_in_gamut(self):
        rgb, in_gamut = lab_to_srgb255_checked(50, 0, 0)
        assert in_gamut is True

    def test_out_of_gamut(self):
        """Very saturated Lab should be out of sRGB gamut."""
        _, in_gamut = lab_to_srgb255_checked(50, 128, 128)
        assert in_gamut is False


class TestHex:
    """Hex parsing and formatting."""

    def test_parse_hash(self):
        assert hex_to_rgb("#FF0000") == (255, 0, 0)

    def test_parse_no_hash(self):
        assert hex_to_rgb("00FF00") == (0, 255, 0)

    def test_parse_short(self):
        assert hex_to_rgb("#F00") == (255, 0, 0)

    def test_format(self):
        assert rgb_to_hex(255, 0, 0) == "#FF0000"
        assert rgb_to_hex(0, 128, 255) == "#0080FF"

    def test_round_trip(self):
        for hex_val in ["#000000", "#FFFFFF", "#1A2B3C", "#FF8800"]:
            r, g, b = hex_to_rgb(hex_val)
            assert rgb_to_hex(r, g, b) == hex_val

    def test_invalid(self):
        with pytest.raises(ValueError):
            hex_to_rgb("#ZZZZZZ")

    def test_invalid_length(self):
        with pytest.raises(ValueError):
            hex_to_rgb("#FFFF")


class TestHSL:
    """RGB ↔ HSL conversion tests."""

    def test_red(self):
        h, s, l = rgb_to_hsl(255, 0, 0)
        assert abs(h - 0.0) < 0.1
        assert abs(s - 100.0) < 0.1
        assert abs(l - 50.0) < 0.1

    def test_green(self):
        h, s, l = rgb_to_hsl(0, 255, 0)
        assert abs(h - 120.0) < 0.1

    def test_blue(self):
        h, s, l = rgb_to_hsl(0, 0, 255)
        assert abs(h - 240.0) < 0.1

    def test_white(self):
        h, s, l = rgb_to_hsl(255, 255, 255)
        assert abs(l - 100.0) < 0.1
        assert abs(s - 0.0) < 0.1

    def test_black(self):
        h, s, l = rgb_to_hsl(0, 0, 0)
        assert abs(l - 0.0) < 0.1

    def test_round_trip(self):
        for rgb in [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                     (128, 64, 32), (200, 200, 200)]:
            h, s, l = rgb_to_hsl(*rgb)
            r, g, b = hsl_to_rgb(h, s, l)
            assert abs(r - rgb[0]) <= 1
            assert abs(g - rgb[1]) <= 1
            assert abs(b - rgb[2]) <= 1
