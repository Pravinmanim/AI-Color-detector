"""
Colour-space conversions: sRGB ↔ XYZ ↔ CIELAB.

All conversions use:
  • sRGB with the IEC 61966-2-1 transfer function
  • D65 illuminant, CIE 1931 2° observer
  • Reference white Xn=0.95047, Yn=1.0, Zn=1.08883

Functions are pure (no global state) and work on single colours
(tuples/lists of 3 floats).
"""

from __future__ import annotations
import math
from typing import Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import D65_XN, D65_YN, D65_ZN

# ──────────────────────────────────────────────
# sRGB ↔ Linear RGB
# ──────────────────────────────────────────────

def _srgb_compand(c: float) -> float:
    """Linear → sRGB companding (IEC 61966-2-1)."""
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def _srgb_inverse_compand(c: float) -> float:
    """sRGB → linear (inverse companding)."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def srgb_to_linear(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Convert sRGB [0-1] to linear RGB [0-1]."""
    return (
        _srgb_inverse_compand(r),
        _srgb_inverse_compand(g),
        _srgb_inverse_compand(b),
    )


def linear_to_srgb(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Convert linear RGB [0-1] to sRGB [0-1]."""
    return (
        _srgb_compand(r),
        _srgb_compand(g),
        _srgb_compand(b),
    )


# ──────────────────────────────────────────────
# Linear RGB ↔ XYZ  (sRGB matrix, D65)
# ──────────────────────────────────────────────
# Matrix from IEC 61966-2-1 / Wikipedia "sRGB"

def linear_rgb_to_xyz(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Linear RGB [0-1] → CIE XYZ (D65)."""
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return (X, Y, Z)


def xyz_to_linear_rgb(X: float, Y: float, Z: float) -> Tuple[float, float, float]:
    """CIE XYZ (D65) → linear RGB [0-1]."""
    r =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    return (r, g, b)


# ──────────────────────────────────────────────
# XYZ ↔ CIELAB  (D65, 2°)
# ──────────────────────────────────────────────
_EPSILON = 216.0 / 24389.0   # ≈ 0.008856
_KAPPA = 24389.0 / 27.0      # ≈ 903.3


def _lab_f(t: float) -> float:
    """CIE Lab forward transform helper."""
    if t > _EPSILON:
        return t ** (1.0 / 3.0)
    return (_KAPPA * t + 16.0) / 116.0


def _lab_f_inv(t: float) -> float:
    """CIE Lab inverse transform helper."""
    t3 = t * t * t
    if t3 > _EPSILON:
        return t3
    return (116.0 * t - 16.0) / _KAPPA


def xyz_to_lab(
    X: float, Y: float, Z: float,
    Xn: float = D65_XN, Yn: float = D65_YN, Zn: float = D65_ZN,
) -> Tuple[float, float, float]:
    """CIE XYZ → CIELAB (D65, 2° by default)."""
    fx = _lab_f(X / Xn)
    fy = _lab_f(Y / Yn)
    fz = _lab_f(Z / Zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return (L, a, b)


def lab_to_xyz(
    L: float, a: float, b: float,
    Xn: float = D65_XN, Yn: float = D65_YN, Zn: float = D65_ZN,
) -> Tuple[float, float, float]:
    """CIELAB → CIE XYZ (D65, 2° by default)."""
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    X = _lab_f_inv(fx) * Xn
    Y = _lab_f_inv(fy) * Yn
    Z = _lab_f_inv(fz) * Zn
    return (X, Y, Z)


# ──────────────────────────────────────────────
# Convenience: sRGB 0-255 ↔ Lab
# ──────────────────────────────────────────────

def srgb255_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """sRGB [0-255] integer → CIELAB."""
    lr, lg, lb = srgb_to_linear(r / 255.0, g / 255.0, b / 255.0)
    X, Y, Z = linear_rgb_to_xyz(lr, lg, lb)
    return xyz_to_lab(X, Y, Z)


def lab_to_srgb255(L: float, a: float, b: float) -> Tuple[int, int, int]:
    """
    CIELAB → sRGB [0-255] integer, clamped to [0, 255].
    Returns a tuple and a boolean `in_gamut`.
    """
    X, Y, Z = lab_to_xyz(L, a, b)
    lr, lg, lb = xyz_to_linear_rgb(X, Y, Z)
    sr, sg, sb = linear_to_srgb(lr, lg, lb)
    # Clamp
    r = int(round(max(0.0, min(1.0, sr)) * 255))
    g = int(round(max(0.0, min(1.0, sg)) * 255))
    b = int(round(max(0.0, min(1.0, sb)) * 255))
    return (r, g, b)


def lab_to_srgb255_checked(
    L: float, a: float, b: float,
) -> Tuple[Tuple[int, int, int], bool]:
    """
    CIELAB → sRGB [0-255] with gamut check.
    Returns ((R,G,B), in_gamut).
    """
    X, Y, Z = lab_to_xyz(L, a, b)
    lr, lg, lb = xyz_to_linear_rgb(X, Y, Z)
    sr, sg, sb = linear_to_srgb(lr, lg, lb)
    in_gamut = all(0.0 <= v <= 1.0 for v in (sr, sg, sb))
    r = int(round(max(0.0, min(1.0, sr)) * 255))
    g = int(round(max(0.0, min(1.0, sg)) * 255))
    b = int(round(max(0.0, min(1.0, sb)) * 255))
    return ((r, g, b), in_gamut)


# ──────────────────────────────────────────────
# HEX helpers
# ──────────────────────────────────────────────

def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Parse '#RRGGBB' or 'RRGGBB' → (R, G, B) 0-255."""
    h = hex_str.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    if len(h) != 6:
        raise ValueError(f"Invalid hex colour: {hex_str!r}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """(R, G, B) 0-255 → '#RRGGBB'."""
    return f"#{r:02X}{g:02X}{b:02X}"


# ──────────────────────────────────────────────
# RGB ↔ HSL
# ──────────────────────────────────────────────

def rgb_to_hsl(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """(R, G, B) 0-255 → (H 0-360, S 0-100, L 0-100)."""
    r1, g1, b1 = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r1, g1, b1)
    cmin = min(r1, g1, b1)
    delta = cmax - cmin
    # Lightness
    light = (cmax + cmin) / 2.0
    if delta == 0:
        hue = 0.0
        sat = 0.0
    else:
        # Saturation
        sat = delta / (1.0 - abs(2.0 * light - 1.0))
        # Hue
        if cmax == r1:
            hue = 60.0 * (((g1 - b1) / delta) % 6.0)
        elif cmax == g1:
            hue = 60.0 * ((b1 - r1) / delta + 2.0)
        else:
            hue = 60.0 * ((r1 - g1) / delta + 4.0)
    if hue < 0:
        hue += 360.0
    return (round(hue, 1), round(sat * 100, 1), round(light * 100, 1))


def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    """(H 0-360, S 0-100, L 0-100) → (R, G, B) 0-255."""
    s1 = s / 100.0
    l1 = l / 100.0
    c = (1.0 - abs(2.0 * l1 - 1.0)) * s1
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = l1 - c / 2.0
    if h < 60:
        r1, g1, b1 = c, x, 0.0
    elif h < 120:
        r1, g1, b1 = x, c, 0.0
    elif h < 180:
        r1, g1, b1 = 0.0, c, x
    elif h < 240:
        r1, g1, b1 = 0.0, x, c
    elif h < 300:
        r1, g1, b1 = x, 0.0, c
    else:
        r1, g1, b1 = c, 0.0, x
    return (
        int(round((r1 + m) * 255)),
        int(round((g1 + m) * 255)),
        int(round((b1 + m) * 255)),
    )
