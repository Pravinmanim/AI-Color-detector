"""
CIEDE2000 colour-difference formula.

Implements the formula described in:
  Sharma, G., Wu, W., & Dalal, E. N. (2005).
  "The CIEDE2000 color-difference formula: Implementation notes,
   supplementary test data, and mathematical observations."
  Color Research & Application, 30(1), 21-30.

The unit tests (tests/test_delta_e.py) verify against the 34 reference
pairs published in that paper's Table 1.

All inputs are CIELAB (L*, a*, b*) under D65 / 2° observer.
Parametric weighting factors kL, kC, kH default to 1.0 (reference
conditions: illumination ≈1000 lx, uniform surround, normal vision).
"""

from __future__ import annotations
import math
from typing import Tuple


def ciede2000(
    L1: float, a1: float, b1: float,
    L2: float, a2: float, b2: float,
    kL: float = 1.0, kC: float = 1.0, kH: float = 1.0,
) -> float:
    """
    Compute CIEDE2000 ΔE*₀₀ between two CIELAB colours.

    Parameters
    ----------
    L1, a1, b1 : float   – CIELAB of colour 1
    L2, a2, b2 : float   – CIELAB of colour 2
    kL, kC, kH : float   – parametric weighting factors (default 1.0)

    Returns
    -------
    float – ΔE*₀₀ (always ≥ 0)
    """
    # Step 1 ─ Calculate C'ab, h'ab
    C1ab = math.sqrt(a1 * a1 + b1 * b1)
    C2ab = math.sqrt(a2 * a2 + b2 * b2)
    Cab_mean = (C1ab + C2ab) / 2.0

    Cab_mean_pow7 = Cab_mean ** 7
    _25_pow7 = 25.0 ** 7   # 6103515625
    G = 0.5 * (1.0 - math.sqrt(Cab_mean_pow7 / (Cab_mean_pow7 + _25_pow7)))

    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)

    C1p = math.sqrt(a1p * a1p + b1 * b1)
    C2p = math.sqrt(a2p * a2p + b2 * b2)

    h1p = math.degrees(math.atan2(b1, a1p)) % 360.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360.0

    # Step 2 ─ Calculate ΔL', ΔC', ΔH'
    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0.0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180.0:
        dhp = h2p - h1p
    elif h2p - h1p > 180.0:
        dhp = h2p - h1p - 360.0
    else:
        dhp = h2p - h1p + 360.0

    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

    # Step 3 ─ Calculate CIEDE2000 ΔE00
    Lp_mean = (L1 + L2) / 2.0
    Cp_mean = (C1p + C2p) / 2.0

    if C1p * C2p == 0.0:
        hp_mean = h1p + h2p
    elif abs(h1p - h2p) <= 180.0:
        hp_mean = (h1p + h2p) / 2.0
    elif h1p + h2p < 360.0:
        hp_mean = (h1p + h2p + 360.0) / 2.0
    else:
        hp_mean = (h1p + h2p - 360.0) / 2.0

    T = (
        1.0
        - 0.17 * math.cos(math.radians(hp_mean - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * hp_mean))
        + 0.32 * math.cos(math.radians(3.0 * hp_mean + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * hp_mean - 63.0))
    )

    SL = 1.0 + 0.015 * (Lp_mean - 50.0) ** 2 / math.sqrt(20.0 + (Lp_mean - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cp_mean
    SH = 1.0 + 0.015 * Cp_mean * T

    Cp_mean_pow7 = Cp_mean ** 7
    RC = 2.0 * math.sqrt(Cp_mean_pow7 / (Cp_mean_pow7 + _25_pow7))

    d_theta = 30.0 * math.exp(-(((hp_mean - 275.0) / 25.0) ** 2))
    RT = -math.sin(math.radians(2.0 * d_theta)) * RC

    dE = math.sqrt(
        (dLp / (kL * SL)) ** 2
        + (dCp / (kC * SC)) ** 2
        + (dHp / (kH * SH)) ** 2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )
    return dE


def delta_e00(
    lab1: Tuple[float, float, float],
    lab2: Tuple[float, float, float],
) -> float:
    """Convenience wrapper: takes two (L,a,b) tuples."""
    return ciede2000(lab1[0], lab1[1], lab1[2], lab2[0], lab2[1], lab2[2])


def classify_delta_e(de: float) -> Tuple[str, str]:
    """
    Return (description, status) for a ΔE*₀₀ value using the
    thresholds from config.DE_BANDS.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import DE_BANDS
    for lo, hi, desc, status in DE_BANDS:
        if lo <= de < hi:
            return (desc, status)
    return ("Unknown", "unknown")
