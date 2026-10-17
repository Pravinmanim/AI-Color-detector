"""
Kubelka-Munk (two-flux) forward model for R:B:Y:W paint mixing.

Physics:
  Each pigment has absorption (K) and scattering (S) spectra.
  For a mixture: K_mix/S_mix = sum(ci*Ki) / sum(ci*Si)
  Reflectance R = 1 + K/S - sqrt((K/S)^2 + 2*K/S)

We use 36 wavelength channels (380-730 nm, 10 nm steps).
Spectral integration uses the CIE 1931 2-degree observer CMFs under D65
to produce XYZ, then standard xyz_to_lab (D65 Xn=0.95047, Yn=1.0, Zn=1.08883).

Pigment K/S spectra are SIMULATED (hand-tuned). NOT from real paint.
DATA SOURCE: SIMULATED
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DISPENSE_ERROR_PCT, SCAN_NOISE_L, SCAN_NOISE_AB, LOT_SHIFT_SIGMA, SRC_SIMULATED

# ──────────────────────────────────────────────
# Wavelength grid: 380-730 nm, 10 nm steps (36 channels)
# ──────────────────────────────────────────────
WAVELENGTHS = list(range(380, 740, 10))
N_CHANNELS  = len(WAVELENGTHS)  # 36

# ──────────────────────────────────────────────
# CIE 1931 2-degree observer colour-matching functions
# Sampled at 380-730 nm, 10 nm steps.
# Source: CIE 15:2004, Table T.2 (Wyszecki & Stiles 2nd ed.)
# DATA SOURCE: PUBLIC-MEASURED (CIE standard)
#
# IMPORTANT: x-bar has TWO peaks:
#   First  peak ~450 nm  (~0.34)
#   Second peak ~600 nm  (~1.06)
# The second peak is what makes red paint show positive a*.
# ──────────────────────────────────────────────
CMF_X = [
    # 380  390   400   410   420   430   440   450   460   470
    0.01360,0.01986,0.04000,0.08230,0.16490,0.29080,0.40450,0.48100,0.48420,0.43060,
    # 480  490   500   510   520   530   540   550   560   570
    0.34010,0.22840,0.13438,0.08450,0.04847,0.02399,0.00927,0.00000,0.00000,0.01147,
    # 580  590   600   610   620   630   640   650   660   670
    0.04818,0.11578,0.24613,0.43076,0.62070,0.81630,0.95360,1.02480,1.05140,0.99230,
    # 680  690   700   710   720   730
    0.84330,0.63600,0.44790,0.28480,0.16490,0.08740,
]

CMF_Y = [
    # 380  390   400   410   420   430   440   450   460   470
    0.000039,0.000120,0.000396,0.001210,0.004000,0.011600,0.023000,0.038000,0.060000,0.090980,
    # 480  490   500   510   520   530   540   550   560   570
    0.13902,0.20802,0.32300,0.50300,0.71000,0.86200,0.95400,0.99495,1.00000,0.95200,
    # 580  590   600   610   620   630   640   650   660   670
    0.87000,0.75700,0.63100,0.50300,0.38100,0.26500,0.17500,0.10700,0.06100,0.03200,
    # 680  690   700   710   720   730
    0.01700,0.00821,0.00410,0.00210,0.00105,0.00050,
]

CMF_Z = [
    # 380  390   400   410   420   430   440   450   460   470
    0.06500,0.10050,0.20050,0.43330,0.97390,1.84450,2.62200,3.07550,2.95210,2.48340,
    # 480  490   500   510   520   530   540   550   560   570
    1.74510,1.10130,0.64560,0.34960,0.17230,0.08220,0.03680,0.01920,0.00830,0.00380,
    # 580  590   600   610   620   630   640   650   660   670
    0.00190,0.00060,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,
    # 680  690   700   710   720   730
    0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,
]

# D65 illuminant SPD, 380-730 nm, 10 nm steps
# Source: CIE 15:2004, Table T.1
# DATA SOURCE: PUBLIC-MEASURED (CIE standard)
D65_SPD = [
    # 380  390   400   410   420   430   440   450   460   470
    49.975,52.312,54.648,68.701,82.755,87.120,91.486,92.459,93.432,90.057,
    # 480  490   500   510   520   530   540   550   560   570
    86.682,95.774,104.865,110.936,117.008,117.410,117.812,116.336,114.861,115.392,
    # 580  590   600   610   620   630   640   650   660   670
    115.923,112.367,108.811,109.082,109.354,108.578,107.802,106.296,104.790,106.239,
    # 680  690   700   710   720   730
    107.689,106.047,104.405,104.225,104.046,102.023,
]

# ──────────────────────────────────────────────
# Verify CMF dimensions
# ──────────────────────────────────────────────
assert len(CMF_X) == N_CHANNELS, f"CMF_X length {len(CMF_X)} != {N_CHANNELS}"
assert len(CMF_Y) == N_CHANNELS
assert len(CMF_Z) == N_CHANNELS
assert len(D65_SPD) == N_CHANNELS

# ──────────────────────────────────────────────
# Scale factors: map truncated-integral XYZ to
# standard CIE D65 white point.
# This allows xyz_to_lab to use the correct Xn=0.95047.
# ──────────────────────────────────────────────
def _compute_km_scale() -> Tuple[float, float, float]:
    N  = sum(D65_SPD[i] * CMF_Y[i] for i in range(N_CHANNELS))
    X0 = sum(D65_SPD[i] * CMF_X[i] for i in range(N_CHANNELS)) / N
    Y0 = 1.0
    Z0 = sum(D65_SPD[i] * CMF_Z[i] for i in range(N_CHANNELS)) / N
    # Standard D65 Xn/Yn/Zn (CIE 1931 2-degree, full spectrum 360-830 nm)
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    return Xn / X0, Yn / Y0, Zn / Z0

_KM_SX, _KM_SY, _KM_SZ = _compute_km_scale()


# ──────────────────────────────────────────────
# SIMULATED pigment K and S spectra
# NOT measured from real paint.
# ──────────────────────────────────────────────
def _make_pigment_spectra() -> Dict[str, Dict[str, List[float]]]:
    """
    SIMULATED K/S spectra for Red, Blue, Yellow, White.

    Design ensures correct Lab hue quadrants:
      Red    -> a* > 0  (absorbs green 500-560 nm, reflects red 570-730 nm)
      Blue   -> b* < 0  (absorbs red+green, reflects blue 380-480 nm)
      Yellow -> b* > 0  (absorbs blue 380-460 nm, reflects yellow-red 470-730 nm)
      White  -> L* ~98, a* ~0, b* ~0

    DATA SOURCE: SIMULATED
    """
    pigments = {}

    # WHITE: near-zero absorption everywhere
    K_white = [0.001] * N_CHANNELS
    S_white = [1.0]   * N_CHANNELS
    pigments["white"] = {"K": K_white, "S": S_white}

    # RED: absorbs 380-560 nm, reflects 560-730 nm
    K_red, S_red = [], []
    for i, w in enumerate(WAVELENGTHS):
        if w <= 510:
            K_red.append(15.0);  S_red.append(0.05)
        elif w <= 560:
            t = (w - 510) / 50.0
            K_red.append(15.0*(1-t) + 0.05*t)
            S_red.append(0.05*(1-t) + 1.0*t)
        else:
            K_red.append(0.05);  S_red.append(1.0)
    pigments["red"] = {"K": K_red, "S": S_red}

    # BLUE: reflects 380-470 nm, absorbs 480-730 nm
    K_blue, S_blue = [], []
    for i, w in enumerate(WAVELENGTHS):
        if w <= 460:
            K_blue.append(0.05);  S_blue.append(1.0)
        elif w <= 510:
            t = (w - 460) / 50.0
            K_blue.append(0.05*(1-t) + 15.0*t)
            S_blue.append(1.0*(1-t) + 0.05*t)
        else:
            K_blue.append(15.0);  S_blue.append(0.05)
    pigments["blue"] = {"K": K_blue, "S": S_blue}

    # YELLOW: absorbs 380-460 nm, reflects 470-730 nm
    K_yellow, S_yellow = [], []
    for i, w in enumerate(WAVELENGTHS):
        if w <= 440:
            K_yellow.append(15.0);  S_yellow.append(0.05)
        elif w <= 490:
            t = (w - 440) / 50.0
            K_yellow.append(15.0*(1-t) + 0.05*t)
            S_yellow.append(0.05*(1-t) + 1.0*t)
        else:
            K_yellow.append(0.05);  S_yellow.append(1.0)
    pigments["yellow"] = {"K": K_yellow, "S": S_yellow}

    return pigments


# Module-level pigment data (computed once)
PIGMENT_SPECTRA = _make_pigment_spectra()


# ──────────────────────────────────────────────
# Core KM mixing
# ──────────────────────────────────────────────
def km_mix_reflectance(
    red_pct: float, blue_pct: float,
    yellow_pct: float, white_pct: float,
    pigment_spectra: Optional[Dict] = None,
) -> List[float]:
    """KM mixture -> reflectance spectrum [0-1] at 36 wavelengths. DATA SOURCE: SIMULATED"""
    if pigment_spectra is None:
        pigment_spectra = PIGMENT_SPECTRA
    fracs = {"red": red_pct, "blue": blue_pct, "yellow": yellow_pct, "white": white_pct}
    total = sum(fracs.values())
    if total < 1e-12:
        return [0.0] * N_CHANNELS
    fracs = {k: v / total for k, v in fracs.items()}

    reflectance = []
    for i in range(N_CHANNELS):
        K_mix = sum(fracs[p] * pigment_spectra[p]["K"][i] for p in fracs)
        S_mix = sum(fracs[p] * pigment_spectra[p]["S"][i] for p in fracs)
        if S_mix < 1e-12:
            reflectance.append(0.0)
            continue
        KS = K_mix / S_mix
        R  = 1.0 + KS - math.sqrt(KS*KS + 2.0*KS)
        reflectance.append(max(0.0, min(1.0, R)))
    return reflectance


def reflectance_to_xyz(reflectance: List[float]) -> Tuple[float, float, float]:
    """
    Reflectance spectrum x D65 x CMF -> XYZ, scaled to standard D65 white point.
    Perfect reflector (R=1 everywhere) gives XYZ = (0.95047, 1.0, 1.08883).
    """
    X = Y = Z = N = 0.0
    for i in range(N_CHANNELS):
        w = D65_SPD[i] * reflectance[i]
        X += w * CMF_X[i]
        Y += w * CMF_Y[i]
        Z += w * CMF_Z[i]
        N += D65_SPD[i] * CMF_Y[i]
    if N < 1e-12:
        return (0.0, 0.0, 0.0)
    return (X * _KM_SX / N, Y * _KM_SY / N, Z * _KM_SZ / N)


def reflectance_to_lab(reflectance: List[float]) -> Tuple[float, float, float]:
    """Reflectance -> CIELAB (D65, 2-degree)."""
    from color.conversions import xyz_to_lab
    return xyz_to_lab(*reflectance_to_xyz(reflectance))


def recipe_to_lab(
    red_pct: float, blue_pct: float,
    yellow_pct: float, white_pct: float,
    pigment_spectra: Optional[Dict] = None,
) -> Tuple[float, float, float]:
    """
    Forward model: R:B:Y:W percentages -> CIELAB.
    DATA SOURCE: SIMULATED
    """
    total = red_pct + blue_pct + yellow_pct + white_pct
    if total < 1e-12:
        return (0.0, 0.0, 0.0)
    refl = km_mix_reflectance(red_pct, blue_pct, yellow_pct, white_pct, pigment_spectra)
    return reflectance_to_lab(refl)


def recipe_to_lab_with_noise(
    red_pct: float, blue_pct: float,
    yellow_pct: float, white_pct: float,
    dispense_error: float = DISPENSE_ERROR_PCT,
    scan_noise_L: float = SCAN_NOISE_L,
    scan_noise_ab: float = SCAN_NOISE_AB,
    lot_shift: Optional[Tuple[float, float, float]] = None,
    rng=None,
    pigment_spectra: Optional[Dict] = None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    Forward model with realistic noise:
      1. Dispensing error (±dispense_error fraction per pigment)
      2. Scan noise (Gaussian on L*, a*, b*)
      3. Lot-to-lot shift (fixed Lab offset)

    Returns: (Lab_noisy, actual_pcts)
    DATA SOURCE: SIMULATED
    """
    import random
    if rng is None:
        rng = random.Random()

    # 1. Dispensing error
    pcts  = [red_pct, blue_pct, yellow_pct, white_pct]
    actual = [max(0.0, p * (1.0 + rng.gauss(0, dispense_error))) for p in pcts]
    total  = sum(actual)
    if total < 1e-12:
        actual = [25.0, 25.0, 25.0, 25.0]; total = 100.0
    actual = [a / total * 100.0 for a in actual]

    # 2. KM forward model
    L, a, b = recipe_to_lab(actual[0], actual[1], actual[2], actual[3], pigment_spectra)

    # 3. Lot shift
    if lot_shift is not None:
        L += lot_shift[0]; a += lot_shift[1]; b += lot_shift[2]

    # 4. Scan noise
    L += rng.gauss(0, scan_noise_L)
    a += rng.gauss(0, scan_noise_ab)
    b += rng.gauss(0, scan_noise_ab)
    L = max(0.0, min(100.0, L))

    return ((L, a, b), tuple(actual))
