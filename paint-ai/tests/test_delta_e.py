"""
Unit tests for CIEDE2000.

The 34 reference pairs are from Table 1 of:
  Sharma, G., Wu, W., & Dalal, E. N. (2005).
  "The CIEDE2000 color‐difference formula: Implementation notes,
   supplementary test data, and mathematical observations."
  Color Research & Application, 30(1), 21–30.

Each row: (L1, a1, b1, L2, a2, b2, expected_dE00)
Tolerance is set to 0.0001 matching the published precision.

DATA SOURCE: PUBLIC-MEASURED (published reference data)
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from color.delta_e import ciede2000, delta_e00, classify_delta_e

# ──────────────────────────────────────────────
# Sharma-Wu-Dalal Table 1  (34 reference pairs)
# ──────────────────────────────────────────────
SHARMA_PAIRS = [
    # Pair 1
    (50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425),
    # Pair 2
    (50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615),
    # Pair 3
    (50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412),
    # Pair 4
    (50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000),
    # Pair 5
    (50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000),
    # Pair 6
    (50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000),
    # Pair 7
    (50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669),
    # Pair 8
    (50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669),
    # Pair 9
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792),
    # Pair 10
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792),
    # Pair 11
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195),
    # Pair 12
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195),
    # Pair 13
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045),
    # Pair 14
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045),
    # Pair 15
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461),
    # Pair 16
    (50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065),
    # Pair 17
    (50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492),
    # Pair 18
    (50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977),
    # Pair 19
    (50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030),
    # Pair 20
    (50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535),
    # Pair 21
    (50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000),
    # Pair 22
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000),
    # Pair 23
    (50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000),
    # Pair 24
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000),
    # Pair 25
    (60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644),
    # Pair 26
    (63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630),
    # Pair 27
    (61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731),
    # Pair 28
    (35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645),
    # Pair 29
    (22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373),
    # Pair 30
    (36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146),
    # Pair 31
    (90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441),
    # Pair 32
    (90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381),
    # Pair 33
    (6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377),
    # Pair 34
    (2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082),
]


@pytest.mark.parametrize(
    "L1,a1,b1,L2,a2,b2,expected",
    SHARMA_PAIRS,
    ids=[f"pair-{i+1}" for i in range(len(SHARMA_PAIRS))],
)
def test_ciede2000_sharma_pairs(L1, a1, b1, L2, a2, b2, expected):
    """Each Sharma-Wu-Dalal reference pair must match to 4 decimal places."""
    result = ciede2000(L1, a1, b1, L2, a2, b2)
    assert abs(result - expected) < 0.0001, (
        f"CIEDE2000({L1},{a1},{b1} vs {L2},{a2},{b2}) = {result:.4f}, "
        f"expected {expected:.4f}"
    )


def test_ciede2000_identical_colours():
    """ΔE of identical colours must be exactly 0."""
    assert ciede2000(50, 25, -10, 50, 25, -10) == 0.0


def test_ciede2000_symmetry():
    """ΔE(a,b) == ΔE(b,a) to machine precision."""
    L1, a1, b1 = 50.0, 2.5, 0.0
    L2, a2, b2 = 73.0, 25.0, -18.0
    d1 = ciede2000(L1, a1, b1, L2, a2, b2)
    d2 = ciede2000(L2, a2, b2, L1, a1, b1)
    assert abs(d1 - d2) < 1e-10


def test_ciede2000_non_negative():
    """ΔE must never be negative."""
    import random
    random.seed(42)
    for _ in range(200):
        vals = [random.uniform(-128, 128) for _ in range(4)]
        L1 = random.uniform(0, 100)
        L2 = random.uniform(0, 100)
        d = ciede2000(L1, vals[0], vals[1], L2, vals[2], vals[3])
        assert d >= 0.0


def test_delta_e00_tuple_wrapper():
    """The tuple wrapper should give the same result as the raw function."""
    lab1 = (50.0, 2.6772, -79.7751)
    lab2 = (50.0, 0.0, -82.7485)
    assert abs(delta_e00(lab1, lab2) - 2.0425) < 0.0001


def test_classify_delta_e():
    """Classification bands must work correctly."""
    desc, status = classify_delta_e(0.5)
    assert desc == "Excellent"
    assert status == "accepted"
    desc, status = classify_delta_e(1.5)
    assert desc == "Very close"
    desc, status = classify_delta_e(2.5)
    assert desc == "Needs minor correction"
    desc, status = classify_delta_e(4.0)
    assert desc == "Noticeable difference"
    desc, status = classify_delta_e(10.0)
    assert desc == "Large difference"
    assert status == "rejected"


def test_ciede2000_black_vs_white():
    """Black (L=0) vs White (L=100) should give a large ΔE."""
    d = ciede2000(0, 0, 0, 100, 0, 0)
    assert d > 50  # Should be very large


def test_ciede2000_near_grey_axis():
    """Colours near the grey axis (a≈0, b≈0) should handle the hue correctly."""
    d = ciede2000(50, 0.001, 0.001, 50, -0.001, -0.001)
    assert d < 0.01  # Should be very close to 0
