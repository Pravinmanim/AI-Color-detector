"""
Unit tests for colour name resolution.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from color.names import resolve_colour_name
from color.conversions import srgb255_to_lab


class TestExactCSS:
    """Exact CSS name resolution → REFERENCE source."""

    def test_red(self):
        r = resolve_colour_name("red")
        assert r is not None
        assert r.hex == "#FF0000"
        assert r.rgb == (255, 0, 0)
        assert r.source == "REFERENCE"
        assert abs(r.lab[0] - 53.23) < 0.5

    def test_case_insensitive(self):
        r = resolve_colour_name("DodgerBlue")
        assert r is not None
        assert r.hex == "#1E90FF"
        assert r.source == "REFERENCE"

    def test_cornflowerblue(self):
        r = resolve_colour_name("cornflowerblue")
        assert r is not None
        assert r.hex == "#6495ED"
        assert r.source == "REFERENCE"


class TestCommonNames:
    """Extended common name resolution → REFERENCE source."""

    def test_terracotta(self):
        r = resolve_colour_name("terracotta")
        assert r is not None
        assert r.source == "REFERENCE"

    def test_navy_blue(self):
        r = resolve_colour_name("navy blue")
        assert r is not None
        assert r.source == "REFERENCE"

    def test_cream(self):
        r = resolve_colour_name("cream")
        assert r is not None
        assert r.source == "REFERENCE"


class TestHexInput:
    """Direct hex input → REFERENCE source."""

    def test_hash_hex(self):
        r = resolve_colour_name("#FF8800")
        assert r is not None
        assert r.rgb == (255, 136, 0)
        assert r.source == "REFERENCE"

    def test_rgb_func(self):
        r = resolve_colour_name("rgb(100, 200, 50)")
        assert r is not None
        assert r.rgb == (100, 200, 50)
        assert r.source == "REFERENCE"

    def test_hsl_func(self):
        r = resolve_colour_name("hsl(0, 100, 50)")
        assert r is not None
        assert r.rgb == (255, 0, 0)
        assert r.source == "REFERENCE"


class TestModifiers:
    """Modifier rules → ESTIMATED source."""

    def test_dark_red(self):
        r = resolve_colour_name("dark red")
        assert r is not None
        assert r.source == "ESTIMATED"
        # Should be darker than pure red
        red_lab = srgb255_to_lab(255, 0, 0)
        assert r.lab[0] < red_lab[0]  # L* should be lower

    def test_light_blue(self):
        # "lightblue" is a CSS name, but "light blue" with space triggers modifier
        r = resolve_colour_name("light blue")
        assert r is not None
        assert r.source == "ESTIMATED"

    def test_deep_ocean_blue(self):
        """This is the acceptance criterion test: must show ESTIMATED."""
        r = resolve_colour_name("deep ocean blue")
        assert r is not None
        assert r.source == "ESTIMATED"
        assert len(r.steps) >= 2  # Should have modifier + base steps
        assert any("deep" in s.lower() for s in r.steps)


class TestFuzzyMatch:
    """Fuzzy matching → ESTIMATED source."""

    def test_typo_bleu(self):
        r = resolve_colour_name("bleu")
        assert r is not None
        assert r.source == "ESTIMATED"
        assert any("fuzzy" in s.lower() for s in r.steps)

    def test_typo_yelow(self):
        r = resolve_colour_name("yelow")
        assert r is not None
        assert r.source == "ESTIMATED"


class TestUnresolvable:
    """Names that cannot be resolved should return None."""

    def test_nonsense(self):
        r = resolve_colour_name("xyzzyflurp")
        assert r is None

    def test_empty(self):
        r = resolve_colour_name("")
        assert r is None


class TestResolutionSteps:
    """The resolution should always include traceable steps."""

    def test_steps_populated(self):
        r = resolve_colour_name("coral")
        assert r is not None
        assert len(r.steps) > 0
        assert r.source == "REFERENCE"

    def test_to_dict(self):
        r = resolve_colour_name("red")
        d = r.to_dict()
        assert "name" in d
        assert "hex" in d
        assert "lab" in d
        assert "source" in d
        assert "steps" in d
