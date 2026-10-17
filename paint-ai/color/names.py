"""
Colour-name resolution pipeline.

Resolution order:
  1. Exact CSS named colour (case-insensitive)
  2. Extended common-names list (≥ 60 entries: terracotta, navy blue, cream …)
  3. Fuzzy match (edit-distance ≤ 2)
  4. Modifier rules applied in CIELAB space:
        dark   → L* -= 15
        light  → L* += 20
        deep   → L* -= 10, C* += 10
        pale   → L* += 25, C* -= 15
        bright → C* += 15

Each result carries a `source` label:
  REFERENCE  – exact CSS or extended-list match
  ESTIMATED  – fuzzy match or modifier-derived
"""

from __future__ import annotations
import math
import re
from typing import Dict, List, Optional, Tuple

from color.conversions import srgb255_to_lab, lab_to_srgb255_checked, hex_to_rgb

# ──────────────────────────────────────────────
# CSS named colours (all 148 CSS4 / X11 names)
# Only a representative subset is shown inline;
# the full table is generated programmatically.
# ──────────────────────────────────────────────
CSS_COLORS: Dict[str, str] = {
    "aliceblue": "#F0F8FF", "antiquewhite": "#FAEBD7", "aqua": "#00FFFF",
    "aquamarine": "#7FFFD4", "azure": "#F0FFFF", "beige": "#F5F5DC",
    "bisque": "#FFE4C4", "black": "#000000", "blanchedalmond": "#FFEBCD",
    "blue": "#0000FF", "blueviolet": "#8A2BE2", "brown": "#A52A2A",
    "burlywood": "#DEB887", "cadetblue": "#5F9EA0", "chartreuse": "#7FFF00",
    "chocolate": "#D2691E", "coral": "#FF7F50", "cornflowerblue": "#6495ED",
    "cornsilk": "#FFF8DC", "crimson": "#DC143C", "cyan": "#00FFFF",
    "darkblue": "#00008B", "darkcyan": "#008B8B", "darkgoldenrod": "#B8860B",
    "darkgray": "#A9A9A9", "darkgreen": "#006400", "darkgrey": "#A9A9A9",
    "darkkhaki": "#BDB76B", "darkmagenta": "#8B008B", "darkolivegreen": "#556B2F",
    "darkorange": "#FF8C00", "darkorchid": "#9932CC", "darkred": "#8B0000",
    "darksalmon": "#E9967A", "darkseagreen": "#8FBC8F", "darkslateblue": "#483D8B",
    "darkslategray": "#2F4F4F", "darkslategrey": "#2F4F4F", "darkturquoise": "#00CED1",
    "darkviolet": "#9400D3", "deeppink": "#FF1493", "deepskyblue": "#00BFFF",
    "dimgray": "#696969", "dimgrey": "#696969", "dodgerblue": "#1E90FF",
    "firebrick": "#B22222", "floralwhite": "#FFFAF0", "forestgreen": "#228B22",
    "fuchsia": "#FF00FF", "gainsboro": "#DCDCDC", "ghostwhite": "#F8F8FF",
    "gold": "#FFD700", "goldenrod": "#DAA520", "gray": "#808080",
    "green": "#008000", "greenyellow": "#ADFF2F", "grey": "#808080",
    "honeydew": "#F0FFF0", "hotpink": "#FF69B4", "indianred": "#CD5C5C",
    "indigo": "#4B0082", "ivory": "#FFFFF0", "khaki": "#F0E68C",
    "lavender": "#E6E6FA", "lavenderblush": "#FFF0F5", "lawngreen": "#7CFC00",
    "lemonchiffon": "#FFFACD", "lightblue": "#ADD8E6", "lightcoral": "#F08080",
    "lightcyan": "#E0FFFF", "lightgoldenrodyellow": "#FAFAD2", "lightgray": "#D3D3D3",
    "lightgreen": "#90EE90", "lightgrey": "#D3D3D3", "lightpink": "#FFB6C1",
    "lightsalmon": "#FFA07A", "lightseagreen": "#20B2AA", "lightskyblue": "#87CEFA",
    "lightslategray": "#778899", "lightslategrey": "#778899", "lightsteelblue": "#B0C4DE",
    "lightyellow": "#FFFFE0", "lime": "#00FF00", "limegreen": "#32CD32",
    "linen": "#FAF0E6", "magenta": "#FF00FF", "maroon": "#800000",
    "mediumaquamarine": "#66CDAA", "mediumblue": "#0000CD", "mediumorchid": "#BA55D3",
    "mediumpurple": "#9370DB", "mediumseagreen": "#3CB371", "mediumslateblue": "#7B68EE",
    "mediumspringgreen": "#00FA9A", "mediumturquoise": "#48D1CC",
    "mediumvioletred": "#C71585", "midnightblue": "#191970", "mintcream": "#F5FFFA",
    "mistyrose": "#FFE4E1", "moccasin": "#FFE4B5", "navajowhite": "#FFDEAD",
    "navy": "#000080", "oldlace": "#FDF5E6", "olive": "#808000",
    "olivedrab": "#6B8E23", "orange": "#FFA500", "orangered": "#FF4500",
    "orchid": "#DA70D6", "palegoldenrod": "#EEE8AA", "palegreen": "#98FB98",
    "paleturquoise": "#AFEEEE", "palevioletred": "#DB7093", "papayawhip": "#FFEFD5",
    "peachpuff": "#FFDAB9", "peru": "#CD853F", "pink": "#FFC0CB",
    "plum": "#DDA0DD", "powderblue": "#B0E0E6", "purple": "#800080",
    "rebeccapurple": "#663399", "red": "#FF0000", "rosybrown": "#BC8F8F",
    "royalblue": "#4169E1", "saddlebrown": "#8B4513", "salmon": "#FA8072",
    "sandybrown": "#F4A460", "seagreen": "#2E8B57", "seashell": "#FFF5EE",
    "sienna": "#A0522D", "silver": "#C0C0C0", "skyblue": "#87CEEB",
    "slateblue": "#6A5ACD", "slategray": "#708090", "slategrey": "#708090",
    "snow": "#FFFAFA", "springgreen": "#00FF7F", "steelblue": "#4682B4",
    "tan": "#D2B48C", "teal": "#008080", "thistle": "#D8BFD8",
    "tomato": "#FF6347", "turquoise": "#40E0D0", "violet": "#EE82EE",
    "wheat": "#F5DEB3", "white": "#FFFFFF", "whitesmoke": "#F5F5F5",
    "yellow": "#FFFF00", "yellowgreen": "#9ACD32",
}

# ──────────────────────────────────────────────
# Extended common names (at least 60 extra)
# ──────────────────────────────────────────────
COMMON_COLORS: Dict[str, str] = {
    "terracotta": "#E2725B", "cream": "#FFFDD0", "navy blue": "#000080",
    "sky blue": "#87CEEB", "baby blue": "#89CFF0", "royal blue": "#4169E1",
    "ocean blue": "#0077BE", "ice blue": "#99C5C4", "steel blue": "#4682B4",
    "cobalt": "#0047AB", "sapphire": "#0F52BA", "cerulean": "#007BA7",
    "burgundy": "#800020", "wine": "#722F37", "ruby": "#E0115F",
    "scarlet": "#FF2400", "crimson": "#DC143C", "rose": "#FF007F",
    "blush": "#DE5D83", "coral": "#FF7F50", "peach": "#FFCBA4",
    "apricot": "#FBCEB1", "tangerine": "#FF9966", "rust": "#B7410E",
    "copper": "#B87333", "bronze": "#CD7F32", "cinnamon": "#D2691E",
    "amber": "#FFBF00", "honey": "#EB9605", "mustard": "#FFDB58",
    "saffron": "#F4C430", "lemon": "#FFF44F", "canary": "#FFEF00",
    "buttercup": "#F9E154", "maize": "#FBEC5D", "sand": "#C2B280",
    "khaki": "#C3B091", "taupe": "#483C32", "mocha": "#967969",
    "coffee": "#6F4E37", "espresso": "#3C2415", "chocolate": "#7B3F00",
    "caramel": "#FFD59A", "ivory": "#FFFFF0", "eggshell": "#F0EAD6",
    "bone": "#E3DAC9", "champagne": "#F7E7CE", "vanilla": "#F3E5AB",
    "mint": "#98FF98", "sage": "#BCB88A", "olive": "#808000",
    "moss": "#8A9A5B", "emerald": "#50C878", "jade": "#00A86B",
    "forest green": "#228B22", "hunter green": "#355E3B", "lime green": "#32CD32",
    "pine": "#01796F", "teal": "#008080", "turquoise": "#40E0D0",
    "aquamarine": "#7FFFD4", "seafoam": "#93E9BE", "lavender": "#E6E6FA",
    "lilac": "#C8A2C8", "mauve": "#E0B0FF", "plum": "#8E4585",
    "grape": "#6F2DA8", "eggplant": "#614051", "mulberry": "#C54B8C",
    "magenta": "#FF00FF", "fuchsia": "#FF00FF", "charcoal": "#36454F",
    "slate": "#708090", "ash": "#B2BEB5", "pewter": "#96A8A1",
    "graphite": "#383838",
}

# ──────────────────────────────────────────────
# Modifier rules  (applied in CIELAB space)
# ──────────────────────────────────────────────
MODIFIERS = {
    "dark":   {"dL": -15, "dC": 0},
    "light":  {"dL": +20, "dC": 0},
    "deep":   {"dL": -10, "dC": +10},
    "pale":   {"dL": +25, "dC": -15},
    "bright": {"dL": 0,   "dC": +15},
    "vivid":  {"dL": -5,  "dC": +20},
    "muted":  {"dL": +5,  "dC": -20},
    "pastel": {"dL": +30, "dC": -25},
    "neon":   {"dL": +10, "dC": +25},
    "dusty":  {"dL": +5,  "dC": -10},
}


# ──────────────────────────────────────────────
# Levenshtein distance (small strings only)
# ──────────────────────────────────────────────
def _edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            ins = prev[j + 1] + 1
            rem = curr[j] + 1
            sub = prev[j] + (0 if ca == cb else 1)
            curr.append(min(ins, rem, sub))
        prev = curr
    return prev[-1]


# ──────────────────────────────────────────────
# Apply a modifier to a Lab colour
# ──────────────────────────────────────────────
def _apply_modifier(L: float, a: float, b: float, mod: dict) -> Tuple[float, float, float]:
    """Shift L* and chroma in CIELAB space."""
    C = math.sqrt(a * a + b * b)
    h = math.atan2(b, a)
    L_new = max(0.0, min(100.0, L + mod["dL"]))
    C_new = max(0.0, C + mod["dC"])
    a_new = C_new * math.cos(h)
    b_new = C_new * math.sin(h)
    return (L_new, a_new, b_new)


# ──────────────────────────────────────────────
# Main resolver
# ──────────────────────────────────────────────

class ResolvedColour:
    """Result of a colour-name lookup."""
    __slots__ = ("name", "hex", "rgb", "lab", "source", "steps", "in_gamut")

    def __init__(
        self,
        name: str, hex_: str, rgb: Tuple[int, int, int],
        lab: Tuple[float, float, float],
        source: str, steps: List[str], in_gamut: bool = True,
    ):
        self.name = name
        self.hex = hex_
        self.rgb = rgb
        self.lab = lab
        self.source = source   # REFERENCE | ESTIMATED
        self.steps = steps
        self.in_gamut = in_gamut

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "hex": self.hex,
            "rgb": list(self.rgb),
            "lab": [round(v, 2) for v in self.lab],
            "source": self.source,
            "steps": self.steps,
            "in_gamut": self.in_gamut,
        }


def resolve_colour_name(text: str) -> Optional[ResolvedColour]:
    """
    Try to resolve a free-text colour description to an sRGB colour.

    Returns None only if nothing could be matched at all.
    """
    raw = text.strip()
    key = raw.lower()
    steps: List[str] = []

    # ── 0. Direct hex / rgb() / hsl() ────────────────────
    if key.startswith("#") or (len(key) in (3, 6) and all(c in "0123456789abcdef" for c in key)):
        try:
            hex_str = raw if raw.startswith("#") else f"#{raw}"
            rgb = hex_to_rgb(hex_str)
            lab = srgb255_to_lab(*rgb)
            steps.append(f"Parsed hex value {hex_str}")
            from color.conversions import rgb_to_hex
            return ResolvedColour(raw, rgb_to_hex(*rgb), rgb, lab, "REFERENCE", steps)
        except ValueError:
            pass

    m_rgb = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", key)
    if m_rgb:
        rgb = (int(m_rgb.group(1)), int(m_rgb.group(2)), int(m_rgb.group(3)))
        lab = srgb255_to_lab(*rgb)
        steps.append(f"Parsed rgb() value")
        from color.conversions import rgb_to_hex
        return ResolvedColour(raw, rgb_to_hex(*rgb), rgb, lab, "REFERENCE", steps)

    m_hsl = re.match(r"hsl\(\s*([\d.]+)\s*,\s*([\d.]+)%?\s*,\s*([\d.]+)%?\s*\)", key)
    if m_hsl:
        from color.conversions import hsl_to_rgb, rgb_to_hex
        h, s, l = float(m_hsl.group(1)), float(m_hsl.group(2)), float(m_hsl.group(3))
        rgb = hsl_to_rgb(h, s, l)
        lab = srgb255_to_lab(*rgb)
        steps.append(f"Parsed hsl() value")
        return ResolvedColour(raw, rgb_to_hex(*rgb), rgb, lab, "REFERENCE", steps)

    # ── 1. Extract modifiers ─────────────────────────────
    applied_mods: List[Tuple[str, dict]] = []
    remaining = key
    for mod_name, mod_val in MODIFIERS.items():
        if remaining.startswith(mod_name + " "):
            applied_mods.append((mod_name, mod_val))
            remaining = remaining[len(mod_name):].strip()
            steps.append(f"Detected modifier: {mod_name}")

    # ── 2. Exact CSS match ───────────────────────────────
    normalized = remaining.replace(" ", "")
    if normalized in CSS_COLORS:
        hex_val = CSS_COLORS[normalized]
        rgb = hex_to_rgb(hex_val)
        lab = srgb255_to_lab(*rgb)
        steps.append(f"Exact CSS match: {normalized} → {hex_val}")
        if applied_mods:
            for mname, mval in applied_mods:
                lab = _apply_modifier(*lab, mval)
                steps.append(f"Applied modifier '{mname}' in Lab space")
            rgb_checked, in_gamut = lab_to_srgb255_checked(*lab)
            from color.conversions import rgb_to_hex
            return ResolvedColour(raw, rgb_to_hex(*rgb_checked), rgb_checked, lab,
                                  "ESTIMATED", steps, in_gamut)
        return ResolvedColour(raw, hex_val, rgb, lab, "REFERENCE", steps)

    # ── 3. Exact common-names match ──────────────────────
    if remaining in COMMON_COLORS:
        hex_val = COMMON_COLORS[remaining]
        rgb = hex_to_rgb(hex_val)
        lab = srgb255_to_lab(*rgb)
        steps.append(f"Exact common-name match: {remaining} → {hex_val}")
        if applied_mods:
            for mname, mval in applied_mods:
                lab = _apply_modifier(*lab, mval)
                steps.append(f"Applied modifier '{mname}' in Lab space")
            rgb_checked, in_gamut = lab_to_srgb255_checked(*lab)
            from color.conversions import rgb_to_hex
            return ResolvedColour(raw, rgb_to_hex(*rgb_checked), rgb_checked, lab,
                                  "ESTIMATED", steps, in_gamut)
        return ResolvedColour(raw, hex_val, rgb, lab, "REFERENCE", steps)

    # ── 4. Fuzzy match (edit distance ≤ 2) ───────────────
    all_names = {**{k: CSS_COLORS[k] for k in CSS_COLORS}, **COMMON_COLORS}
    best_dist = 999
    best_name = None
    for name in all_names:
        d = _edit_distance(remaining, name)
        if d < best_dist:
            best_dist = d
            best_name = name
    if best_dist <= 2 and best_name is not None:
        hex_val = all_names[best_name]
        rgb = hex_to_rgb(hex_val)
        lab = srgb255_to_lab(*rgb)
        steps.append(f"Fuzzy match: '{remaining}' → '{best_name}' (edit distance {best_dist})")
        if applied_mods:
            for mname, mval in applied_mods:
                lab = _apply_modifier(*lab, mval)
                steps.append(f"Applied modifier '{mname}' in Lab space")
            rgb_checked, in_gamut = lab_to_srgb255_checked(*lab)
            from color.conversions import rgb_to_hex
            return ResolvedColour(raw, rgb_to_hex(*rgb_checked), rgb_checked, lab,
                                  "ESTIMATED", steps, in_gamut)
        from color.conversions import rgb_to_hex
        return ResolvedColour(raw, rgb_to_hex(*rgb), rgb, lab, "ESTIMATED", steps)

    # ── 5. Last resort: try treating multi-word as base + modifiers ──
    # If we have modifiers but no match, try the last word alone
    words = remaining.split()
    if len(words) >= 2:
        for i in range(len(words) - 1):
            possible_mod = words[i]
            if possible_mod in MODIFIERS and possible_mod not in [m[0] for m in applied_mods]:
                applied_mods.append((possible_mod, MODIFIERS[possible_mod]))
                steps.append(f"Detected additional modifier: {possible_mod}")
        base = words[-1].replace(" ", "")
        if base in CSS_COLORS:
            hex_val = CSS_COLORS[base]
        elif " ".join(words[len(applied_mods):]) in COMMON_COLORS:
            base_phrase = " ".join(words[len(applied_mods):])
            hex_val = COMMON_COLORS[base_phrase]
            base = base_phrase
        else:
            return None
        rgb = hex_to_rgb(hex_val)
        lab = srgb255_to_lab(*rgb)
        steps.append(f"Base colour: {base}")
        for mname, mval in applied_mods:
            lab = _apply_modifier(*lab, mval)
            steps.append(f"Applied modifier '{mname}' in Lab space")
        rgb_checked, in_gamut = lab_to_srgb255_checked(*lab)
        from color.conversions import rgb_to_hex
        return ResolvedColour(raw, rgb_to_hex(*rgb_checked), rgb_checked, lab,
                              "ESTIMATED", steps, in_gamut)

    return None
