"""
Central configuration for the Paint-AI system.
All tuneable parameters live here — no magic numbers scattered in code.
"""

# ──────────────────────────────────────────────
# Delta-E thresholds
# ──────────────────────────────────────────────
TARGET_DE = 1.0        # Accepted match
LOOSE_DE = 2.0         # Acceptable but not perfect
MAX_ITERATIONS = 5     # Correction loop cap
BATCH_G = 100          # Default batch size in grams

# ──────────────────────────────────────────────
# Delta-E quality bands  (upper bound exclusive)
# ──────────────────────────────────────────────
DE_BANDS = [
    (0.0, 1.0,  "Excellent",              "accepted"),
    (1.0, 2.0,  "Very close",             "accepted"),
    (2.0, 3.5,  "Needs minor correction",  "correction"),
    (3.5, 5.0,  "Noticeable difference",   "correction"),
    (5.0, 1e9,  "Large difference",        "rejected"),
]

# ──────────────────────────────────────────────
# Colour science constants  (sRGB, D65, 2°)
# ──────────────────────────────────────────────
ILLUMINANT = "D65"
OBSERVER = "2°"
COLOUR_SPACE = "sRGB"

# D65 white point (CIE 1931 2° observer)
D65_XN = 0.95047
D65_YN = 1.00000
D65_ZN = 1.08883

# ──────────────────────────────────────────────
# Simulator noise parameters
# ──────────────────────────────────────────────
DISPENSE_ERROR_PCT = 0.015     # ±1.5 % mass error
SCAN_NOISE_L = 0.35            # Lab-unit noise σ on L*
SCAN_NOISE_AB = 0.35           # Lab-unit noise σ on a*, b*
LOT_SHIFT_SIGMA = 1.5          # Lab-unit lot-to-lot shift σ

# ──────────────────────────────────────────────
# Model / training
# ──────────────────────────────────────────────
ENSEMBLE_SIZE = 5
PRETRAIN_LR = 1e-3
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 15

# ──────────────────────────────────────────────
# Inverse optimisation
# ──────────────────────────────────────────────
OPTUNA_TRIALS = 500
CORRECTION_FRACTION = 0.80     # Apply 80 % of first correction
QUANTISE_G = 0.1               # Round to 0.1 g

# ──────────────────────────────────────────────
# Vision
# ──────────────────────────────────────────────
MAX_IMAGE_PX = 800             # Resize longest edge
KMEANS_K = 4                   # Dominant-colour clusters
MERGE_DE_THRESHOLD = 3.0       # Merge clusters closer than this

# ──────────────────────────────────────────────
# Data source labels
# ──────────────────────────────────────────────
SRC_SIMULATED = "SIMULATED"
SRC_PUBLIC = "PUBLIC-MEASURED"
SRC_REAL = "REAL-OWN"
SRC_REFERENCE = "REFERENCE"
SRC_IMAGE = "IMAGE-DERIVED"
SRC_ESTIMATED = "ESTIMATED"

# ──────────────────────────────────────────────
# Model versioning
# ──────────────────────────────────────────────
MODEL_VERSION = "0.1.0-phase1"
