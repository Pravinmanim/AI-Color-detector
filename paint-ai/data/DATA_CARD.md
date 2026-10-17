# Data Card: Paint Colour Measurement Protocol

## Overview
This document describes how to collect REAL-OWN measurement data for the
paint-AI system. Following this protocol ensures consistency and allows
the fine-tuning stage to learn accurate pigment properties.

---

## Equipment Needed
- Digital kitchen scale (0.1 g resolution minimum)
- 4 paint tubes: **Red**, **Blue**, **Yellow**, **White** (same brand/batch)
- Mixing palette or disposable cups
- Flat, smooth white card stock (for paint swatches)
- Phone or camera with manual white balance
- Colour reference chart (optional, for camera CCM calibration)
- Thermometer (optional, for recording ambient temperature)

## Paint Preparation
1. **Label each paint tube** with a batch code (e.g., `LOT-001`).
2. **Shake/mix** each tube thoroughly before dispensing.
3. Use the **same brand and type** throughout a measurement session.

## Mixing Protocol
1. **Weigh each pigment individually** on a tared scale.
2. Record the **actual weighed values** (not the target), to 0.1 g precision.
3. Mix thoroughly for at least 60 seconds until visually uniform.
4. Apply a **uniform layer** onto white card stock, approximately 2 mm thick.
5. **Thickness matters**: too thin → substrate shows through; too thick → wasteful.

## Drying / Wet State
- **Choose ONE state and use it for ALL measurements**: either wet or fully dry.
- If dry: allow at least 2 hours for acrylics, 24 hours for oils.
- **Record which state** in the `measurement_condition` column (`WET` or `DRY`).
- Do not mix wet and dry measurements in the same training set.

## Colour Measurement

### Camera-Based (software-only approach)
1. Set up **fixed, diffuse lighting** (overcast daylight or two 5000K lamps at 45°).
2. Place a **white card** next to the swatch for white-balance reference.
3. Take a photo from directly above (avoid shadows, glare, angles).
4. In the app, use the **white-card balance** feature before reading colour.
5. Read the **central region** of the swatch (avoid edges).
6. Record L*, a*, b* values from the app.

### Spectrophotometer (if available)
1. Use D65 illuminant, 2° observer, specular excluded.
2. Average 3 readings per swatch.
3. Record L*, a*, b* values directly.

## Repeatability
- **At least 15% of recipes** should be measured 2–3 times (different swatches
  from the same mixture) to estimate measurement noise.
- Use the same `recipe_id` but different `repeat_id` for repeats.

## Recording in the CSV

| Column | Description | Example |
|---|---|---|
| `sample_id` | Unique ID per measurement | `REAL-000001` |
| `recipe_id` | Groups repeats of the same recipe | `R-00001` |
| `repeat_id` | 0, 1, 2 for repeats | `0` |
| `red_pct` | **Actually weighed** red % | `32.5` |
| `blue_pct` | **Actually weighed** blue % | `0.0` |
| `yellow_pct` | **Actually weighed** yellow % | `17.5` |
| `white_pct` | **Actually weighed** white % | `50.0` |
| `L` | Measured L* | `62.34` |
| `a` | Measured a* | `38.12` |
| `b` | Measured b* | `22.67` |
| `paint_batch` | Batch code of paint used | `LOT-001` |
| `temperature_c` | Ambient temperature °C | `23.5` |
| `measurement_condition` | `WET`, `DRY`, or `CAMERA-WB` | `DRY` |
| `source` | Always `REAL-OWN` | `REAL-OWN` |

## Important Notes
- **Percentages must sum to 100** (within rounding): the system will warn if
  they deviate by more than 0.5%.
- **Never adjust measurements to look better.** Record what you actually see.
- Keep the **lighting identical** across all sessions.
- Store the CSV in `data/real/` and re-run training with the retrain command.

## Recommended Minimum Recipes

For a useful fine-tuning set, measure at least:

| Category | Count | Purpose |
|---|---|---|
| Pure pigments (100% each) | 4 | Anchor the endpoints |
| Two-pigment blends (25/75, 50/50, 75/25) | 18 | Map two-way interactions |
| Tint series (pigment + white, 10% steps) | 27 | Map lightness behaviour |
| Random 3- and 4-pigment mixes | 20–50 | Fill the interior of the gamut |
| **Repeats** (15% of above) | ~10–15 | Estimate noise |
| **Total minimum** | **~70–115** | |

For the learning-curve comparison, aim for 50, 100, 200, and 400 samples
in separate files or use the `recipe_id` column to create train/test splits.

## Data Integrity Checks
The system will automatically check:
- All percentages sum to 100 ± 0.5
- L* is in [0, 100]
- No missing values in required columns
- `source` is always `REAL-OWN`
- At least 10% of recipe_ids have multiple repeat_ids
