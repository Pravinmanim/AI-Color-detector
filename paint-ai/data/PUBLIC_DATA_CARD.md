# Public Dataset Evaluation & Licensing

This document outlines the evaluation, usage constraints, and integration plan for public multispectral datasets.

## 1. Max Planck Institute (MPI) Multispectral Image Database
- **Content:** Contains multispectral images of everyday objects, including paint swatches and colour checkers.
- **Licence:** Typically restricted to non-commercial academic research (e.g., CC BY-NC-SA or custom institutional licence). 
- **Constraint Check:** MUST verify the specific licence of the dataset downloaded. If strictly Non-Commercial, it cannot be used for a commercial product.
- **Integration Plan:** If the licence permits, we extract the reflectance spectra (typically 400-700nm at 10nm intervals) of the paint/colour patches, convert them to XYZ (using our standard `reflectance_to_xyz` with D65), then to CIELAB. The source column MUST be `PUBLIC-MEASURED`.

## 2. NTU (Nanyang Technological University) Multispectral Dataset
- **Content:** Multispectral imaging data for varying materials.
- **Licence:** Often open for research, but commercial usage requires explicit verification.
- **Constraint Check:** Verify exact terms.
- **Integration Plan:** Similar to MPI, extract the 31 or 36 spectral bands, integrate under D65, and store as CIELAB with `PUBLIC-MEASURED` source.

## Strict Rule Adherence
> **NEVER** invent dataset contents, papers, or licences. If a dataset's licence is unavailable or ambiguous, we **will not** include it in the training pipeline. 

## Standardized Target Format
Any converted public dataset will be saved in `data/public/` using the following schema:
- `sample_id`: Unique ID (e.g., `PUB-MPI-001`)
- `L`, `a`, `b`: Converted CIELAB values (D65, 2-degree observer)
- `source`: Strictly `PUBLIC-MEASURED`
- `dataset_name`: Name of the source (e.g., `MaxPlanck`)
- `licence`: Shortcode of the verified licence

*Note: Since we do not have the physical dataset files downloaded locally at this moment, the `convert_public_data.py` script serves as a robust template/pipeline for when the data is legally acquired and mounted.*
