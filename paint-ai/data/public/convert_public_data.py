import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.kubelka_munk import reflectance_to_xyz
from color.conversions import xyz_to_lab
from config import SRC_PUBLIC_MEASURED

def process_multispectral_dataset(
    input_csv_path: str, 
    output_csv_path: str,
    dataset_name: str,
    licence_verified: str
):
    """
    Template for converting a public multispectral dataset to our CIELAB standard.
    
    Expected input format:
    A CSV where each row is a sample, containing columns for reflectance at 
    specific wavelengths (e.g., '400nm', '410nm', ... '700nm').
    
    If the dataset uses 400-700nm, we pad with zeros for 380, 390, 710, 720, 730nm
    to match our 36-channel 380-730nm grid for D65 integration.
    """
    print(f"[{SRC_PUBLIC_MEASURED}] Processing {dataset_name}...")
    
    if not os.path.exists(input_csv_path):
        print(f"Warning: Dataset not found at {input_csv_path}. Please acquire legally and verify {licence_verified} licence.")
        return

    df = pd.read_csv(input_csv_path)
    output_rows = []
    
    # 36 channels: 380 to 730 nm in 10 nm steps
    target_wavelengths = list(range(380, 740, 10))
    
    for idx, row in df.iterrows():
        reflectance = []
        for w in target_wavelengths:
            col_name = f"{w}nm"
            if col_name in df.columns:
                reflectance.append(float(row[col_name]))
            else:
                # Pad missing boundary wavelengths with 0 or nearest neighbor if needed
                reflectance.append(0.0)
                
        # Convert to XYZ then Lab
        X, Y, Z = reflectance_to_xyz(reflectance)
        # xyz_to_lab handles standard D65 
        L, a, b = xyz_to_lab(X, Y, Z)
        
        output_rows.append({
            "sample_id": f"PUB-{dataset_name}-{idx:05d}",
            "L": round(L, 4),
            "a": round(a, 4),
            "b": round(b, 4),
            "dataset_name": dataset_name,
            "licence": licence_verified,
            "source": SRC_PUBLIC_MEASURED
        })

    out_df = pd.DataFrame(output_rows)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)
    print(f"[{SRC_PUBLIC_MEASURED}] Saved {len(out_df)} samples to {output_csv_path}")

if __name__ == "__main__":
    # Example usage for when datasets are acquired:
    # process_multispectral_dataset(
    #     "data/public/raw/mpi_spectra.csv",
    #     "data/public/mpi_lab.csv",
    #     "MaxPlanck",
    #     "CC-BY-NC-SA"
    # )
    print("Public Data Converter is ready. Awaiting legal acquisition of datasets.")
