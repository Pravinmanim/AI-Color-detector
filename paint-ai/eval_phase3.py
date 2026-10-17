import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.baselines import eval_km_baseline, eval_xgboost_baseline
from models.forward_mlp import ForwardMLP, ResidualKMMLP, PaintDataset, train_model
from simulator.kubelka_munk import recipe_to_lab

def main():
    data_path = os.path.join("data", "simulated", "simulated_dataset.csv")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    
    # Feature columns and Target columns
    X = df[['red_pct', 'blue_pct', 'yellow_pct', 'white_pct']]
    y = df[['L', 'a', 'b']]
    
    print("\n--- Baseline 1: Analytical Kubelka-Munk ---")
    km_metrics = eval_km_baseline(X, y)
    for k, v in km_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    print("\n--- Baseline 2: XGBoost Regressor ---")
    xgb_metrics = eval_xgboost_baseline(X, y)
    for k, v in xgb_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    print("\n--- Baseline 3: MLP (From Scratch) ---")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = PaintDataset(X_train, y_train)
    val_dataset = PaintDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    mlp = ForwardMLP(hidden_dim=128)
    mlp_metrics, _ = train_model(
        mlp, train_loader, val_loader, 
        epochs=100, lr=1e-3, device=device
    )
    for k, v in mlp_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    print("\n--- Stage A: Residual MLP (KM + MLP) ---")
    # Pre-calculate KM predictions for train and val
    def get_km_preds(df_x):
        preds = []
        for _, row in df_x.iterrows():
            L, a, b = recipe_to_lab(row['red_pct'], row['blue_pct'], row['yellow_pct'], row['white_pct'])
            preds.append([L, a, b])
        return torch.tensor(preds, dtype=torch.float32)
        
    print("Pre-computing KM predictions...")
    km_train_preds = get_km_preds(X_train)
    km_val_preds = get_km_preds(X_val)
    
    residual_mlp = ResidualKMMLP(hidden_dim=64)
    res_metrics, _ = train_model(
        residual_mlp, train_loader, val_loader, 
        epochs=100, lr=1e-3, device=device,
        use_residual=True, km_train=km_train_preds, km_val=km_val_preds
    )
    for k, v in res_metrics.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
