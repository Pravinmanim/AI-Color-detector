"""
End-to-end training pipeline for Paint AI.

Usage:
    python train.py                       # Train with defaults
    python train.py --epochs 200          # Override epochs
    python train.py --regenerate-data     # Force re-generate simulated data
    python train.py --hidden-dim 128      # Larger network
    python train.py --model-tag v1        # Custom model filename tag

Pipeline:
    1. Generate simulated dataset (if missing or --regenerate-data)
    2. Pre-compute KM predictions for residual formulation
    3. Train ResidualKMMLP (KM + MLP correction) with early stopping
    4. Save best model checkpoint to models/checkpoints/
    5. Evaluate on held-out test set with CIEDE2000 metrics
    6. Print summary report

DATA SOURCE: SIMULATED (training data from KM simulator)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PRETRAIN_LR, WEIGHT_DECAY, EARLY_STOP_PATIENCE,
    MODEL_VERSION, TARGET_DE, LOOSE_DE, DE_BANDS,
)
from models.forward_mlp import ForwardMLP, ResidualKMMLP, PaintDataset
from simulator.kubelka_munk import recipe_to_lab
from simulator.dataset_generator import generate_dataset
from color.delta_e import ciede2000


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def compute_km_predictions(df_x: pd.DataFrame) -> torch.Tensor:
    """Pre-compute Kubelka-Munk Lab predictions for all recipes."""
    preds = []
    for _, row in df_x.iterrows():
        L, a, b = recipe_to_lab(
            row['red_pct'], row['blue_pct'],
            row['yellow_pct'], row['white_pct']
        )
        preds.append([L, a, b])
    return torch.tensor(preds, dtype=torch.float32)


def evaluate_ciede2000(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute CIEDE2000 metrics between true and predicted Lab values."""
    errors = []
    for (L1, a1, b1), (L2, a2, b2) in zip(y_true, y_pred):
        errors.append(ciede2000(L1, a1, b1, L2, a2, b2))
    errors = np.array(errors)

    # Count quality bands
    band_counts = {}
    for lo, hi, label, action in DE_BANDS:
        count = int(np.sum((errors >= lo) & (errors < hi)))
        band_counts[label] = count

    return {
        "mean_de00": float(np.mean(errors)),
        "median_de00": float(np.median(errors)),
        "p95_de00": float(np.percentile(errors, 95)),
        "p99_de00": float(np.percentile(errors, 99)),
        "max_de00": float(np.max(errors)),
        "pct_under_1": float(np.mean(errors < TARGET_DE) * 100),
        "pct_under_2": float(np.mean(errors < LOOSE_DE) * 100),
        "n_samples": len(errors),
        "quality_bands": band_counts,
    }


def print_metrics(metrics: dict, title: str = "Evaluation"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    print(f"  Mean  dE00:  {metrics['mean_de00']:.4f}")
    print(f"  Median dE00: {metrics['median_de00']:.4f}")
    print(f"  95th % dE00: {metrics['p95_de00']:.4f}")
    print(f"  99th % dE00: {metrics['p99_de00']:.4f}")
    print(f"  Max    dE00: {metrics['max_de00']:.4f}")
    print(f"  %% < 1.0 dE: {metrics['pct_under_1']:.1f}%")
    print(f"  %% < 2.0 dE: {metrics['pct_under_2']:.1f}%")
    print(f"  Samples:     {metrics['n_samples']}")
    print(f"  " + "-" * 37)
    print(f"  Quality breakdown:")
    for band, count in metrics['quality_bands'].items():
        pct = count / metrics['n_samples'] * 100
        print(f"    {band:30s}  {count:5d}  ({pct:.1f}%)")
    print(f"{'=' * 50}\n")


# ──────────────────────────────────────────────
# Training loop with early stopping
# ──────────────────────────────────────────────

def train_residual_model(
    model: ResidualKMMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    km_train: torch.Tensor,
    km_val: torch.Tensor,
    epochs: int = 100,
    lr: float = PRETRAIN_LR,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = EARLY_STOP_PATIENCE,
    device: torch.device = torch.device('cpu'),
) -> tuple:
    """
    Train the ResidualKMMLP with early stopping on validation loss.
    Returns: (best_model_state_dict, training_history)
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    stale_epochs = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    print(f"\nTraining ResidualKMMLP for up to {epochs} epochs (patience={patience})")
    print(f"  LR: {lr}, Weight decay: {weight_decay}")
    print(f"  Device: {device}")
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    print("-" * 55)

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for i, (X_b, y_b) in enumerate(train_loader):
            X_b, y_b = X_b.to(device), y_b.to(device)
            start = i * train_loader.batch_size
            end = start + X_b.size(0)
            km_b = km_train[start:end].to(device)

            optimizer.zero_grad()
            preds = model(X_b, km_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_b.size(0)

        train_loss /= len(train_loader.dataset)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (X_b, y_b) in enumerate(val_loader):
                X_b, y_b = X_b.to(device), y_b.to(device)
                start = i * val_loader.batch_size
                end = start + X_b.size(0)
                km_b = km_val[start:end].to(device)

                preds = model(X_b, km_b)
                loss = criterion(preds, y_b)
                val_loss += loss.item() * X_b.size(0)

        val_loss /= len(val_loader.dataset)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
            marker = " * best"
        else:
            stale_epochs += 1
            marker = ""

        if (epoch + 1) % 10 == 0 or epoch == 0 or stale_epochs == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"lr={current_lr:.2e}{marker}")

        if stale_epochs >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    return best_state, history


# ──────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────

def run_pipeline(args):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Step 1: Data ──
    data_path = os.path.join("data", "simulated", "simulated_dataset.csv")

    if args.regenerate_data or not os.path.exists(data_path):
        print("\n[1/5] Generating simulated dataset...")
        data_path = generate_dataset(
            n_dirichlet=args.n_samples,
            n_repeats=1,
            include_edges=True,
            include_tints=True,
            n_lot_batches=3,
            seed=42,
        )
    else:
        print(f"\n[1/5] Using existing dataset: {data_path}")

    df = pd.read_csv(data_path)
    print(f"      Loaded {len(df)} samples")

    X = df[['red_pct', 'blue_pct', 'yellow_pct', 'white_pct']]
    y = df[['L', 'a', 'b']]

    # ── Step 2: Split ──
    print("\n[2/5] Splitting data (70/15/15 train/val/test)...")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, random_state=42  # 0.176 ≈ 15/85
    )
    print(f"      Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Step 3: Pre-compute KM predictions ──
    print("\n[3/5] Pre-computing Kubelka-Munk predictions...")
    km_train = compute_km_predictions(X_train)
    km_val = compute_km_predictions(X_val)
    km_test = compute_km_predictions(X_test)
    print("      Done.")

    # Evaluate KM baseline on test set
    km_baseline_metrics = evaluate_ciede2000(y_test.values, km_test.numpy())
    print_metrics(km_baseline_metrics, "KM Analytical Baseline (Test Set)")

    # ── Step 4: Train ──
    print("[4/5] Training ResidualKMMLP...")
    train_dataset = PaintDataset(X_train, y_train)
    val_dataset = PaintDataset(X_val, y_val)

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ResidualKMMLP(hidden_dim=args.hidden_dim)
    best_state, history = train_residual_model(
        model, train_loader, val_loader,
        km_train, km_val,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
    )

    # ── Step 5: Evaluate on test set ──
    print("\n[5/5] Evaluating on held-out test set...")
    model.eval()
    model.to(device)

    test_dataset = PaintDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for i, (X_b, _) in enumerate(test_loader):
            X_b = X_b.to(device)
            start = i * batch_size
            end = start + X_b.size(0)
            km_b = km_test[start:end].to(device)
            preds = model(X_b, km_b)
            all_preds.append(preds.cpu().numpy())

    y_pred = np.vstack(all_preds)
    test_metrics = evaluate_ciede2000(y_test.values, y_pred)
    print_metrics(test_metrics, "ResidualKMMLP (Test Set)")

    # ── Save checkpoint ──
    checkpoint_dir = os.path.join("models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    tag = args.model_tag or f"h{args.hidden_dim}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"residual_km_mlp_{tag}_{timestamp}.pt"
    model_path = os.path.join(checkpoint_dir, model_filename)

    checkpoint = {
        "model_state_dict": best_state,
        "hidden_dim": args.hidden_dim,
        "model_version": MODEL_VERSION,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "test_metrics": test_metrics,
        "km_baseline_metrics": km_baseline_metrics,
        "training_history": {
            "final_train_loss": history["train_loss"][-1],
            "best_val_loss": min(history["val_loss"]),
            "epochs_trained": len(history["train_loss"]),
        },
        "args": vars(args),
        "timestamp": timestamp,
    }
    torch.save(checkpoint, model_path)
    print(f"  Model saved: {model_path}")

    # Also save as "latest" symlink / copy for API
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(checkpoint, latest_path)
    print(f"  Latest copy: {latest_path}")

    # ── Save metrics report ──
    report_path = os.path.join(checkpoint_dir, f"report_{tag}_{timestamp}.json")
    report = {
        "model_file": model_filename,
        "km_baseline": km_baseline_metrics,
        "residual_km_mlp": test_metrics,
        "improvement": {
            "mean_de00_reduction": km_baseline_metrics["mean_de00"] - test_metrics["mean_de00"],
            "pct_under_1_improvement": test_metrics["pct_under_1"] - km_baseline_metrics["pct_under_1"],
        },
        "config": vars(args),
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Report:      {report_path}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 50}")

    return test_metrics, model_path


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train Paint AI forward model (ResidualKMMLP)"
    )
    # Data
    parser.add_argument("--regenerate-data", action="store_true",
                        help="Force regenerate simulated dataset")
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Number of Dirichlet samples to generate (default: 5000)")

    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden dimension for MLP (default: 64)")

    # Training
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max training epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=PRETRAIN_LR,
                        help=f"Learning rate (default: {PRETRAIN_LR})")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help=f"Weight decay (default: {WEIGHT_DECAY})")
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE,
                        help=f"Early stopping patience (default: {EARLY_STOP_PATIENCE})")

    # Output
    parser.add_argument("--model-tag", type=str, default=None,
                        help="Tag for model filename (default: h{hidden_dim})")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
