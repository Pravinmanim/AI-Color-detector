"""Quick end-to-end test: load trained model -> inverse solve -> verify."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models.forward_mlp import ResidualKMMLP
from models.inverse_solver import InverseSolver

# Load trained checkpoint
ckpt_path = os.path.join("models", "checkpoints", "latest.pt")
assert os.path.exists(ckpt_path), f"No checkpoint at {ckpt_path}"

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = ResidualKMMLP(hidden_dim=ckpt["hidden_dim"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded model (version={ckpt['model_version']}, "
      f"test mean dE00={ckpt['test_metrics']['mean_de00']:.3f})")

solver = InverseSolver(model, torch.device("cpu"))

# Test cases: target Lab values
test_targets = [
    (50.0, 30.0, 20.0, "Warm red-orange"),
    (70.0, -10.0, 40.0, "Muted yellow-green"),
    (85.0, 5.0, 10.0, "Light warm tint"),
    (30.0, 10.0, -30.0, "Dark blue-violet"),
    (95.0, 0.0, 2.0,   "Near-white"),
]

print(f"\n{'='*70}")
print(f"  Inverse Solver Test (trained model)")
print(f"{'='*70}")

all_pass = True
for L, a, b, desc in test_targets:
    result = solver.solve(L, a, b, n_starts=5, steps=200)
    r_pct = result["recipe_RBYW"]
    pred = result["predicted_Lab"]
    de = result["predicted_deltaE00"]
    status = result["status"]
    
    ok = "PASS" if de < 2.5 else "WARN"
    if de >= 5.0:
        ok = "FAIL"
        all_pass = False
    
    print(f"\n  [{ok}] {desc}")
    print(f"    Target:  L={L:.1f}  a={a:.1f}  b={b:.1f}")
    print(f"    Predict: L={pred[0]:.2f}  a={pred[1]:.2f}  b={pred[2]:.2f}")
    print(f"    Recipe:  R={r_pct[0]:.1f}%  B={r_pct[1]:.1f}%  Y={r_pct[2]:.1f}%  W={r_pct[3]:.1f}%")
    print(f"    dE00={de:.4f}  status={status}")

# Test correction
print(f"\n{'='*70}")
print(f"  Correction Test")
print(f"{'='*70}")

current_g = [3.0, 1.0, 2.0, 4.0]  # 10g batch
target_L, target_a, target_b = 70.0, -10.0, 40.0

correction = solver.solve_correction(current_g, target_L, target_a, target_b, steps=200)
adds = correction["additions_g_RBYW"]
c_pred = correction["predicted_Lab"]
c_de = correction["predicted_deltaE00"]

print(f"\n  Current batch: R={current_g[0]}g  B={current_g[1]}g  Y={current_g[2]}g  W={current_g[3]}g")
print(f"  Target: L={target_L}  a={target_a}  b={target_b}")
print(f"  Add:    R=+{adds[0]:.2f}g  B=+{adds[1]:.2f}g  Y=+{adds[2]:.2f}g  W=+{adds[3]:.2f}g")
print(f"  Result: L={c_pred[0]:.2f}  a={c_pred[1]:.2f}  b={c_pred[2]:.2f}")
print(f"  dE00={c_de:.4f}  status={correction['status']}")

print(f"\n{'='*70}")
print(f"  All tests {'PASSED' if all_pass else 'had warnings'}")
print(f"{'='*70}")
