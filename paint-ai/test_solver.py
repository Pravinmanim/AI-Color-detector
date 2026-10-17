import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.forward_mlp import ResidualKMMLP
from models.inverse_solver import InverseSolver

def test_inverse_solver():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model (untrained is fine for a mechanics test)
    model = ResidualKMMLP(hidden_dim=64).to(device)
    model.eval()
    
    solver = InverseSolver(model, device)
    
    # Test target (e.g., some shade of red)
    target_L, target_a, target_b = 45.0, 40.0, 25.0
    print(f"Target Lab: L={target_L}, a={target_a}, b={target_b}")
    
    print("\n--- Testing Phase 6: Initial Recipe Solving ---")
    result = solver.solve(target_L, target_a, target_b, n_starts=3, steps=100)
    
    print("\nOptimization Result:")
    print(f"Predicted Recipe (R,B,Y,W): {result['recipe_RBYW']}")
    print(f"Predicted Lab: {result['predicted_Lab']}")
    print(f"Predicted CIEDE2000: {result['predicted_deltaE00']:.4f}")
    print(f"Status: {result['status']}")

    print("\n--- Testing Phase 7: Closed-loop Correction ---")
    # Suppose we mixed the predicted recipe in 10g batch:
    initial_recipe = [r * 0.1 for r in result['recipe_RBYW']]
    print(f"Current Mixed Amounts (g): {initial_recipe}")
    
    # Now ask the solver to fix the error (assuming the initial recipe wasn't perfect)
    correction = solver.solve_correction(
        current_amounts_g=initial_recipe,
        target_L=target_L, target_a=target_a, target_b=target_b,
        steps=100
    )
    
    print("\nCorrection Result:")
    print(f"Add-only amounts (g) (R,B,Y,W): {correction['additions_g_RBYW']}")
    print(f"Predicted New Lab: {correction['predicted_Lab']}")
    print(f"Predicted New CIEDE2000: {correction['predicted_deltaE00']:.4f}")
    print(f"Status: {correction['status']}")

if __name__ == "__main__":
    test_inverse_solver()
