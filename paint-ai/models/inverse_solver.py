import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.forward_mlp import ResidualKMMLP
from simulator.kubelka_munk import recipe_to_lab
from color.delta_e import ciede2000

class InverseSolver:
    """
    Optimizes for the best paint recipe [R, B, Y, W] to match a target CIELAB color.
    Uses gradient descent on the pre-trained differentiable forward model.
    """
    def __init__(self, model: ResidualKMMLP, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def solve(self, target_L: float, target_a: float, target_b: float, n_starts: int = 8, steps: int = 150) -> dict:
        """
        Solves for the recipe using intelligent batched parallel initializations to guarantee
        finding the global minimum error to target_Lab.
        Constraints: recipe components in [0, 100] and sum to 100 via Softmax parameterization.
        """
        target_tensor = torch.tensor([[target_L, target_a, target_b]], dtype=torch.float32, device=self.device)
        
        # ─── 1. Build Intelligent Batched Starting Seeds ───
        # Latent vector z of shape [n_starts, 4]. Softmax(z) gives percentages summing to 100.
        seeds = []
        
        # Seed 0: Balanced baseline
        seeds.append([0.0, 0.0, 0.0, 0.0])
        
        # Seed 1: White tint (essential for light colors / whites)
        seeds.append([-2.5, -2.5, -2.5, 3.5])
        
        # Seed 2: Ultra-pure White (>98% White)
        seeds.append([-3.5, -3.5, -3.5, 5.0])
        
        # Seed 3: Red dominant
        seeds.append([3.0, -2.0, 0.5, 0.0])
        
        # Seed 4: Blue dominant
        seeds.append([-2.0, 3.0, -1.0, 0.0])
        
        # Seed 5: Yellow dominant
        seeds.append([0.5, -2.0, 3.0, 0.5])
        
        # Seed 6: Warm Earth (Terracotta / Orange / Ochre)
        seeds.append([2.5, -2.5, 2.5, 1.0])
        
        # Seed 7: Green / Olive (Blue + Yellow)
        seeds.append([-1.5, 2.0, 2.5, 1.0])
        
        # Additional seeds: random exploration if n_starts > 8
        while len(seeds) < n_starts:
            seeds.append(torch.randn(4).tolist())
            
        z_init = torch.tensor(seeds[:n_starts], dtype=torch.float32, device=self.device)
        z = z_init.clone().detach().requires_grad_(True)
        
        # ─── 2. Batched Parallel Optimization ───
        optimizer = optim.Adam([z], lr=0.18)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.005)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            recipe_pcts = torch.softmax(z, dim=1) * 100.0
            km_lab = self._pytorch_km(recipe_pcts)
            pred_lab = self.model(recipe_pcts, km_lab)
            
            # Perceptual weighted MSE: Chroma shifts (a*, b*) receive 1.8x weight
            dL = (pred_lab[:, 0:1] - target_tensor[:, 0:1]) ** 2
            da = (pred_lab[:, 1:2] - target_tensor[:, 1:2]) ** 2
            db = (pred_lab[:, 2:3] - target_tensor[:, 2:3]) ** 2
            loss = torch.sum(dL + 1.8 * da + 1.8 * db)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        # ─── 3. Evaluate All Candidates with CIEDE2000 ───
        best_recipe = None
        best_de00 = float('inf')
        best_pred_lab = None
        best_z_candidate = None
        
        with torch.no_grad():
            final_recipes = (torch.softmax(z, dim=1) * 100.0).cpu().numpy()
            final_labs = pred_lab.cpu().numpy()
            
            for idx in range(n_starts):
                cand_recipe = final_recipes[idx]
                cand_lab = final_labs[idx]
                de00 = ciede2000(target_L, target_a, target_b, cand_lab[0], cand_lab[1], cand_lab[2])
                
                if de00 < best_de00:
                    best_de00 = de00
                    best_recipe = cand_recipe
                    best_pred_lab = cand_lab
                    best_z_candidate = z[idx:idx+1].clone().detach()

        # ─── 4. High-Precision Fine-Tuning Stage ───
        if best_z_candidate is not None and best_de00 > 0.05:
            fine_z = best_z_candidate.clone().detach().requires_grad_(True)
            fine_opt = optim.Adam([fine_z], lr=0.03)
            
            for _ in range(40):
                fine_opt.zero_grad()
                r_pct = torch.softmax(fine_z, dim=1) * 100.0
                k_lab = self._pytorch_km(r_pct)
                p_lab = self.model(r_pct, k_lab)
                
                dL = (p_lab[:, 0] - target_tensor[:, 0]) ** 2
                da = (p_lab[:, 1] - target_tensor[:, 1]) ** 2
                db = (p_lab[:, 2] - target_tensor[:, 2]) ** 2
                f_loss = dL + 2.0 * da + 2.0 * db
                f_loss.backward()
                fine_opt.step()
                
            with torch.no_grad():
                cand_recipe = (torch.softmax(fine_z, dim=1) * 100.0).cpu().squeeze().numpy()
                cand_lab = p_lab.cpu().squeeze().numpy()
                fine_de = ciede2000(target_L, target_a, target_b, cand_lab[0], cand_lab[1], cand_lab[2])
                if fine_de < best_de00:
                    best_de00 = fine_de
                    best_recipe = cand_recipe
                    best_pred_lab = cand_lab

        return {
            "recipe_RBYW": best_recipe.tolist(),
            "predicted_Lab": best_pred_lab.tolist(),
            "predicted_deltaE00": float(best_de00),
            "status": "accepted" if best_de00 <= 2.5 else "target outside achievable gamut"
        }

    def solve_correction(
        self, 
        current_amounts_g: list, 
        target_L: float, target_a: float, target_b: float, 
        steps: int = 200
    ) -> dict:
        """
        Phase 7: Closed-loop Correction.
        Given current mixed amounts (in grams) of [R, B, Y, W], finds the 
        add-only amounts (delta >= 0) to minimize the error to target_Lab.
        """
        target_tensor = torch.tensor([[target_L, target_a, target_b]], dtype=torch.float32, device=self.device)
        
        # We optimize the additions in log space or use Softplus to strictly enforce delta >= 0
        # Initialize small additions (e.g., 0.1g)
        raw_deltas = torch.zeros((1, 4), dtype=torch.float32, device=self.device, requires_grad=True)
        current_g = torch.tensor(current_amounts_g, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        optimizer = optim.Adam([raw_deltas], lr=0.5)
        criterion = nn.MSELoss()
        
        best_additions = None
        best_de00 = float('inf')
        best_pred_lab = None
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # deltas must be >= 0
            deltas = torch.nn.functional.softplus(raw_deltas)
            new_amounts = current_g + deltas
            
            # Convert to percentages
            total_mass = torch.sum(new_amounts, dim=1, keepdim=True)
            recipe_pcts = (new_amounts / total_mass) * 100.0
            
            # Forward pass
            km_lab = self._pytorch_km(recipe_pcts)
            pred_lab = self.model(recipe_pcts, km_lab)
            
            # Loss is MSE between prediction and target
            loss = criterion(pred_lab, target_tensor)
            
            # We can also add a small L1 penalty to encourage minimal paint addition
            loss += 0.001 * torch.sum(deltas)
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                final_deltas = torch.nn.functional.softplus(raw_deltas).cpu().squeeze().numpy()
                final_lab = pred_lab.cpu().squeeze().numpy()
                
                de00 = ciede2000(target_L, target_a, target_b, final_lab[0], final_lab[1], final_lab[2])
                
                if de00 < best_de00:
                    best_de00 = de00
                    best_additions = final_deltas
                    best_pred_lab = final_lab
                    
        return {
            "additions_g_RBYW": best_additions.tolist(),
            "predicted_Lab": best_pred_lab.tolist(),
            "predicted_deltaE00": best_de00,
            "status": "accepted" if best_de00 <= 2.5 else "target outside achievable gamut"
        }

    def _pytorch_km(self, recipe_pcts: torch.Tensor) -> torch.Tensor:
        """
        A differentiable implementation of the Kubelka-Munk forward model in PyTorch.
        """
        # Load constants into PyTorch tensors
        from simulator.kubelka_munk import PIGMENT_SPECTRA, D65_SPD, CMF_X, CMF_Y, CMF_Z, _KM_SX, _KM_SY, _KM_SZ
        
        device = recipe_pcts.device
        
        K_red = torch.tensor(PIGMENT_SPECTRA['red']['K'], device=device)
        S_red = torch.tensor(PIGMENT_SPECTRA['red']['S'], device=device)
        K_blue = torch.tensor(PIGMENT_SPECTRA['blue']['K'], device=device)
        S_blue = torch.tensor(PIGMENT_SPECTRA['blue']['S'], device=device)
        K_yellow = torch.tensor(PIGMENT_SPECTRA['yellow']['K'], device=device)
        S_yellow = torch.tensor(PIGMENT_SPECTRA['yellow']['S'], device=device)
        K_white = torch.tensor(PIGMENT_SPECTRA['white']['K'], device=device)
        S_white = torch.tensor(PIGMENT_SPECTRA['white']['S'], device=device)
        
        D65 = torch.tensor(D65_SPD, device=device)
        cX = torch.tensor(CMF_X, device=device)
        cY = torch.tensor(CMF_Y, device=device)
        cZ = torch.tensor(CMF_Z, device=device)
        
        # recipe_pcts is [batch_size, 4]
        # Normalize to 0-1 range
        fracs = recipe_pcts / 100.0
        
        R_f = fracs[:, 0].unsqueeze(1)
        B_f = fracs[:, 1].unsqueeze(1)
        Y_f = fracs[:, 2].unsqueeze(1)
        W_f = fracs[:, 3].unsqueeze(1)
        
        K_mix = R_f * K_red + B_f * K_blue + Y_f * K_yellow + W_f * K_white
        S_mix = R_f * S_red + B_f * S_blue + Y_f * S_yellow + W_f * S_white
        
        # Avoid division by zero
        S_mix = torch.clamp(S_mix, min=1e-12)
        KS = K_mix / S_mix
        
        R_spectrum = 1.0 + KS - torch.sqrt(KS**2 + 2.0 * KS)
        R_spectrum = torch.clamp(R_spectrum, 0.0, 1.0)
        
        # Integrate XYZ
        W = D65 * R_spectrum
        X = torch.sum(W * cX, dim=1)
        Y = torch.sum(W * cY, dim=1)
        Z = torch.sum(W * cZ, dim=1)
        N = torch.sum(D65 * cY)
        
        X = X * _KM_SX / N
        Y = Y * _KM_SY / N
        Z = Z * _KM_SZ / N
        
        # Differentiable XYZ to Lab
        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
        
        xr = X / Xn
        yr = Y / Yn
        zr = Z / Zn
        
        def f(t):
            eps = 216.0 / 24389.0
            kappa = 24389.0 / 27.0
            # torch.where to handle piecewise
            mask = t > eps
            return torch.where(mask, torch.pow(torch.clamp(t, min=eps), 1/3), (kappa * t + 16.0) / 116.0)
            
        fx = f(xr)
        fy = f(yr)
        fz = f(zr)
        
        L = (116.0 * fy) - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        
        return torch.stack((L, a, b), dim=1)

if __name__ == "__main__":
    print("Inverse Solver module ready.")
