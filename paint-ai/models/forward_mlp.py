import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from color.delta_e import ciede2000
from simulator.kubelka_munk import recipe_to_lab

class PaintDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ForwardMLP(nn.Module):
    """
    Direct MLP: R,B,Y,W (percentages) -> L,a,b
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        
    def forward(self, x):
        # x is assumed to be normalized or bounded between 0-100
        # Normalizing to 0-1 can help, assuming input sums to 100
        x_norm = x / 100.0
        return self.net(x_norm)

class ResidualKMMLP(nn.Module):
    """
    Residual formulation for Stage A.
    L,a,b = KM(R,B,Y,W) + MLP(R,B,Y,W)
    The MLP learns to correct the analytical model's errors (e.g., interaction effects).
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.mlp = ForwardMLP(hidden_dim=hidden_dim)
        
    def forward(self, x, km_preds):
        """
        x: Recipe percentages [batch, 4]
        km_preds: Pre-calculated Kubelka-Munk predictions for x [batch, 3]
        """
        # The MLP predicts the delta to add to KM predictions
        correction = self.mlp(x)
        return km_preds + correction

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cpu', use_residual=False, km_train=None, km_val=None):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, (X_b, y_b) in enumerate(train_loader):
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            
            if use_residual:
                # Need KM predictions for the batch
                km_b = km_train[i*train_loader.batch_size:(i+1)*train_loader.batch_size].to(device)
                preds = model(X_b, km_b)
            else:
                preds = model(X_b)
                
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_b.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for i, (X_b, y_b) in enumerate(val_loader):
                X_b, y_b = X_b.to(device), y_b.to(device)
                if use_residual:
                    km_b = km_val[i*val_loader.batch_size:(i+1)*val_loader.batch_size].to(device)
                    preds = model(X_b, km_b)
                else:
                    preds = model(X_b)
                
                loss = criterion(preds, y_b)
                val_loss += loss.item() * X_b.size(0)
                val_preds.append(preds.cpu().numpy())
                val_trues.append(y_b.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # Load best model and evaluate CIEDE2000
    model.load_state_dict(best_model_state)
    model.eval()
    
    val_preds = np.vstack(val_preds)
    val_trues = np.vstack(val_trues)
    
    de2000_errors = []
    for (L1, a1, b1), (L2, a2, b2) in zip(val_trues, val_preds):
        de2000_errors.append(ciede2000(L1, a1, b1, L2, a2, b2))
        
    return {
        "val_loss": best_val_loss,
        "mean_de00": np.mean(de2000_errors),
        "median_de00": np.median(de2000_errors),
        "max_de00": np.max(de2000_errors),
        "95th_p_de00": np.percentile(de2000_errors, 95)
    }, model
