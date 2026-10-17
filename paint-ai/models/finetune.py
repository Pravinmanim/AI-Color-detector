import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.forward_mlp import ResidualKMMLP

def load_pretrained_model(model_path: str, device: torch.device) -> ResidualKMMLP:
    """
    Loads the Stage A pre-trained ResidualKMMLP model.
    """
    model = ResidualKMMLP(hidden_dim=64)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded pretrained model from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Using uninitialized model.")
    model.to(device)
    return model

def fine_tune_on_real_data(
    model: ResidualKMMLP, 
    real_train_loader: torch.utils.data.DataLoader,
    real_km_preds: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4
):
    """
    Fine-tunes the ResidualKMMLP model exclusively on REAL-OWN data.
    By default, we allow fine-tuning of the MLP weights with a very low learning rate,
    preventing catastrophic forgetting of the physics/synthetic baseline.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    print("Starting Stage B Fine-tuning on REAL-OWN data...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (X_b, y_b) in enumerate(real_train_loader):
            X_b, y_b = X_b.to(device), y_b.to(device)
            km_b = real_km_preds[i*real_train_loader.batch_size:(i+1)*real_train_loader.batch_size].to(device)
            
            optimizer.zero_grad()
            preds = model(X_b, km_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * X_b.size(0)
            
        epoch_loss /= len(real_train_loader.dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Fine-tune Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")
            
    print("Fine-tuning complete.")
    return model

if __name__ == "__main__":
    print("Fine-tuning module ready.")
