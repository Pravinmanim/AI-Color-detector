import os
import sys
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import torch
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.forward_mlp import ResidualKMMLP
from models.inverse_solver import InverseSolver
from vision.extractor import extract_lab_from_image
from color.names import resolve_colour_name, CSS_COLORS, COMMON_COLORS
from color.conversions import hex_to_rgb, srgb255_to_lab, lab_to_srgb255
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Paint AI API", description="Predicts R:B:Y:W paint recipes for target colours.")

# Load models at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model checkpoint if available
_checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "models", "checkpoints", "latest.pt")
if os.path.exists(_checkpoint_path):
    _ckpt = torch.load(_checkpoint_path, map_location=device, weights_only=False)
    _hidden_dim = _ckpt.get("hidden_dim", 64)
    model = ResidualKMMLP(hidden_dim=_hidden_dim).to(device)
    model.load_state_dict(_ckpt["model_state_dict"])
    print(f"[Paint AI] Loaded trained model from {_checkpoint_path}")
    if "test_metrics" in _ckpt:
        print(f"[Paint AI] Test dE00: mean={_ckpt['test_metrics']['mean_de00']:.3f}, "
              f"median={_ckpt['test_metrics']['median_de00']:.3f}")
else:
    print(f"[Paint AI] No checkpoint found at {_checkpoint_path}, using untrained model")
    model = ResidualKMMLP(hidden_dim=64).to(device)

model.eval()
solver = InverseSolver(model, device)

class TargetColor(BaseModel):
    L: float
    a: float
    b: float

class RGBColor(BaseModel):
    r: int
    g: int
    b: int

class CorrectionRequest(BaseModel):
    current_R_g: float
    current_B_g: float
    current_Y_g: float
    current_W_g: float
    target_L: float
    target_a: float
    target_b: float

def get_nearest_color_name(r, g, b):
    best_dist = float('inf')
    best_name = "Unknown"
    all_colors = {**CSS_COLORS, **COMMON_COLORS}
    for name, hex_val in all_colors.items():
        cr, cg, cb = hex_to_rgb(hex_val)
        dist = (r - cr)**2 + (g - cg)**2 + (b - cb)**2
        if dist < best_dist:
            best_dist = dist
            best_name = name.title()
    return best_name

@app.post("/api/rgb_to_lab")
def rgb_to_lab(color: RGBColor):
    """Convert RGB to Lab and return formatted values."""
    L, a, b = srgb255_to_lab(color.r, color.g, color.b)
    hex_str = f"#{color.r:02x}{color.g:02x}{color.b:02x}"
    return {
        "L": round(L, 2),
        "a": round(a, 2),
        "b": round(b, 2),
        "rgb": [color.r, color.g, color.b],
        "hex": hex_str,
        "name": get_nearest_color_name(color.r, color.g, color.b)
    }

@app.post("/api/predict_recipe")
def predict_recipe(target: TargetColor):
    """Phase 6: Solve for the initial R:B:Y:W recipe."""
    try:
        result = solver.solve(target.L, target.a, target.b, n_starts=3, steps=100)
        
        target_rgb = lab_to_srgb255(target.L, target.a, target.b)
        pred_rgb = lab_to_srgb255(*result["predicted_Lab"])
        
        result["target_rgb"] = list(target_rgb)
        result["target_hex"] = f"#{target_rgb[0]:02x}{target_rgb[1]:02x}{target_rgb[2]:02x}"
        result["predicted_rgb"] = list(pred_rgb)
        result["predicted_hex"] = f"#{pred_rgb[0]:02x}{pred_rgb[1]:02x}{pred_rgb[2]:02x}"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/correct_recipe")
def correct_recipe(req: CorrectionRequest):
    """Phase 7: Compute add-only corrections to fix a mixed batch."""
    try:
        current_g = [req.current_R_g, req.current_B_g, req.current_Y_g, req.current_W_g]
        result = solver.solve_correction(
            current_g, 
            req.target_L, req.target_a, req.target_b,
            steps=100
        )
        pred_rgb = lab_to_srgb255(*result["predicted_Lab"])
        result["predicted_rgb"] = list(pred_rgb)
        result["predicted_hex"] = f"#{pred_rgb[0]:02x}{pred_rgb[1]:02x}{pred_rgb[2]:02x}"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract_color")
async def extract_color(file: UploadFile = File(...)):
    """Phase 8: Vision - Extract Lab color from uploaded image."""
    try:
        os.makedirs("temp", exist_ok=True)
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        (L, a, b_val), rgb = extract_lab_from_image(file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
            
        hex_val = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        return {
            "L": round(L, 2),
            "a": round(a, 2),
            "b": round(b_val, 2),
            "rgb": list(rgb),
            "hex": hex_val,
            "name": get_nearest_color_name(*rgb)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/resolve_name")
def resolve_name(name_or_hex: str = Form(...)):
    """Resolve a colour name or hex string to Lab."""
    raw = name_or_hex.strip()
    
    # Check for hex
    clean_hex = raw.lstrip("#")
    if len(clean_hex) in (3, 6) and all(c in "0123456789abcdefABCDEF" for c in clean_hex):
        try:
            r, g, b = hex_to_rgb(raw)
            L, a, b_val = srgb255_to_lab(r, g, b)
            hex_str = f"#{r:02x}{g:02x}{b:02x}"
            return {
                "L": round(L, 2), "a": round(a, 2), "b": round(b_val, 2),
                "rgb": [r, g, b], "hex": hex_str, "name": hex_str,
                "source": "hex"
            }
        except ValueError:
            pass
            
    resolved = resolve_colour_name(raw)
    if not resolved:
        raise HTTPException(status_code=404, detail="Colour name not found.")
        
    L, a, b_val = resolved.lab
    return {
        "L": round(L, 2), "a": round(a, 2), "b": round(b_val, 2), 
        "rgb": list(resolved.rgb), "hex": resolved.hex,
        "name": resolved.name, "source": resolved.source
    }

# Mount static files UI at root
ui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")
if os.path.exists(ui_path):
    app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui")

