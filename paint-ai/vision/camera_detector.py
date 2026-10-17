"""
Real-time OpenCV Camera Color Detector & Paint Recipe Analyzer.

Uses a connected webcam to analyze colors in real time, extract sRGB
and CIELAB color values, identify the nearest named color, and compute
physics-informed R:B:Y:W paint mixing recipes via the trained model.
"""

from __future__ import annotations
import os
import sys
import time
import argparse
from typing import Tuple, Optional, Dict, List
import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from color.conversions import srgb255_to_lab, lab_to_srgb255
from color.names import CSS_COLORS, COMMON_COLORS

# Optional PyTorch inverse solver import
try:
    import torch
    from models.forward_mlp import ResidualKMMLP
    from models.inverse_solver import InverseSolver
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def build_color_database() -> List[Tuple[str, str, Tuple[int, int, int]]]:
    """
    Builds a precomputed list of (name, hex, (r, g, b)) for fast nearest-color lookup.
    """
    db = []
    seen_hex = set()

    for name, hex_str in {**CSS_COLORS, **COMMON_COLORS}.items():
        clean_hex = hex_str.lstrip("#")
        if clean_hex in seen_hex:
            continue
        seen_hex.add(clean_hex)
        r = int(clean_hex[0:2], 16)
        g = int(clean_hex[2:4], 16)
        b = int(clean_hex[4:6], 16)
        db.append((name.title(), hex_str.upper(), (r, g, b)))
    return db


COLOR_DB = build_color_database()


def find_nearest_color(r: int, g: int, b: int) -> Tuple[str, str, float]:
    """
    Finds the closest color name in the database using weighted Euclidean distance.
    (Weighted for human eye sensitivity: 0.30 R + 0.59 G + 0.11 B)
    """
    best_name = "Unknown"
    best_hex = f"#{r:02X}{g:02X}{b:02X}"
    best_dist = float("inf")

    for name, hex_str, (db_r, db_g, db_b) in COLOR_DB:
        dr = r - db_r
        dg = g - db_g
        db = b - db_b
        # Human visual perception weighting
        dist = 0.3 * (dr ** 2) + 0.59 * (dg ** 2) + 0.11 * (db ** 2)
        if dist < best_dist:
            best_dist = dist
            best_name = name
            best_hex = hex_str

    return best_name, best_hex, best_dist ** 0.5


class CameraColorDetector:
    """
    OpenCV-based real-time camera color analyzer.
    Features:
      - Center crosshair or mouse-click ROI sampling
      - Temporal noise smoothing
      - Real-time RGB, CIELAB, HEX, and closest color name
      - Paint AI recipe prediction using trained ResidualKMMLP
      - Heads-Up Display (HUD) overlay
    """

    def __init__(self, camera_index: int = 0, sample_radius: int = 15, enable_solver: bool = True):
        self.camera_index = camera_index
        self.sample_radius = sample_radius
        self.enable_solver = enable_solver

        # Target point: None means center of frame
        self.target_point: Optional[Tuple[int, int]] = None
        self.is_center_mode = True

        # Temporal smoothing
        self.smoothed_rgb = np.array([128.0, 128.0, 128.0], dtype=np.float32)
        self.smoothing_factor = 0.35  # Exponential moving average

        # White-point calibration and lighting compensation
        self.white_ref = [255.0, 255.0, 255.0]
        self.is_white_calibrated = False
        self.exposure_gain = 1.25
        self.saturation_gain = 1.15
        self.raw_sampled_rgb = (128, 128, 128)

        # State
        self.is_locked = False
        self.locked_color = None
        self.latest_recipe = None

        # Initialize solver if available
        self.solver = None
        self.device = None
        if self.enable_solver and TORCH_AVAILABLE:
            self._load_solver()

    def _load_solver(self):
        ckpt_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "latest.pt")
        if not os.path.exists(ckpt_path):
            print(f"[CameraDetector] Note: No checkpoint found at {ckpt_path}. Solver disabled.")
            return

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            hidden_dim = ckpt.get("hidden_dim", 64)
            model = ResidualKMMLP(hidden_dim=hidden_dim).to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.solver = InverseSolver(model, self.device)
            print(f"[CameraDetector] Loaded trained Paint AI model on {self.device}")
        except Exception as e:
            print(f"[CameraDetector] Could not load model: {e}")

    def on_mouse(self, event, x, y, flags, param):
        """Mouse callback: sample color where user clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.target_point = (x, y)
            self.is_center_mode = False
            self.is_locked = False
            self.latest_recipe = None

    def sample_roi(self, frame: np.ndarray, cx: int, cy: int) -> Tuple[int, int, int]:
        """
        Samples a window around (cx, cy), applies median blur to suppress
        camera sensor noise, and returns (R, G, B) integer tuple.
        """
        h, w = frame.shape[:2]
        r = self.sample_radius

        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(w, cx + r)
        y2 = min(h, cy + r)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 128, 128, 128

        # Median filter to eliminate pixel sparkle/grain
        filtered = cv2.medianBlur(roi, 5) if min(roi.shape[:2]) >= 5 else roi
        mean_bgr = np.mean(filtered, axis=(0, 1))

        # Exponential smoothing
        raw_rgb = np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float32)
        self.smoothed_rgb = (1 - self.smoothing_factor) * self.smoothed_rgb + self.smoothing_factor * raw_rgb

        r_raw = int(round(np.clip(self.smoothed_rgb[0], 0, 255)))
        g_raw = int(round(np.clip(self.smoothed_rgb[1], 0, 255)))
        b_raw = int(round(np.clip(self.smoothed_rgb[2], 0, 255)))
        self.raw_sampled_rgb = (r_raw, g_raw, b_raw)

        return self.apply_calibration(r_raw, g_raw, b_raw)

    def apply_calibration(self, r_raw: int, g_raw: int, b_raw: int) -> Tuple[int, int, int]:
        """Applies chromatic adaptation and exposure/vibrancy boost."""
        r_cal = r_raw * (255.0 / self.white_ref[0]) * self.exposure_gain
        g_cal = g_raw * (255.0 / self.white_ref[1]) * self.exposure_gain
        b_cal = b_raw * (255.0 / self.white_ref[2]) * self.exposure_gain

        # Saturation compensation
        gray = 0.299 * r_cal + 0.587 * g_cal + 0.114 * b_cal
        r_cal = gray + (r_cal - gray) * self.saturation_gain
        g_cal = gray + (g_cal - gray) * self.saturation_gain
        b_cal = gray + (b_cal - gray) * self.saturation_gain

        return (
            int(np.clip(round(r_cal), 0, 255)),
            int(np.clip(round(g_cal), 0, 255)),
            int(np.clip(round(b_cal), 0, 255)),
        )

    def calibrate_white_point(self):
        """Sets reference white point to eliminate gloomy indoor lighting."""
        raw_r, raw_g, raw_b = self.raw_sampled_rgb
        self.white_ref = [max(float(raw_r), 35.0), max(float(raw_g), 35.0), max(float(raw_b), 35.0)]
        self.is_white_calibrated = True
        print(f"\n[CALIBRATION] Calibrated White Reference: R={raw_r}, G={raw_g}, B={raw_b}")
        print("              Gloominess eliminated! Colors now normalized against D65 reference.")

    def reset_white_point(self):
        """Resets calibration to standard defaults."""
        self.white_ref = [255.0, 255.0, 255.0]
        self.is_white_calibrated = False
        self.exposure_gain = 1.25
        print("\n[CALIBRATION] Reset calibration to default exposure.")

    def solve_recipe(self, L: float, a: float, b_val: float):
        """Solves for the R:B:Y:W paint recipe using upgraded high-precision solver."""
        if not self.solver:
            return
        try:
            res = self.solver.solve(L, a, b_val, n_starts=8, steps=150)
            self.latest_recipe = res
        except Exception as e:
            print(f"[CameraDetector] Solve error: {e}")

    def draw_hud(self, frame: np.ndarray, cx: int, cy: int, r: int, g: int, b: int,
                 L: float, a: float, b_val: float, name: str, hex_str: str) -> np.ndarray:
        """Draws the Heads-Up Display (HUD) overlay on the frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 1. Draw Sampling Reticle
        rad = self.sample_radius
        x1, y1 = max(0, cx - rad), max(0, cy - rad)
        x2, y2 = min(w, cx + rad), min(h, cy + rad)

        reticle_color = (0, 255, 128) if self.is_locked else (0, 200, 255)
        # Corner brackets
        corner_len = max(6, rad // 2)
        # Top-left
        cv2.line(overlay, (x1, y1), (x1 + corner_len, y1), reticle_color, 2)
        cv2.line(overlay, (x1, y1), (x1, y1 + corner_len), reticle_color, 2)
        # Top-right
        cv2.line(overlay, (x2, y1), (x2 - corner_len, y1), reticle_color, 2)
        cv2.line(overlay, (x2, y1), (x2, y1 + corner_len), reticle_color, 2)
        # Bottom-left
        cv2.line(overlay, (x1, y2), (x1 + corner_len, y2), reticle_color, 2)
        cv2.line(overlay, (x1, y2), (x1, y2 - corner_len), reticle_color, 2)
        # Bottom-right
        cv2.line(overlay, (x2, y2), (x2 - corner_len, y2), reticle_color, 2)
        cv2.line(overlay, (x2, y2), (x2, y2 - corner_len), reticle_color, 2)
        # Center dot
        cv2.circle(overlay, (cx, cy), 2, (0, 255, 255), -1)

        # 2. Bottom Information Panel (Dark Glass)
        panel_h = 130
        panel_top = h - panel_h
        cv2.rectangle(overlay, (0, panel_top), (w, h), (14, 18, 28), -1)
        # Border line
        cv2.line(overlay, (0, panel_top), (w, panel_top), (60, 75, 105), 1)

        # 3. Color Swatch Box (Left side of panel)
        swatch_x, swatch_y = 16, panel_top + 16
        swatch_size = 96
        # Draw detected color
        bgr_color = (b, g, r)
        cv2.rectangle(overlay, (swatch_x, swatch_y), (swatch_x + swatch_size, swatch_y + swatch_size), bgr_color, -1)
        cv2.rectangle(overlay, (swatch_x, swatch_y), (swatch_x + swatch_size, swatch_y + swatch_size), (255, 255, 255), 2)

        # 4. Color Text Readouts
        text_x = swatch_x + swatch_size + 20
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Line 1: Color Name & Lock Status & Calibration Badge
        lock_txt = "[LOCKED]" if self.is_locked else "[LIVE]"
        lock_color = (0, 255, 128) if self.is_locked else (0, 200, 255)
        calib_txt = "[WHITE BALANCED]" if self.is_white_calibrated else f"[BOOST: {self.exposure_gain:.2f}x]"
        calib_color = (255, 255, 255) if self.is_white_calibrated else (120, 180, 240)

        cv2.putText(overlay, f"{name}", (text_x, panel_top + 34), font, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, lock_txt, (text_x + 210, panel_top + 34), font, 0.5, lock_color, 2, cv2.LINE_AA)
        cv2.putText(overlay, calib_txt, (text_x + 300, panel_top + 34), font, 0.45, calib_color, 1, cv2.LINE_AA)

        # Line 2: RGB & HEX
        rgb_hex_text = f"RGB: ({r}, {g}, {b})   HEX: {hex_str}"
        cv2.putText(overlay, rgb_hex_text, (text_x, panel_top + 64), font, 0.55, (200, 220, 240), 1, cv2.LINE_AA)

        # Line 3: CIELAB
        lab_text = f"CIELAB: L*={L:.1f}  a*={a:.1f}  b*={b_val:.1f}"
        cv2.putText(overlay, lab_text, (text_x, panel_top + 90), font, 0.5, (140, 160, 190), 1, cv2.LINE_AA)

        # Line 4: Instructions
        instr = "SPACE: Lock/Recipe | W: Calibrate White | R: Reset Calib | B: Brightness | S: Save | Q: Quit"
        cv2.putText(overlay, instr, (text_x, panel_top + 116), font, 0.38, (120, 140, 170), 1, cv2.LINE_AA)

        # 5. Right Side: Paint Recipe Bar Display (if calculated)
        if self.latest_recipe:
            recipe = self.latest_recipe["recipe_RBYW"]
            de00 = self.latest_recipe["predicted_deltaE00"]
            bar_start_x = w - 210
            bar_w = 110
            bar_h = 12

            cv2.putText(overlay, f"Paint AI Recipe (dE00: {de00:.2f})", (bar_start_x - 10, panel_top + 26),
                        font, 0.42, (0, 255, 200), 1, cv2.LINE_AA)

            colors_meta = [
                ("R", recipe[0], (40, 40, 220)),
                ("B", recipe[1], (220, 120, 30)),
                ("Y", recipe[2], (30, 210, 240)),
                ("W", recipe[3], (230, 230, 230)),
            ]

            for idx, (lbl, pct, col) in enumerate(colors_meta):
                by = panel_top + 46 + (idx * 20)
                cv2.putText(overlay, f"{lbl}:", (bar_start_x - 10, by + 10), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.rectangle(overlay, (bar_start_x + 10, by), (bar_start_x + 10 + bar_w, by + bar_h), (35, 45, 60), -1)
                fill_w = int((pct / 100.0) * bar_w)
                if fill_w > 0:
                    cv2.rectangle(overlay, (bar_start_x + 10, by), (bar_start_x + 10 + fill_w, by + bar_h), col, -1)
                cv2.putText(overlay, f"{pct:.1f}%", (bar_start_x + 15 + bar_w, by + 10), font, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

        # Alpha blend HUD overlay
        alpha = 0.92
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def run(self):
        """Starts the live camera loop."""
        print(f"[CameraDetector] Opening camera index {self.camera_index}...")
        # Use DirectShow backend on Windows to prevent MSMF sample drop/empty frame errors
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.camera_index)
        else:
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print(f"[CameraDetector] ERROR: Could not open camera {self.camera_index}.")
            print("Please ensure your webcam is plugged in and not in use by another application.")
            return

        window_name = "Paint AI - Real-time Color & Recipe Detector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        cv2.setMouseCallback(window_name, self.on_mouse)

        print("[CameraDetector] Live camera started!")
        print("  • Point camera at any color swatch / object.")
        print("  • Press [W] while pointing at white paper to Calibrate White Reference!")
        print("  • Press [B] to toggle exposure / brightness boost.")
        print("  • Press [SPACE] to lock color and calculate Paint Recipe.")
        print("  • Press [C] to return to center reticle.")
        print("  • Press [S] to save snapshot.")
        print("  • Press [Q] or [ESC] to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[CameraDetector] Warning: Empty frame received.")
                    break

                h, w = frame.shape[:2]

                # Determine target coordinate
                if self.is_center_mode or self.target_point is None:
                    cx, cy = w // 2, h // 2
                else:
                    cx, cy = self.target_point
                    cx = max(0, min(w - 1, cx))
                    cy = max(0, min(h - 1, cy))

                # If locked, retain locked values; otherwise sample
                if not self.is_locked:
                    r, g, b = self.sample_roi(frame, cx, cy)
                    L, a, b_val = srgb255_to_lab(r, g, b)
                    hex_str = f"#{r:02X}{g:02X}{b:02X}"
                    name, _, _ = find_nearest_color(r, g, b)
                    self.locked_color = (r, g, b, L, a, b_val, name, hex_str)
                else:
                    r, g, b, L, a, b_val, name, hex_str = self.locked_color

                # Draw HUD
                annotated = self.draw_hud(frame, cx, cy, r, g, b, L, a, b_val, name, hex_str)
                cv2.imshow(window_name, annotated)

                key = cv2.waitKey(15) & 0xFF

                # Q or ESC -> Exit
                if key in (ord('q'), ord('Q'), 27):
                    break

                # SPACE -> Lock & Solve
                elif key == ord(' '):
                    self.is_locked = not self.is_locked
                    if self.is_locked:
                        print(f"\n[LOCKED] RGB: ({r}, {g}, {b}) | HEX: {hex_str} | Name: {name}")
                        print(f"         CIELAB: L*={L:.2f}, a*={a:.2f}, b*={b_val:.2f}")
                        if self.solver:
                            print("         Solving paint recipe with neural network...")
                            self.solve_recipe(L, a, b_val)
                            if self.latest_recipe:
                                rec = self.latest_recipe["recipe_RBYW"]
                                de = self.latest_recipe["predicted_deltaE00"]
                                print(f"         Recipe: Red={rec[0]:.1f}%, Blue={rec[1]:.1f}%, Yellow={rec[2]:.1f}%, White={rec[3]:.1f}% (dE00: {de:.2f})")
                    else:
                        print("[UNLOCKED] Returning to live sampling.")
                        self.latest_recipe = None

                # W -> Calibrate White Point
                elif key in (ord('w'), ord('W')):
                    self.calibrate_white_point()

                # R -> Reset White Point Calibration
                elif key in (ord('r'), ord('R')):
                    self.reset_white_point()

                # B -> Cycle Exposure Gain Boost
                elif key in (ord('b'), ord('B')):
                    if self.exposure_gain < 1.2:
                        self.exposure_gain = 1.35
                    elif self.exposure_gain < 1.5:
                        self.exposure_gain = 1.6
                    else:
                        self.exposure_gain = 1.0
                    print(f"[EXPOSURE] Boost set to {self.exposure_gain:.2f}x")

                # C -> Reset to Center
                elif key in (ord('c'), ord('C')):
                    self.is_center_mode = True
                    self.target_point = None
                    self.is_locked = False
                    self.latest_recipe = None
                    print("[MODE] Reset target to center reticle.")

                # S -> Save snapshot
                elif key in (ord('s'), ord('S')):
                    os.makedirs("snapshots", exist_ok=True)
                    filename = f"snapshots/color_{hex_str.lstrip('#')}_{int(time.time())}.png"
                    cv2.imwrite(filename, annotated)
                    print(f"[SAVED] Screenshot saved to: {filename}")

                # + / = -> Increase sampling radius
                elif key in (ord('+'), ord('=')):
                    self.sample_radius = min(60, self.sample_radius + 5)
                    print(f"[RADIUS] Sample window: {self.sample_radius * 2}x{self.sample_radius * 2} px")

                # - / _ -> Decrease sampling radius
                elif key in (ord('-'), ord('_')):
                    self.sample_radius = max(5, self.sample_radius - 5)
                    print(f"[RADIUS] Sample window: {self.sample_radius * 2}x{self.sample_radius * 2} px")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[CameraDetector] Camera released. Clean exit.")


def main():
    parser = argparse.ArgumentParser(description="Paint AI Real-time OpenCV Camera Color Detector")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--radius", "-r", type=int, default=15, help="Sampling radius in pixels (default: 15)")
    parser.add_argument("--no-solver", action="store_true", help="Disable recipe solver")
    args = parser.parse_args()

    detector = CameraColorDetector(
        camera_index=args.camera,
        sample_radius=args.radius,
        enable_solver=not args.no_solver
    )
    detector.run()


if __name__ == "__main__":
    main()
