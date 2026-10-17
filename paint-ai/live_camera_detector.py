"""
Paint AI - Real-time OpenCV Camera Color & Recipe Detector.

Run this script to start your webcam:
    python live_camera_detector.py

Controls:
    • Point camera at any color swatch or object.
    • Click with mouse anywhere on the frame to sample that exact pixel/region.
    • Press [SPACE] to lock color and solve the R:B:Y:W paint recipe.
    • Press [C] to return to the center reticle.
    • Press [+] / [-] to adjust sample box size.
    • Press [S] to save an annotated snapshot image.
    • Press [Q] or [ESC] to quit.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision.camera_detector import main

if __name__ == "__main__":
    main()
