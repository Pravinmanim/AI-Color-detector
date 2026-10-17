from PIL import Image
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from color.conversions import srgb255_to_lab

def extract_lab_from_image(image_path: str, box: tuple = None) -> tuple:
    """
    Reads an image, optionally crops it to a bounding box (left, top, right, bottom),
    calculates the average sRGB color, and converts it to CIELAB.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    img = Image.open(image_path).convert('RGB')
    
    if box:
        img = img.crop(box)
        
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Calculate average RGB across all pixels in the crop/image
    # (In a real system, you might want to cluster or find the dominant color instead of mean)
    mean_rgb = np.mean(img_array, axis=(0, 1))
    
    # Convert to standard sRGB 255 space
    r, g, b = int(round(mean_rgb[0])), int(round(mean_rgb[1])), int(round(mean_rgb[2]))
    
    # Convert to CIELAB
    L, a, b_val = srgb255_to_lab(r, g, b)
    
    return (L, a, b_val), (r, g, b)

if __name__ == "__main__":
    print("Vision extractor ready.")
