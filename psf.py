import numpy as np
import cv2
import os

def generate_disk_psf(size=200, radius=50):
    """
    Generate a circular disk Point Spread Function (PSF).
    Bright disk on black background, normalized to sum = 1.
    """
    psf = np.zeros((size, size), dtype=np.float32)
    y, x = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    psf[mask] = 1.0
    psf /= psf.sum()  # normalize so sum = 1
    return psf

def save_psf_image(path, size=200, radius=50):
    psf = generate_disk_psf(size, radius)
    # Scale to 0-255 for saving as PNG
    psf_img = (psf / psf.max() * 255).astype(np.uint8)
    cv2.imwrite(path, psf_img)
    print(f"PSF image saved at {path}")

if __name__ == "__main__":
    output_dir = "test-images"
    os.makedirs(output_dir, exist_ok=True)
    
    save_psf_image(os.path.join(output_dir, "psf.png"), size=200, radius=50)
