import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# ==============================
# PART (A): PSF Preparation
# ==============================

def load_and_normalize_psf(path):
    """
    Load PSF image, convert to float, normalize sum to 1.
    """
    psf = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if psf is None:
        raise FileNotFoundError(f"Cannot load PSF image at {path}")
    
    psf = psf.astype(np.float32)
    psf /= psf.sum()  # normalize so sum = 1
    return psf

def resize_psf(psf, sizes, output_dir="part2_output"):
    """
    Resize PSF to multiple sizes and save them.
    sizes: list of integers for square PSFs (e.g., [200, 100, 50, 10])
    """
    os.makedirs(output_dir, exist_ok=True)
    resized_versions = {}
    
    for s in sizes:
        resized = cv2.resize(psf, (s, s), interpolation=cv2.INTER_AREA)
        resized /= resized.sum()  # normalize again
        psf_img = (resized / resized.max() * 255).astype(np.uint8)
        out_path = os.path.join(output_dir, f"psf_{s}x{s}.png")
        cv2.imwrite(out_path, psf_img)
        resized_versions[s] = resized
        print(f"Saved normalized PSF at {out_path}")
    
    return resized_versions


# ==============================
# PART (B): Spatial-Domain Convolution
# ==============================

def create_impulse_image(size=(200, 200), impulses=[(100, 100), (50, 150), (150, 50)]):
    """
    Create an image with a few isolated impulses.
    """
    img = np.zeros(size, dtype=np.float32)
    for (y, x) in impulses:
        if 0 <= y < size[0] and 0 <= x < size[1]:
            img[y, x] = 1.0  # bright impulses
    return img

def save_image_normalized(img, path):
    """
    Normalize an image for visibility before saving.
    """
    img_norm = img / img.max() if img.max() > 0 else img
    img_uint8 = (img_norm * 255).astype(np.uint8)
    cv2.imwrite(path, img_uint8)

def convolve_spatial(img, psf):
    return convolve2d(img, psf, mode='same', boundary='symm')

def main():
    output_dir = "part2_output"
    os.makedirs(output_dir, exist_ok=True)

    # -----------------
    # Part (a)
    # -----------------
    psf_path = "test-images/psf.png"  # Original 200x200 PSF
    psf = load_and_normalize_psf(psf_path)
    sizes = [200, 100, 50, 10]
    psf_versions = resize_psf(psf, sizes, output_dir=output_dir)

    # -----------------
    # Part (b)
    # -----------------
    # Create impulse image once and save
    impulse_path = os.path.join(output_dir, "impulse_original.png")
    if not os.path.exists(impulse_path):
        impulse_img = create_impulse_image(size=(200, 200))
        save_image_normalized(impulse_img, impulse_path)
        print(f"Created impulse image at {impulse_path}")
    else:
        impulse_img = cv2.imread(impulse_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        print(f"Loaded existing impulse image from {impulse_path}")

    # Load a rectangular photograph
    photo_path = "test-images/stars.jpeg"
    photo = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
    if photo is None:
        raise FileNotFoundError('Cannot find the {photo_path} image.')
    photo = photo.astype(np.float32) / 255.0
    print(f"Loaded photograph shape: {photo.shape}")

    # Use smallest PSF (10x10)
    psf_small = psf_versions[10]

    # Convolve with impulse image
    blurred_impulse = convolve_spatial(impulse_img, psf_small)
    save_image_normalized(blurred_impulse, os.path.join(output_dir, "blurred_impulse.png"))

    # Convolve with real photo
    blurred_photo = convolve_spatial(photo, psf_small)
    save_image_normalized(blurred_photo, os.path.join(output_dir, "blurred_photo.png"))

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(impulse_img, cmap='gray')
    plt.title("Original Impulse Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(blurred_impulse, cmap='gray', vmin=0, vmax=blurred_impulse.max())
    plt.title("Blurred Impulse Image (with PSF)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(blurred_photo, cmap='gray')
    plt.title("Blurred Photograph (Spatial Convolution)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "part2b_results.png"), dpi=200)
    plt.show()

    print("✅ Part (b) results saved in:", output_dir)


if __name__ == "__main__":
    main()