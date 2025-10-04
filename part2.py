import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# ==============================
# PART (A): PSF Preparation
# ==============================

def load_and_normalize_psf(path):
    psf = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if psf is None:
        raise FileNotFoundError(f"Cannot load PSF image at {path}")
    psf = psf.astype(np.float32)
    psf /= psf.sum()
    return psf

def resize_psf(psf, sizes, output_dir="part2_output"):
    os.makedirs(output_dir, exist_ok=True)
    resized_versions = {}
    for s in sizes:
        resized = cv2.resize(psf, (s, s), interpolation=cv2.INTER_AREA)
        resized /= resized.sum()
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
    img = np.zeros(size, dtype=np.float32)
    for (y, x) in impulses:
        if 0 <= y < size[0] and 0 <= x < size[1]:
            img[y, x] = 1.0
    return img

def save_image_normalized(img, path):
    img_norm = img / img.max() if img.max() > 0 else img
    img_uint8 = (img_norm * 255).astype(np.uint8)
    cv2.imwrite(path, img_uint8)

def convolve_spatial(img, psf):
    return convolve2d(img, psf, mode='same', boundary='symm')


# ==============================
# PART (C): Frequency-Domain Convolution
# ==============================

def convolve_frequency(img, psf, output_dir, label):
    H, W = img.shape
    h, w = psf.shape

    psf_padded = np.zeros_like(img)
    y0, x0 = H//2 - h//2, W//2 - w//2
    psf_padded[y0:y0+h, x0:x0+w] = psf

    F_img = np.fft.fft2(img)
    F_psf = np.fft.fft2(np.fft.ifftshift(psf_padded))
    F_blurred = F_img * F_psf
    blurred = np.real(np.fft.ifft2(F_blurred))

    def spectrum(x):
        return np.log(1 + np.abs(np.fft.fftshift(x)))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(spectrum(F_img), cmap='gray')
    plt.title("FFT of Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(spectrum(F_psf), cmap='gray')
    plt.title("FFT of PSF")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(spectrum(F_blurred), cmap='gray')
    plt.title("FFT of Product (Blurred)")
    plt.axis('off')

    plt.tight_layout()
    inter_path = os.path.join(output_dir, f"fft_intermediates_{label}.png")
    plt.savefig(inter_path, dpi=200)
    plt.show()
    print(f"✅ Saved intermediate FFT visualization: {inter_path}")

    return blurred


# ==============================
# PART (D): Quantitative Comparison
# ==============================

def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def psnr(img1, img2):
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return float('inf')
    max_pixel = 1.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_val))

def compare_results(spatial, freq, name, output_dir):
    diff = np.abs(spatial - freq)
    mse_val = mse(spatial, freq)
    psnr_val = psnr(spatial, freq)
    max_diff = np.max(diff)

    print(f"\n=== Comparison for {name} ===")
    print(f"Mean Squared Error (MSE): {mse_val:.6e}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr_val:.2f} dB")
    print(f"Maximum Absolute Difference: {max_diff:.6e}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(spatial, cmap='gray')
    plt.title(f"{name}: Spatial Convolution")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(freq, cmap='gray')
    plt.title(f"{name}: Frequency Convolution")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap='hot')
    plt.title(f"{name}: Absolute Difference")
    plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"comparison_{name}.png")
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"✅ Saved comparison visualization for {name} at {save_path}")


# ==============================
# MAIN
# ==============================

def main():
    output_dir = "part2_output"
    os.makedirs(output_dir, exist_ok=True)

    # --- Part (a)
    psf_path = "test-images/psf.png"
    psf = load_and_normalize_psf(psf_path)
    sizes = [200, 100, 50, 10]
    psf_versions = resize_psf(psf, sizes, output_dir=output_dir)

    # --- Part (b)
    print("\n=== Running Part (b): Spatial-Domain Convolution ===")
    impulse_path = os.path.join(output_dir, "impulse_original.png")
    if not os.path.exists(impulse_path):
        impulse_img = create_impulse_image(size=(200, 200))
        save_image_normalized(impulse_img, impulse_path)
        print(f"Created impulse image at {impulse_path}")
    else:
        impulse_img = cv2.imread(impulse_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        print(f"Loaded existing impulse image from {impulse_path}")

    photo_path = "test-images/stars.jpeg"
    photo = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
    if photo is None:
        raise FileNotFoundError(f"Cannot find image {photo_path}")
    photo = photo.astype(np.float32) / 255.0
    psf_small = psf_versions[10]

    blurred_impulse_spatial = convolve_spatial(impulse_img, psf_small)
    blurred_photo_spatial = convolve_spatial(photo, psf_small)
    save_image_normalized(blurred_impulse_spatial, os.path.join(output_dir, "blurred_impulse_spatial.png"))
    save_image_normalized(blurred_photo_spatial, os.path.join(output_dir, "blurred_photo_spatial.png"))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(blurred_impulse_spatial, cmap='gray')
    plt.title("Part (b): Impulse Blurred (Spatial)")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(blurred_photo_spatial, cmap='gray')
    plt.title("Part (b): Photo Blurred (Spatial)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "part2b_results.png"), dpi=200)
    plt.show()

    # --- Part (c)
    print("\n=== Running Part (c): Frequency-Domain Convolution ===")
    blurred_impulse_freq = convolve_frequency(impulse_img, psf_small, output_dir, label="impulse")
    blurred_photo_freq = convolve_frequency(photo, psf_small, output_dir, label="photo")

    save_image_normalized(blurred_impulse_freq, os.path.join(output_dir, "blurred_impulse_freq.png"))
    save_image_normalized(blurred_photo_freq, os.path.join(output_dir, "blurred_photo_freq.png"))

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(blurred_impulse_spatial, cmap='gray')
    plt.title("Impulse - Spatial")
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(blurred_impulse_freq, cmap='gray')
    plt.title("Impulse - Frequency")
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(blurred_photo_spatial, cmap='gray')
    plt.title("Photo - Spatial")
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(blurred_photo_freq, cmap='gray')
    plt.title("Photo - Frequency")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "part2c_final_comparison.png"), dpi=200)
    plt.show()

    # --- Part (d)
    print("\n=== Running Part (d): Quantitative Comparison ===")
    compare_results(blurred_impulse_spatial, blurred_impulse_freq, "Impulse", output_dir)
    compare_results(blurred_photo_spatial, blurred_photo_freq, "Photo", output_dir)


if __name__ == "__main__":
    main()