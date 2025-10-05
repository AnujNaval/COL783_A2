import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os

# --- Helper Functions ---

def calculate_psnr(img1, img2):
    """Calculates the Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def add_gaussian_noise_psnr(image, target_psnr):
    """Adds Gaussian noise to an image to achieve a specific target PSNR."""
    image_power = np.mean(np.square(image.astype(np.float64)))
    noise_power = image_power / (10**(target_psnr / 10))
    noise_sigma = np.sqrt(noise_power)
    
    noise = np.random.normal(0, noise_sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def create_ideal_filter(shape, low_cutoff, high_cutoff=None):
    """Creates an ideal circular filter (low-pass or band-pass) in the frequency domain."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Create a meshgrid of coordinates
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    dist_from_center = np.sqrt(x*x + y*y)
    
    # Create low-pass mask
    mask_low = dist_from_center <= low_cutoff
    
    if high_cutoff is None: # Low-pass filter
        return mask_low.astype(float)
    else: # Band-pass filter
        mask_high = dist_from_center <= high_cutoff
        return (mask_high ^ mask_low).astype(float) # XOR for the band

def visualize_frequency_bands(image, num_bands, title, output_folder):
    """Generates and visualizes low-pass and band-pass filtered images."""
    print(f"Visualizing frequency bands for '{title}'...")
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)
    
    fig, axes = plt.subplots(2, num_bands, figsize=(num_bands * 3, 6))
    fig.suptitle(title, fontsize=16)
    
    l_prev = np.zeros_like(image, dtype=np.float64)

    for i in range(num_bands):
        cutoff = (2**(i+1)) - 1
        
        # Low-pass filter l_i
        lp_filter = create_ideal_filter(image.shape, low_cutoff=cutoff)
        l_i_freq = F_shifted * lp_filter
        l_i = np.fft.ifft2(np.fft.ifftshift(l_i_freq)).real
        
        # Band-pass image b_i
        b_i = l_i - l_prev
        
        # Store for next iteration
        l_prev = l_i
        
        # Visualize
        axes[0, i].imshow(np.clip(l_i, 0, 255), cmap='gray')
        axes[0, i].set_title(f'$l_{i+1}$ (0-{cutoff})')
        axes[0, i].axis('off')
        
        # Add 128 for visualization of band-pass image
        b_i_vis = np.clip(b_i + 128, 0, 255)
        axes[1, i].imshow(b_i_vis, cmap='gray')
        axes[1, i].set_title(f'$b_{i+1}$')
        axes[1, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filepath = os.path.join(output_folder, f"freq_bands_{title.replace(' ', '_').lower()}.png")
    plt.savefig(filepath)
    print(f"  -> Saved band visualization to {filepath}")
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    output_folder = "part5_output"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Results will be saved in '{output_folder}'")

    # --- Setup: Load original image and create blur kernel ---
    try:
        # url = "https://upload.wikimedia.org/wikipedia/commons/3/30/Scarlet_Macaw_and_Blue-and-yellow_Macaw.jpg"
        img_f_path = "test-images/blurred_photo.png"
        # urllib.request.urlretrieve(url, img_f_path)
        img_f = cv2.imread(img_f_path, cv2.IMREAD_GRAYSCALE)
        img_f = cv2.resize(img_f, (256, 256))
        cv2.imwrite(os.path.join(output_folder, "0_original_f.png"), img_f)
    except Exception as e:
        print(f"Could not download image. Using a synthetic image. Error: {e}")
        img_f = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img_f, (64, 64), (192, 192), 255, -1)
    
    # Create a significant Gaussian blur kernel
    blur_kernel_1d = cv2.getGaussianKernel(ksize=25, sigma=8)
    blur_kernel_h = blur_kernel_1d @ blur_kernel_1d.T
    
    # Apply blur
    img_blurred = cv2.filter2D(img_f, -1, blur_kernel_h)
    cv2.imwrite(os.path.join(output_folder, "1_blurred.png"), img_blurred)
    
    # Get Fourier Transform of the blur kernel (padded to image size)
    H_fft = np.fft.fft2(blur_kernel_h, s=img_f.shape)

    # --- Loop through different noise levels (for part e) ---
    for psnr_level in [20, 30, 10]:
        print("\n" + "="*60)
        print(f"Processing for Degraded Image with PSNR = {psnr_level} dB")
        print("="*60)

        # Create degraded image g
        img_g = add_gaussian_noise_psnr(img_blurred, target_psnr=psnr_level)
        cv2.imwrite(os.path.join(output_folder, f"2_degraded_g_psnr{psnr_level}.png"), img_g)
        G_fft = np.fft.fft2(img_g)

        # --- Part (a): Visualize Frequency Bands ---
        visualize_frequency_bands(img_f, 4, "Original f", output_folder)
        visualize_frequency_bands(img_g, 4, f"Degraded g (PSNR {psnr_level})", output_folder)

        # --- Part (b): Inverse Filtering ---
        print("\n--- (b) Inverse Filtering ---")
        epsilon = 1e-8 # Avoid division by zero
        F_hat_inv_fft = G_fft / (H_fft + epsilon)
        f_hat_inv = np.fft.ifft2(F_hat_inv_fft).real
        psnr_inv = calculate_psnr(img_f, np.clip(f_hat_inv, 0, 255))
        print(f"  PSNR of Inverse Filtered Image: {psnr_inv:.2f} dB")
        cv2.imwrite(os.path.join(output_folder, f"3_restored_inverse_psnr{psnr_level}.png"), f_hat_inv)
        visualize_frequency_bands(f_hat_inv, 4, f"Restored Inverse (PSNR {psnr_level})", output_folder)
        print("  Comment: Inverse filtering heavily amplifies noise, especially in high-frequency bands where the blur kernel H(u,v) is close to zero.")

        # --- Part (d): Wiener Filtering (finding best K) ---
        print("\n--- (d) Wiener Filtering ---")
        K_values = np.logspace(-4, 2, 50)
        psnr_wiener_scores = []
        for K in K_values:
            wiener_filter = np.conj(H_fft) / (np.abs(H_fft)**2 + K)
            F_hat_wiener_fft = G_fft * wiener_filter
            f_hat_wiener = np.fft.ifft2(F_hat_wiener_fft).real
            psnr_wiener_scores.append(calculate_psnr(img_f, np.clip(f_hat_wiener, 0, 255)))
        
        best_K = K_values[np.argmax(psnr_wiener_scores)]
        best_psnr_wiener = np.max(psnr_wiener_scores)
        print(f"  Best K for Wiener Filter: {best_K:.4f} -> Best PSNR: {best_psnr_wiener:.2f} dB")
        
        # Plot PSNR vs K
        plt.figure()
        plt.semilogx(K_values, psnr_wiener_scores)
        plt.title(f'Wiener Filter Performance (PSNR {psnr_level})')
        plt.xlabel('K (S_n/S_f ratio)')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)
        k_path = os.path.join(output_folder, f"4_plot_wiener_K_psnr{psnr_level}.png")
        plt.savefig(k_path)
        print(f"  Saved Wiener K-plot to {k_path}")
        plt.close()

        # Restore with best K
        wiener_filter = np.conj(H_fft) / (np.abs(H_fft)**2 + best_K)
        f_hat_best_wiener = np.fft.ifft2(G_fft * wiener_filter).real
        cv2.imwrite(os.path.join(output_folder, f"5_restored_wiener_psnr{psnr_level}.png"), f_hat_best_wiener)
        visualize_frequency_bands(f_hat_best_wiener, 4, f"Restored Wiener (PSNR {psnr_level})", output_folder)

        # --- Part (f): Regularized Deconvolution (Optional) ---
        print("\n--- (f) Regularized Deconvolution ---")
        laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        P_fft = np.fft.fft2(laplacian_kernel, s=img_f.shape)

        lambda_values = np.logspace(-8, -2, 50)
        psnr_reg_scores = []
        for lam in lambda_values:
            reg_filter = np.conj(H_fft) / (np.abs(H_fft)**2 + lam * np.abs(P_fft)**2)
            F_hat_reg_fft = G_fft * reg_filter
            f_hat_reg = np.fft.ifft2(F_hat_reg_fft).real
            psnr_reg_scores.append(calculate_psnr(img_f, np.clip(f_hat_reg, 0, 255)))
            
        best_lambda = lambda_values[np.argmax(psnr_reg_scores)]
        best_psnr_reg = np.max(psnr_reg_scores)
        print(f"  Best lambda for Regularization: {best_lambda:.2e} -> Best PSNR: {best_psnr_reg:.2f} dB")

        # Plot PSNR vs Lambda
        plt.figure()
        plt.semilogx(lambda_values, psnr_reg_scores)
        plt.title(f'Regularized Filter Performance (PSNR {psnr_level})')
        plt.xlabel('Lambda (Regularization Parameter)')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)
        lam_path = os.path.join(output_folder, f"6_plot_regularized_lambda_psnr{psnr_level}.png")
        plt.savefig(lam_path)
        print(f"  Saved Regularization lambda-plot to {lam_path}")
        plt.close()

        # Restore with best lambda
        reg_filter = np.conj(H_fft) / (np.abs(H_fft)**2 + best_lambda * np.abs(P_fft)**2)
        f_hat_best_reg = np.fft.ifft2(G_fft * reg_filter).real
        cv2.imwrite(os.path.join(output_folder, f"7_restored_regularized_psnr{psnr_level}.png"), f_hat_best_reg)

    print("\nAll tasks completed successfully.")
