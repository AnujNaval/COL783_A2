import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os

# --- Part 1: Helper Functions (PSNR, Noise Generation, Filters) ---

def calculate_psnr(img1, img2):
    """Calculates the Peak Signal-to-Noise Ratio between two images."""
    # Ensure images are float type for mse calculation
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def add_noise(image, noise_type, target_psnr=20.0):
    """
    Adds specified noise to an image to achieve a target PSNR.
    """
    print(f"Adding {noise_type} noise to achieve PSNR approx {target_psnr} dB...")
    
    # Calculate the target MSE from the target PSNR
    target_mse = (255.0 ** 2) / (10 ** (target_psnr / 10.0))
    
    if noise_type == "uniform":
        # For zero-mean uniform noise U(-L, L), variance = L^2 / 3 = MSE
        L = np.sqrt(3 * target_mse)
        noise = np.random.uniform(-L, L, image.shape)
        noisy_image = image + noise
    
    elif noise_type == "gaussian":
        # For Gaussian noise, variance = sigma^2 = MSE
        sigma = np.sqrt(target_mse)
        noise = np.random.normal(0, sigma, image.shape)
        noisy_image = image + noise
        
    elif noise_type == "salt_pepper":
        noisy_image = image.copy()
        # For salt-and-pepper, we need to find the density 'p'
        # MSE for S&P is approx. p * ( (255-I)^2 + (0-I)^2 ) / 2
        # We can solve for 'p' or use an iterative approach.
        # Here we use an iterative search to find the best density.
        best_p = 0
        min_psnr_diff = float('inf')

        for p_candidate in np.linspace(0.01, 1.0, 100):
            temp_img = image.copy()
            s_vs_p = 0.5  # Equal salt vs pepper
            amount = p_candidate
            
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            temp_img[tuple(coords)] = 255

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            temp_img[tuple(coords)] = 0

            current_psnr = calculate_psnr(image, temp_img)
            diff = abs(current_psnr - target_psnr)
            if diff < min_psnr_diff:
                min_psnr_diff = diff
                best_p = p_candidate
        
        # Now generate the final noisy image with the best p found
        s_vs_p = 0.5
        amount = best_p
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[tuple(coords)] = 255
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[tuple(coords)] = 0
    else:
        raise ValueError("Unknown noise type")

    # Clip values to be in the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    actual_psnr = calculate_psnr(image, noisy_image)
    print(f"  -> Actual PSNR: {actual_psnr:.2f} dB")
    return noisy_image

def mean_filter(image, w):
    """Applies an arithmetic mean filter of size w x w from scratch."""
    pad = w // 2
    img_padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    denoised_image = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = img_padded[i:i+w, j:j+w]
            denoised_image[i, j] = np.mean(neighborhood)
    return denoised_image.astype(np.uint8)

def median_filter(image, w):
    """Applies a median filter of size w x w from scratch."""
    pad = w // 2
    img_padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    denoised_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = img_padded[i:i+w, j:j+w]
            denoised_image[i, j] = np.median(neighborhood)
    return denoised_image

def non_local_means_filter(image, search_window_m, patch_m, h):
    """
    Applies a Non-Local Means filter from scratch.
    search_window_m: Side length of the search window (e.g., 21).
    patch_m: Side length of the patch for comparison (e.g., 7).
    h: Degree of filtering, related to noise standard deviation.
    """
    print("Applying Non-Local Means filter (this is slow)...")
    img_float = image.astype(np.float64)
    denoised_image = np.zeros_like(img_float)
    
    patch_rad = patch_m // 2
    search_rad = search_window_m // 2
    
    # Pad the image to handle borders.
    # CRITICAL FIX: The padding must be done on the float image to ensure
    # all subsequent calculations are done with high precision.
    pad_width = search_rad + patch_rad
    img_padded = cv2.copyMakeBorder(img_float, pad_width, pad_width, pad_width, pad_width, cv2.BORDER_REFLECT)

    h2 = h * h
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Center coordinates in the padded image
            pi, pj = i + pad_width, j + pad_width
            
            # Extract the main patch P(i,j)
            patch_main = img_padded[pi-patch_rad : pi+patch_rad+1, pj-patch_rad : pj+patch_rad+1]
            
            # Dictionaries to store weighted sums and total weights
            total_weight = 0.0
            pixel_weighted_sum = 0.0
            
            # Iterate through the search window
            for sr in range(-search_rad, search_rad + 1):
                for sc in range(-search_rad, search_rad + 1):
                    # Coordinates of the comparison patch's center
                    qi, qj = pi + sr, pj + sc
                    
                    # Extract the comparison patch P(q)
                    patch_comp = img_padded[qi-patch_rad : qi+patch_rad+1, qj-patch_rad : qj+patch_rad+1]
                    
                    # Calculate squared Euclidean distance between patches
                    # This now correctly uses float64 arithmetic, preventing overflow.
                    dist_sq = np.sum((patch_main - patch_comp)**2)
                    
                    # Calculate weight
                    weight = np.exp(-dist_sq / h2)
                    
                    total_weight += weight
                    pixel_weighted_sum += weight * img_padded[qi, qj]
            
            # Normalize and assign the new pixel value
            # Add a small epsilon to prevent division by zero in case all weights are zero
            denoised_image[i, j] = pixel_weighted_sum / (total_weight + 1e-8)
        
        # Progress indicator
        if (i + 1) % (image.shape[0] // 10 or 1) == 0:
            print(f"  ...processed NLM row {i+1}/{image.shape[0]}")
            
    print("NLM filtering complete.")
    return np.clip(denoised_image, 0, 255).astype(np.uint8)


def analyze_and_plot(original_image, noisy_image, noise_type, image_name, output_folder):
    """Runs filters, calculates PSNR, finds the best, and plots results."""
    print(f"\n--- Analyzing {image_name} with {noise_type} noise ---")
    w_range = range(3, 22, 2)  # Filter widths w = 3, 5, ..., 21
    psnr_mean = []
    psnr_median = []
    
    best_psnr_mean = -1
    best_img_mean = None
    best_w_mean = 0

    best_psnr_median = -1
    best_img_median = None
    best_w_median = 0

    for w in w_range:
        print(f"Testing filter width w={w}...")
        # Mean filter
        denoised_mean = mean_filter(noisy_image, w)
        current_psnr_mean = calculate_psnr(original_image, denoised_mean)
        psnr_mean.append(current_psnr_mean)
        if current_psnr_mean > best_psnr_mean:
            best_psnr_mean = current_psnr_mean
            best_img_mean = denoised_mean
            best_w_mean = w

        # Median filter
        denoised_median = median_filter(noisy_image, w)
        current_psnr_median = calculate_psnr(original_image, denoised_median)
        psnr_median.append(current_psnr_median)
        if current_psnr_median > best_psnr_median:
            best_psnr_median = current_psnr_median
            best_img_median = denoised_median
            best_w_median = w
    
    # Save the best result images
    mean_filename = os.path.join(output_folder, f"best_mean_{image_name}_{noise_type}.png")
    median_filename = os.path.join(output_folder, f"best_median_{image_name}_{noise_type}.png")
    cv2.imwrite(mean_filename, best_img_mean)
    cv2.imwrite(median_filename, best_img_median)
    print(f"Saved best mean-filtered image to {mean_filename} (w={best_w_mean}, PSNR={best_psnr_mean:.2f})")
    print(f"Saved best median-filtered image to {median_filename} (w={best_w_median}, PSNR={best_psnr_median:.2f})")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(w_range, psnr_mean, 'o-', label='Arithmetic Mean Filter')
    plt.plot(w_range, psnr_median, 's-', label='Median Filter')
    plt.title(f'Denoising Performance on {image_name} with {noise_type.title()} Noise')
    plt.xlabel('Filter Width (w)')
    plt.ylabel('PSNR (dB)')
    plt.xticks(w_range)
    plt.grid(True)
    plt.legend()
    plot_filename = os.path.join(output_folder, f"plot_{image_name}_{noise_type}.png")
    plt.savefig(plot_filename)
    print(f"Saved plot to {plot_filename}")
    plt.close() # Close plot to free memory
    
    return best_img_mean, best_img_median


# --- Main Execution ---
if __name__ == "__main__":
    output_folder = "part4_output"
    os.makedirs(output_folder, exist_ok=True)

    # --- Part a: Image & Noise Creation ---
    # Create constant image 'c'
    img_c = np.full((256, 256), 128, dtype=np.uint8)
    cv2.imwrite(os.path.join(output_folder, "0_original_constant.png"), img_c)

    # Download and load natural image 'f'
    try:
        # url = "https://upload.wikimedia.org/wikipedia/commons/3/30/Scarlet_Macaw_and_Blue-and-yellow_Macaw.jpg"
        # img_f_path = os.path.join(output_folder, "0_original_natural.jpg")
        # urllib.request.urlretrieve(url, img_f_path)
        img_f_path = "./test-images/macaw.jpg"
        img_f_bgr = cv2.imread(img_f_path)
        img_f = cv2.cvtColor(img_f_bgr, cv2.COLOR_BGR2GRAY)
        # Resize for faster processing, especially for NLM
        cv2.imshow("image",img_f)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img_f = cv2.resize(img_f, (256, 256))
        cv2.imwrite(os.path.join(output_folder, "0_original_natural_gray.png"), img_f)
    except Exception as e:
        print(f"Could not download natural image, using constant image instead. Error: {e}")
        img_f = img_c.copy()

    # Generate noisy images
    c1 = add_noise(img_c, "uniform")
    c2 = add_noise(img_c, "gaussian")
    c3 = add_noise(img_c, "salt_pepper")
    cv2.imwrite(os.path.join(output_folder, "c1_uniform.png"), c1)
    cv2.imwrite(os.path.join(output_folder, "c2_gaussian.png"), c2)
    cv2.imwrite(os.path.join(output_folder, "c3_salt_pepper.png"), c3)

    f1 = add_noise(img_f, "uniform")
    f2 = add_noise(img_f, "gaussian")
    f3 = add_noise(img_f, "salt_pepper")
    cv2.imwrite(os.path.join(output_folder, "f1_uniform.png"), f1)
    cv2.imwrite(os.path.join(output_folder, "f2_gaussian.png"), f2)
    cv2.imwrite(os.path.join(output_folder, "f3_salt_pepper.png"), f3)

    # --- Part b: Denoise Constant Images & Plot ---
    analyze_and_plot(img_c, c1, "uniform", "constant", output_folder)
    analyze_and_plot(img_c, c2, "gaussian", "constant", output_folder)
    analyze_and_plot(img_c, c3, "salt_pepper", "constant", output_folder)

    # --- Part c: Denoise Natural Images & Plot ---
    print("\n" + "="*50)
    print("Explanation for Part (c): Why PSNR increases and then decreases with 'w' for natural images.")
    print("Initially, as the filter width 'w' increases, the filter is large enough to average out more of the random noise, causing the PSNR to rise.")
    print("However, beyond an optimal width, the filter becomes too large. It starts to average over important image features (edges, textures), causing significant blurring.")
    print("This blurring represents a deviation from the original image, increasing the Mean Squared Error and thus causing the PSNR to drop.")
    print("="*50 + "\n")
    analyze_and_plot(img_f, f1, "uniform", "natural", output_folder)
    analyze_and_plot(img_f, f2, "gaussian", "natural", output_folder)
    analyze_and_plot(img_f, f3, "salt_pepper", "natural", output_folder)
    
    # --- Part d: Optional - Non-local Means ---
    print("\n--- Part (d): Non-Local Means Filter Demonstration ---")
    # NLM is most effective against Gaussian-like noise. We will test it on f2.
    # The 'h' parameter is crucial. A good starting point is the standard deviation of the noise.
    # For PSNR=20, MSE is ~650, so noise sigma is sqrt(650) ~ 25.5.
    # We will use search_m=21, patch_m=7 as is common.
    nlm_denoised = non_local_means_filter(f2, search_window_m=21, patch_m=7, h=25.5)
    nlm_psnr = calculate_psnr(img_f, nlm_denoised)
    nlm_filename = os.path.join(output_folder, "d_nlm_denoised_natural_gaussian.png")
    cv2.imwrite(nlm_filename, nlm_denoised)
    print(f"Saved Non-Local Means result to {nlm_filename}")
    print(f"  -> NLM Filter PSNR on Gaussian noise: {nlm_psnr:.2f} dB")
    print("Compare this to the best results from Mean and Median filters for the same image.")
    print("NLM typically preserves edges and textures much better, resulting in a higher quality image and often a higher PSNR.")

    print("\nAll tasks complete. Check the 'denoising_results' folder.")
    plt.show() # Display any plots that were generated if running interactively

