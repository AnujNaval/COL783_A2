import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os


def save_image(filepath, img_array):
    """
    Saves a NumPy array as an image file. Handles normalization for float arrays.
    """
    # Normalize float arrays from [0, 1] to [0, 255] for saving
    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
        # Check for negative values which can occur in intermediate steps like 'eta'
        # Normalize based on min/max to ensure visibility
        min_val, max_val = np.min(img_array), np.max(img_array)
        if min_val < 0 or max_val > 1:
            img_normalized = 255 * (img_array - min_val) / (max_val - min_val)
        else:
            img_normalized = img_array * 255.0
        
        img_to_save = img_normalized.astype(np.uint8)
    else:
        img_to_save = img_array.astype(np.uint8)
        
    cv2.imwrite(filepath, img_to_save)
    print(f"Saved image to: {filepath}")

def visualize_spectrum(Fshift, title):
    """Visualizes the magnitude spectrum of a Fourier-transformed image."""
    # Use a log scale to enhance visibility of details
    magnitude_spectrum = np.log1p(np.abs(Fshift))

    plt.figure(figsize=(8, 8))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(title)
    plt.xlabel('u')
    plt.ylabel('v')
    plt.colorbar(label='Log Magnitude')
    plt.show()

def select_spikes_interactively(Fshift):
    """
    Allows the user to manually select spike points by clicking on the spectrum image.
    
    Returns:
        list of tuples: A list of (row, col) coordinates for the selected spikes.
    """
    print("\n--- Interactive Spike Selection ---")
    print("INSTRUCTIONS: Click on the center of each spike you want to filter.")
    print("A red 'x' will mark your selection. Close the plot window when you are finished.")

    magnitude_spectrum = np.log1p(np.abs(Fshift))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(magnitude_spectrum, cmap='gray')
    ax.set_title("Click on spikes to select, then close this window")
    
    # List to store the coordinates of the clicks
    spike_points = []

    def onclick(event):
        """Event handler for mouse clicks."""
        # Check if the click was inside the axes
        if event.inaxes:
            # event.xdata and event.ydata are float coordinates in data space.
            # We want integer pixel coordinates.
            col = int(round(event.xdata))
            row = int(round(event.ydata))
            
            spike_points.append((row, col))
            print(f"Point selected at (row={row}, col={col})")
            
            # Draw a marker on the plot for visual feedback
            ax.plot(col, row, 'rx', markersize=10, mew=2) # mew is markeredgewidth
            fig.canvas.draw()

    # Connect the event handler to the figure
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Show the plot. The script will pause here until the window is closed.
    plt.show()

    # Disconnect the event handler after the plot is closed
    fig.canvas.mpl_disconnect(cid)
    
    print(f"\nFinished selection. {len(spike_points)} points were selected.")
    return spike_points

def create_notch_filter(shape, notch_points, D0, filter_type='gaussian'):
    """
    Creates a notch reject filter.

    Args:
        shape (tuple): The shape of the image (rows, cols).
        notch_points (list of tuples): A list of (u, v) coordinates for the notches.
        D0 (float): The cutoff radius or width of the notches.
        filter_type (str): 'gaussian', 'butterworth' (n=2), or 'ideal'.

    Returns:
        np.ndarray: The generated notch filter mask.
    """
    M, N = shape
    H = np.ones((M, N), dtype=np.float32)

    # Create coordinate grids
    u = np.arange(M)
    v = np.arange(N)
    U, V = np.meshgrid(v, u)

    # The filter is a product of individual notch filters
    for uk, vk in notch_points:
        # Calculate distance from the current notch center
        D_k = np.sqrt((U - vk)**2 + (V - uk)**2)
        D_minus_k = np.sqrt((U - (N - vk))**2 + (V - (M - uk))**2)

        if filter_type == 'ideal':
            H[D_k <= D0] = 0
            H[D_minus_k <= D0] = 0
        elif filter_type == 'butterworth':
            # Use n=2 as specified in the assignment
            n = 2
            H_k = 1 / (1 + (D0 / (D_k + 1e-8))**(2 * n))
            H_minus_k = 1 / (1 + (D0 / (D_minus_k + 1e-8))**(2 * n))
            H *= H_k * H_minus_k
        elif filter_type == 'gaussian':
            H_k = 1 - np.exp(-(D_k**2) / (2 * D0**2))
            H_minus_k = 1 - np.exp(-(D_minus_k**2) / (2 * D0**2))
            H *= H_k * H_minus_k
            
    return H

def optimum_notch_filtering(g, eta, neighborhood_size=15):
    """
    Performs optimum notch filtering.
    
    Args:
        g (np.ndarray): The original (degraded) image.
        eta (np.ndarray): The estimated noise pattern.
        neighborhood_size (int): The side length of the square neighborhood.
        
    Returns:
        tuple: (f_hat, w), where f_hat is the restored image and w is the weight mask.
    """
    # Ensure inputs are float64 for precision as recommended
    g = g.astype(np.float64)
    eta = eta.astype(np.float64)
    
    # Pad images to handle borders during neighborhood processing
    pad = neighborhood_size // 2
    g_padded = np.pad(g, pad, mode='reflect')
    eta_padded = np.pad(eta, pad, mode='reflect')
    
    M, N = g.shape
    var_eta = np.zeros_like(g, dtype=np.float64)
    cov_g_eta = np.zeros_like(g, dtype=np.float64)
    
    print("Performing optimum notch filtering (this may take a while)...")
    
    # Iterate over each pixel to compute local statistics
    for i in range(M):
        for j in range(N):
            # Define the local neighborhood
            i_start, i_end = i, i + neighborhood_size
            j_start, j_end = j, j + neighborhood_size
            
            # Extract local regions
            g_local = g_padded[i_start:i_end, j_start:j_end]
            eta_local = eta_padded[i_start:i_end, j_start:j_end]
            
            # Compute local variance and covariance
            mean_eta_local = np.mean(eta_local)
            var_eta[i, j] = np.mean((eta_local - mean_eta_local)**2)
            
            mean_g_local = np.mean(g_local)
            cov_g_eta[i, j] = np.mean((g_local - mean_g_local) * (eta_local - mean_eta_local))

    # Compute the weight mask w(x,y)
    # Add a small epsilon to denominator to avoid division by zero
    w = cov_g_eta / (var_eta + 1e-8)
    
    # Clamp weights to be between 0 and 1 for stability
    w = np.clip(w, 0, 1)

    # Estimate the final image
    f_hat = g - w * eta
    
    print("Optimum notch filtering complete.")
    return f_hat, w, cov_g_eta, var_eta


# --- Main Execution ---
if __name__ == "__main__":
    image_path = "./test-images/Batyrbek_Gutnov.jpg"

    if image_path:
        # Create a directory to save results
        output_folder = "part3_output"
        os.makedirs(output_folder, exist_ok=True)
        print(f"Results will be saved in the '{output_folder}' directory.")

        # Load the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_float = img.astype(np.float32) / 255.0

        # --- Part (a): Visualize the Fourier Spectrum & Select Spikes ---
        print("\n--- Part (a): Fourier Spectrum ---")
        # Compute 2D FFT and shift the zero-frequency component to the center
        F = np.fft.fft2(img_float)
        Fshift = np.fft.fftshift(F)
        
        # Allow user to manually select spike locations by clicking on the spectrum
        spike_points = select_spikes_interactively(Fshift)

        # Check if user selected any points before proceeding
        if not spike_points:
            print("No spike points were selected. Exiting.")
        else:
            print(f"Using manually selected spike locations (row, col): {spike_points}")

            # --- Part (b): Gaussian Notch Filter ---
            print("\n--- Part (b): Gaussian Notch Filter ---")
            D0_gaussian = 10  # Width of the notch. This is a tunable parameter.
            gaussian_notch_filter = create_notch_filter(img.shape, spike_points, D0_gaussian, 'gaussian')
            
            Fshift_filtered_gaussian = Fshift * gaussian_notch_filter
            F_filtered_gaussian = np.fft.ifftshift(Fshift_filtered_gaussian)
            img_filtered_gaussian = np.abs(np.fft.ifft2(F_filtered_gaussian))

            # --- Part (c): Ideal Notch Filter ---
            print("\n--- Part (c): Ideal Notch Filter ---")
            D0_ideal = D0_gaussian
            ideal_notch_filter = create_notch_filter(img.shape, spike_points, D0_ideal, 'ideal')

            Fshift_filtered_ideal = Fshift * ideal_notch_filter
            F_filtered_ideal = np.fft.ifftshift(Fshift_filtered_ideal)
            img_filtered_ideal = np.abs(np.fft.ifft2(F_filtered_ideal))
            
            print("Comparing Gaussian vs Ideal: The ideal filter has a sharp cutoff, which can introduce 'ringing' artifacts.")

            # --- Part (d): Optimum Notch Filtering ---
            print("\n--- Part (d): Optimum Notch Filtering ---")
            eta = img_float - img_filtered_gaussian
            f_hat, w, cov_g_eta, var_eta = optimum_notch_filtering(img_float, eta, neighborhood_size=21)

            # --- Save individual result images ---
            print("\n--- Saving individual result images ---")
            save_image(os.path.join(output_folder, 'original_image.png'), img)
            save_image(os.path.join(output_folder, 'b_filtered_gaussian.png'), img_filtered_gaussian)
            save_image(os.path.join(output_folder, 'c_filtered_ideal.png'), img_filtered_ideal)
            save_image(os.path.join(output_folder, 'd_estimated_noise_eta.png'), eta)
            save_image(os.path.join(output_folder, 'd_weight_w.png'), w)
            save_image(os.path.join(output_folder, 'd_covariance.png'), cov_g_eta)
            save_image(os.path.join(output_folder, 'd_variance_eta.png'), var_eta)
            save_image(os.path.join(output_folder, 'd_final_optimum_notch.png'), f_hat)
            save_image(os.path.join(output_folder, 'b_gaussian_filter_mask.png'), gaussian_notch_filter)
            save_image(os.path.join(output_folder, 'c_ideal_filter_mask.png'), ideal_notch_filter)


            # --- Display and save summary plots ---
            print("\n--- Displaying and saving summary plots ---")

            # Window 1: Original Data
            fig1 = plt.figure(figsize=(12, 6))
            plt.suptitle("Part (a): Original Image and Spectrum", fontsize=16)
            plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original Halftone Image')
            plt.subplot(1, 2, 2); plt.imshow(np.log1p(np.abs(Fshift)), cmap='gray'); plt.title('Original Spectrum')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig1.savefig(os.path.join(output_folder, 'summary_01_original.png'))
            print(f"Saved plot to: {os.path.join(output_folder, 'summary_01_original.png')}")


            # Window 2: Gaussian Filter Results
            fig2 = plt.figure(figsize=(18, 6))
            plt.suptitle("Part (b): Gaussian Notch Filter Results", fontsize=16)
            plt.subplot(1, 3, 1); plt.imshow(gaussian_notch_filter, cmap='gray'); plt.title(f'Gaussian Notch Filter (D0={D0_gaussian})')
            plt.subplot(1, 3, 2); plt.imshow(np.log1p(np.abs(Fshift_filtered_gaussian)), cmap='gray'); plt.title('Filtered Spectrum (Gaussian)')
            plt.subplot(1, 3, 3); plt.imshow(img_filtered_gaussian, cmap='gray', vmin=0, vmax=1); plt.title('Filtered Image (Gaussian)')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig2.savefig(os.path.join(output_folder, 'summary_02_gaussian.png'))
            print(f"Saved plot to: {os.path.join(output_folder, 'summary_02_gaussian.png')}")

            # Window 3: Ideal Filter Results
            fig3 = plt.figure(figsize=(18, 6))
            plt.suptitle("Part (c): Ideal Notch Filter Results", fontsize=16)
            plt.subplot(1, 3, 1); plt.imshow(ideal_notch_filter, cmap='gray'); plt.title(f'Ideal Notch Filter (D0={D0_ideal})')
            plt.subplot(1, 3, 2); plt.imshow(np.log1p(np.abs(Fshift_filtered_ideal)), cmap='gray'); plt.title('Filtered Spectrum (Ideal)')
            plt.subplot(1, 3, 3); plt.imshow(img_filtered_ideal, cmap='gray', vmin=0, vmax=1); plt.title('Filtered Image (Ideal)')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig3.savefig(os.path.join(output_folder, 'summary_03_ideal.png'))
            print(f"Saved plot to: {os.path.join(output_folder, 'summary_03_ideal.png')}")

            # Window 4: Optimum Notch Filtering Results
            fig4 = plt.figure(figsize=(15, 10))
            plt.suptitle("Part (d): Optimum Notch Filtering Results", fontsize=16)
            plt.subplot(2, 3, 1); plt.imshow(eta, cmap='gray'); plt.title('Estimated Noise (η)')
            plt.subplot(2, 3, 2); plt.imshow(cov_g_eta, cmap='gray'); plt.title('cov(g, η)'); plt.colorbar()
            plt.subplot(2, 3, 3); plt.imshow(var_eta, cmap='gray'); plt.title('var(η)'); plt.colorbar()
            plt.subplot(2, 3, 4); plt.imshow(w, cmap='gray'); plt.title('Computed Weight (w)'); plt.colorbar()
            plt.subplot(2, 3, 5); plt.imshow(f_hat, cmap='gray', vmin=0, vmax=1); plt.title('Final Image (Optimum Notch)')
            plt.subplot(2,3,6).axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig4.savefig(os.path.join(output_folder, 'summary_04_optimum_notch.png'))
            print(f"Saved plot to: {os.path.join(output_folder, 'summary_04_optimum_notch.png')}")

            plt.show()

