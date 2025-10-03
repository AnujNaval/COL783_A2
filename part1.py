import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_gray_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image at {path}")
    return img.astype(np.float32) / 255.0  # normalize to 0..1 float

def ideal_rect_lowpass_fft(img, k):
    M, N = img.shape
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)

    u = np.linspace(-0.5, 0.5, M, endpoint=False)
    v = np.linspace(-0.5, 0.5, N, endpoint=False)
    U, V = np.meshgrid(u, v, indexing='ij')

    cutoff = 0.5 * k
    mask = (np.abs(U) <= cutoff) & (np.abs(V) <= cutoff)

    Fshift_masked = Fshift * mask
    Fmasked = np.fft.ifftshift(Fshift_masked)
    img_filtered = np.fft.ifft2(Fmasked)
    img_filtered = np.real(img_filtered)

    return np.clip(img_filtered, 0.0, 1.0)

def resize_and_restore(img, k, filtered=False):
    M, N = img.shape
    newM = max(1, int(round(M * k)))
    newN = max(1, int(round(N * k)))

    if filtered and k < 1.0:
        img_proc = ideal_rect_lowpass_fft(img, k)
    else:
        img_proc = img

    down = cv2.resize((img_proc*255).astype(np.uint8), (newN, newM), interpolation=cv2.INTER_LINEAR)
    down = down.astype(np.float32) / 255.0

    restored = cv2.resize((down*255).astype(np.uint8), (N, M), interpolation=cv2.INTER_NEAREST)
    restored = restored.astype(np.float32) / 255.0

    return down, restored

def show_compare(orig, naive_restored, filt_restored, k, save_prefix=None):
    plt.figure(figsize=(12,5))
    plt.suptitle(f"Scale k = {k} (downsample -> upsample back with nearest neighbour)", fontsize=14)

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(orig, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Naive (no prefilter)")
    plt.imshow(naive_restored, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Freq-domain ideal low-pass")
    plt.imshow(filt_restored, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_prefix:
        # Ensure the folder exists
        folder = os.path.dirname(save_prefix)
        if folder != "" and not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(f"{save_prefix}_k{str(k).replace('.','_')}.png", dpi=200)
    plt.show()

def main():
    image_path = "test-images/barbara.bmp"
    output_folder = "part1_output"

    img = load_gray_image(image_path)
    M, N = img.shape
    print("Loaded image shape:", img.shape)

    ks = [0.5, 0.25, 0.125]  # 1/2, 1/4, 1/8
    for k in ks:
        # Naive
        down_naive = cv2.resize((img*255).astype(np.uint8), 
                                (max(1,int(round(N*k))), max(1,int(round(M*k)))), 
                                interpolation=cv2.INTER_LINEAR)
        down_naive = down_naive.astype(np.float32)/255.0
        restored_naive = cv2.resize((down_naive*255).astype(np.uint8), (N, M), interpolation=cv2.INTER_NEAREST)
        restored_naive = restored_naive.astype(np.float32)/255.0

        # Filtered
        down_filt, restored_filt = resize_and_restore(img, k, filtered=True)

        # Compare and save in output folder
        save_prefix = os.path.join(output_folder, "resize_demo")
        show_compare(img, restored_naive, restored_filt, k, save_prefix=save_prefix)

if __name__ == "__main__":
    main()
