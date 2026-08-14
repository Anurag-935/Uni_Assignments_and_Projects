"""
Stage 2: Embedded Hardware Simulation - Demo A (Static Image Enhancement)
Loads a low-contrast, low-light sample image, converts it to 8-bit grayscale, 
and processes it using:
1. Global Histogram Equalization (GHE)
2. OpenCV's Reference CLAHE
3. Our Custom CLAHE Implementation (from scratch)

Saves comparison results and prints performance timings.
"""

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import custom implementations
from clahe_core import (
    global_histogram_equalization,
    clahe_custom
)

def run_static_enhancement(image_path, grid_size=(8, 8), clip_limit=4.0):
    print("="*60)
    print("      STAGE 2: DEMO A - STATIC IMAGE CONTRAST ENHANCEMENT     ")
    print("="*60)
    
    # 1. Load the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {os.path.abspath(image_path)}")
        
    print(f"Loading image: {os.path.abspath(image_path)}")
    color_img = cv2.imread(image_path)
    if color_img is None:
        raise ValueError(f"Could not load image at {image_path}. Check file format.")
        
    # 2. Convert to 8-bit Grayscale (reduces matrix data size by 66.7% for embedded optimization)
    # Using standard NTSC formula (0.299 * R + 0.587 * G + 0.114 * B)
    start_time = time.perf_counter()
    if len(color_img.shape) == 3:
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = color_img.copy()
    gray_conv_time = (time.perf_counter() - start_time) * 1000.0
    print(f"Grayscale conversion completed in: {gray_conv_time:.2f} ms")
    print(f"Resolution: {gray_img.shape[1]}x{gray_img.shape[0]} pixels")
    
    # 3. Apply Global Histogram Equalization (GHE)
    start_time = time.perf_counter()
    ghe_img = global_histogram_equalization(gray_img)
    ghe_time = (time.perf_counter() - start_time) * 1000.0
    print(f"Global Histogram Equalization (GHE) completed in: {ghe_time:.2f} ms")
    
    # 4. Apply OpenCV's CLAHE (Golden Reference)
    start_time = time.perf_counter()
    clahe_cv = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    opencv_clahe_img = clahe_cv.apply(gray_img)
    opencv_clahe_time = (time.perf_counter() - start_time) * 1000.0
    print(f"OpenCV CLAHE (Golden Reference) completed in: {opencv_clahe_time:.2f} ms")
    
    # 5. Apply Our Custom CLAHE Implementation (Vectorized math verification)
    start_time = time.perf_counter()
    custom_clahe_img = clahe_custom(gray_img, grid_size=grid_size, clip_limit=clip_limit)
    custom_clahe_time = (time.perf_counter() - start_time) * 1000.0
    print(f"Custom Vectorized CLAHE completed in: {custom_clahe_time:.2f} ms")
    
    # 6. Verify correctness by measuring Mean Squared Error (MSE) between custom and OpenCV implementation
    mse = np.mean((opencv_clahe_img.astype(float) - custom_clahe_img.astype(float)) ** 2)
    print(f"Mean Squared Error (MSE) vs OpenCV CLAHE: {mse:.4f} (Close to 0 proves mathematical parity)")
    
    # 7. Generate and save the comparison figure
    print("\nSaving comparison plot...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original Grayscale
    axs[0, 0].imshow(gray_img, cmap='gray')
    axs[0, 0].set_title(f"Original Grayscale (Low Light)\nSize: {gray_img.shape[1]}x{gray_img.shape[0]}")
    axs[0, 0].axis('off')
    
    # Global Histogram Equalization (GHE)
    axs[0, 1].imshow(ghe_img, cmap='gray')
    axs[0, 1].set_title(f"Global Histogram Equalization (GHE)\nTime: {ghe_time:.2f} ms (Over-enhanced/Noisy)")
    axs[0, 1].axis('off')
    
    # OpenCV CLAHE
    axs[1, 0].imshow(opencv_clahe_img, cmap='gray')
    axs[1, 0].set_title(f"OpenCV CLAHE Reference\nTime: {opencv_clahe_time:.2f} ms (Adaptive/Noise-Limited)")
    axs[1, 0].axis('off')
    
    # Custom CLAHE
    axs[1, 1].imshow(custom_clahe_img, cmap='gray')
    axs[1, 1].set_title(f"Custom Vectorized CLAHE\nTime: {custom_clahe_time:.2f} ms (MSE: {mse:.4f})")
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    os.makedirs("../output", exist_ok=True)
    comparison_path = "../output/stage2_comparison.png"
    plt.savefig(comparison_path, dpi=150)
    plt.close()
    
    # Extract filename without extension to save custom results
    basename = os.path.splitext(os.path.basename(image_path))[0]
    custom_output_path = f"../output/{basename}_enhanced.png"
    
    # Save processed image files directly for raw quality inspection
    cv2.imwrite("../output/img_original_gray.png", gray_img)
    cv2.imwrite("../output/img_ghe.png", ghe_img)
    cv2.imwrite("../output/img_opencv_clahe.png", opencv_clahe_img)
    cv2.imwrite(custom_output_path, custom_clahe_img)
    
    print(f"Standalone enhanced image saved to: {os.path.abspath(custom_output_path)}")
    print(f"Outputs successfully generated under: {os.path.abspath('../output/')}")
    print("="*60)

if __name__ == "__main__":
    import argparse
    import sys
    
    # Change working directory to file directory to resolve relative paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser(description="Enhance static images using GHE and CLAHE.")
    parser.add_argument("--image", type=str, default=None, help="Path to input low-light image file")
    parser.add_argument("--select", action="store_true", help="Launch GUI file browser to select an image")
    parser.add_argument("--clip", type=float, default=4.0, help="Clip limit for CLAHE (default: 4.0)")
    parser.add_argument("--grid", type=int, nargs=2, default=[8, 8], help="Grid layout (rows cols) (default: 8 8)")
    args = parser.parse_args()
    
    input_path = args.image
    
    # Fallback to tkinter UI file dialog if user requested --select
    if args.select:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            print("Opening file selector dialog...")
            input_path = filedialog.askopenfilename(
                title="Select Low-Light Image to Enhance",
                filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")]
            )
            if not input_path:
                print("No file selected. Exiting.")
                sys.exit(0)
        except Exception as e:
            print(f"Error launching GUI selector: {e}. Falling back to default.")
            
    # Default fallback
    if not input_path:
        input_path = "../data/low_light_sample.png"
        
    run_static_enhancement(
        image_path=input_path,
        grid_size=tuple(args.grid),
        clip_limit=args.clip
    )
