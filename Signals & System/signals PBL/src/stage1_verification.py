"""
Stage 1: Software Verification
Demonstrates the mathematics of Global Histogram Equalization (GHE) 
and Contrast Limited Adaptive Histogram Equalization (CLAHE) 
on a micro 8x8 grayscale matrix.
Saves verification plots to output/stage1_plots.png.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Import custom implementations
from clahe_core import (
    global_histogram_equalization,
    clahe_custom,
    compute_tile_lut
)

def run_verification():
    print("="*60)
    print("         STAGE 1: SOFTWARE VERIFICATION (8x8 MICRO MATRIX)        ")
    print("="*60)
    
    # 1. Define low-contrast 8x8 test matrix
    # Grayscale intensities clustered tightly between 52 and 64 (representing poor contrast)
    test_matrix = np.array([
        [52, 53, 54, 54, 55, 55, 56, 57],
        [53, 54, 54, 55, 55, 56, 57, 58],
        [54, 54, 55, 55, 56, 57, 58, 59],
        [54, 55, 55, 56, 57, 58, 59, 60],
        [55, 55, 56, 57, 58, 59, 60, 61],
        [55, 56, 57, 58, 59, 60, 61, 62],
        [56, 57, 58, 59, 60, 61, 62, 63],
        [57, 58, 59, 60, 61, 62, 63, 64]
    ], dtype=np.uint8)
    
    print("\n--- [1] Original 8x8 Grayscale Matrix ---")
    print(test_matrix)
    
    # 2. Run Global Histogram Equalization (GHE)
    ghe_result = global_histogram_equalization(test_matrix)
    print("\n--- [2] Global Histogram Equalization (GHE) Result ---")
    print(ghe_result)
    
    # 3. Run CLAHE with 2x2 grid and clip limit 32.0 (translates to bin limit of 2 for 16-pixel tiles)
    # Average bin height = 16 pixels / 256 bins = 0.0625
    # Limit = clip_limit * average bin height = 32.0 * 0.0625 = 2.0 pixels
    grid_size = (2, 2)
    clip_limit = 32.0
    
    # Trace details of the Top-Left 4x4 tile
    top_left_tile = test_matrix[0:4, 0:4]
    print("\n--- [3] CLAHE Step-by-Step (Top-Left 4x4 Tile) ---")
    print("Tile contents:\n", top_left_tile)
    
    # Compute raw histogram
    raw_hist, _ = np.histogram(top_left_tile, bins=256, range=(0, 256))
    nonzero_bins = np.where(raw_hist > 0)[0]
    print("\nRaw Histogram (Non-zero bins):")
    for b in nonzero_bins:
        print(f"  Intensity {b:3d}: count = {raw_hist[b]}")
        
    # Compute limit and clipped histogram
    limit = int(clip_limit * top_left_tile.size / 256)
    print(f"\nClip Limit Count (Limit = {limit}):")
    
    clipped_hist = np.minimum(raw_hist, limit)
    excess = np.sum(raw_hist) - np.sum(clipped_hist)
    print(f"  Total Pixels = {top_left_tile.size}")
    print(f"  Clipped Excess to Redistribute = {excess}")
    
    # Detailed redistribution simulation
    lut = compute_tile_lut(top_left_tile, clip_limit)
    print("\nLocal Normalized Mapping LUT (Non-zero values):")
    for v in range(256):
        if lut[v] > 0 or v in nonzero_bins:
            print(f"  Input Intensity {v:3d} -> Equalized Output {lut[v]:3d}")
            
    # Apply full CLAHE pipeline
    clahe_result = clahe_custom(test_matrix, grid_size=grid_size, clip_limit=clip_limit)
    print("\n--- [4] CLAHE Custom Equalization Result (2x2 Grid) ---")
    print(clahe_result)
    
    # 4. Generate CDF Comparison Plots
    print("\n--- [5] Generating Visualization Plots ---")
    
    # Flatten matrices to calculate CDFs
    def get_cdf_for_plot(mat):
        hist, _ = np.histogram(mat, bins=256, range=(0, 256))
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]
        return hist, cdf_normalized
        
    hist_orig, cdf_orig = get_cdf_for_plot(test_matrix)
    hist_ghe, cdf_ghe = get_cdf_for_plot(ghe_result)
    hist_clahe, cdf_clahe = get_cdf_for_plot(clahe_result)
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    
    # Top Row: Histograms
    # Original
    axs[0, 0].bar(range(256), hist_orig, color='gray', width=1.0)
    axs[0, 0].set_title("Original Histogram (Squeezed)")
    axs[0, 0].set_xlim(45, 70)
    axs[0, 0].set_ylim(0, 15)
    
    # GHE
    axs[0, 1].bar(range(256), hist_ghe, color='blue', width=1.0)
    axs[0, 1].set_title("GHE Histogram (Stretched / Gaps)")
    axs[0, 1].set_xlim(0, 256)
    axs[0, 1].set_ylim(0, 15)
    
    # CLAHE
    axs[0, 2].bar(range(256), hist_clahe, color='green', width=1.0)
    axs[0, 2].set_title("CLAHE Histogram (Smooth & Localized)")
    axs[0, 2].set_xlim(0, 256)
    axs[0, 2].set_ylim(0, 15)
    
    # Bottom Row: Cumulative Distribution Functions (CDFs)
    # Original
    axs[1, 0].plot(range(256), cdf_orig, color='black', linewidth=2)
    axs[1, 0].set_title("Original CDF (Sharp Step-Function)")
    axs[1, 0].set_xlim(45, 70)
    axs[1, 0].grid(True)
    
    # GHE
    axs[1, 1].plot(range(256), cdf_ghe, color='blue', linewidth=2)
    axs[1, 1].set_title("GHE CDF (Diagonal Linear Trend)")
    axs[1, 1].set_xlim(0, 256)
    axs[1, 1].grid(True)
    
    # CLAHE
    axs[1, 2].plot(range(256), cdf_clahe, color='green', linewidth=2)
    axs[1, 2].set_title("CLAHE CDF (Controlled Linear Trend)")
    axs[1, 2].set_xlim(0, 256)
    axs[1, 2].grid(True)
    
    plt.tight_layout()
    
    # Ensure output folder exists
    os.makedirs("../output", exist_ok=True)
    plot_path = "../output/stage1_plots.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plots successfully generated and saved to: {os.path.abspath(plot_path)}")
    print("="*60)

if __name__ == "__main__":
    # Change working directory to file directory to resolve relative paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_verification()
