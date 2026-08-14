import cv2
import numpy as np
import math

def calculate_entropy(img):
    """Calculates the Shannon Entropy of a grayscale image."""
    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    prob = hist / hist.sum()
    entropy = -sum([p * math.log2(p) for p in prob if p > 0])
    return entropy

def analyze_results(comparison_path):
    # Load the side-by-side comparison image
    comparison = cv2.imread(comparison_path, cv2.IMREAD_GRAYSCALE)
    if comparison is None:
        print(f"Error: Could not load {comparison_path}")
        return
        
    h, w = comparison.shape
    half_w = w // 2
    
    # Split the side-by-side image back into Original (left) and Enhanced (right)
    original = comparison[:, :half_w]
    enhanced = comparison[:, half_w:]
    
    # Calculate metrics
    mean_orig = np.mean(original)
    mean_enh = np.mean(enhanced)
    
    std_orig = np.std(original)
    std_enh = np.std(enhanced)
    
    entropy_orig = calculate_entropy(original)
    entropy_enh = calculate_entropy(enhanced)
    
    # Print results formatted for slides
    print("\n" + "="*50)
    print("        MATHEMATICAL PERFORMANCE METRICS")
    print("="*50)
    print(f"1. Mean Brightness (Average Light Level):")
    print(f"   * Original Image: {mean_orig:.2f}")
    print(f"   * Enhanced Image: {mean_enh:.2f}  (Increase of {((mean_enh - mean_orig)/mean_orig)*100:.1f}%)")
    print("\n2. Standard Deviation (Contrast/Dynamic Range):")
    print(f"   * Original Image: {std_orig:.2f}")
    print(f"   * Enhanced Image: {std_enh:.2f}  (Contrast stretched by {((std_enh - std_orig)/std_orig)*100:.1f}%)")
    print("\n3. Information Entropy (Amount of Visible Detail):")
    print(f"   * Original Image: {entropy_orig:.3f} bits/pixel")
    print(f"   * Enhanced Image: {entropy_enh:.3f} bits/pixel (Detail recovery success)")
    print("="*50 + "\n")

if __name__ == "__main__":
    analyze_results("data/enhanced_output.png")
