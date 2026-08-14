"""
Core signal processing module containing custom implementations of:
1. Global Histogram Equalization (GHE)
2. Contrast Limited Adaptive Histogram Equalization (CLAHE)
   - Tiling
   - Clip-limit enforcement with iterative redistribution
   - Min-Max CDF normalization
   - Vectorized Bilinear Interpolation
"""

import numpy as np

def global_histogram_equalization(image):
    """
    Applies Global Histogram Equalization to an 8-bit grayscale image.
    
    Parameters:
        image (numpy.ndarray): 2D grayscale image.
        
    Returns:
        numpy.ndarray: Equalized image.
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be a 2D grayscale array.")
        
    h, w = image.shape
    total_pixels = h * w
    
    # 1. Compute Histogram (PMF)
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    
    # 2. Compute Cumulative Distribution Function (CDF)
    cdf = hist.cumsum()
    
    # 3. Min-Max Scaling LUT building
    cdf_min = cdf[cdf > 0]
    if len(cdf_min) == 0:
        return np.zeros_like(image)
    cdf_min_val = cdf_min[0]
    
    # Formula: round( ((CDF(v) - CDF_min) / (Total - CDF_min)) * 255 )
    denominator = total_pixels - cdf_min_val
    if denominator == 0:
        lut = np.zeros(256, dtype=np.uint8)
    else:
        lut = np.round((cdf - cdf_min_val) / denominator * 255).astype(np.uint8)
        
    # 4. Map the input pixels using the LUT
    return lut[image]


def compute_tile_lut(tile, clip_limit):
    """
    Computes the Contrast-Limited Lookup Table (LUT) for a single image tile.
    
    Parameters:
        tile (numpy.ndarray): 2D sub-array representing an image tile.
        clip_limit (float): Normalization factor for clipping (e.g. 4.0).
        
    Returns:
        numpy.ndarray: 256-element mapping array (LUT).
    """
    # 1. Compute local histogram
    hist, _ = np.histogram(tile, bins=256, range=(0, 256))
    total_pixels = tile.size
    
    # 2. Enforce clip limit and redistribute excess
    if clip_limit > 0:
        # Standard clip limit count: clip_limit * average bin height
        limit = max(1, int(clip_limit * total_pixels / 256))
        
        # Calculate initial excess and clip bins
        excess = 0
        for i in range(256):
            if hist[i] > limit:
                excess += hist[i] - limit
                hist[i] = limit
                
        # Redistribute excess iteratively to avoid any bin re-exceeding the limit
        while excess > 0:
            step = excess // 256
            if step < 1:
                step = 1
                
            for i in range(256):
                if excess > 0 and hist[i] < limit:
                    available = limit - hist[i]
                    to_add = min(step, available, excess)
                    hist[i] += to_add
                    excess -= to_add
            
            # Escape if all bins are somehow full (mathematically impossible if limit * 256 >= total_pixels)
            if np.all(hist >= limit):
                break
                
    # 3. Compute CDF
    cdf = hist.cumsum()
    
    # 4. Min-Max Scaling Formula
    cdf_min = cdf[cdf > 0]
    if len(cdf_min) == 0:
        cdf_min_val = 0
    else:
        cdf_min_val = cdf_min[0]
        
    denominator = total_pixels - cdf_min_val
    if denominator == 0:
        lut = np.zeros(256, dtype=np.uint8)
    else:
        lut = np.round((cdf - cdf_min_val) / denominator * 255).astype(np.uint8)
        
    return lut


def get_interp_params(coords, centers):
    """
    Computes 1D interpolation indices and weights for a list of coordinates 
    relative to a list of sorted tile center coordinates.
    
    Parameters:
        coords (numpy.ndarray): 1D array of pixel indices (height or width).
        centers (numpy.ndarray): 1D array of sorted tile center coordinates.
        
    Returns:
        tuple: (i0, i1, weights) where:
            i0 (numpy.ndarray): index of left/upper tile center
            i1 (numpy.ndarray): index of right/lower tile center
            weights (numpy.ndarray): interpolation weight [0, 1]
    """
    n = len(coords)
    num_centers = len(centers)
    
    i0 = np.zeros(n, dtype=np.int32)
    i1 = np.zeros(n, dtype=np.int32)
    w = np.zeros(n, dtype=np.float32)
    
    for idx, val in enumerate(coords):
        if val < centers[0]:
            # Beyond first tile center (Top/Left border or corner) -> map directly
            i0[idx] = 0
            i1[idx] = 0
            w[idx] = 0.0
        elif val >= centers[-1]:
            # Beyond last tile center (Bottom/Right border or corner) -> map directly
            i0[idx] = num_centers - 1
            i1[idx] = num_centers - 1
            w[idx] = 0.0
        else:
            # Interior -> find the enclosing center interval
            k = 0
            while k < num_centers - 1 and centers[k+1] <= val:
                k += 1
            i0[idx] = k
            i1[idx] = k + 1
            w[idx] = (val - centers[k]) / (centers[k+1] - centers[k])
            
    return i0, i1, w


def clahe_custom(image, grid_size=(8, 8), clip_limit=4.0):
    """
    Performs Contrast Limited Adaptive Histogram Equalization from scratch.
    
    Parameters:
        image (numpy.ndarray): 2D grayscale image (uint8).
        grid_size (tuple): Number of tiles (rows, cols) - default is 8x8.
        clip_limit (float): Normalized clip limit factor - default is 4.0.
        
    Returns:
        numpy.ndarray: Enhanced 2D grayscale image (uint8).
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be a 2D grayscale array.")
        
    h, w = image.shape
    rows, cols = grid_size
    
    # Calculate tile sizes
    tile_h = h // rows
    tile_w = w // cols
    
    # Calculate center coordinates for all tiles
    # Tile center is located at: start_index + (tile_size - 1) / 2
    tc_y = np.array([r * tile_h + (tile_h - 1) / 2 for r in range(rows)], dtype=np.float32)
    tc_x = np.array([c * tile_w + (tile_w - 1) / 2 for c in range(cols)], dtype=np.float32)
    
    # Step 1: Precompute the LUT for each tile
    # Shape: (rows, cols, 256)
    tile_luts = np.zeros((rows, cols, 256), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            tile = image[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w]
            tile_luts[r, c] = compute_tile_lut(tile, clip_limit)
            
    # Step 2: Compute 1D interpolation parameters for rows and columns
    # This reduces complexity from O(H*W) loops to O(H) + O(W) setup and O(H*W) vectorized operations
    r0, r1, b = get_interp_params(np.arange(h), tc_y)
    c0, c1, a = get_interp_params(np.arange(w), tc_x)
    
    # Expand indices to 2D for broadcasting
    r0_2d = r0[:, np.newaxis]  # (h, 1)
    r1_2d = r1[:, np.newaxis]  # (h, 1)
    c0_2d = c0[np.newaxis, :]  # (1, w)
    c1_2d = c1[np.newaxis, :]  # (1, w)
    
    b_2d = b[:, np.newaxis]    # (h, 1)
    a_2d = a[np.newaxis, :]    # (1, w)
    
    # Step 3: Retrieve mapped values using advanced numpy indexing
    # Extract LUT values for each pixel at the 4 neighboring tile centers
    s_tl = tile_luts[r0_2d, c0_2d, image]
    s_tr = tile_luts[r0_2d, c1_2d, image]
    s_bl = tile_luts[r1_2d, c0_2d, image]
    s_br = tile_luts[r1_2d, c1_2d, image]
    
    # Step 4: Perform Bilinear Interpolation
    s_top = (1.0 - a_2d) * s_tl + a_2d * s_tr
    s_bottom = (1.0 - a_2d) * s_bl + a_2d * s_br
    enhanced = (1.0 - b_2d) * s_top + b_2d * s_bottom
    
    return np.round(enhanced).astype(np.uint8)
