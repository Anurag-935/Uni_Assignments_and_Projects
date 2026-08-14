# Signals PBL: Edge-Based Contrast Enhancement

This document serves as a complete project summary and context dump. It details the core concepts, mathematical models, implementation mechanics, and optimization strategies of the project. It has been stripped of local file paths and environment-specific folder structures for maximum portability across different machines.

---

## 1. Project Overview

The project implements and validates a high-performance **Contrast Limited Adaptive Histogram Equalization (CLAHE)** signal processing pipeline. It is optimized for resource-constrained edge hardware deployments (such as a Raspberry Pi 3 B+ running DietPi) but is fully portable and runnable on any standard Windows/macOS/Linux Python environment.

---

## 2. Core Concepts & Design Decisions

* **The Problem with Global Histogram Equalization (GHE):** GHE applies a single mapping function across the entire image. In non-uniform lighting conditions, it causes **over-enhancement**, blowing out highlights, crushing shadows, and heavily amplifying background noise in flat regions.
* **The Selection of CLAHE:** CLAHE operates locally by dividing the image into contextual blocks called **tiles**. It computes histograms and mapping functions independently for each tile to adapt to local lighting variance.
* **Noise Control (Histogram Clipping):** In flat regions, standard adaptive histogram equalization amplifies grainy sensor noise. CLAHE enforces a **Clip Limit** on histogram bins. Pixel counts exceeding the limit are clipped and redistributed evenly across all bins to keep noise under control.
* **Boundary Smoothing (Bilinear Interpolation):** Processing tiles independently creates blocky "checkerboard" artifacts at tile boundaries. Bilinear interpolation blends the mapped values from the four nearest tile centers to create smooth spatial transitions.

---

## 3. Mathematical Foundations

### Probability Mass Function (PMF) & Histogram
Within an $M \times N$ pixel tile, the frequency of grayscale intensity value $v \in [0, 255]$ is counted:

$$H(v) = \sum_{y=0}^{M-1} \sum_{x=0}^{N-1} \delta\Big(I(y, x) - v\Big)$$

where $\delta(0) = 1$ and $\delta(k) = 0$ for $k \neq 0$.

### Cumulative Distribution Function (CDF)
The CDF is the running cumulative sum of the histogram:

$$CDF(v) = \sum_{i=0}^{v} H(i)$$

### Min-Max CDF Scaling Formula
To map the cumulative ranks back to the 8-bit grayscale spectrum, each intensity level ($v$) is normalized:

$$h(v) = \text{round} \left( \frac{CDF(v) - CDF_{min}}{(M \times N) - CDF_{min}} \times 255 \right)$$

where $CDF_{min}$ forces the lowest observed intensity in the tile to resolve directly to $0$ (pure black), stretching the dynamic range.

### Bilinear Interpolation
To find the final intensity for a pixel at coordinates $(x, y)$, we interpolate the mapped values from the four nearest tile centers $C_{tl}, C_{tr}, C_{bl}, C_{br}$ located at coordinates $(x_0, y_0), (x_1, y_0), (x_0, y_1), (x_1, y_1)$:

Calculate relative fractional distances:
$$a = \frac{x - x_0}{x_1 - x_0}, \quad b = \frac{y - y_0}{y_1 - y_0}$$

First, interpolate horizontally along the top and bottom boundaries:
$$s_{top} = (1 - a) \cdot s_{tl} + a \cdot s_{tr}$$
$$s_{bottom} = (1 - a) \cdot s_{bl} + a \cdot s_{br}$$

Finally, interpolate vertically between the top and bottom values:
$$s_{final} = (1 - b) \cdot s_{top} + b \cdot s_{bottom}$$

*(where $s_{tl}, s_{tr}, s_{bl}, s_{br}$ are the equalized values obtained by looking up the pixel intensity in the respective tile LUTs).*

---

## 4. Hardware Optimization Strategy

* **Grayscale Conversion:** Converting 3-channel BGR inputs to 8-bit Grayscale reduces matrix data size by $66.7\%$. This conserves memory bandwidth and protects embedded CPUs from overheating.
* **Lookup Tables (LUT):** Normalization scaling calculations are executed exactly 256 times per tile to build a static LUT. The mapping of pixels to equalized values is then resolved using instant $O(1)$ memory retrieval.
* **Vectorized Bilinear Interpolation:** In Python, looping over every pixel is too slow. Our implementation computes 1D row and column interpolation weights ($O(H) + O(W)$) and projects them into 2D broadcasts using NumPy. This vectorized calculation runs a 1-Megapixel image in **75 ms** (~13 FPS) in pure Python.

---

## 5. Codebase Walkthrough

### 1. `clahe_core.py` (Math Engine)
* **`global_histogram_equalization(image)`**: Standard global histogram equalization.
* **`compute_tile_lut(tile, clip_limit)`**:
  * Calculates the tile histogram.
  * Clips bins exceeding the limit and redistributes excess pixels iteratively until no bin exceeds the limit.
  * Generates a 256-element LUT using the Min-Max CDF scaling formula.
* **`get_interp_params(coords, centers)`**: Computes the 1D neighboring tile indices and interpolation weights. Clamps weights to `0.0` at the borders/corners to prevent out-of-bounds errors.
* **`clahe_custom(image, grid_size, clip_limit)`**:
  * Splits the image into a grid of tiles (e.g. $8 \times 8$).
  * Precomputes tile LUTs.
  * Performs 2D vectorized bilinear interpolation across the entire image to output the final pixel matrix.

### 2. `stage1_verification.py` (Micro Math Verification)
* Defines a micro $8 \times 8$ grayscale test matrix with highly squeezed contrast (values 52 to 64).
* Computes GHE and CLAHE (2x2 grid, limit=32.0) and prints intermediate states (histograms, CDFs, and LUT mappings) step-by-step.
* Generates plots showing the linearization of the CDF and saves them to the output folder.

### 3. `stage2_static.py` (Static Image Benchmark & Custom Selector)
* Enhances static low-contrast images and benchmarks processing times.
* Validates math correctness against OpenCV's C++ CLAHE, achieving a very low Mean Squared Error (MSE) of `13.8`.
* **Custom Run modes**:
  * Run with `--image "path/to/img.png"` -> Processes custom image and saves Standalone enhanced result.
  * Run with `--select` -> Launches a native file chooser dialog.

### 4. `stage2_live.py` (Real-Time Edge Processing)
* Runs a live contrast enhancement loop using a webcam (640x480).
* If no webcam is found, it automatically falls back to a **Synthetic Simulation Mode**, generating a low-contrast noisy moving scene to simulate night-time security feeds.
* Displays a side-by-side feed (Original vs. Enhanced) with a live FPS counter (~27.5 FPS). Press `c` to toggle between Custom CLAHE and OpenCV CLAHE, and `q` to quit.

### 5. `app_gui.py` (Unified Desktop GUI Dashboard)
* Built using native `tkinter` (Dark Theme) and Matplotlib.
* Contains a "Select Image" button, sliders for adjusting CLAHE parameters (Clip Limit and Grid Size), and a 2x2 image comparison grid.
* Embeds a Matplotlib canvas on the right side showing active histograms and CDF lines, which update dynamically as sliders are dragged.

---

## 6. How to Run the Code

To run the codebase, open a terminal in the root directory of the project:

* **Launch the GUI Dashboard (Recommended):**
  ```bash
  python src/app_gui.py
  ```
* **Run 8x8 Math Verification:**
  ```bash
  python src/stage1_verification.py
  ```
* **Enhance a Custom Image:**
  ```bash
  python src/stage2_static.py --select
  ```
* **Run Live Video Stream Demo:**
  ```bash
  python src/stage2_live.py
  ```
