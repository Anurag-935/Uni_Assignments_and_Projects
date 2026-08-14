# Signals PBL: Edge-Based Contrast Enhancement using CLAHE

This project implements and validates a high-performance **Contrast Limited Adaptive Histogram Equalization (CLAHE)** signal processing pipeline optimized for embedded edge deployments (e.g., Raspberry Pi 3 B+ running DietPi). It addresses the limitations of Global Histogram Equalization (GHE) by locally adapting to lighting conditions while preventing noise over-amplification through clip-limit histogram redistribution and boundary smoothing via bilinear interpolation.

---

## 1. Project Directory Structure
```
signals PBL/
├── README.md                  # Project documentation (Theory, Setup, Run Guide)
├── requirements.txt           # Python package dependencies
├── data/
│   └── low_light_sample.png   # Low-contrast benchmark test image
├── src/
│   ├── __init__.py
│   ├── clahe_core.py          # Math library (GHE, CLAHE, interpolation, LUTs)
│   ├── stage1_verification.py # 8x8 micro matrix math verification & CDF plots
│   ├── stage2_static.py       # Benchmark static image processing & visual comparison
│   └── stage2_live.py         # Real-time processing stream (Webcam + Synthetic fallback)
└── output/
    ├── stage1_plots.png       # Matplotlib histogram and CDF linearization plots
    ├── stage2_comparison.png  # Visual comparison chart (Original, GHE, OpenCV, Custom)
    ├── img_original_gray.png  # Grayscale input image
    ├── img_ghe.png            # Global Equalization output
    ├── img_opencv_clahe.png   # OpenCV reference output
    └── img_custom_clahe.png   # Custom CLAHE output
```

---

## 2. Mathematical Foundations

### Probability Mass Function (PMF) & Histogram
For an $M \times N$ pixel tile, the frequency of grayscale intensity value $v \in [0, 255]$ is counted:
$$H(v) = \sum_{y=0}^{M-1} \sum_{x=0}^{N-1} \delta(I(y, x) - v)$$
where $\delta(0) = 1$ and $\delta(n) = 0$ for $n \neq 0$.

### Cumulative Distribution Function (CDF)
The CDF is the running cumulative sum of the histogram:
$$CDF(v) = \sum_{i=0}^{v} H(i)$$

### Min-Max CDF Scaling Formula
To map the cumulative ranks back onto the 8-bit spectrum $[0, 255]$, each intensity level $v$ is normalized:
$$h(v) = \text{round}\left( \frac{CDF(v) - CDF_{min}}{(M \times N) - CDF_{min}} \times 255 \right)$$
where $CDF_{min}$ maps the lowest observed intensity in the tile to $0$ (pure black).

### Bilinear Interpolation
To blend tile boundaries and remove blocking artifacts, pixel intensities are interpolated using the four closest tile centers $C_{tl}, C_{tr}, C_{bl}, C_{br}$ with coordinates $(x_0, y_0), (x_1, y_0), (x_0, y_1), (x_1, y_1)$:
$$a = \frac{x - x_0}{x_1 - x_0}, \quad b = \frac{y - y_0}{y_1 - y_0}$$
$$s_{top} = (1 - a) \cdot s_{tl} + a \cdot s_{tr}$$
$$s_{bottom} = (1 - a) \cdot s_{bl} + a \cdot s_{br}$$
$$s_{final} = (1 - b) \cdot s_{top} + b \cdot s_{bottom}$$
*Where $s_{tl}, s_{tr}, s_{bl}, s_{br}$ are the mapped intensities retrieved from the corresponding tile LUTs.*

---

## 3. Hardware Optimization Strategies

1. **Grayscale Conversion**: Converts 3-channel RGB data to 1-channel Grayscale using the NTSC formula:
   $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
   This reduces matrix size by **$66.7\%$**, optimizing RAM footprint and preventing thermal throttling on low-spec processors.
2. **Lookup Table (LUT) Processing**: To avoid re-evaluating the CDF scaling math for every pixel in a tile, the mapping formula is executed exactly 256 times per tile to build a static translation array (LUT). Mapping is then resolved in $O(1)$ memory lookup time.
3. **Vectorized Bilinear Interpolation**: Precomputes row and column mapping indices/weights in 1D arrays ($O(H) + O(W)$) and projects them into 2D broadcasts using NumPy. This eliminates expensive double pixel loops in Python.

---

## 4. Setup & Installation

Ensure you have Python 3.7+ installed. Clone or copy the project folder to your local machine, open a shell in the root `signals PBL` directory, and run:

```bash
pip install -r requirements.txt
```

---

## 5. Execution Guide

### Interactive GUI Dashboard (Recommended)
Launches the full desktop dashboard app where you can upload images, adjust sliders (Clip Limit, Grid Layout), and view the comparison matrix and graphs in a single window.
```bash
python src/app_gui.py
```

### Stage 1: Software Verification (8x8 Micro Matrix)
Runs the mathematical pipeline on a micro-scale 8x8 matrix. It prints step-by-step intermediate histograms, clipping limits, redistribution counts, mapping tables, and exports CDF linearization plots.
```bash
python src/stage1_verification.py
```
*Output plot:* `output/stage1_plots.png`

### Stage 2: Static Image Enhancement (Demo A)
Enhances low-light or poor-contrast images. By default, it runs on the benchmark image, but you can feed it any image of your own:
* **Default Run**:
  ```bash
  python src/stage2_static.py
  ```
* **Custom Image Run**:
  ```bash
  python src/stage2_static.py --image path/to/your/image.png
  ```
* **GUI File Chooser Mode**:
  ```bash
  python src/stage2_static.py --select
  ```
*Output files:* 
* Comparison sheet: `output/stage2_comparison.png`
* Standalone enhanced output: `output/[your_image_name]_enhanced.png`

### Stage 2: Live Real-Time Processing (Demo B)
Runs a real-time capture loop. If a webcam is available, it processes the camera stream. If no camera is detected, it automatically initiates a **Synthetic Simulation Mode** generating a moving shape sequence with added camera sensor noise to demonstrate live contrast recovery.
```bash
python src/stage2_live.py
```
* **Controls**:
  * Press **`c`** to toggle between **Custom CLAHE** and **OpenCV CLAHE** processing engines.
  * Press **`q`** to quit the live feed.

---

## 6. Future Scope (Embedded Scaling)
For highly demanding applications (such as 4K video streams at 60 FPS), the CPU-bound tiling and interpolation steps can be offloaded to **Field Programmable Gate Arrays (FPGAs)** (like Xilinx Zynq-7000 SoCs) using hardware-software co-design:
* **Hardware Logic**: Parallel local histogram buffers, clipping circuits, and bilinear interpolators implemented in fabric logic blocks.
* **Software Core**: Adaptive threshold parameters and overall stream coordination handled by an ARM processor core.
