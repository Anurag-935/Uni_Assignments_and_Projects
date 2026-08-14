"""
Signals PBL: Desktop GUI Application
Provides a unified dashboard allowing users to:
1. Load a low-light/poor-contrast image.
2. Control CLAHE parameters (Clip Limit, Grid Size) interactively via sliders.
3. View the 4-way comparison (Original, GHE, OpenCV, Custom CLAHE) in a grid.
4. View the corresponding Histograms and CDF graphs in real time.
5. Inspect execution times and MSE metrics.
"""

import os
import time
import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import serial
from tkinter.scrolledtext import ScrolledText



# Embed Matplotlib in Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import custom signal processing functions
from clahe_core import (
    global_histogram_equalization,
    clahe_custom
)

class ContrastEnhancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signals PBL: Edge-Based Contrast Enhancement Dashboard")
        self.root.geometry("1350x800")
        self.root.configure(bg="#1e1e24")
        
        # Application state
        self.raw_img = None        # Original BGR OpenCV image
        self.gray_img = None       # Grayscale input image
        self.ghe_img = None        # GHE output
        self.opencv_img = None     # OpenCV CLAHE output
        self.custom_img = None     # Custom CLAHE output
        
        self.grid_size = (8, 8)
        self.clip_limit = 4.0
        
        # Configure Grid Weights
        self.root.columnconfigure(0, weight=6)  # Left Panel (Images & Controls)
        self.root.columnconfigure(1, weight=5)  # Right Panel (Matplotlib Math)
        self.root.rowconfigure(0, weight=1)
        
        self.create_styles()
        self.build_gui()
        
        # Load default image on startup
        default_path = "../data/low_light_sample.png"
        if os.path.exists(default_path):
            self.load_image_from_path(default_path)

    def create_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Configure ttk widget styling for dark theme
        self.style.configure(".", bg="#1e1e24", fg="#ffffff")
        self.style.configure("TFrame", background="#2a2a35")
        self.style.configure("Control.TFrame", background="#1e1e24")
        
        self.style.configure("TLabel", background="#2a2a35", foreground="#ffffff", font=("Helvetica", 10))
        self.style.configure("Title.TLabel", background="#1e1e24", foreground="#ffffff", font=("Helvetica", 16, "bold"))
        self.style.configure("Section.TLabel", background="#2a2a35", foreground="#fdcb6e", font=("Helvetica", 11, "bold"))
        self.style.configure("Stats.TLabel", background="#181820", foreground="#00b894", font=("Consolas", 10))
        
        self.style.configure("TButton", background="#00b894", foreground="#ffffff", font=("Helvetica", 10, "bold"), borderwidth=0)
        self.style.map("TButton", background=[("active", "#009473")])

    def build_gui(self):
        # ----------------- LEFT PANEL (Controls + 2x2 Grid) -----------------
        left_panel = ttk.Frame(self.root, padding=15)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        left_panel.rowconfigure(0, weight=0)  # Title / Controls
        left_panel.rowconfigure(1, weight=0)  # Sliders
        left_panel.rowconfigure(2, weight=1)  # 2x2 Image Grid
        left_panel.rowconfigure(3, weight=0)  # Stats Box
        left_panel.columnconfigure(0, weight=1)
        
        # Controls Frame (Row 0)
        controls_frame = ttk.Frame(left_panel, style="Control.TFrame")
        controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        # Title
        title_lbl = ttk.Label(controls_frame, text="Signals & Systems PBL: Contrast Enhancer", style="Title.TLabel")
        title_lbl.pack(side="left", padx=5)
        
        # Buttons
        self.load_btn = ttk.Button(controls_frame, text="Select Image", command=self.upload_image)
        self.load_btn.pack(side="right", padx=5, ipadx=10, ipady=4)
        
        self.stm32_btn = ttk.Button(controls_frame, text="STM32 Co-Processor", command=self.open_stm32_window)
        self.stm32_btn.pack(side="right", padx=5, ipadx=10, ipady=4)

        
        # Sliders Frame (Row 1)
        sliders_frame = ttk.Frame(left_panel, padding=10)
        sliders_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        # Clip Limit Slider
        ttk.Label(sliders_frame, text="CLAHE Clip Limit:").grid(row=0, column=0, sticky="w", padx=5)
        self.clip_slider = tk.Scale(sliders_frame, from_=1.0, to=10.0, resolution=0.5, orient="horizontal",
                                    bg="#2a2a35", fg="#ffffff", highlightthickness=0, command=self.on_slider_change)
        self.clip_slider.set(self.clip_limit)
        self.clip_slider.grid(row=0, column=1, sticky="ew", padx=10)
        
        # Grid Size Slider
        ttk.Label(sliders_frame, text="Tile Grid Layout:").grid(row=0, column=2, sticky="w", padx=5)
        self.grid_slider = tk.Scale(sliders_frame, from_=2, to=16, resolution=2, orient="horizontal",
                                    bg="#2a2a35", fg="#ffffff", highlightthickness=0, command=self.on_slider_change)
        self.grid_slider.set(self.grid_size[0])
        self.grid_slider.grid(row=0, column=3, sticky="ew", padx=10)
        
        sliders_frame.columnconfigure(1, weight=1)
        sliders_frame.columnconfigure(3, weight=1)
        
        # 2x2 Image Display Grid (Row 2)
        grid_frame = ttk.Frame(left_panel)
        grid_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        
        for r in range(2):
            grid_frame.rowconfigure(r, weight=1)
            grid_frame.columnconfigure(r, weight=1)
            
        # Individual image frames
        # Original Grayscale
        self.orig_card = ttk.LabelFrame(grid_frame, text=" Original Grayscale (Low-Light) ", padding=5)
        self.orig_card.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.orig_lbl = ttk.Label(self.orig_card, text="[No Image Loaded]", anchor="center")
        self.orig_lbl.pack(fill="both", expand=True)
        
        # GHE Frame
        self.ghe_card = ttk.LabelFrame(grid_frame, text=" Global Histogram Equalization (GHE) ", padding=5)
        self.ghe_card.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.ghe_lbl = ttk.Label(self.ghe_card, text="[No Image Loaded]", anchor="center")
        self.ghe_lbl.pack(fill="both", expand=True)
        
        # OpenCV CLAHE Frame
        self.cv_card = ttk.LabelFrame(grid_frame, text=" OpenCV CLAHE Reference ", padding=5)
        self.cv_card.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.cv_lbl = ttk.Label(self.cv_card, text="[No Image Loaded]", anchor="center")
        self.cv_lbl.pack(fill="both", expand=True)
        
        # Custom CLAHE Frame
        self.cust_card = ttk.LabelFrame(grid_frame, text=" Custom Vectorized CLAHE ", padding=5)
        self.cust_card.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.cust_lbl = ttk.Label(self.cust_card, text="[No Image Loaded]", anchor="center")
        self.cust_lbl.pack(fill="both", expand=True)
        
        # Performance Stats box (Row 3)
        self.stats_frame = ttk.Frame(left_panel, padding=10)
        self.stats_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.stats_lbl = ttk.Label(self.stats_frame, text="Ready. Load an image to start processing.", style="Stats.TLabel")
        self.stats_lbl.pack(anchor="w")

        # ----------------- RIGHT PANEL (Matplotlib Math Graphs) -----------------
        self.right_panel = ttk.Frame(self.root, padding=15)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.right_panel.rowconfigure(0, weight=1)
        self.right_panel.columnconfigure(0, weight=1)
        
        # Create matplotlib figure with dark background
        self.fig = Figure(figsize=(5, 6), dpi=100)
        self.fig.patch.set_facecolor("#2a2a35")
        
        # Layout plots: Top: Histograms, Bottom: CDFs
        self.ax_hist = self.fig.add_subplot(211)
        self.ax_cdf = self.fig.add_subplot(212)
        self.style_axes(self.ax_hist, "Grayscale Histograms", "Intensity Value", "Frequency")
        self.style_axes(self.ax_cdf, "Cumulative Distribution Function (CDF)", "Intensity Value", "Probability Density")
        
        # Adjust layout to prevent title and axis label overlap
        self.fig.tight_layout()
        
        # Canvas Aggregator to bridge Matplotlib & Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
    def style_axes(self, ax, title, xlabel, ylabel):
        ax.set_facecolor("#1e1e24")
        ax.set_title(title, color="#ffffff", fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, color="#b0b0c0", fontsize=9)
        ax.set_ylabel(ylabel, color="#b0b0c0", fontsize=9)
        ax.tick_params(colors="#b0b0c0", labelsize=8)
        ax.spines['bottom'].set_color('#404050')
        ax.spines['top'].set_color('#404050')
        ax.spines['left'].set_color('#404050')
        ax.spines['right'].set_color('#404050')
        ax.grid(True, color="#2d2d38", linestyle="--", linewidth=0.5)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Upload Low-Light Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")]
        )
        if file_path:
            self.load_image_from_path(file_path)
            
    def load_image_from_path(self, file_path):
        self.raw_img = cv2.imread(file_path)
        if self.raw_img is None:
            self.stats_lbl.configure(text="Error loading selected image file.")
            return
            
        # Convert to grayscale
        if len(self.raw_img.shape) == 3:
            self.gray_img = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_img = self.raw_img.copy()
            
        self.process_image()

    def on_slider_change(self, val):
        # Read values from Tk Scale sliders
        self.clip_limit = float(self.clip_slider.get())
        g_size = int(self.grid_slider.get())
        self.grid_size = (g_size, g_size)
        
        # If image is already loaded, re-process with new sliders
        if self.gray_img is not None:
            self.process_image()

    def process_image(self):
        if self.gray_img is None:
            return
            
        # Time and execute Global Histogram Equalization (GHE)
        t0 = time.perf_counter()
        self.ghe_img = global_histogram_equalization(self.gray_img)
        t_ghe = (time.perf_counter() - t0) * 1000.0
        
        # Time and execute OpenCV CLAHE (Reference)
        t0 = time.perf_counter()
        clahe_cv = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
        self.opencv_img = clahe_cv.apply(self.gray_img)
        t_cv = (time.perf_counter() - t0) * 1000.0
        
        # Time and execute Custom Vectorized CLAHE
        t0 = time.perf_counter()
        self.custom_img = clahe_custom(self.gray_img, grid_size=self.grid_size, clip_limit=self.clip_limit)
        t_custom = (time.perf_counter() - t0) * 1000.0
        
        # Calculate MSE
        mse = np.mean((self.opencv_img.astype(float) - self.custom_img.astype(float)) ** 2)
        
        # Update Status Footer
        self.stats_lbl.configure(
            text=f"Resolution: {self.gray_img.shape[1]}x{self.gray_img.shape[0]} px | "
                 f"GHE: {t_ghe:.1f}ms | OpenCV CLAHE: {t_cv:.1f}ms | "
                 f"Custom CLAHE: {t_custom:.1f}ms | MSE vs Reference: {mse:.3f}"
        )
        
        # Render images to grid
        self.update_image_display(self.orig_lbl, self.gray_img)
        self.update_image_display(self.ghe_lbl, self.ghe_img)
        self.update_image_display(self.cv_lbl, self.opencv_img)
        self.update_image_display(self.cust_lbl, self.custom_img)
        
        # Render plots
        self.update_plots()

    def update_image_display(self, label_widget, cv_img):
        # Resize image to fit inside Card limits while keeping aspect ratio
        h, w = cv_img.shape
        max_size = 280
        
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert OpenCV Gray Mat to PIL Image -> ImageTk
        pil_img = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        label_widget.configure(image=tk_img, text="")
        label_widget.image = tk_img  # Store reference to prevent garbage collection

    def update_plots(self):
        # Clear previous plots
        self.ax_hist.clear()
        self.ax_cdf.clear()
        
        # Restyle
        self.style_axes(self.ax_hist, "Grayscale Histograms", "Intensity Value", "Frequency")
        self.style_axes(self.ax_cdf, "Cumulative Distribution Function (CDF)", "Intensity Value", "Probability Density")
        
        # Compute histograms and CDFs
        def calc_data(mat):
            hist, _ = np.histogram(mat, bins=256, range=(0, 256))
            cdf = hist.cumsum()
            cdf_norm = cdf / cdf[-1]
            return hist, cdf_norm
            
        hist_orig, cdf_orig = calc_data(self.gray_img)
        hist_ghe, cdf_ghe = calc_data(self.ghe_img)
        hist_cust, cdf_cust = calc_data(self.custom_img)
        
        # Plot Histograms
        # Original (gray fill)
        self.ax_hist.fill_between(range(256), hist_orig, color="gray", alpha=0.3, label="Original")
        # GHE (blue outline)
        self.ax_hist.plot(range(256), hist_ghe, color="#0097e6", linewidth=1.5, label="GHE")
        # Custom CLAHE (green outline)
        self.ax_hist.plot(range(256), hist_cust, color="#00b894", linewidth=1.5, label="Custom CLAHE")
        self.ax_hist.legend(loc="upper right", facecolor="#1e1e24", edgecolor="none", fontsize=8, labelcolor="#ffffff")
        
        # Plot CDFs
        self.ax_cdf.plot(range(256), cdf_orig, color="gray", linewidth=1.5, linestyle="--", label="Original")
        self.ax_cdf.plot(range(256), cdf_ghe, color="#0097e6", linewidth=2, label="GHE")
        self.ax_cdf.plot(range(256), cdf_cust, color="#00b894", linewidth=2, label="Custom CLAHE")
        self.ax_cdf.legend(loc="lower right", facecolor="#1e1e24", edgecolor="none", fontsize=8, labelcolor="#ffffff")
        
        # Adjust layout to prevent title and axis label overlap during redraws
        self.fig.tight_layout()
        
        # Redraw matplotlib canvas
        self.canvas.draw()

    def open_stm32_window(self):
        if self.gray_img is None:
            tk.messagebox.showerror("Error", "Please load an image in the main window first.")
            return

        # Create new window
        self.stm_win = tk.Toplevel(self.root)
        self.stm_win.title("STM32 Hardware Co-Processor Panel")
        self.stm_win.geometry("1100x750")
        self.stm_win.configure(bg="#1e1e24")
        self.stm_win.grab_set() # Focus this window

        # Configure row/col weights for the window so it expands when full screen
        self.stm_win.columnconfigure(0, weight=1)
        self.stm_win.rowconfigure(0, weight=0) # Top Controls
        self.stm_win.rowconfigure(1, weight=3) # Images (expands most)
        self.stm_win.rowconfigure(2, weight=2) # Logs/Info (expands less)

        # 1. Top Controls Frame (Row 0)
        top_frame = ttk.Frame(self.stm_win, padding=10)
        top_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=5)

        ttk.Label(top_frame, text="STM32 COM Port:").pack(side="left", padx=5)
        self.port_entry = ttk.Entry(top_frame, width=10)
        self.port_entry.insert(0, "COM10")
        self.port_entry.pack(side="left", padx=5)

        self.proc_btn = ttk.Button(top_frame, text="Transmit & Process Image", command=self.run_stm32_processing)
        self.proc_btn.pack(side="left", padx=15, ipadx=10)

        # 2. Main 1x2 Image Grid (Row 1)
        grid_frame = ttk.Frame(self.stm_win, padding=5)
        grid_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=5)
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)
        grid_frame.rowconfigure(0, weight=1)

        # Left Card (Original)
        orig_card = ttk.LabelFrame(grid_frame, text=" Original High-Resolution Grayscale ")
        orig_card.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.stm_orig_lbl = ttk.Label(orig_card, text="[Waiting for Process]", anchor="center")
        self.stm_orig_lbl.pack(fill="both", expand=True)

        # Right Card (Enhanced)
        enh_card = ttk.LabelFrame(grid_frame, text=" Fully Interpolated High-Resolution STM32 Enhanced ")
        enh_card.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.stm_enh_lbl = ttk.Label(enh_card, text="[Waiting for Process]", anchor="center")
        self.stm_enh_lbl.pack(fill="both", expand=True)

        # 3. Bottom Logs & Explanation Area (Row 2)
        bottom_frame = ttk.Frame(self.stm_win, padding=5)
        bottom_frame.grid(row=2, column=0, sticky="nsew", padx=15, pady=5)
        bottom_frame.columnconfigure(0, weight=3) # Text log
        bottom_frame.columnconfigure(1, weight=2) # Explanation
        bottom_frame.rowconfigure(0, weight=1)

        # Log Terminal
        log_frame = ttk.LabelFrame(bottom_frame, text=" Live Data Transmission Log (Hex Dumps) ")
        log_frame.grid(row=0, column=0, sticky="nsew", padx=5)
        
        # ScrolledText handles mousewheel and scroll bar binding automatically
        self.log_text = ScrolledText(log_frame, height=9, bg="black", fg="#00b894", font=("Consolas", 9), insertbackground="white")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.log_text.insert("end", "System Ready. Select COM port and click 'Transmit & Process'.\n")
        self.log_text.see("end")

        # Protocol Explanation Card
        info_frame = ttk.LabelFrame(bottom_frame, text=" Hardware Pipeline Explanation ")
        info_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        
        info_text = (
            "1. Downsampling: Laptop resizes image to 128x128 (16KB) to fit STM32 RAM.\n"
            "2. Packet Tx: Laptop sends: Header [AA BB CC DD] -> Pixel Data -> Checksum.\n"
            "3. Board Action: STM32 computes local histograms, clips contrast, and generates "
            "mapping curves (Lookup Tables) for the 8x8 tile layout.\n"
            "4. LUT Rx: STM32 returns the 16KB of mapping tables (LUTs) over USB VCP.\n"
            "5. Laptop Rendering: Laptop performs bilinear interpolation using the "
            "STM32-accelerated LUTs directly on the original high-resolution image, "
            "preserving full quality!"
        )
        info_lbl = ttk.Label(info_frame, text=info_text, justify="left", wraplength=400)
        info_lbl.pack(fill="both", expand=True, padx=8, pady=5)

        # Show initial original image
        self.update_stm_image_display(self.stm_orig_lbl, self.gray_img)

    def write_to_log(self, text):
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.update_idletasks()
        print(text) # Automatically mirror to PowerShell terminal

    def run_stm32_processing(self):
        port = self.port_entry.get().strip()
        self.log_text.delete("1.0", "end")
        
        # Clear console and print clear headers in PowerShell
        print("\n" + "="*60)
        print("        STM32 HARDWARE CO-PROCESSOR TRANSMISSION LOG")
        print("="*60)
        
        self.write_to_log(f"[INIT] Starting Co-processor execution on {port}...")

        # 1. Downsample
        self.write_to_log("[STEP 1] Downsampling image to 128x128 pixels (16KB)...")
        img_resized = cv2.resize(self.gray_img, (128, 128))
        raw_bytes = img_resized.tobytes()
        checksum = sum(raw_bytes) % 256

        # 2. Build Packet
        header = bytes([0xAA, 0xBB, 0xCC, 0xDD])
        packet = header + raw_bytes + bytes([checksum])
        
        self.write_to_log("[STEP 2] Preparing Data Packet:")
        self.write_to_log(f"  * Header: {header.hex().upper()}")
        self.write_to_log(f"  * Checksum: {hex(checksum).upper()}")
        self.write_to_log(f"  * TX Hex Dump (First 16 bytes): {raw_bytes[:16].hex(' ').upper()}")

        # 3. Connection
        self.write_to_log(f"[STEP 3] Opening Serial COM Port {port}...")
        try:
            ser = serial.Serial(port, baudrate=115200, timeout=3.0)
            time.sleep(0.5) # Stabilization delay
            
            self.write_to_log("Transmitting data packet...")
            start_time = time.perf_counter()
            ser.write(packet)
            
            self.write_to_log("Waiting for STM32 calculation response...")
            response = ser.read(128 * 128)
            elapsed_time = (time.perf_counter() - start_time) * 1000.0
            ser.close()
            
            if len(response) == 128 * 128:
                self.write_to_log(f"[STEP 4] Received calculated data from STM32 in {elapsed_time:.1f} ms:")
                self.write_to_log(f"  * Bytes Received: {len(response)} bytes")
                self.write_to_log(f"  * RX Hex Dump (First 16 bytes): {response[:16].hex(' ').upper()}")
                
                # Reconstruct LUTs
                tile_luts = np.frombuffer(response, dtype=np.uint8).reshape((8, 8, 256))
                self.write_to_log("  * Successfully reconstructed 8x8x256 Local Mapping Lookup Tables.")
                
                # 4. Bilinear Interpolation on original high-res image
                self.write_to_log(f"[STEP 5] Applying STM32 LUTs to original high-res image on Laptop CPU...")
                start_interp = time.perf_counter()
                enhanced_highres = self.apply_luts_to_highres(self.gray_img, tile_luts)
                interp_time = (time.perf_counter() - start_interp) * 1000.0
                self.write_to_log(f"  * High-res Bilinear Rendering completed in {interp_time:.1f} ms.")
                
                # Display enhanced high-res image
                self.update_stm_image_display(self.stm_enh_lbl, enhanced_highres)
                self.write_to_log("[SUCCESS] Rendering complete. Images updated successfully!")
                print("="*60 + "\n")
            else:
                self.write_to_log(f"[ERROR] Incomplete packet received: {len(response)} bytes. Check connection.")
                print("="*60 + "\n")
                
        except Exception as e:
            self.write_to_log(f"[ERROR] Serial Communication Failed: {e}")
            print("="*60 + "\n")


    def apply_luts_to_highres(self, img_highres, tile_luts):
        h, w = img_highres.shape
        rows, cols, _ = tile_luts.shape
        
        tile_h = h / rows
        tile_w = w / cols
        
        tc_y = np.array([r * tile_h + (tile_h - 1) / 2 for r in range(rows)], dtype=np.float32)
        tc_x = np.array([c * tile_w + (tile_w - 1) / 2 for c in range(cols)], dtype=np.float32)
        
        def get_interp_params(coords, centers, tile_size):
            n = len(coords)
            num_centers = len(centers)
            i0 = np.zeros(n, dtype=np.int32)
            i1 = np.zeros(n, dtype=np.int32)
            weight = np.zeros(n, dtype=np.float32)
            
            for idx, val in enumerate(coords):
                if val < centers[0]:
                    i0[idx] = 0
                    i1[idx] = 0
                    weight[idx] = 0.0
                elif val >= centers[-1]:
                    i0[idx] = num_centers - 1
                    i1[idx] = num_centers - 1
                    weight[idx] = 0.0
                else:
                    k = int((val - centers[0]) / tile_size)
                    if k < 0: k = 0
                    if k > num_centers - 2: k = num_centers - 2
                    i0[idx] = k
                    i1[idx] = k + 1
                    weight[idx] = (val - centers[k]) / (centers[k+1] - centers[k])
            return i0, i1, weight
            
        r0, r1, b = get_interp_params(np.arange(h), tc_y, tile_h)
        c0, c1, a = get_interp_params(np.arange(w), tc_x, tile_w)
        
        r0_2d = r0[:, np.newaxis]
        r1_2d = r1[:, np.newaxis]
        c0_2d = c0[np.newaxis, :]
        c1_2d = c1[np.newaxis, :]
        
        b_2d = b[:, np.newaxis]
        a_2d = a[np.newaxis, :]
        
        s_tl = tile_luts[r0_2d, c0_2d, img_highres]
        s_tr = tile_luts[r0_2d, c1_2d, img_highres]
        s_bl = tile_luts[r1_2d, c0_2d, img_highres]
        s_br = tile_luts[r1_2d, c1_2d, img_highres]
        
        s_top = (1.0 - a_2d) * s_tl + a_2d * s_tr
        s_bottom = (1.0 - a_2d) * s_bl + a_2d * s_br
        enhanced = (1.0 - b_2d) * s_top + b_2d * s_bottom
        
        return np.round(enhanced).astype(np.uint8)

    def update_stm_image_display(self, label_widget, cv_img):
        # Resize to fit in STM32 window cards (max size 450x450)
        h, w = cv_img.shape
        max_size = 450
        
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        pil_img = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        label_widget.configure(image=tk_img, text="")
        label_widget.image = tk_img


if __name__ == "__main__":
    # Change working directory to file directory to resolve relative paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    root = tk.Tk()
    app = ContrastEnhancerApp(root)
    root.mainloop()
