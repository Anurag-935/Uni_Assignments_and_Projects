import serial
import time
import numpy as np
import cv2

# Configure your STM32 COM port
PORT = 'COM10'
BAUDRATE = 115200
TIMEOUT = 3.0

def apply_luts_to_highres(img_highres, tile_luts):
    """
    Applies the 8x8 local LUTs calculated by the STM32 to the original 
    high-resolution image on the laptop using fast vectorized bilinear interpolation.
    """
    h, w = img_highres.shape
    rows, cols, _ = tile_luts.shape
    
    # Calculate tile sizes for high-res image
    tile_h = h / rows
    tile_w = w / cols
    
    # Calculate tile center coordinates on high-res grid
    tc_y = np.array([r * tile_h + (tile_h - 1) / 2 for r in range(rows)], dtype=np.float32)
    tc_x = np.array([c * tile_w + (tile_w - 1) / 2 for c in range(cols)], dtype=np.float32)
    
    # Calculate 1D interpolation parameters
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
    
    # Expand to 2D for fast broadcasting
    r0_2d = r0[:, np.newaxis]
    r1_2d = r1[:, np.newaxis]
    c0_2d = c0[np.newaxis, :]
    c1_2d = c1[np.newaxis, :]
    
    b_2d = b[:, np.newaxis]
    a_2d = a[np.newaxis, :]
    
    # Retrieve mapped intensities from the 4 surrounding tile LUTs
    s_tl = tile_luts[r0_2d, c0_2d, img_highres]
    s_tr = tile_luts[r0_2d, c1_2d, img_highres]
    s_bl = tile_luts[r1_2d, c0_2d, img_highres]
    s_br = tile_luts[r1_2d, c1_2d, img_highres]
    
    # Bilinear interpolation
    s_top = (1.0 - a_2d) * s_tl + a_2d * s_tr
    s_bottom = (1.0 - a_2d) * s_bl + a_2d * s_br
    enhanced = (1.0 - b_2d) * s_top + b_2d * s_bottom
    
    return np.round(enhanced).astype(np.uint8)

def run_coprocessor_demo(image_path):
    # 1. Load high-res image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from path '{image_path}'")
        return
        
    h_orig, w_orig = img.shape
    print(f"\n[STEP 1] Loaded Original Image. Resolution: {w_orig}x{h_orig}")
    
    # 2. Downsample to 128x128 for STM32 processing
    img_resized = cv2.resize(img, (128, 128))
    raw_bytes = img_resized.tobytes()
    checksum = sum(raw_bytes) % 256
    
    # 3. Serial Packet Setup
    header = bytes([0xAA, 0xBB, 0xCC, 0xDD])
    packet = header + raw_bytes + bytes([checksum])
    
    print(f"\n[STEP 2] Prepared STM32 Data Packet:")
    print(f"  * Header: {header.hex().upper()}")
    print(f"  * Payload Size: {len(raw_bytes)} bytes (128x128 pixels)")
    print(f"  * Checksum: {hex(checksum).upper()}")
    print(f"  * TX Hex Dump (First 16 bytes): {raw_bytes[:16].hex(' ').upper()}")
    
    # 4. Transmission
    try:
        print(f"\n[STEP 3] Opening Serial Port {PORT}...")
        ser = serial.Serial(PORT, baudrate=BAUDRATE, timeout=TIMEOUT)
        time.sleep(1) # stabilization
        
        print("Transmitting packet to STM32...")
        start_time = time.time()
        ser.write(packet)
        
        print("Waiting for calculated LUTs from STM32...")
        response = ser.read(128 * 128) # Reads 16384 bytes of tile_luts
        elapsed_time = (time.time() - start_time) * 1000
        ser.close()
        
        if len(response) == 128 * 128:
            print(f"\n[STEP 4] Received Data Pack from STM32 in {elapsed_time:.1f} ms:")
            print(f"  * Bytes Received: {len(response)} bytes")
            print(f"  * RX Hex Dump (First 16 bytes): {response[:16].hex(' ').upper()}")
            
            # Reconstruct 8x8x256 LUTs
            tile_luts = np.frombuffer(response, dtype=np.uint8).reshape((8, 8, 256))
            print("  * Reconstructed 8x8x256 Local Mapping Lookup Tables (LUTs).")
            
            # 5. Apply LUTs to original high-res image
            print(f"\n[STEP 5] Applying STM32 LUTs to original {w_orig}x{h_orig} image on Laptop CPU...")
            start_interp = time.time()
            enhanced_highres = apply_luts_to_highres(img, tile_luts)
            interp_time = (time.time() - start_interp) * 1000
            print(f"  * High-res Bilinear Rendering completed in {interp_time:.1f} ms.")
            
            # 6. Save results
            output_path = "data/enhanced_output.png"
            # Compare original high-res vs enhanced high-res side-by-side
            comparison = np.hstack((img, enhanced_highres))
            # Resize comparison for screen-fitting if it's too large
            if comparison.shape[1] > 1920:
                comparison_disp = cv2.resize(comparison, (1280, int(1280 * comparison.shape[0] / comparison.shape[1])))
            else:
                comparison_disp = comparison
                
            cv2.imwrite(output_path, comparison)
            print(f"\n[SUCCESS] Saved high-resolution comparison to '{output_path}'.")
            
        else:
            print(f"\n[ERROR] Incomplete packet received: {len(response)} bytes. Connection failed.")
            
    except Exception as e:
        print(f"\n[ERROR] Serial Communication Failed: {e}")

if __name__ == "__main__":
    run_coprocessor_demo("C:/Users/anura/OneDrive/Desktop/signals PBL/data/low_light_sample.png")
