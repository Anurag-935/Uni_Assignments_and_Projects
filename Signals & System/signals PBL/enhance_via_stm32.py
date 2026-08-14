import serial
import time
import numpy as np
import cv2

# Configure your STM32 COM port
PORT = 'COM10'
BAUDRATE = 115200
TIMEOUT = 3.0

def enhance_image_on_stm32(image_path):
    # 1. Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from path '{image_path}'")
        return
        
    print(f"Original image size: {img.shape[1]}x{img.shape[0]}")
    
    # 2. Resize to 128x128 for STM32 processing
    img_resized = cv2.resize(img, (128, 128))
    raw_bytes = img_resized.tobytes()
    
    # 3. Compute checksum
    checksum = sum(raw_bytes) % 256
    
    # 4. Open Serial port
    print(f"Connecting to STM32 on {PORT}...")
    try:
        ser = serial.Serial(PORT, baudrate=BAUDRATE, timeout=TIMEOUT)
        time.sleep(1) # wait for connection
        
        # 5. Build packet: Header (0xAA 0xBB 0xCC 0xDD) + Data (16384 bytes) + Checksum (1 byte)
        header = bytes([0xAA, 0xBB, 0xCC, 0xDD])
        packet = header + raw_bytes + bytes([checksum])
        
        print("Sending image data to STM32 (16,389 bytes)...")
        start_time = time.time()
        ser.write(packet)
        
        # 6. Read response (16384 bytes)
        print("Waiting for processed image from STM32...")
        response = ser.read(128 * 128)
        elapsed_time = (time.time() - start_time) * 1000 # in ms
        
        ser.close()
        
        if len(response) == 128 * 128:
            print(f"Success! Received enhanced image in {elapsed_time:.1f} ms.")
            
            # Reconstruct image
            enhanced_img = np.frombuffer(response, dtype=np.uint8).reshape((128, 128))
            
            # Upscale both to 512x512 using smooth Bicubic interpolation
            original_display = cv2.resize(img_resized, (512, 512), interpolation=cv2.INTER_CUBIC)
            enhanced_display = cv2.resize(enhanced_img, (512, 512), interpolation=cv2.INTER_CUBIC)
            
            # Concatenate side-by-side
            comparison = np.hstack((original_display, enhanced_display))
            
            # Save results
            output_path = "data/enhanced_output.png"
            cv2.imwrite(output_path, comparison)
            print(f"Enhanced image saved successfully as '{output_path}'.")
            
        else:
            print(f"Error: Received incomplete data ({len(response)} bytes). Checksum or transmission error.")
            
    except Exception as e:
        print(f"Serial Error: {e}")

if __name__ == "__main__":
    # Test using the sample image in your workspace
    enhance_image_on_stm32("data/low_light_sample.png")
