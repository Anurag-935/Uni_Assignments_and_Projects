"""
Stage 2: Embedded Hardware Simulation - Demo B (Live Edge Processing)
Simulates real-time video processing on the edge (Raspberry Pi / PC).
Features:
1. Captures live frames from a physical camera (webcam).
2. Automatics fallback: if no camera is available, generates a synthetic, 
   low-contrast, noisy dynamic scene to simulate low-light CCTV footage.
3. Converts frames to grayscale to optimize CPU cycle usage.
4. Applies CLAHE (with custom / OpenCV toggle) in real-time.
5. Displays original vs enhanced streams side-by-side with real-time FPS overlay.
"""

import os
import time
import cv2
import numpy as np

# Import custom implementations
from clahe_core import clahe_custom

def generate_synthetic_frame(width=640, height=480, frame_count=0):
    """
    Generates a synthetic, low-contrast, noisy video frame.
    Simulates a low-light camera scene with moving objects.
    """
    # Background: dark gray (intensity 40)
    frame = np.ones((height, width), dtype=np.uint8) * 40
    
    # Draw a static background grid of very low contrast
    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), 42, 1)
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), 42, 1)
        
    # Moving Circle (low contrast: intensity 55)
    cx = int(width / 2 + 120 * np.cos(frame_count * 0.04))
    cy = int(height / 2 + 80 * np.sin(frame_count * 0.03))
    cv2.circle(frame, (cx, cy), 60, 52, -1)
    # Add a slightly brighter inner core (intensity 58)
    cv2.circle(frame, (cx, cy), 20, 56, -1)
    
    # Moving Rectangle (low contrast: intensity 48)
    rx = int(width / 2 - 160 + 60 * np.sin(frame_count * 0.02))
    ry = int(height / 2 - 60 + 40 * np.cos(frame_count * 0.05))
    cv2.rectangle(frame, (rx, ry), (rx + 90, ry + 90), 46, -1)
    
    # Stationary low-contrast test card at corner (intensity range 38-48)
    for i in range(5):
        cv2.rectangle(frame, (20 + i*15, 20), (35 + i*15, 50), 38 + i*2, -1)
        
    # Add Gaussian/Sensor Noise (simulates sensor noise in low-light environments)
    noise = np.random.normal(0, 1.8, (height, width)).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return frame

def run_live_processing(test_mode=False):
    print("="*60)
    print("        STAGE 2: DEMO B - LIVE REAL-TIME EDGE PROCESSING      ")
    print("="*60)
    if test_mode:
        print("Running in TEST MODE (10 frames, no GUI)...")
    else:
        print("Controls:")
        print("  Press 'c' to toggle between Custom CLAHE and OpenCV CLAHE")
        print("  Press 'q' to quit the demo")
    print("="*60)
    
    # Initialize camera capture
    cap = cv2.VideoCapture(0)
    camera_active = cap.isOpened()
    
    if camera_active:
        print("Camera detected. Capturing real-time feed from Device 0...")
        # Set resolution to 640x480 for edge performance optimization
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        print("No camera detected. Launching in Synthetic Simulation Mode...")
        cap.release()
        
    frame_count = 0
    fps = 0.0
    fps_start = time.perf_counter()
    fps_frames = 0
    
    use_custom = True
    total_algo_time = 0.0
    
    while True:
        loop_start = time.perf_counter()
        
        # 1. Fetch input frame
        if camera_active:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting loop...")
                break
            # Convert to Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            # Generate synthetic low-contrast frame
            gray = generate_synthetic_frame(640, 480, frame_count)
            
        frame_count += 1
        
        # 2. Apply chosen algorithm
        algo_name = "Custom CLAHE" if use_custom else "OpenCV CLAHE"
        algo_start = time.perf_counter()
        if use_custom:
            enhanced = clahe_custom(gray, grid_size=(8, 8), clip_limit=4.0)
        else:
            # OpenCV CLAHE reference
            clahe_cv = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced = clahe_cv.apply(gray)
        algo_time = (time.perf_counter() - algo_start) * 1000.0
        total_algo_time += algo_time
        
        # 3. Compute FPS
        fps_frames += 1
        elapsed = time.perf_counter() - fps_start
        if elapsed >= 1.0:
            fps = fps_frames / elapsed
            fps_frames = 0
            fps_start = time.perf_counter()
            
        if test_mode:
            print(f"Frame {frame_count:2d}/10: Processing time = {algo_time:.2f} ms")
            if frame_count >= 10:
                avg_time = total_algo_time / frame_count
                print(f"\nTest Run complete. Average algorithm time: {avg_time:.2f} ms (~{1000.0/avg_time:.1f} FPS theoretical)")
                break
            continue
            
        # 4. Format display
        # Draw side-by-side comparison
        combined = np.hstack((gray, enhanced))
        
        # Convert combined back to BGR to write colored text
        display_frame = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        
        # Add labels and statistics overlays
        h, w, _ = display_frame.shape
        cv2.putText(display_frame, "ORIGINAL (Low-Contrast)", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if camera_active else (200, 200, 200), 2)
        cv2.putText(display_frame, f"ENHANCED ({algo_name})", (w // 2 + 20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Overlay stats on the bottom left
        stats_bg = np.zeros((100, 260, 3), dtype=np.uint8)
        # alpha blend stats box onto display
        display_frame[h-120:h-20, 20:280] = cv2.addWeighted(
            display_frame[h-120:h-20, 20:280], 0.5, stats_bg, 0.5, 0
        )
        
        cv2.putText(display_frame, f"Resolution: {gray.shape[1]}x{gray.shape[0]}", (30, h - 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Algorithm: {algo_time:.1f} ms", (30, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Frame Rate: {fps:.1f} FPS", (30, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Show window
        cv2.imshow("Edge-Based Contrast Enhancement - Live Demo", display_frame)
        
        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            use_custom = not use_custom
            print(f"Switched processing engine to: {'Custom CLAHE' if use_custom else 'OpenCV CLAHE'}")
            
    if camera_active:
        cap.release()
    if not test_mode:
        cv2.destroyAllWindows()
    print("Live demo closed successfully.")
    print("="*60)

if __name__ == "__main__":
    import sys
    test_mode = len(sys.argv) > 1 and sys.argv[1] == '--test'
    run_live_processing(test_mode)
