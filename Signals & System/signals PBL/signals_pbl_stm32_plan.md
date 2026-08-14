# STM32F411 Black Pill Image Processing Project Plan

This document outlines the design, implementation, and verification steps for migrating the CLAHE (Contrast Limited Adaptive Histogram Equalization) image enhancement project to the STM32F411CEU6 Black Pill microcontroller.

---

## 1. Hardware Interface & Flashing Options

There are two ways to flash and debug the Black Pill board:

### Option A: Direct USB-C Flashing (Built-in DFU Bootloader)
The STM32F411 has a ROM-based system bootloader that allows flashing directly over USB without a hardware debugger.

1. **Hardware Connection:** Plug a USB-C cable from your laptop into the Black Pill.
2. **Driver Configuration (Windows only - Critical):**
   * If STM32CubeProgrammer does not see your device in DFU mode, download **Zadig** (zadig.akeo.ie).
   * Put the board in DFU mode, select "List All Devices" in Zadig, find the STM32 Bootloader, and change its driver to **WinUSB**.
3. **Enter DFU Mode:**
   * Press and hold the **BOOT0** button.
   * Press and release the **NRST** (Reset) button.
   * Release the **BOOT0** button.
4. **Flash Firmware:** Open **STM32CubeProgrammer**, select **USB**, click **Connect**, load your `.bin` file, and flash.

### Option B: Hardware Debugger (ST-Link V2) - Recommended for Debugging
If you obtain an ST-Link V2 debugger, you can flash and debug directly inside STM32CubeIDE.

* Connect the 4 pins at the bottom of the Black Pill to the ST-Link:
  * **3V3** -> **3.3V** on ST-Link
  * **GND** -> **GND** on ST-Link
  * **SWDIO** -> **SWDIO** on ST-Link
  * **SWCLK** -> **SWCLK** on ST-Link
* Press **F11** in STM32CubeIDE to build, flash, and start a live debugging session (breakpoints, variable watch, etc.).

---

## 2. STM32 Firmware Design (C Code)

### Clock Configuration (USB Compatibility)
The STM32F411 requires a precise 48 MHz clock for USB operations. The Black Pill has a **25 MHz Crystal Oscillator (HSE)**.
* **SYSCLK:** Configure to **96 MHz** (using PLL).
* **USB OTG FS Clock:** Configure to exactly **48 MHz**.

### Robust USB CDC Serial State Machine
To prevent packet drop over USB Virtual COM Port (VCP), we implement a state-machine parser in `usbd_cdc_if.c`:

```c
typedef enum {
    STATE_WAIT_HEADER,
    STATE_RECEIVE_DATA,
    STATE_RECEIVE_CHECKSUM
} RX_State_t;

#define IMG_SIZE 16384 // 128x128 pixels
uint8_t image_in[IMG_SIZE];
uint8_t image_out[IMG_SIZE];
volatile uint8_t data_ready = 0;

static int rx_index = 0;
static uint32_t running_checksum = 0;
static RX_State_t rx_state = STATE_WAIT_HEADER;
static uint8_t header_buf[4] = {0};

// Called automatically by USB interrupt when bytes arrive
static int8_t CDC_Receive_FS(uint8_t* Buf, uint32_t *Len) {
    for (uint32_t i = 0; i < *Len; i++) {
        uint8_t byte = Buf[i];
        
        switch (rx_state) {
            case STATE_WAIT_HEADER:
                header_buf[0] = header_buf[1];
                header_buf[1] = header_buf[2];
                header_buf[2] = header_buf[3];
                header_buf[3] = byte;
                
                if (header_buf[0] == 0xAA && header_buf[1] == 0xBB && 
                    header_buf[2] == 0xCC && header_buf[3] == 0xDD) {
                    rx_state = STATE_RECEIVE_DATA;
                    rx_index = 0;
                    running_checksum = 0;
                }
                break;
                
            case STATE_RECEIVE_DATA:
                image_in[rx_index++] = byte;
                running_checksum += byte;
                if (rx_index >= IMG_SIZE) {
                    rx_state = STATE_RECEIVE_CHECKSUM;
                }
                break;
                
            case STATE_RECEIVE_CHECKSUM:
                uint8_t expected_checksum = (uint8_t)(running_checksum % 256);
                if (byte == expected_checksum) {
                    data_ready = 1; // Signal main loop to process
                } else {
                    // Checksum error - reset and wait for next header
                }
                rx_state = STATE_WAIT_HEADER;
                break;
        }
    }
    return (USBD_OK);
}
```

### Main Application Loop (`main.c`)
The main loop waits for `data_ready`, runs the CLAHE algorithm, transmits the result back, and resets the flag:
```c
while (1) {
    if (data_ready) {
        HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_RESET); // Turn ON LED (PC13 is active low)
        
        // Execute CLAHE Image Enhancement
        run_clahe(image_in, image_out, 128, 128);
        
        // Send processed data back to laptop
        CDC_Transmit_FS(image_out, IMG_SIZE);
        
        HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_SET); // Turn OFF LED
        data_ready = 0;
    }
}
```

---

## 3. Laptop Application Design (Python)

The Python script uses the `pyserial` library to send the grayscale data to the STM32 and receive the processed output.

```python
import serial
import time
import numpy as np

def process_on_stm32(image_gray, port='COM3'):
    # Ensure image is exactly 128x128
    img_resized = cv2.resize(image_gray, (128, 128))
    raw_bytes = img_resized.tobytes()
    
    # Calculate checksum
    checksum = sum(raw_bytes) % 256
    
    # Open Serial connection
    ser = serial.Serial(port, baudrate=115200, timeout=2.0)
    
    # Send packet: Header + Data + Checksum
    header = bytes([0xAA, 0xBB, 0xCC, 0xDD])
    ser.write(header + raw_bytes + bytes([checksum]))
    
    # Read response (16384 bytes)
    response = ser.read(128 * 128)
    ser.close()
    
    if len(response) == 128 * 128:
        # Reconstruct enhanced image
        enhanced_img = np.frombuffer(response, dtype=np.uint8).reshape((128, 128))
        return enhanced_img
    else:
        raise TimeoutError("STM32 did not respond in time or data was corrupted.")
```

---

## 4. CLAHE Math Implementation on STM32 (C Code)

The CLAHE logic on the microcontroller follows these steps:
1. **Tiling:** Subdivide the $128 \times 128$ image buffer into an $8 \times 8$ grid of tiles (each tile contains $16 \times 16 = 256$ pixels).
2. **Local Histogram:** Build a 256-bin histogram for each tile.
3. **Clip Limit & Redistribution:** Clip bin values exceeding the user-defined threshold and distribute the clipped count equally across all bins.
4. **Cumulative Distribution Function (CDF):** Calculate the running sum of the histogram to form the local mapping function.
5. **Bilinear Interpolation:** Iterate through each pixel in the image and interpolate mappings from the 4 surrounding tiles to eliminate block boundaries.

---

## 5. Verification Plan

1. **Verification Stage 1 (Serial Loopback):** Configure the STM32 to simply copy `image_in` to `image_out` immediately. Verify that the Python script receives the identical image file back over USB.
2. **Verification Stage 2 (Algorithm Comparison):** Process the same image on both the Python prototype and the STM32. Verify that the outputs are numerically identical (MSE approx 0).
3. **Verification Stage 3 (Real-Time Visual Validation):** Integrate the `process_on_stm32` function into your Tkinter GUI comparison dashboard. Adjust parameters on the GUI sliders, transmit to the STM32, and observe real-time screen updates.
