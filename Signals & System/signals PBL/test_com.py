import serial
import time

# Configure the port (Use COM10 as detected by your Device Manager)
PORT = 'COM10'
BAUDRATE = 115200

try:
    print(f"Opening port {PORT}...")
    ser = serial.Serial(PORT, baudrate=BAUDRATE, timeout=2.0)
    time.sleep(1) # Wait for connection to stabilize
    
    test_message = b"Hello STM32 Black Pill!"
    print(f"Sending: {test_message.decode()}")
    
    # Write to STM32
    ser.write(test_message)
    
    # Read the response (should be the exact same message echoed back)
    response = ser.read(len(test_message))
    
    print(f"Received: {response.decode()}")
    
    if response == test_message:
        print("\nSUCCESS! The loopback test passed perfectly.")
        print("The USB connection is fast, stable, and bi-directional.")
    else:
        print("\nFAILED: Received data does not match sent data.")
        
    ser.close()

except Exception as e:
    print(f"\nError: {e}")
    print("Make sure the STM32 is plugged in, and no other program (like a serial monitor) is using COM10.")
