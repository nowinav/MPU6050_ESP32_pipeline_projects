import serial, serial.tools.list_ports
import numpy as np

def find_esp32_port():
    for port in serial.tools.list_ports.comports():
        if "USB" in port.device or "ACM" in port.device: return port.device
    return None

SERIAL_PORT = find_esp32_port()
if not SERIAL_PORT:
    print("❌ ESP32 not found. Close Arduino IDE.")
    exit()

ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
print(f"🎧 Listening on {SERIAL_PORT}... Swipe the sensor!")

while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if "->" in line: line = line.split("->")[-1].strip()
        try:
            parts = line.split(',')
            if len(parts) == 7:
                # Extract raw Accel X, Y, Z
                ax, ay, az = float(parts[0]), float(parts[1]), float(parts[2])
                
                # Calculate Variance (how much it's moving)
                variance = np.var([ax, ay, az])
                
                # Create a simple visual bar for the variance
                bar = "█" * int(variance * 100)
                
                # Print live data
                print(f"Accel: X={ax:>6.2f} Y={ay:>6.2f} Z={az:>6.2f} | Variance: {variance:.4f} | {bar}")
        except ValueError:
            pass