import serial
import serial.tools.list_ports
import numpy as np
from scipy import signal
import csv
import time

def find_esp32_port():
    for port in serial.tools.list_ports.comports():
        if "USB" in port.device or "ACM" in port.device:
            return port.device
    return None

SERIAL_PORT = find_esp32_port()
BAUD_RATE = 115200
SAMPLE_RATE = 100
CUTOFF_FREQ = 15.0
nyquist = 0.5 * SAMPLE_RATE
b, a = signal.butter(2, CUTOFF_FREQ / nyquist, btype='low', analog=False)
zi = np.zeros((max(len(a), len(b)) - 1, 7)) 

print("\n--- DATA COLLECTION ---")
print("Classes: 0=Healthy, 1=Resting Tremor, 2=Postural Tremor, 3=Kinetic Tremor")
label = int(input("Enter the class label (0, 1, 2, or 3) you want to record: "))
duration = int(input("Enter recording duration in seconds (e.g., 30): "))

filename = f"data_class_{label}.csv"
print(f"Recording to {filename} for {duration} seconds...")

if SERIAL_PORT is None:
    print("❌ No serial device found.")
    exit()

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
ser.flush()

data_to_save = []
start_time = time.time()

try:
    while (time.time() - start_time) < duration:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if "->" in line:
                line = line.split("->")[-1].strip()
            
            try:
                parts = line.split(',')
                if len(parts) == 7:
                    raw_vals = np.array([float(x) for x in parts])
                    
                    # Apply DSP Filter
                    filtered, zi = signal.lfilter(b, a, raw_vals.reshape(1, -1), axis=0, zi=zi)
                    
                    # Save the 7 filtered features (Ax, Ay, Az, Rx, Ry, Rz, Temp)
                    data_to_save.append(filtered[0])
            except ValueError:
                continue

    print(f"✅ Captured {len(data_to_save)} samples.")
    
    # Write to CSV
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in data_to_save:
            writer.writerow(row)
    print(f"💾 Saved to {filename}")

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    ser.close()