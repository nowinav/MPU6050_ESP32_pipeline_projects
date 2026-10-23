import serial, serial.tools.list_ports, numpy as np, csv
from scipy import signal

def find_esp32_port():
    for port in serial.tools.list_ports.comports():
        if "USB" in port.device or "ACM" in port.device: return port.device
    return None

SERIAL_PORT = find_esp32_port()
if not SERIAL_PORT:
    print("❌ ESP32 not found. Close Arduino IDE and check USB.")
    exit()

# 15Hz Low-pass filter to remove high-frequency noise
b, a = signal.butter(2, 15.0 / 50.0, btype='low', analog=False)
zi = np.zeros((max(len(a), len(b)) - 1, 7))

print("\n--- DIRECTION RECORDING ---")
print("Options: LEFT, RIGHT, UP, DOWN, FORWARD, BACKWARD")
direction = input("Enter the direction to record: ").upper()
target_strokes = 10  # 10 swipes per direction is enough for simple movements
strokes_captured = 0

ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
print(f"Recording '{direction}'. Swipe firmly and PAUSE between swipes.")

current_stroke = []
is_moving = False
still_frames = 0

with open('direction_dataset.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    while strokes_captured < target_strokes:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if "->" in line: line = line.split("->")[-1].strip()
            try:
                parts = line.split(',')
                if len(parts) == 7:
                    raw = np.array([float(x) for x in parts])
                    filtered, zi = signal.lfilter(b, a, raw.reshape(1, -1), axis=0, zi=zi)
                    sample = filtered[0]
                    
                    # 0.1 variance is a good threshold for a deliberate swipe
                    variance = np.var([sample[0], sample[1], sample[2]])
                    
                    if variance > 0.1: 
                        is_moving = True
                        still_frames = 0
                        current_stroke.append(list(sample[:6]))
                    else:
                        if is_moving:
                            still_frames += 1
                            if still_frames > 10: # Pen up
                                is_moving = False
                                if len(current_stroke) > 10: # Must have at least some movement
                                    for s in current_stroke:
                                        writer.writerow([direction, strokes_captured] + s)
                                    strokes_captured += 1
                                    print(f"✅ Captured {direction}: {strokes_captured}/{target_strokes}")
                                current_stroke = []
            except ValueError:
                pass
print("Done!")