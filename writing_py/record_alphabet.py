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

b, a = signal.butter(2, 15.0 / 50.0, btype='low', analog=False)
zi = np.zeros((max(len(a), len(b)) - 1, 7))

letter = input("Enter the letter to record (e.g., A): ").upper()
target_strokes = 3  # <--- REDUCED TO 3 STROKES FOR QUICK TESTING
strokes_captured = 0

ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
print(f"Recording '{letter}' (3 strokes). Write in the air and PAUSE between strokes.")

current_stroke = []
is_pen_down = False
still_frames = 0

with open('airwrite_dataset.csv', mode='a', newline='') as file:
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
                    
                    variance = np.var([sample[0], sample[1], sample[2]])
                    
                    if variance > 0.05:
                        is_pen_down = True
                        still_frames = 0
                        current_stroke.append(list(sample[:6]))
                    else:
                        if is_pen_down:
                            still_frames += 1
                            if still_frames > 10:
                                is_pen_down = False
                                if len(current_stroke) > 15:
                                    for s in current_stroke:
                                        writer.writerow([letter, strokes_captured] + s)
                                    strokes_captured += 1
                                    print(f"✅ Captured {letter}: stroke {strokes_captured}/{target_strokes}")
                                current_stroke = []
            except ValueError:
                pass
print("Done!")