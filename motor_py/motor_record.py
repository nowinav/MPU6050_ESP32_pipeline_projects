import serial, serial.tools.list_ports, numpy as np, csv, time
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# ==========================================
# 1. SETUP
# ==========================================
def find_esp32_port():
    for port in serial.tools.list_ports.comports():
        if "USB" in port.device or "ACM" in port.device: return port.device
    return None

SERIAL_PORT = find_esp32_port()
if not SERIAL_PORT:
    print("❌ ESP32 not found. Close Arduino IDE.")
    exit()

print("\n--- MOTOR VIBRATION RECORDING ---")
print("Classes: 0 = Stable, 1 = Warning (Slight Rattle), 2 = Critical (Violent Shake)")
label = int(input("Enter the class label (0, 1, or 2): "))
duration = int(input("Enter recording duration in seconds (e.g., 20): "))

# DSP Filter (20Hz low pass to isolate motor vibration)
b, a = signal.butter(2, 20.0 / 50.0, btype='low', analog=False)
zi = np.zeros((max(len(a), len(b)) - 1, 7))

# Graph buffers
MAX_POINTS = 150
x_data, y_data, z_data = deque(maxlen=MAX_POINTS), deque(maxlen=MAX_POINTS), deque(maxlen=MAX_POINTS)

# Recording State
data_to_save = []
start_time = time.time()
is_recording = True

try:
    ser = serial.Serial(SERIAL_PORT, 115200, timeout=0.01)
    ser.flush()
except Exception as e:
    print(f"❌ Failed to open serial: {e}")
    exit()

# ==========================================
# 2. GRAPH SETUP
# ==========================================
fig, ax = plt.subplots(figsize=(10, 5))
fig.canvas.manager.set_window_title('Motor Live Vibration Monitor')
line_x, = ax.plot([], [], label='Accel X', color='red', alpha=0.7)
line_y, = ax.plot([], [], label='Accel Y', color='green', alpha=0.7)
line_z, = ax.plot([], [], label='Accel Z', color='blue', alpha=0.7)

ax.set_ylim(-4.0, 4.0) # Adjust based on motor vibration intensity
ax.set_xlim(0, MAX_POINTS)
ax.set_title(f"Recording Class {label}... Time left: {duration}s")
ax.set_ylabel("Acceleration (g)")
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.6)

# ==========================================
# 3. ANIMATION & RECORDING LOOP
# ==========================================
def update(frame):
    global zi, is_recording, start_time
    
    if not is_recording:
        return line_x, line_y, line_z
        
    # Check time
    time_left = duration - (time.time() - start_time)
    if time_left <= 0:
        is_recording = False
        print("\n✅ Recording complete. You can close the graph window.")
        ax.set_title("Recording Complete!")
        save_data()
        return line_x, line_y, line_z
        
    ax.set_title(f"Recording Class {label}... Time left: {int(time_left)}s")

    # Read Serial
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if "->" in line: line = line.split("->")[-1].strip()
            parts = line.split(',')
            
            if len(parts) == 7:
                raw = np.array([float(x) for x in parts])
                filtered, zi = signal.lfilter(b, a, raw.reshape(1, -1), axis=0, zi=zi)
                sample = filtered[0]
                
                # Append to graph
                x_data.append(sample[0])
                y_data.append(sample[1])
                z_data.append(sample[2])
                
                # Append to save buffer (Ax, Ay, Az, Gx, Gy, Gz)
                data_to_save.append(list(sample[:6]))
        except ValueError:
            pass

    line_x.set_data(range(len(x_data)), x_data)
    line_y.set_data(range(len(y_data)), y_data)
    line_z.set_data(range(len(z_data)), z_data)
    
    return line_x, line_y, line_z

def save_data():
    filename = 'motor_vibration_dataset.csv'
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for row in data_to_save:
            writer.writerow([label] + row)
    print(f"💾 Saved {len(data_to_save)} samples to {filename}")

ani = animation.FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)

print(f"\n👉 Motor is recording for {duration} seconds. DO NOT CLOSE THE GRAPH WINDOW YET.")
plt.show()

ser.close()