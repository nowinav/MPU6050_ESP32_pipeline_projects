import serial, serial.tools.list_ports, numpy as np, csv, time
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
def find_esp32_port():
    for port in serial.tools.list_ports.comports():
        if "USB" in port.device or "ACM" in port.device: return port.device
    return None

SERIAL_PORT = find_esp32_port()
if not SERIAL_PORT:
    print("❌ ESP32 not found. Close Arduino IDE and check USB.")
    exit()

direction = input("Enter the direction to record (e.g., LEFT): ").upper()
target_strokes = 5  # How many swipes you want to record

# DSP Filter (15Hz low pass)
b, a = signal.butter(2, 15.0 / 50.0, btype='low', analog=False)
zi = np.zeros((max(len(a), len(b)) - 1, 7))

# Data buffers for the live graph
MAX_POINTS = 150
x_data, y_data, z_data = deque(maxlen=MAX_POINTS), deque(maxlen=MAX_POINTS), deque(maxlen=MAX_POINTS)

# Recording state
current_stroke = []
is_moving = False
still_frames = 0
strokes_captured = 0

# ==========================================
# 2. SETUP MATPLOTLIB GRAPH
# ==========================================
fig, ax = plt.subplots(figsize=(10, 5))
fig.canvas.manager.set_window_title(f'Recording {direction}')
line_x, = ax.plot([], [], label='Accel X', color='red', alpha=0.7)
line_y, = ax.plot([], [], label='Accel Y', color='green', alpha=0.7)
line_z, = ax.plot([], [], label='Accel Z', color='blue', alpha=0.7)

ax.set_ylim(-2.0, 2.0) # Y-axis limits for acceleration (g)
ax.set_xlim(0, MAX_POINTS)
ax.set_title(f"Live Waveform - Recording '{direction}' ({strokes_captured}/{target_strokes})")
ax.set_ylabel("Acceleration (g)")
ax.set_xlabel("Samples")
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.6)

# Open serial connection
try:
    ser = serial.Serial(SERIAL_PORT, 115200, timeout=0.01)
    ser.flush()
except Exception as e:
    print(f"❌ Failed to open serial: {e}")
    exit()

# Open CSV file globally so we can append to it in the animation loop
file = open('direction_dataset.csv', mode='a', newline='')
writer = csv.writer(file)

# ==========================================
# 3. THE LIVE ANIMATION LOOP
# ==========================================
def update(frame):
    global is_moving, still_frames, current_stroke, strokes_captured, zi
    
    if strokes_captured >= target_strokes:
        return line_x, line_y, line_z

    # Read all available serial data
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if "->" in line: line = line.split("->")[-1].strip()
            parts = line.split(',')
            
            if len(parts) == 7:
                raw = np.array([float(x) for x in parts])
                # Apply DSP Filter
                filtered, zi = signal.lfilter(b, a, raw.reshape(1, -1), axis=0, zi=zi)
                sample = filtered[0]
                
                # Add to graph buffers
                x_data.append(sample[0])
                y_data.append(sample[1])
                z_data.append(sample[2])
                
                # --- RECORDING LOGIC ---
                variance = np.var([sample[0], sample[1], sample[2]])
                
                if variance > 0.1: # Movement threshold
                    is_moving = True
                    still_frames = 0
                    current_stroke.append(list(sample[:6]))
                else:
                    if is_moving:
                        still_frames += 1
                        if still_frames > 10:
                            is_moving = False
                            if len(current_stroke) > 10:
                                for s in current_stroke:
                                    writer.writerow([direction, strokes_captured] + s)
                                strokes_captured += 1
                                ax.set_title(f"Live Waveform - Recording '{direction}' ({strokes_captured}/{target_strokes})")
                                print(f"✅ Saved swipe {strokes_captured}/{target_strokes}")
                            current_stroke = []
        except ValueError:
            pass

    # Update the graph lines
    line_x.set_data(range(len(x_data)), x_data)
    line_y.set_data(range(len(y_data)), y_data)
    line_z.set_data(range(len(z_data)), z_data)
    
    return line_x, line_y, line_z

# Start the animation
ani = animation.FuncAnimation(fig, update, interval=20, blit=True, cache_frame_data=False)

print(f"\n👉 Graph opened. Swipe {direction.upper()} {target_strokes} times.")
print("👉 The script will auto-save the strokes and close when done.")
print("👉 Close the graph window manually to stop early.\n")

try:
    plt.show()
except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
finally:
    ser.close()
    file.close()
    print("🎉 Recording complete. Data saved to direction_dataset.csv.")