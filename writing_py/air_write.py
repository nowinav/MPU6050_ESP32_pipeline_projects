import serial
import serial.tools.list_ports
import numpy as np
from scipy import signal
from collections import deque
import torch
import torch.nn as nn
import time

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
def find_esp32_port():
    for port in serial.tools.list_ports.comports():
        if "USB" in port.device or "ACM" in port.device:
            return port.device
    return None

SERIAL_PORT = find_esp32_port()
BAUD_RATE = 115200
SAMPLE_RATE = 100

# Segmentation (Pen Up/Down) Thresholds
STILLNESS_THRESHOLD = 0.15  # Variance threshold to detect if pen is moving
MIN_STROKE_LENGTH = 20      # Minimum samples to consider a valid letter (0.2s)
STILLNESS_FRAMES = 10       # Frames of stillness to end a stroke (0.1s)

CLASSES = ["A", "B", "C", "D", "E"] # Add all letters you train on

# ==========================================
# 2. NEURAL NETWORK DEFINITION
# ==========================================
class AirWriteCNN(nn.Module):
    def __init__(self, num_features=6, num_classes=26): # 6 IMU axes (ignore temp)
        super(AirWriteCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # Adaptive pooling means it can handle variable stroke lengths!
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x shape: (batch, 6, sequence_length)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.global_pool(x) # Reduces to (batch, 64, 1)
        x = x.flatten(1)        # Flattens to (batch, 64)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = AirWriteCNN()
# Uncomment to load trained weights:
# model.load_state_dict(torch.load('airwrite_model.pth'))
model.eval()

# ==========================================
# 3. DSP FILTERS
# ==========================================
CUTOFF_FREQ = 12.0  # Human writing is rarely faster than 12 Hz
nyquist = 0.5 * SAMPLE_RATE
b, a = signal.butter(2, CUTOFF_FREQ / nyquist, btype='low', analog=False)
zi = np.zeros((max(len(a), len(b)) - 1, 7)) 

# ==========================================
# 4. DATA PIPELINE & SEGMENTATION
# ==========================================
current_stroke = []
is_pen_down = False
still_frames_count = 0
rolling_buffer = deque(maxlen=5) # For calculating variance

def classify_stroke(stroke_data):
    """Passes the segmented letter into the NN."""
    if len(stroke_data) < MIN_STROKE_LENGTH:
        return # Too short, ignore
    
    # Format: (batch=1, channels=6, length)
    # We exclude temperature (index 6) for the NN
    tensor_data = torch.tensor(stroke_data, dtype=torch.float32).T.unsqueeze(0)[:, :6, :]
    
    with torch.no_grad():
        output = model(tensor_data)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        if confidence.item() > 0.5: # Only print if confident
            print(f"✍️  LETTER DETECTED: '{CLASSES[predicted_idx.item()]}' (Confidence: {confidence.item()*100:.1f}%)")
        else:
            print("🤷 Gesture too ambiguous.")

def process_stream():
    global is_pen_down, current_stroke, still_frames_count, zi
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"🎧 Listening on {SERIAL_PORT}. Start writing in the air...")

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if "->" in line:
                line = line.split("->")[-1].strip()
            
            try:
                parts = line.split(',')
                if len(parts) == 7:
                    raw_data = np.array([float(x) for x in parts])
                    
                    # 1. Low-Pass Filter
                    filtered, zi = signal.lfilter(b, a, raw_data.reshape(1, -1), axis=0, zi=zi)
                    sample = filtered[0]
                    
                    # 2. Pen-Up / Pen-Down Logic
                    # Calculate the magnitude of acceleration (Ax, Ay, Az)
                    acc_mag = np.linalg.norm(sample[:3])
                    rolling_buffer.append(acc_mag)
                    
                    # Calculate variance to see if the pen is moving
                    variance = np.var(rolling_buffer)
                    
                    if variance > STILLNESS_THRESHOLD:
                        # PEN DOWN: User is actively writing
                        is_pen_down = True
                        still_frames_count = 0
                        current_stroke.append(sample[:6]) # Save Ax, Ay, Az, Gx, Gy, Gz
                    else:
                        if is_pen_down:
                            # PEN UP: User stopped writing
                            still_frames_count += 1
                            
                            # If they've been still for long enough, the letter is finished
                            if still_frames_count > STILLNESS_FRAMES:
                                is_pen_down = False
                                if len(current_stroke) > MIN_STROKE_LENGTH:
                                    classify_stroke(current_stroke)
                                current_stroke = [] # Reset for next letter
            except ValueError:
                pass

if __name__ == "__main__":
    if SERIAL_PORT:
        process_stream()
    else:
        print("❌ ESP32 not found.")