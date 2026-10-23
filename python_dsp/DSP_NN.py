import serial
import serial.tools.list_ports
import numpy as np
from scipy import signal
from collections import deque
import torch
import torch.nn as nn

# ==========================================
# 1. CONFIGURATION & AUTO-DETECT
# ==========================================
def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "USB" in port.device or "ACM" in port.device:
            return port.device
    return None

SERIAL_PORT = find_esp32_port()
BAUD_RATE = 115200
HARDWARE_SAMPLE_RATE = 100     # What the ESP32 is sending (Hz)

# DOWNSAMPLING CONFIGURATION
DOWNSAMPLE_FACTOR = 2          # Process every 2nd sample (100Hz -> 50Hz)
PROCESSING_RATE = HARDWARE_SAMPLE_RATE / DOWNSAMPLE_FACTOR 
WINDOW_SIZE = 50               # 50 samples at 50Hz = 1.0 second window

# 4 PARKINSON'S CLASSES
CLASS_NAMES = ["Healthy", "Resting Tremor", "Postural Tremor", "Kinetic Tremor"]

if SERIAL_PORT is None:
    print("❌ No serial device found! Check USB cable and connections.")
    exit()
print(f"✅ Found device at: {SERIAL_PORT}")

# ==========================================
# 2. NEURAL NETWORK (4 CLASSES)
# ==========================================
class IMU_CNN(nn.Module):
    def __init__(self, num_features=7, num_classes=4): # Changed to 4 classes
        super(IMU_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (WINDOW_SIZE // 2), 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = IMU_CNN(num_features=7, num_classes=4)

# 🚨 IMPORTANT: If you have trained your model, uncomment the line below to load the weights!
model.load_state_dict(torch.load('parkinsons_model.pth'))
# model.eval() 

# ==========================================
# 3. SIGNAL PROCESSING (DSP) - REAL-TIME FILTER
# ==========================================
# We filter BEFORE downsampling to prevent aliasing (Nyquist frequency)
# Cutoff at 15 Hz (preserves 3-12 Hz tremors perfectly)
CUTOFF_FREQ = 15.0
nyquist = 0.5 * HARDWARE_SAMPLE_RATE
normal_cutoff = CUTOFF_FREQ / nyquist
b, a = signal.butter(2, normal_cutoff, btype='low', analog=False)
zi = np.zeros((max(len(a), len(b)) - 1, 7)) 

# ==========================================
# 4. DATA PIPELINE
# ==========================================
data_buffer = deque(maxlen=WINDOW_SIZE)
sample_counter = 0

def parse_line(raw_line):
    try:
        if "->" in raw_line:
            raw_line = raw_line.split("->")[-1].strip()
        parts = raw_line.split(',')
        if len(parts) == 7:
            return np.array([float(val) for val in parts])
    except ValueError:
        pass 
    return None

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"🎧 Listening. Downsampling from {HARDWARE_SAMPLE_RATE}Hz to {PROCESSING_RATE}Hz...")
except Exception as e:
    print(f"❌ Error opening serial port: {e}")
    exit()

try:
    while True:
        if ser.in_waiting > 0:
            raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
            sensor_data = parse_line(raw_line)
            
            if sensor_data is not None:
                # 1. APPLY DSP FILTER
                filtered_data, zi = signal.lfilter(b, a, sensor_data.reshape(1, -1), axis=0, zi=zi)
                filtered_sample = filtered_data[0]
                
                # 2. DOWNSAMPLE (Only process every Nth sample)
                sample_counter += 1
                if sample_counter >= DOWNSAMPLE_FACTOR:
                    sample_counter = 0  # Reset counter
                    
                    data_buffer.append(filtered_sample)
                    
                    # 3. RUN NEURAL NETWORK (When buffer is full)
                    if len(data_buffer) == WINDOW_SIZE:
                        # Format for CNN: (batch, channels, length)
                        input_tensor = torch.tensor(np.array(data_buffer), dtype=torch.float32).T.unsqueeze(0)
                        
                        with torch.no_grad():
                            output = model(input_tensor)
                            probabilities = torch.softmax(output, dim=1)
                            
                            # --- THIS IS THE FINAL CLASSIFICATION ---
                            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                            confidence = probabilities[0][predicted_class_idx].item()
                            
                            # Map the index to the actual name
                            prediction_label = CLASS_NAMES[predicted_class_idx]
                        
                        # Print Results
                        print(f"Ax: {filtered_sample[0]:.3f} | "
                              f"Prediction: {prediction_label} "
                              f"({confidence*100:.1f}%)")

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
finally:
    ser.close()