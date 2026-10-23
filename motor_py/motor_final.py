import serial, serial.tools.list_ports, numpy as np, torch, torch.nn as nn
from scipy import signal
from collections import deque

# --- 1. Load Model ---
class MotorCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(MotorCNN, self).__init__()
        self.conv1 = nn.Conv1d(6, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
    def forward(self, x):
        x = self.relu(self.conv1(x)); x = self.pool(x)
        x = self.relu(self.conv2(x)); x = self.global_pool(x)
        x = x.flatten(1); x = self.relu(self.fc1(x)); x = self.fc2(x)
        return x

model = MotorCNN()
model.load_state_dict(torch.load('motor_model.pth'))
model.eval()
CLASSES = ["🟢 STABLE", "🟡 WARNING (PRE-FAULT)", "🔴 CRITICAL"]

# --- 2. Setup Serial & DSP ---
def find_esp32_port():
    for port in serial.tools.list_ports.comports():
        if "USB" in port.device or "ACM" in port.device: return port.device

SERIAL_PORT = find_esp32_port()
b, a = signal.butter(2, 20.0 / 50.0, btype='low', analog=False)
zi = np.zeros((max(len(a), len(b)) - 1, 7))

ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
print(f"🎧 Listening on {SERIAL_PORT}. Start the motor!")

# Sliding window buffer
data_buffer = deque(maxlen=100)

while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if "->" in line: line = line.split("->")[-1].strip()
        try:
            parts = line.split(',')
            if len(parts) == 7:
                raw = np.array([float(x) for x in parts])
                filtered, zi = signal.lfilter(b, a, raw.reshape(1, -1), axis=0, zi=zi)
                sample = filtered[0]
                
                data_buffer.append(list(sample[:6]))
                
                # Every time the buffer fills up (100 samples = 1 second)
                if len(data_buffer) == 100:
                    tensor = torch.tensor(np.array(data_buffer), dtype=torch.float32).T.unsqueeze(0)
                    
                    with torch.no_grad():
                        out = model(tensor)
                        prob = torch.softmax(out, dim=1)
                        conf, idx = torch.max(prob, 1)
                        
                        # Print status
                        status = CLASSES[idx.item()]
                        print(f"Status: {status} (Confidence: {conf.item()*100:.1f}%)")
                        
                    # Clear half the buffer so we get a 50% overlapping window next time
                    for _ in range(50):
                        data_buffer.popleft()
        except ValueError:
            pass