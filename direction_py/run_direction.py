import serial, serial.tools.list_ports, numpy as np, torch, torch.nn as nn
from scipy import signal

# --- 1. Load Model ---
class DirectionCNN(nn.Module):
    def __init__(self, num_classes):
        super(DirectionCNN, self).__init__()
        self.conv1 = nn.Conv1d(6, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
    def forward(self, x):
        x = self.relu(self.conv1(x)); x = self.pool(x)
        x = self.relu(self.conv2(x)); x = self.global_pool(x)
        x = x.flatten(1); x = self.relu(self.fc1(x)); x = self.fc2(x)
        return x

checkpoint = torch.load('direction_model.pth')
CLASSES = checkpoint['classes']
model = DirectionCNN(num_classes=len(CLASSES))
model.load_state_dict(checkpoint['model_state'])
model.eval()
print(f"🧠 Model loaded. Ready to detect: {CLASSES}")

# --- 2. Setup ---
def find_esp32_port():
    for port in serial.tools.list_ports.comports():
        if "USB" in port.device or "ACM" in port.device: return port.device

SERIAL_PORT = find_esp32_port()
b, a = signal.butter(2, 15.0 / 50.0, btype='low', analog=False)
zi = np.zeros((max(len(a), len(b)) - 1, 7))

ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
print(f"🎧 Listening on {SERIAL_PORT}. Swipe the sensor!")

current_stroke = []
is_moving = False
still_frames = 0

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
                
                variance = np.var([sample[0], sample[1], sample[2]])
                
                if variance > 0.1:
                    is_moving = True
                    still_frames = 0
                    current_stroke.append(list(sample[:6]))
                else:
                    if is_moving:
                        still_frames += 1
                        if still_frames > 10:
                            is_moving = False
                            if len(current_stroke) > 10:
                                if len(current_stroke) > 100: data = current_stroke[:100]
                                else: data = np.pad(current_stroke, ((0, 100 - len(current_stroke)), (0, 0)), mode='constant')
                                
                                tensor = torch.tensor(data, dtype=torch.float32).T.unsqueeze(0)
                                with torch.no_grad():
                                    out = model(tensor)
                                    prob = torch.softmax(out, dim=1)
                                    conf, idx = torch.max(prob, 1)
                                    
                                    if conf.item() > 0.6:
                                        print(f"🧭 DETECTED: {CLASSES[idx.item()]} ({conf.item()*100:.1f}%)")
                                    else:
                                        print("🤷 Unrecognized movement.")
                            current_stroke = []
        except ValueError:
            pass