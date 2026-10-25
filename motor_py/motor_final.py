import serial
import serial.tools.list_ports
import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from collections import deque

# Define the same architecture used during training
class MotorCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(MotorCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.global_pool(x)
        
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

def find_port():
    # Just looking for the USB serial adapter
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if "USB" in p.device or "ACM" in p.device:
            return p.device
    return None

if __name__ == "__main__":
    # 1. Load the weights
    model = MotorCNN()
    model.load_state_dict(torch.load('motor_model.pth'))
    model.eval()
    
    class_names = ["STABLE", "WARNING (PRE-FAULT)", "CRITICAL"]

    # 2. Setup serial
    port = find_port()
    if not port:
        print("Could not find the ESP32 port. Check the cable!")
        exit(1)

    # Setup a lowpass filter (20Hz cutoff)
    b, a = signal.butter(2, 20.0 / 50.0, btype='low', analog=False)
    zi = np.zeros((max(len(a), len(b)) - 1, 7))

    try:
        ser = serial.Serial(port, 115200, timeout=1)
        print(f"Connected to {port}")
        print("Start the motor...")
    except Exception as e:
        print(f"Failed to connect to serial: {e}")
        exit(1)

    # Buffer to hold exactly 1 second of data (100 samples)
    data_buffer = deque(maxlen=100)

    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # The Arduino serial monitor prepends timestamps sometimes
                if "->" in line:
                    line = line.split("->")[-1].strip()
                    
                try:
                    parts = line.split(',')
                    if len(parts) == 7:
                        raw_data = np.array([float(x) for x in parts])
                        
                        # Filter the data
                        filtered_data, zi = signal.lfilter(b, a, raw_data.reshape(1, -1), axis=0, zi=zi)
                        clean_sample = filtered_data[0]
                        
                        # Only keep accel and gyro (ignore temp)
                        data_buffer.append(list(clean_sample[:6]))
                        
                        # Once we have 100 samples, run the prediction
                        if len(data_buffer) == 100:
                            # Prep tensor: (1, 6, 100)
                            input_tensor = torch.tensor(np.array(data_buffer), dtype=torch.float32).T.unsqueeze(0)
                            
                            with torch.no_grad():
                                output = model(input_tensor)
                                probabilities = torch.softmax(output, dim=1)
                                confidence, predicted_idx = torch.max(probabilities, 1)
                                
                                status = class_names[predicted_idx.item()]
                                conf_val = confidence.item() * 100
                                
                                print(f"Status: {status} | Conf: {conf_val:.1f}%")
                            
                            # Drop the oldest 50 samples to slide the window forward by 0.5s
                            for i in range(50):
                                data_buffer.popleft()
                                
                except ValueError:
                    # Probably just a weird empty line from the serial buffer, skip it
                    pass
                    
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ser.close()
