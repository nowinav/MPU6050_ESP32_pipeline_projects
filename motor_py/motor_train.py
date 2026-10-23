import torch, torch.nn as nn, torch.optim as optim, pandas as pd, numpy as np
import os
from torch.utils.data import Dataset, DataLoader

CSV_FILE = 'motor_vibration_dataset.csv'
WINDOW_SIZE = 100 # 1 second at 100Hz

if not os.path.exists(CSV_FILE):
    print(f"❌ ERROR: '{CSV_FILE}' not found. Run motor_record.py first.")
    exit()

class MotorDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, header=None)
        self.windows, self.labels = [], []
        
        # Slice continuous data into overlapping windows
        for label in df[0].unique():
            class_data = df[df[0] == label].iloc[:, 1:].values # Get Ax, Ay, Az, Gx, Gy, Gz
            
            # Create windows with 50% overlap
            for i in range(0, len(class_data) - WINDOW_SIZE, WINDOW_SIZE // 2):
                window = class_data[i : i + WINDOW_SIZE]
                self.windows.append(torch.tensor(window, dtype=torch.float32).T)
                self.labels.append(int(label))
                
        self.windows = torch.stack(self.windows)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        print(f"✅ Loaded {len(self.windows)} windows across {len(df[0].unique())} classes.")
        print(f"📊 Class distribution:\n{df[0].value_counts().to_string()}\n")

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx): return self.windows[idx], self.labels[idx]

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

dataset = MotorDataset(CSV_FILE)
loader = DataLoader(dataset, batch_size=32, shuffle=True) 

model = MotorCNN(num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) 

print("🚀 Training Motor Predictive Model...")
for epoch in range(30):
    epoch_loss = 0
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/30 | Loss: {epoch_loss/len(loader):.4f}")

torch.save(model.state_dict(), 'motor_model.pth')
print("💾 Model saved as 'motor_model.pth'!")