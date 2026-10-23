import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
WINDOW_SIZE = 50
NUM_CLASSES = 4
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# ==========================================
# 1. LOAD AND WINDOW THE DATA
# ==========================================
class IMUDataset(Dataset):
    def __init__(self):
        self.windows = []
        self.labels = []
        
        # Load all 4 CSV files
        for label in range(4):
            try:
                df = pd.read_csv(f"data_class_{label}.csv", header=None)
                data = df.values # Shape: (Total_Samples, 7)
                
                # Slice into windows of 50 (with 50% overlap for more data)
                for i in range(0, len(data) - WINDOW_SIZE, WINDOW_SIZE // 2):
                    window = data[i : i + WINDOW_SIZE]
                    self.windows.append(window)
                    self.labels.append(label)
            except FileNotFoundError:
                print(f"⚠️ Warning: data_class_{label}.csv not found. Skipping.")

        self.windows = np.array(self.windows) # Shape: (Num_Windows, 50, 7)
        self.labels = np.array(self.labels)
        print(f"✅ Loaded {len(self.windows)} total windows.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # PyTorch expects shape (Channels, Length) -> (7, 50)
        x = torch.tensor(self.windows[idx], dtype=torch.float32).T
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ==========================================
# 2. DEFINE THE MODEL (Must match real-time!)
# ==========================================
class IMU_CNN(nn.Module):
    def __init__(self, num_features=7, num_classes=4):
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

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train():
    dataset = IMUDataset()
    if len(dataset) == 0:
        print("❌ No data found. Run record_data.py first.")
        return

    # Split into 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = IMU_CNN(num_features=7, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("🚀 Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Val Accuracy: {val_acc:.2f}%")

    # Save the model
    torch.save(model.state_dict(), 'parkinsons_model.pth')
    print("💾 Model saved as 'parkinsons_model.pth'")

if __name__ == "__main__":
    train()