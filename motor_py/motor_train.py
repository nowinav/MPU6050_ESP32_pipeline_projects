import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Config
CSV_FILE = 'motor_vibration_dataset.csv'
WINDOW_SIZE = 100  # 1 second of data assuming 100Hz sample rate

class MotorDataset(Dataset):
    def __init__(self, csv_file):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Could not find {csv_file}. Did you run the recording script?")
            
        df = pd.read_csv(csv_file, header=None)
        self.windows = []
        self.labels = []
        
        # Slice the continuous data into overlapping windows
        for label in df[0].unique():
            # Grab Ax, Ay, Az, Gx, Gy, Gz (ignoring the label column)
            class_data = df[df[0] == label].iloc[:, 1:].values
            
            # 50% overlap so we get more training data
            for i in range(0, len(class_data) - WINDOW_SIZE, WINDOW_SIZE // 2):
                window = class_data[i : i + WINDOW_SIZE]
                
                # PyTorch Conv1d expects shape (Channels, Length)
                self.windows.append(torch.tensor(window, dtype=torch.float32).T)
                self.labels.append(int(label))
                
        self.windows = torch.stack(self.windows)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        print(f"Loaded {len(self.windows)} windows across {len(df[0].unique())} classes.")
        print("Class distribution:")
        print(df[0].value_counts().to_string())
        print("-" * 30)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]

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
        
        # Flatten before the fully connected layers
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    try:
        dataset = MotorDataset(CSV_FILE)
    except FileNotFoundError as e:
        print(e)
        exit()

    # Batch size 32 is a decent default
    loader = DataLoader(dataset, batch_size=32, shuffle=True) 

    model = MotorCNN(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    print("Starting training...")
    epochs = 30
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for inputs, labels in loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # Save the trained weights
    torch.save(model.state_dict(), 'motor_model.pth')
    print("Training complete. Model saved to 'motor_model.pth'.")
