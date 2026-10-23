import torch, torch.nn as nn, torch.optim as optim, pandas as pd, numpy as np
import os
from torch.utils.data import Dataset, DataLoader

CSV_FILE = 'airwrite_dataset.csv'

# 1. Check if data exists
if not os.path.exists(CSV_FILE):
    print(f"❌ ERROR: '{CSV_FILE}' not found. Please run record_alphabet.py first to record some letters.")
    exit()

class AirWriteDataset(Dataset):
    def __init__(self, csv_file):
        # Read the CSV. No headers, so we use header=None
        df = pd.read_csv(csv_file, header=None)
        self.strokes, self.labels = [], []
        
        # Automatically find whatever letters are in the CSV
        self.letters = sorted(df[0].unique())
        self.letter_to_idx = {l: i for i, l in enumerate(self.letters)}
        
        print(f"✅ Found data for {len(self.letters)} letters: {self.letters}")
        
        # Count how many strokes we have per letter
        stroke_counts = df.groupby(0)[1].nunique()
        print(f"📊 Strokes per letter:\n{stroke_counts.to_string()}\n")

        # Group by Letter (col 0) and Stroke_ID (col 1)
        for (letter, stroke_id), group in df.groupby([0, 1]):
            data = group.iloc[:, 2:].values # Extract Ax, Ay, Az, Gx, Gy, Gz
            
            # Pad or truncate to exactly 200 samples
            if len(data) > 200: 
                data = data[:200]
            else: 
                data = np.pad(data, ((0, 200 - len(data)), (0, 0)), mode='constant')
            
            self.strokes.append(torch.tensor(data, dtype=torch.float32).T)
            self.labels.append(self.letter_to_idx[letter])

    def __len__(self): 
        return len(self.strokes)
        
    def __getitem__(self, idx): 
        return self.strokes[idx], torch.tensor(self.labels[idx], dtype=torch.long)

class AirWriteCNN(nn.Module):
    def __init__(self, num_classes):
        super(AirWriteCNN, self).__init__()
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

# 2. Load Data
dataset = AirWriteDataset(CSV_FILE)

if len(dataset) == 0:
    print("❌ No strokes found in the dataset.")
    exit()

# Use batch size 1 if the dataset is tiny, else 8
batch_size = 1 if len(dataset) < 10 else 8
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

# 3. Initialize Model
model = AirWriteCNN(num_classes=len(dataset.letters))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005) 

# 4. Train
print("🚀 Training...")
epochs = 100 if len(dataset) < 10 else 30 # Train longer if dataset is tiny
for epoch in range(epochs):
    epoch_loss = 0
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    if (epoch+1) % 20 == 0 or (epoch+1) == epochs:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f}")

# 5. Save
torch.save({'model_state': model.state_dict(), 'letters': dataset.letters}, 'airwrite_model.pth')
print(f"💾 Model successfully trained on {len(dataset.letters)} letters and saved as 'airwrite_model.pth'!")