import torch, torch.nn as nn, torch.optim as optim, pandas as pd, numpy as np
import os
from torch.utils.data import Dataset, DataLoader

CSV_FILE = 'direction_dataset.csv'

if not os.path.exists(CSV_FILE):
    print(f"❌ ERROR: '{CSV_FILE}' not found. Run record_direction.py first.")
    exit()

class DirectionDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, header=None)
        self.strokes, self.labels = [], []
        
        self.classes = sorted(df[0].unique())
        self.class_to_idx = {l: i for i, l in enumerate(self.classes)}
        
        print(f"✅ Found data for {len(self.classes)} directions: {self.classes}")
        stroke_counts = df.groupby(0)[1].nunique()
        print(f"📊 Swipes per direction:\n{stroke_counts.to_string()}\n")

        for (label, stroke_id), group in df.groupby([0, 1]):
            data = group.iloc[:, 2:].values
            
            # Directions are faster, so we can truncate/pad to 100 samples (1 second)
            if len(data) > 100: data = data[:100]
            else: data = np.pad(data, ((0, 100 - len(data)), (0, 0)), mode='constant')
            
            self.strokes.append(torch.tensor(data, dtype=torch.float32).T)
            self.labels.append(self.class_to_idx[label])

    def __len__(self): return len(self.strokes)
    def __getitem__(self, idx): return self.strokes[idx], torch.tensor(self.labels[idx], dtype=torch.long)

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

dataset = DirectionDataset(CSV_FILE)
if len(dataset) == 0:
    print("❌ No strokes found.")
    exit()

batch_size = 1 if len(dataset) < 10 else 8
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

model = DirectionCNN(num_classes=len(dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005) 

print("🚀 Training...")
epochs = 100 if len(dataset) < 10 else 30
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

torch.save({'model_state': model.state_dict(), 'classes': dataset.classes}, 'direction_model.pth')
print(f"💾 Model saved as 'direction_model.pth'!")