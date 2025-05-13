import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

# --- Dataset ---
class MelSpectrogramDataset(Dataset):
    def __init__(self, img_dir, files, class_to_idx, transform=None):
        self.img_dir = img_dir
        self.files = files
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        label_name = img_name.split('_')[2]  # e.g., car_horn
        label = self.class_to_idx[label_name]
        if self.transform:
            img = self.transform(img)
        return img, label

# --- Model ---
from conv_encoder import ConvEncoder

class ConvEncoderClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = ConvEncoder(input_channels=3)
        self.classifier = nn.Linear(512, num_classes)  # d_model=512

    def forward(self, x):
        x = self.encoder(x)  # [B, seq_len, d_model]
        x = x.mean(dim=1)    # Global average pooling over sequence
        x = self.classifier(x)
        return x

# --- Prepare data ---
img_dir = "mel_spectrograms_clean"
all_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
random.shuffle(all_files)

# Extract all class names
class_names = sorted(list({f.split('_')[2] for f in all_files}))
class_to_idx = {cls: i for i, cls in enumerate(class_names)}

# Split 90% train, 10% test
split_idx = int(0.9 * len(all_files))
train_files = all_files[:split_idx]
test_files = all_files[split_idx:]

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure all images are the same size
    transforms.ToTensor(),
])

train_dataset = MelSpectrogramDataset(img_dir, train_files, class_to_idx, transform)
test_dataset = MelSpectrogramDataset(img_dir, test_files, class_to_idx, transform)

batch_size = 32
epochs = 20
lr = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# --- Training ---
model = ConvEncoderClassifier(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, epochs+1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total
    print(f"Epoch {epoch}: Train Loss: {running_loss/total:.4f}, Train Acc: {train_acc:.4f}")

    # --- Evaluation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total
    print(f"Epoch {epoch}: Test Acc: {test_acc:.4f}") 