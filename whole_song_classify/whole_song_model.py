import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SongToSliceTripletDataset, SpectrogramTower

# --- Configuration ---
METADATA_FILE = os.path.join('metadata.csv')  # Adjust path as needed
SPECTROGRAM_DIR = '.'  # Current directory (spectrograms)
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
EMBEDDING_DIM = 128
SPECTROGRAM_CHANNELS = 1

# --- Dataset for classification (only anchors and song_ids) ---
class SongAnchorClassificationDataset(SongToSliceTripletDataset):
    def __init__(self, metadata_file, spectrogram_dir, max_slices=68, min_slices=4):
        super().__init__(metadata_file, spectrogram_dir)
        self.max_slices = max_slices
        self.min_slices = min_slices
        # Filter out songs with fewer than min_slices
        self.song_ids = [sid for sid in self.song_ids if len(self.song_id_to_slices[sid]) >= self.min_slices]

    def __getitem__(self, idx):
        song_id = self.song_ids[idx]
        slice_paths = [p for _, p in self.song_id_to_slices[song_id]]
        tensors = []
        for spec_path in slice_paths[:self.max_slices]:  # Truncate if too long
            tensor = torch.load(spec_path)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.float()  # Ensure float32
            print(f"Loaded tensor shape for {spec_path}: {tensor.shape}")  # Print shape of each slice
            tensors.append(tensor)
        # Padding if too short
        num_to_pad = self.max_slices - len(tensors)
        if num_to_pad > 0:
            pad_shape = tensors[0].shape if tensors else (1, 256, 256)
            for _ in range(num_to_pad):
                tensors.append(torch.zeros(pad_shape, dtype=torch.float32))  # Ensure float32
        anchor = torch.cat(tensors, dim=-1)
        print(f"Final anchor tensor shape for song_id {song_id}: {anchor.shape}")  # Print final anchor shape
        return anchor, song_id

# --- Classifier Model ---
class SongIDClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, in_channels=1):
        super().__init__()
        self.tower = SpectrogramTower(embedding_dim=embedding_dim, in_channels=in_channels)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.tower(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare dataset and dataloader
    dataset = SongAnchorClassificationDataset(METADATA_FILE, SPECTROGRAM_DIR)
    num_classes = len(dataset.song_ids)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Model, loss, optimizer
    model = SongIDClassifier(EMBEDDING_DIM, num_classes, in_channels=SPECTROGRAM_CHANNELS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for anchors, song_ids in dataloader:
            anchors = anchors.to(device)
            song_ids = song_ids.to(device)

            optimizer.zero_grad()
            outputs = model(anchors)
            loss = criterion(outputs, song_ids)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * anchors.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == song_ids).sum().item()
            total += anchors.size(0)

        epoch_loss = running_loss / total
        accuracy = correct / total
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {epoch_loss:.4f} Accuracy: {accuracy:.4f}")

    print("Training finished.") 