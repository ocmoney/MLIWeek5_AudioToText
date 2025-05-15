import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from model import SongClassifier
import torchaudio
from tqdm import tqdm
import numpy as np

# --- Config ---
METADATA_FILE = "fma_metadata.csv"
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
MODEL_SAVE_PATH = "song_classifier.pth"
TARGET_TIME = 1291

# --- Dataset ---
class FMASpectrogramDataset(Dataset):
    pad_count = 0
    crop_count = 0
    stats_printed = False

    def __init__(self, metadata_file):
        self.df = pd.read_csv(metadata_file)
        self.paths = self.df['spectrogram_path'].tolist()
        self.song_ids = self.df['song_id'].tolist()
        self.unique_song_ids = sorted(list(set(self.song_ids)))
        self.song_id_to_idx = {sid: i for i, sid in enumerate(self.unique_song_ids)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        spec = np.load(self.paths[idx])  # shape: [channels, n_mels, time] or [n_mels, time]
        if spec.ndim == 2:
            spec = np.expand_dims(spec, 0)
        elif spec.shape[0] > 1:
            spec = np.mean(spec, axis=0, keepdims=True)
        time_dim = spec.shape[-1]
        if time_dim > TARGET_TIME:
            start = (time_dim - TARGET_TIME) // 2
            spec = spec[..., start:start+TARGET_TIME]
            FMASpectrogramDataset.crop_count += 1
        elif time_dim < TARGET_TIME:
            pad_width = TARGET_TIME - time_dim
            spec = np.pad(spec, ((0,0), (0,0), (0, pad_width)), mode='constant')
            FMASpectrogramDataset.pad_count += 1
        spec = torch.from_numpy(spec).float()
        label = self.song_id_to_idx[self.song_ids[idx]]
        return spec, label

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = FMASpectrogramDataset(METADATA_FILE)
    num_classes = len(dataset.unique_song_ids)
    # Print first 10 (label, song_id, file) triplets for verification
    print("First 10 (label, song_id, file, song_name) triplets:")
    # Try to get song name from a CSV if available
    song_id_to_name = {}
    if os.path.exists('tracks.csv'):
        import pandas as pd
        tracks_df = pd.read_csv('tracks.csv', index_col=0, low_memory=False)
        # FMA track_id is the song_id, and 'title' is the song name
        for track_id, row in tracks_df.iterrows():
            song_id_to_name[str(track_id)] = row.get('title', 'Unknown')
    for i in range(min(10, len(dataset))):
        song_id = dataset.song_ids[i]
        label = dataset.song_id_to_idx[song_id]
        file = dataset.paths[i]
        song_name = song_id_to_name.get(str(song_id), 'Unknown')
        print(f"label: {label}, song_id: {song_id}, file: {file}, song_name: {song_name}")

    # Shuffle indices
    num_examples = len(dataset)
    indices = np.arange(num_examples)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)

    # Split
    test_size = int(0.05 * num_examples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = SongClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}: Number of unique songs (classes): {num_classes}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (specs, labels) in enumerate(train_loader):
            specs = specs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * specs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            if (batch_idx + 1) % 10 == 0:
                print(f"  [Train] Epoch {epoch+1} Batch {batch_idx+1}: Loss: {loss.item():.4f}")
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        if not FMASpectrogramDataset.stats_printed:
            print(f"Padded {FMASpectrogramDataset.pad_count} files to {TARGET_TIME} frames.")
            print(f"Cropped {FMASpectrogramDataset.crop_count} files to {TARGET_TIME} frames.")
            FMASpectrogramDataset.stats_printed = True

        # Test loop (if you have a test_loader)
        if 'test_loader' in locals():
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (specs, labels) in enumerate(test_loader):
                specs = specs.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    outputs = model(specs)
                    loss = criterion(outputs, labels)
                test_loss += loss.item() * specs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                if (batch_idx + 1) % 10 == 0:
                    print(f"  [Test] Epoch {epoch+1} Batch {batch_idx+1}: Loss: {loss.item():.4f}")
            test_loss = test_loss / total if total > 0 else 0
            test_acc = correct / total if total > 0 else 0
            print(f"[Test] Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
