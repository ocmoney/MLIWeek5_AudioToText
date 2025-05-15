import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SongToSliceTripletDataset, SpectrogramTower
from tqdm import tqdm
from datasets import load_dataset

# --- Configuration ---
METADATA_FILE = os.path.join('metadata.csv')  # Adjust path as needed
SPECTROGRAM_DIR = '.'  # Current directory (spectrograms)
BATCH_SIZE = 8
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
            tensors.append(tensor)
        # Padding if too short
        num_to_pad = self.max_slices - len(tensors)
        if num_to_pad > 0:
            pad_shape = tensors[0].shape if tensors else (1, 256, 256)
            for _ in range(num_to_pad):
                tensors.append(torch.zeros(pad_shape, dtype=torch.float32))  # Ensure float32
        anchor = torch.cat(tensors, dim=-1)
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

    # Build song_id to song name mapping (lightweight, only once)
    print("Building song_id to song name mapping for analysis...")
    from tqdm import tqdm as tqdm_local
    dataset_hf = load_dataset("teticio/audio-diffusion-instrumental-hiphop-256", split="train")
    audio_name_to_id = {}
    id_to_audio_name = {}
    next_id = 0
    for sample in tqdm_local(dataset_hf, desc="Mapping song IDs"):
        audio_base_name = os.path.splitext(os.path.basename(sample["audio_file"]))[0]
        if audio_base_name not in audio_name_to_id:
            audio_name_to_id[audio_base_name] = next_id
            id_to_audio_name[next_id] = audio_base_name
            next_id += 1
    print(f"Mapping contains {len(id_to_audio_name)} song IDs.")

    # Prepare dataset and dataloader
    dataset = SongAnchorClassificationDataset(METADATA_FILE, SPECTROGRAM_DIR)
    num_classes = len(dataset.song_ids)

    # Preload dataset with tqdm loading bar
    print("Preloading dataset into memory...")
    preload_data = []
    for i in tqdm(range(len(dataset)), desc="Loading dataset"):
        anchor, song_id = dataset[i]
        # Verify mapping: print warning if song_id not in id_to_audio_name
        if song_id not in id_to_audio_name:
            print(f"[WARNING] song_id {song_id} not found in mapping!")
        preload_data.append((anchor, song_id))
    print("Preloading complete.")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

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

            # Print song name for first song in batch (for analysis)
            batch_song_id = song_ids[0].item()
            song_name = id_to_audio_name.get(batch_song_id, "Unknown")

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