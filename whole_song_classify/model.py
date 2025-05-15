import os
import csv
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
# numpy will be needed if spectrograms are PIL Images and need np.array() conversion
# For now, assuming spectrograms are loaded as tensors directly or converted in _load_spectrogram
# import numpy as np 

# --- Configuration (defaults, can be overridden in train.py) ---
EMBEDDING_DIM_DEFAULT = 128

# --- 1. Dataset Definition ---
class SongToSliceTripletDataset(Dataset):
    def __init__(self, metadata_file, spectrogram_dir):
        import csv
        from collections import defaultdict
        self.spectrogram_dir = spectrogram_dir
        self.song_id_to_slices = defaultdict(list)
        with open(metadata_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                song_id = int(row['song_id'])
                slice_idx = int(row['slice'])
                spec_path = row['spectrogram_path']
                self.song_id_to_slices[song_id].append((slice_idx, spec_path))
        # Sort slices for each song
        for song_id in self.song_id_to_slices:
            self.song_id_to_slices[song_id].sort(key=lambda x: x[0])
        self.song_ids = list(self.song_id_to_slices.keys())

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, idx):
        import torch, random
        song_id = self.song_ids[idx]
        # Anchor: concatenate all slices for this song
        slice_paths = [p for _, p in self.song_id_to_slices[song_id]]
        tensors = []
        for spec_path in slice_paths:
            tensor = torch.load(spec_path)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            tensors.append(tensor)
        anchor = torch.cat(tensors, dim=-1)  # Concatenate along time axis

        # Positive: random slice from this song
        pos_path = random.choice(slice_paths)
        positive = torch.load(pos_path)
        if positive.ndim == 2:
            positive = positive.unsqueeze(0)

        # Negative: random slice from a different song
        neg_song_id = random.choice([sid for sid in self.song_ids if sid != song_id])
        neg_path = random.choice([p for _, p in self.song_id_to_slices[neg_song_id]])
        negative = torch.load(neg_path)
        if negative.ndim == 2:
            negative = negative.unsqueeze(0)

        return anchor, positive, negative

    def _load_spectrogram(self, spec_file_path):
        tensor = torch.load(spec_file_path)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[2] < tensor.shape[0] and tensor.shape[2] < tensor.shape[1]:
            tensor = tensor.permute(2, 0, 1)
        
        if tensor.dtype != torch.float32: # Ensure float32 for conv layers
            tensor = tensor.float()
        return tensor

# --- 2. Spectrogram Tower Definition ---
class SpectrogramTower(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM_DEFAULT, in_channels=1):
        super(SpectrogramTower, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, embedding_dim) # 64 is the out_channels of the last conv layer

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x 