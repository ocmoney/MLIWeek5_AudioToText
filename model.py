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
class SongSlicesDataset(Dataset):
    def __init__(self, metadata_file, spectrogram_dir):
        self.spectrogram_dir = spectrogram_dir
        self.slices_info = []  # List of tuples (spectrogram_path, song_id)
        self.songs = {}  # Dict mapping song_id to list of slice_indices in self.slices_info

        with open(metadata_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                song_id = int(row['song_id'])
                spec_path = row['spectrogram_path']
                self.slices_info.append({'path': spec_path, 'song_id': song_id})

                if song_id not in self.songs:
                    self.songs[song_id] = []
                self.songs[song_id].append(idx)

        self.valid_anchor_indices = []
        for i in range(len(self.slices_info)):
            song_id = self.slices_info[i]['song_id']
            if len(self.songs[song_id]) > 1:
                self.valid_anchor_indices.append(i)
        
        if not self.valid_anchor_indices:
            raise ValueError("No songs with multiple slices found. Cannot create triplets.")
            
        print(f"Dataset initialized. Found {len(self.slices_info)} total slices.")
        print(f"Found {len(self.songs)} unique songs.")
        print(f"Number of slices usable as anchors (from songs with >1 slice): {len(self.valid_anchor_indices)}")

    def __len__(self):
        return len(self.valid_anchor_indices)

    def __getitem__(self, idx):
        anchor_info_idx = self.valid_anchor_indices[idx]
        anchor_info = self.slices_info[anchor_info_idx]
        anchor_song_id = anchor_info['song_id']

        possible_positives_indices = [i for i in self.songs[anchor_song_id] if i != anchor_info_idx]
        if not possible_positives_indices:
            print(f"Warning: Could not find a different positive for anchor slice {anchor_info_idx} from song {anchor_song_id}. Re-sampling.")
            return self.__getitem__(random.randint(0, len(self.valid_anchor_indices) - 1))
        positive_info_idx = random.choice(possible_positives_indices)
        positive_info = self.slices_info[positive_info_idx]

        negative_song_id = anchor_song_id
        while negative_song_id == anchor_song_id:
            negative_song_id = random.choice(list(self.songs.keys()))
        
        negative_info_idx = random.choice(self.songs[negative_song_id])
        negative_info = self.slices_info[negative_info_idx]

        anchor_spec = self._load_spectrogram(anchor_info['path'])
        positive_spec = self._load_spectrogram(positive_info['path'])
        negative_spec = self._load_spectrogram(negative_info['path'])
        
        return anchor_spec, positive_spec, negative_spec

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