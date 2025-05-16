import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from model import SongEmbeddingNet
import torchaudio
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter

# --- Config ---
METADATA_FILE = "fma_metadata.csv"
BATCH_SIZE = 64
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

# --- Genre Triplet Dataset ---
class GenreTripletDataset(Dataset):
    def __init__(self, metadata_file, slice_seconds=5, sample_rate=22050, hop_length=512):
        self.df = pd.read_csv(metadata_file)
        self.song_to_path = {row['song_id']: row['spectrogram_path'] for _, row in self.df.iterrows()}
        self.song_to_genre = {row['song_id']: row['genre'] for _, row in self.df.iterrows()}
        self.genre_to_songs = {}
        for song_id, genre in self.song_to_genre.items():
            self.genre_to_songs.setdefault(genre, []).append(song_id)
        self.song_ids = list(self.song_to_path.keys())
        self.slice_seconds = slice_seconds
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.slice_frames = int((slice_seconds * sample_rate) // hop_length)

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, idx):
        anchor_song = self.song_ids[idx]
        anchor_genre = self.song_to_genre[anchor_song]
        anchor_spec = np.load(self.song_to_path[anchor_song])
        if anchor_spec.ndim == 2:  # if it's a 2D array, it's a spectrogram
            anchor_spec = np.expand_dims(anchor_spec, 0) # if it's a 2D array, it's a spectrogram
        time_dim = anchor_spec.shape[-1] # get the time dimension
        max_start = time_dim - self.slice_frames # get the maximum start index
        a_start = np.random.randint(0, max_start) # get a random start index
        anchor_slice = anchor_spec[..., a_start:a_start+self.slice_frames] # get the anchor slice

        # Positive: different song, same genre
        pos_candidates = [sid for sid in self.genre_to_songs[anchor_genre] if sid != anchor_song] # get the positive candidates
        pos_song = np.random.choice(pos_candidates) # get a random positive song
        pos_spec = np.load(self.song_to_path[pos_song]) # load the positive spectrogram
        if pos_spec.ndim == 2: # if it's a 2D array, it's a spectrogram
            pos_spec = np.expand_dims(pos_spec, 0) # if it's a 2D array, it's a spectrogram
        pos_time_dim = pos_spec.shape[-1] # get the time dimension
        pos_max_start = pos_time_dim - self.slice_frames # get the maximum start index
        p_start = np.random.randint(0, pos_max_start) # get a random start index
        positive_slice = pos_spec[..., p_start:p_start+self.slice_frames] # get the positive slice

        # Negative: song from a different genre
        neg_genres = [g for g in self.genre_to_songs if g != anchor_genre] # get the negative genres
        neg_genre = np.random.choice(neg_genres) # get a random negative genre
        neg_song = np.random.choice(self.genre_to_songs[neg_genre]) # get a random negative song
        neg_spec = np.load(self.song_to_path[neg_song]) # load the negative spectrogram
        if neg_spec.ndim == 2: # if it's a 2D array, it's a spectrogram
            neg_spec = np.expand_dims(neg_spec, 0) # if it's a 2D array, it's a spectrogram
        neg_time_dim = neg_spec.shape[-1] # get the time dimension
        neg_max_start = neg_time_dim - self.slice_frames # get the maximum start index
        n_start = np.random.randint(0, neg_max_start) # get a random start index
        negative_slice = neg_spec[..., n_start:n_start+self.slice_frames]

        # Apply Gaussian blur
        # positive_slice = gaussian_filter(positive_slice, sigma=(0, 1, 1))
        # negative_slice = gaussian_filter(negative_slice, sigma=(0, 1, 1))

        return (
            torch.from_numpy(anchor_slice).float(),
            torch.from_numpy(positive_slice).float(),
            torch.from_numpy(negative_slice).float()
        )

class CombinedLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, anchor, positive, negative):
        # Calculate distances
        d_ap = torch.norm(anchor - positive, dim=1)
        d_an = torch.norm(anchor - negative, dim=1)
        
        # Calculate margin as half of the absolute difference of mean distances
        margin = 0.5 * torch.abs(d_ap.mean() - d_an.mean())
        
        # Triplet loss: we want d_ap < d_an - margin
        triplet_loss = torch.clamp(d_ap - d_an + margin, min=0.0).mean()
        
        return triplet_loss, triplet_loss, 0.0  # Return 0.0 for contrastive loss to maintain interface

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use genre-based triplet dataset
    dataset = GenreTripletDataset(METADATA_FILE, slice_seconds=5, sample_rate=22050, hop_length=512)
    num_examples = len(dataset)
    indices = np.arange(num_examples)
    np.random.seed(42)
    np.random.shuffle(indices)
    test_size = int(0.1 * num_examples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = SongEmbeddingNet(temperature=0.07, normalize_embeddings=False).to(device)
    
    # Run a few batches to estimate initial distances
    d_ap_list, d_an_list = [], []
    for i, (anchor, positive, negative) in enumerate(train_loader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        # Use the new triplet interface
        anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
        d_ap = torch.norm(anchor_emb - positive_emb, dim=1)
        d_an = torch.norm(anchor_emb - negative_emb, dim=1)
        d_ap_list.append(d_ap.mean().item())
        d_an_list.append(d_an.mean().item())
        if i == 4:  # Use 5 batches for estimate
            break
    mean_d_ap = np.mean(d_ap_list)
    mean_d_an = np.mean(d_an_list)
    print(f"Initial distances - d_ap: {mean_d_ap:.3f}, d_an: {mean_d_an:.3f}")
    print(f"Initial margin would be: {0.5 * (mean_d_ap - mean_d_an):.3f}")

    criterion = CombinedLoss(temperature=0.07)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1} (Triplet Loss)")
        model.train()
        running_loss = 0.0
        running_triplet_loss = 0.0
        running_acc = 0.0
        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            optimizer.zero_grad()
            
            # Forward pass through model
            anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
            total_loss, triplet_loss, _ = criterion(anchor_emb, positive_emb, negative_emb)
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * anchor.size(0)
            running_triplet_loss += triplet_loss.item() * anchor.size(0)
            
            # Compute accuracy: anchor should be closer to positive than negative
            d_ap = torch.norm(anchor_emb - positive_emb, dim=1)
            d_an = torch.norm(anchor_emb - negative_emb, dim=1)
            acc = (d_ap < d_an).float().mean().item()
            running_acc += acc * anchor.size(0)
            
            if batch_idx == 0:
                print("d_ap (mean):", d_ap.mean().item(), "d_an (mean):", d_an.mean().item())
                print("d_ap (min):", d_ap.min().item(), "d_an (min):", d_an.min().item())
                print("d_ap (max):", d_ap.max().item(), "d_an (max):", d_an.max().item())
            if (batch_idx + 1) % 35 == 0:
                print(f"  [Train] Epoch {epoch+1} Batch {batch_idx+1}: Loss: {total_loss.item():.4f} Acc: {acc:.4f}")
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_triplet_loss = running_triplet_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f}")

        # Optional: test loop (unsupervised, so just report loss and accuracy)
        model.eval()
        test_loss = 0.0
        test_triplet_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for batch_idx, (anchor, positive, negative) in enumerate(test_loader):
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                
                # Forward pass through model
                anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
                total_loss, triplet_loss, _ = criterion(anchor_emb, positive_emb, negative_emb)
                
                test_loss += total_loss.item() * anchor.size(0)
                test_triplet_loss += triplet_loss.item() * anchor.size(0)
                d_ap = torch.norm(anchor_emb - positive_emb, dim=1)
                d_an = torch.norm(anchor_emb - negative_emb, dim=1)
                acc = (d_ap < d_an).float().mean().item()
                test_acc += acc * anchor.size(0)
                if (batch_idx + 1) % 35 == 0:
                    print(f"  [Test] Epoch {epoch+1} Batch {batch_idx+1}: Loss: {total_loss.item():.4f} Acc: {acc:.4f}")
        test_loss = test_loss / len(test_loader.dataset)
        test_triplet_loss = test_triplet_loss / len(test_loader.dataset)
        test_acc = test_acc / len(test_loader.dataset)
        print(f"[Test] Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {test_loss:.4f} Train Acc: {test_acc:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
