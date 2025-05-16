import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
SLICE_SECONDS = 5
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_MELS = 128
SLICE_FRAMES = int((SLICE_SECONDS * SAMPLE_RATE) // HOP_LENGTH)
MODEL_SAVE_PATH = "genre_classifier.pth"
VAL_SPLIT = 0.2  # 20% of data for validation

class GenreDataset(Dataset):
    def __init__(self, metadata_file):
        self.df = pd.read_csv(metadata_file)
        # Filter out Pop genre
        self.df = self.df[self.df['genre'] != 'Pop']
        self.spectrogram_paths = self.df['spectrogram_path'].tolist()
        self.genres = self.df['genre'].tolist()
        
        # Create genre to index mapping
        unique_genres = sorted(list(set(self.genres)))
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}
        self.idx_to_genre = {idx: genre for genre, idx in self.genre_to_idx.items()}
        self.num_classes = len(unique_genres)
        
        # Precompute all valid slices for each song
        self.song_slices = []
        self.song_to_slices = {}  # Map song paths to their slice indices
        
        for spec_path, genre in zip(self.spectrogram_paths, self.genres):
            spec = np.load(spec_path)
            if spec.ndim == 2:
                spec = np.expand_dims(spec, 0)
            
            time_dim = spec.shape[-1]
            num_slices = time_dim // SLICE_FRAMES
            # Use half of the available slices
            num_slices_to_use = max(1, num_slices // 2)
            
            # Calculate valid start indices
            valid_starts = []
            for i in range(num_slices_to_use):
                start_idx = i * SLICE_FRAMES
                if start_idx + SLICE_FRAMES <= time_dim:
                    valid_starts.append(start_idx)
            
            # Add all valid slices for this song
            song_slice_indices = []
            for start_idx in valid_starts:
                slice_idx = len(self.song_slices)
                self.song_slices.append({
                    'path': spec_path,
                    'start_idx': start_idx,
                    'genre': genre
                })
                song_slice_indices.append(slice_idx)
            
            self.song_to_slices[spec_path] = song_slice_indices
        
        print(f"Found {len(unique_genres)} unique genres (excluding Pop)")
        print("Genre mapping:", self.genre_to_idx)
        print(f"Total number of slices: {len(self.song_slices)}")
        # Print genre distribution
        genre_counts = self.df['genre'].value_counts()
        print("\nGenre distribution:")
        for genre, count in genre_counts.items():
            print(f"{genre}: {count} songs")

    def __len__(self):
        return len(self.song_slices)

    def __getitem__(self, idx):
        slice_info = self.song_slices[idx]
        spec_path = slice_info['path']
        start_idx = slice_info['start_idx']
        genre = slice_info['genre']
        
        # Load spectrogram
        spec = np.load(spec_path)
        if spec.ndim == 2:
            spec = np.expand_dims(spec, 0)
        
        # Get the specific slice
        spec_slice = spec[..., start_idx:start_idx+SLICE_FRAMES]
        
        # Get all slices from the same song
        song_slice_indices = self.song_to_slices[spec_path]
        other_slice_indices = [i for i in song_slice_indices if i != idx]
        
        # If there are other slices, randomly select one
        if other_slice_indices:
            other_idx = np.random.choice(other_slice_indices)
            other_slice_info = self.song_slices[other_idx]
            other_spec = np.load(other_slice_info['path'])
            if other_spec.ndim == 2:
                other_spec = np.expand_dims(other_spec, 0)
            other_slice = other_spec[..., other_slice_info['start_idx']:other_slice_info['start_idx']+SLICE_FRAMES]
        else:
            # If no other slices, duplicate the current slice
            other_slice = spec_slice.copy()
        
        # Convert to tensors
        spec_tensor = torch.from_numpy(spec_slice).float()
        other_slice_tensor = torch.from_numpy(other_slice).float()
        
        # Get genre index
        genre_idx = self.genre_to_idx[genre]
        
        return spec_tensor, other_slice_tensor, genre_idx

class SliceEncoder(nn.Module):
    def __init__(self):
        super(SliceEncoder, self).__init__()
        
        # 1D CNN layers for processing individual slices
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(N_MELS, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # Calculate the size of the flattened features
        self._to_linear = None
        x = torch.randn(1, N_MELS, SLICE_FRAMES)
        x = self.conv_layers(x)
        self._to_linear = x.shape[1] * x.shape[2]
        
        # Projection layer
        self.projection = nn.Linear(self._to_linear, 512)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return x

class GenreEncoder(nn.Module):
    def __init__(self, num_classes):
        super(GenreEncoder, self).__init__()
        
        # Two slice encoders
        self.slice_encoder1 = SliceEncoder()
        self.slice_encoder2 = SliceEncoder()
        
        # Attention mechanism for combining slice features
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x1, x2):
        # Encode both slices
        h1 = self.slice_encoder1(x1)
        h2 = self.slice_encoder2(x2)
        
        # Combine features using attention
        combined = torch.stack([h1, h2], dim=1)  # [batch, 2, 512]
        attention_weights = self.attention(combined)  # [batch, 2, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_features = torch.sum(combined * attention_weights, dim=1)  # [batch, 512]
        
        # Classify
        output = self.classifier(weighted_features)
        return output

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs1, inputs2, labels in tqdm(train_loader, desc="Training"):
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs1, inputs2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return epoch_loss, accuracy

def validate_model(model, val_loader, criterion, device, idx_to_genre):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # For detailed metrics
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs1, inputs2, labels in tqdm(val_loader, desc="Validating"):
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store predictions and labels for detailed metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    # Calculate per-genre accuracy
    genre_correct = {genre: 0 for genre in idx_to_genre.values()}
    genre_total = {genre: 0 for genre in idx_to_genre.values()}
    
    for pred, label in zip(all_preds, all_labels):
        genre = idx_to_genre[label]
        genre_total[genre] += 1
        if pred == label:
            genre_correct[genre] += 1
    
    genre_accuracy = {genre: 100. * genre_correct[genre] / genre_total[genre] 
                     for genre in genre_total.keys()}
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return val_loss, accuracy, genre_accuracy, cm

def plot_confusion_matrix(cm, idx_to_genre, save_path='confusion_matrix.png'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(idx_to_genre.values()),
                yticklabels=list(idx_to_genre.values()))
    plt.title('Confusion Matrix')
    plt.ylabel('True Genre')
    plt.xlabel('Predicted Genre')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    print("Loading dataset...")
    dataset = GenreDataset("fma_metadata.csv")
    
    # Split dataset into train and validation
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Total dataset size: {len(dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model
    print("Creating model...")
    model = GenreEncoder(dataset.num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("Starting training...")
    best_val_accuracy = 0.0
    
    for epoch in range(NUM_EPOCHS):
        # Training
        train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_accuracy, genre_accuracy, cm = validate_model(
            model, val_loader, criterion, device, dataset.idx_to_genre)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Print per-genre accuracy
        print("\nPer-genre validation accuracy:")
        for genre, acc in genre_accuracy.items():
            print(f"{genre}: {acc:.2f}%")
        
        # Save best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Plot confusion matrix for best model
            plot_confusion_matrix(cm, dataset.idx_to_genre)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'genre_accuracy': genre_accuracy,
                'genre_mapping': dataset.genre_to_idx
            }, MODEL_SAVE_PATH)
            print(f"\nSaved new best model with validation accuracy: {val_accuracy:.2f}%")
        print()
    
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    main()
