import pandas as pd
from datasets import load_dataset
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchaudio
import torchaudio.transforms as T

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def create_mel_spectrogram(audio_data, sr, n_mels=128, n_fft=2048, hop_length=512):
    """
    Convert audio to log mel spectrogram using GPU acceleration
    """
    # Convert numpy array to torch tensor and move to GPU
    audio_tensor = torch.from_numpy(audio_data).float().to(device)
    
    # Create mel spectrogram transform
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    ).to(device)
    
    # Compute mel spectrogram
    mel_spec = mel_transform(audio_tensor)
    
    # Convert to log scale (dB)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    # Move back to CPU for plotting
    mel_spec_db = mel_spec_db.cpu().numpy()
    
    return mel_spec_db

# Load the dataset
print("Loading dataset...")
dataset = load_dataset("danavery/urbansound8K")
df = pd.DataFrame(dataset['train'])

# Create a directory to store spectrograms
import os
if not os.path.exists('mel_spectrograms'):
    os.makedirs('mel_spectrograms')

# Process audio files in batches
print("Converting audio to mel spectrograms...")
BATCH_SIZE = 10  # Process 10 files at a time

for batch_start in tqdm(range(0, len(df), BATCH_SIZE)):
    batch_end = min(batch_start + BATCH_SIZE, len(df))
    batch_df = df.iloc[batch_start:batch_end]
    
    for idx, row in batch_df.iterrows():
        # Get audio data
        audio_data = row['audio']['array']
        sr = row['audio']['sampling_rate']
        
        # Create mel spectrogram
        mel_spec = create_mel_spectrogram(audio_data, sr)
        
        # Save spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec, 
                               sr=sr,
                               hop_length=512,
                               x_axis='time',
                               y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram - Class: {row["class"]}')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'mel_spectrograms/spectrogram_{idx}.png')
        plt.close()

print("Done! Mel spectrograms have been saved in the 'mel_spectrograms' directory.")

# Print some statistics about the spectrograms
print("\nSpectrogram Statistics:")
print(f"Number of spectrograms created: {len(df)}")
print(f"Shape of each spectrogram: {mel_spec.shape}")
print(f"Sample rate: {sr} Hz")
print(f"Number of mel bands: 128")
print(f"FFT window size: 2048")
print(f"Hop length: 512")

# Display one example spectrogram
print("\nDisplaying example spectrogram...")
# Get a random example
example_idx = np.random.randint(0, len(df))
example_row = df.iloc[example_idx]
audio_data = example_row['audio']['array']
sr = example_row['audio']['sampling_rate']

# Create and display the spectrogram
plt.figure(figsize=(12, 6))
mel_spec = create_mel_spectrogram(audio_data, sr)
librosa.display.specshow(mel_spec, 
                        sr=sr,
                        hop_length=512,
                        x_axis='time',
                        y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Example Mel Spectrogram - Class: {example_row["class"]}')
plt.tight_layout()
plt.show()

