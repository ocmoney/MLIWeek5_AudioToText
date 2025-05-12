import pandas as pd
from datasets import load_dataset
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_mel_spectrogram(audio_data, sr, n_mels=128, n_fft=2048, hop_length=512):
    """
    Convert audio to log mel spectrogram
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

# Load the dataset
print("Loading dataset...")
dataset = load_dataset("danavery/urbansound8K")
df = pd.DataFrame(dataset['train'])

# Create a directory to store spectrograms
import os
if not os.path.exists('mel_spectrograms'):
    os.makedirs('mel_spectrograms')

# Process each audio file
print("Converting audio to mel spectrograms...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
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
