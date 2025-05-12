import pandas as pd
from datasets import load_dataset
import numpy as np
import librosa
import matplotlib.pyplot as plt
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

# Get a random example
example_idx = np.random.randint(0, len(df))
example_row = df.iloc[example_idx]
audio_data = example_row['audio']['array']
sr = example_row['audio']['sampling_rate']

print(f"\nProcessing audio example {example_idx}")
print(f"Class: {example_row['class']}")
print(f"Duration: {len(audio_data)/sr:.2f} seconds")
print(f"Sample rate: {sr} Hz")

# Create and display the spectrogram
print("\nGenerating mel spectrogram...")
mel_spec = create_mel_spectrogram(audio_data, sr)

# Display the spectrogram
plt.figure(figsize=(12, 6))
librosa.display.specshow(mel_spec, 
                        sr=sr,
                        hop_length=512,
                        x_axis='time',
                        y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Mel Spectrogram - Class: {example_row["class"]}')
plt.tight_layout()

# Save the spectrogram
save_path = f'spectrogram_example.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSpectrogram saved to {save_path}")

# Show the plot
plt.show()

# Print spectrogram details
print("\nSpectrogram Details:")
print(f"Shape: {mel_spec.shape}")
print(f"Number of mel bands: 128")
print(f"FFT window size: 2048")
print(f"Hop length: 512") 