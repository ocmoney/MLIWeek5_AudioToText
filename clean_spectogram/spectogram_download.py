import os
import pandas as pd
from datasets import load_dataset
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt  # Added for colormap

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load UrbanSound8K dataset
print("Loading dataset...")
dataset = load_dataset("danavery/urbansound8K")
df = pd.DataFrame(dataset['train'])

# Create output directory
output_dir = 'mel_spectrograms_clean'
os.makedirs(output_dir, exist_ok=True)

# Function to convert audio to mel spectrogram
def create_mel_spectrogram(audio_data, sr, n_mels=128, n_fft=2048, hop_length=512):
    audio_tensor = torch.tensor(audio_data).float().to(device)

    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    ).to(device)

    mel_spec = mel_transform(audio_tensor)
    mel_spec_db = T.AmplitudeToDB()(mel_spec)
    return mel_spec_db.cpu().numpy()

# Generate spectrograms without labels
print("Generating label-free mel spectrograms...")

pad_count = 0
crop_count = 0
target_seconds = 4

for idx, row in tqdm(df.iterrows(), total=len(df)):
    audio_data = row['audio']['array']
    sr = row['audio']['sampling_rate']
    target_length = int(target_seconds * sr)

    # Pad or crop
    if len(audio_data) < target_length:
        pad_count += 1
        pad_width = target_length - len(audio_data)
        audio_data = np.pad(audio_data, (0, pad_width), mode='constant')
    elif len(audio_data) > target_length:
        crop_count += 1
        audio_data = audio_data[:target_length]

    mel_spec = create_mel_spectrogram(audio_data, sr)

    # Normalize and convert to colored image using a colormap
    spec_img = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
    cmap = plt.get_cmap('magma')  # or 'inferno', 'viridis', etc.
    rgba_img = cmap(spec_img)  # Returns an MxNx4 array (RGBA)
    rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)  # Drop alpha, scale to 0-255
    img = Image.fromarray(rgb_img)

    # Build filename with useful metadata
    class_id = row["classID"]  # Capital "ID", based on UrbanSound8K schema
    label = row["class"]
    filename = row["slice_file_name"]
    output_path = os.path.join(output_dir, f"{idx}_{class_id}_{label}_{filename}.png")

    img.save(output_path)

print(f"\n✅ Done! Spectrograms saved in: {output_dir}")
print(f"Padded {pad_count} files to 4 seconds.")
print(f"Cropped {crop_count} files to 4 seconds.")
