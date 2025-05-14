import os
import pandas as pd
from datasets import load_dataset
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

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

# Calculate expected spectrogram size
sample_rate = 22050
hop_length = 512
target_seconds = 4
expected_time_steps = int((target_seconds * sample_rate) / hop_length)
print(f"\nExpected spectrogram size for {target_seconds} seconds:")
print(f"Mel bands: 128")
print(f"Time steps: {expected_time_steps}")
print(f"Total size: 128 x {expected_time_steps}")

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
print("\nGenerating label-free mel spectrograms...")

pad_count = 0
crop_count = 0
target_seconds = 4

# Store first 5 spectrograms for dimension checking
first_five_specs = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    audio_data = row['audio']['array']
    sr = row['audio']['sampling_rate']
    target_length = int(target_seconds * sr)

    # Pad or crop to exactly 4 seconds
    if len(audio_data) < target_length:
        pad_count += 1
        pad_width = target_length - len(audio_data)
        audio_data = np.pad(audio_data, (0, pad_width), mode='constant')
    elif len(audio_data) > target_length:
        crop_count += 1
        # Center crop the audio
        start = (len(audio_data) - target_length) // 2
        audio_data = audio_data[start:start + target_length]
    else:
        # Audio is exactly 4 seconds
        pass

    # Verify the length is exactly 4 seconds
    assert len(audio_data) == target_length, f"Audio length {len(audio_data)} != {target_length}"

    mel_spec = create_mel_spectrogram(audio_data, sr)
    
    # Store first 5 spectrograms
    if len(first_five_specs) < 5:
        first_five_specs.append(mel_spec)

    # Build filename with fold, index, and class metadata
    fold_number = row["fold"]
    class_name = row["class"]
    # Using original DataFrame index (idx) for uniqueness and traceability
    output_filename = f"fold{fold_number}_idx{idx}_{class_name}.npy"
    output_path = os.path.join(output_dir, output_filename)

    # Save the raw mel spectrogram array
    np.save(output_path, mel_spec)

print(f"\n✅ Done! Spectrograms saved in: {output_dir}")
print(f"Padded {pad_count} files to 4 seconds.")
print(f"Cropped {crop_count} files to 4 seconds.")

# Print dimensions of first 5 spectrograms
print("\nActual dimensions of first 5 spectrograms:")
for i, spec in enumerate(first_five_specs):
    print(f"\nSpectrogram {i+1}:")
    print(f"Shape = {spec.shape}")
    print(f"Time length in seconds = {spec.shape[1] * hop_length / sample_rate:.2f} seconds")
    print(f"Expected shape: (128, {expected_time_steps})")
    print(f"Matches expected: {spec.shape == (128, expected_time_steps)}")
