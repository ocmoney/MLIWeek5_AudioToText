import os
import torchaudio
import torch
import numpy as np
from tqdm import tqdm

AUDIO_DIR = "fma_small/fma_small"
OUTPUT_DIR = "fma_spectrograms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sample_rate = 22050
n_mels = 128
n_fft = 2048
hop_length = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels
).to(device)
amp_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

first_five_specs = []
count = 0

# Walk through all mp3 files
for root, dirs, files in os.walk(AUDIO_DIR):
    for file in tqdm(files, desc="Processing audio files"):
        if file.endswith('.mp3'):
            audio_path = os.path.join(root, file)
            out_path = os.path.join(OUTPUT_DIR, file.replace('.mp3', '.npy'))
            if os.path.exists(out_path):
                continue  # Skip if already processed
            try:
                waveform, sr = torchaudio.load(audio_path)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                continue
            if sr != sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.to(device)
            mel_spec = mel_transform(waveform)
            mel_spec_db = amp_to_db(mel_spec)
            mel_spec_db = mel_spec_db.cpu().numpy()
            np.save(out_path, mel_spec_db)
            if count < 5:
                first_five_specs.append(mel_spec_db)
                count += 1

print("\nFirst 5 spectrogram shapes:")
for i, spec in enumerate(first_five_specs):
    print(f"Spectrogram {i+1}: {spec.shape}")
