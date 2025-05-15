import os
import csv
import torch
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from datasets import load_dataset

# Parameters from dataset
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256

METADATA_FILE = '../metadata.csv' if not os.path.exists('metadata.csv') else 'metadata.csv'
SPECTROGRAM_DIR = '../spectrograms' if not os.path.exists('spectrograms') else 'spectrograms'

# Set the song_id you want to reconstruct
SONG_ID = 419  # Change this to the desired song_id

# Rebuild song_id to song name mapping
print("Looking up song name...")
dataset = load_dataset("teticio/audio-diffusion-instrumental-hiphop-256", split="train")
audio_name_to_id = {}
id_to_audio_name = {}
next_id = 0
for sample in dataset:
    audio_base_name = os.path.splitext(os.path.basename(sample["audio_file"]))[0]
    if audio_base_name not in audio_name_to_id:
        audio_name_to_id[audio_base_name] = next_id
        id_to_audio_name[next_id] = audio_base_name
        next_id += 1
song_name = id_to_audio_name.get(SONG_ID, "Unknown")
print(f"Song name for song_id {SONG_ID}: {song_name}")

# 1. Read metadata and get all slices for the song
slices = []
with open(METADATA_FILE, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['song_id']) == SONG_ID:
            slices.append((int(row['slice']), row['spectrogram_path']))

if not slices:
    print(f"No slices found for song_id {SONG_ID}")
    exit(1)

# 2. Sort slices by slice index
slices.sort(key=lambda x: x[0])

# 3. Load and concatenate spectrograms
mel_specs = []
for _, spec_path in slices:
    spec = torch.load(spec_path)
    if spec.ndim == 2:
        spec = spec.unsqueeze(0)
    mel_specs.append(spec.numpy())
mel_spec = np.concatenate(mel_specs, axis=-1)[0]  # shape: (n_mels, time)

# Visualize the spectrogram
plt.figure(figsize=(12, 4))
librosa.display.specshow(mel_spec, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', cmap='magma')
plt.title(f'Mel Spectrogram for song_id {SONG_ID}')
plt.colorbar(format='%+2.0f')
plt.tight_layout()
plt.show()

# 4. Invert mel spectrogram to audio (try several strategies)
mel_spec = mel_spec.astype(np.float32)

# Strategy 1: Linear scaling, Griffin-Lim n_iter=256
mel_linear = mel_spec.copy()
if mel_linear.max() > 2.0:
    mel_linear = mel_linear / 255.0
mel_linear = np.maximum(mel_linear, 1e-5)
audio_linear = librosa.feature.inverse.mel_to_audio(
    mel_linear, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, power=1.0, n_iter=256
)
audio_linear = audio_linear / np.max(np.abs(audio_linear))
sf.write(f'song_{SONG_ID}_reconstructed_linear.wav', audio_linear, SAMPLE_RATE)
print(f"[Linear] Saved: song_{SONG_ID}_reconstructed_linear.wav")

# Strategy 2: Log-mel dB scaling, Griffin-Lim n_iter=256
# Map [0,1] to [-80,0] dB (common for log-mel images)
mel_log = mel_spec.copy()
if mel_log.max() > 2.0:
    mel_log = mel_log / 255.0
mel_log_db = mel_log * 80.0 - 80.0  # [0,1] -> [-80,0] dB
mel_log_power = librosa.db_to_power(mel_log_db)
audio_log = librosa.feature.inverse.mel_to_audio(
    mel_log_power, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, power=1.0, n_iter=256
)
audio_log = audio_log / np.max(np.abs(audio_log))
sf.write(f'song_{SONG_ID}_reconstructed_logmel.wav', audio_log, SAMPLE_RATE)
print(f"[Log-mel] Saved: song_{SONG_ID}_reconstructed_logmel.wav")

# Strategy 3: Default librosa inversion (for reference)
audio_default = librosa.feature.inverse.mel_to_audio(
    mel_spec, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, power=1.0
)
audio_default = audio_default / np.max(np.abs(audio_default))
sf.write(f'song_{SONG_ID}_reconstructed_default.wav', audio_default, SAMPLE_RATE)
print(f"[Default] Saved: song_{SONG_ID}_reconstructed_default.wav")

# Strategy 4: Match original dataset code (uint8 -> log-mel dB -> power -> audio)
top_db = 80.0
mel_spec_uint8 = mel_spec
if mel_spec_uint8.max() <= 1.0:
    mel_spec_uint8 = (mel_spec_uint8 * 255).astype(np.uint8)
elif mel_spec_uint8.max() > 255:
    mel_spec_uint8 = (mel_spec_uint8 / mel_spec_uint8.max() * 255).astype(np.uint8)
else:
    mel_spec_uint8 = mel_spec_uint8.astype(np.uint8)

log_S = mel_spec_uint8.astype("float32") * top_db / 255.0 - top_db
S = librosa.db_to_power(log_S)
audio_matched = librosa.feature.inverse.mel_to_audio(
    S, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=256
)
audio_matched = audio_matched / np.max(np.abs(audio_matched))
sf.write(f'song_{SONG_ID}_reconstructed_matched.wav', audio_matched, SAMPLE_RATE)
print(f"[Matched original code] Saved: song_{SONG_ID}_reconstructed_matched.wav")
