import torch
import torchaudio
import numpy as np
import librosa
import time
from genre_encoder import TwoTowerGenreEncoder, SLICE_FRAMES, N_MELS, SAMPLE_RATE, HOP_LENGTH, add_spectrogram_noise
import matplotlib.pyplot as plt
import os
import pyaudio

# Audio recording constants
CHUNK = 1024  # Number of frames per buffer
RECORD_SECONDS = 5  # Duration of each recording

def load_audio_file(file_path):
    """Load audio from a file."""
    print(f"\nLoading audio file: {file_path}")
    try:
        # Load audio using librosa
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        print(f"Successfully loaded audio file")
        print(f"Duration: {len(audio)/SAMPLE_RATE:.2f} seconds")
        print(f"Sample rate: {sr} Hz")
        return audio
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

def compute_melspectrogram(audio, sample_rate=SAMPLE_RATE):
    """Compute mel spectrogram from audio signal."""
    # Ensure audio is mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Normalize audio
    audio = librosa.util.normalize(audio)
    
    # Compute mel spectrogram with same parameters as training
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=2048,
        fmin=20,
        fmax=8000,
        power=1.0  # Use linear scale
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1] range
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    # Print some debug info
    print(f"\nSpectrogram stats:")
    print(f"Shape: {mel_spec_norm.shape}")
    print(f"Min value: {mel_spec_norm.min():.3f}")
    print(f"Max value: {mel_spec_norm.max():.3f}")
    print(f"Mean value: {mel_spec_norm.mean():.3f}")
    
    return mel_spec_norm

def get_slices(mel_spec, num_slices=5):
    """Get multiple slices from the spectrogram, similar to validation."""
    time_dim = mel_spec.shape[1]
    slices = []
    
    if time_dim <= SLICE_FRAMES:
        # If song is shorter than 5 seconds, pad it
        pad_width = SLICE_FRAMES - time_dim
        padded_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        slices.append(padded_spec)
    else:
        # Get evenly spaced slices, similar to validation
        num_slices_to_use = min(num_slices, time_dim // SLICE_FRAMES)
        for i in range(num_slices_to_use):
            start_idx = int(i * (time_dim - SLICE_FRAMES) / (num_slices_to_use - 1))
            slice_spec = mel_spec[:, start_idx:start_idx+SLICE_FRAMES]
            slices.append(slice_spec)
    
    return slices

def prepare_inputs(audio, sample_rate=SAMPLE_RATE):
    """Prepare inputs for the model."""
    # Compute mel spectrogram
    mel_spec = compute_melspectrogram(audio, sample_rate)
    
    # Get multiple slices
    spec_slices = get_slices(mel_spec)
    
    # Prepare full song (downsampled to match slice length)
    if mel_spec.shape[1] > SLICE_FRAMES:
        # Downsample the full song to match slice length
        full_spec = torch.nn.functional.interpolate(
            torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0),
            size=(mel_spec.shape[0], SLICE_FRAMES),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0).numpy()
    else:
        # Pad if shorter than 5 seconds
        pad_width = SLICE_FRAMES - mel_spec.shape[1]
        full_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    
    # Add noise to slices (matching training conditions)
    spec_slices = [add_spectrogram_noise(slice_spec) for slice_spec in spec_slices]
    
    # Ensure values are in [0, 1] range
    spec_slices = [np.clip(slice_spec, 0, 1) for slice_spec in spec_slices]
    full_spec = np.clip(full_spec, 0, 1)
    
    # Convert to tensors
    slice_tensors = [torch.from_numpy(slice_spec).float() for slice_spec in spec_slices]
    full_tensor = torch.from_numpy(full_spec).float()
    
    return slice_tensors, full_tensor

def plot_spectrograms(slice_specs, full_spec):
    """Plot spectrograms for visualization."""
    n_slices = len(slice_specs)
    fig, axes = plt.subplots(n_slices + 1, 1, figsize=(10, 4 * (n_slices + 1)))
    
    # Plot slices
    for i, slice_spec in enumerate(slice_specs):
        im = axes[i].imshow(slice_spec, aspect='auto', origin='lower', vmin=0, vmax=1)
        axes[i].set_title(f'Slice {i+1} Spectrogram (with noise)')
        axes[i].set_ylabel('Mel Bins')
        axes[i].set_xlabel('Time')
        plt.colorbar(im, ax=axes[i])
    
    # Plot full song
    im = axes[-1].imshow(full_spec, aspect='auto', origin='lower', vmin=0, vmax=1)
    axes[-1].set_title('Full Song Spectrogram (Downsampled)')
    axes[-1].set_ylabel('Mel Bins')
    axes[-1].set_xlabel('Time')
    plt.colorbar(im, ax=axes[-1])
    
    plt.tight_layout()
    plt.show()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    # Initialize model with 7 classes (excluding Experimental)
    model = TwoTowerGenreEncoder(num_classes=7).to(device)
    checkpoint = torch.load("genre_classifier.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get genre mapping from checkpoint
    genre_mapping = checkpoint['genre_mapping']
    idx_to_genre = {v: k for k, v in genre_mapping.items()}
    
    print("Model loaded successfully")
    print("Genre mapping:", genre_mapping)
    
    while True:
        try:
            # Get audio file path from user
            file_path = input("\nEnter the path to your audio file (or 'q' to quit): ")
            if file_path.lower() == 'q':
                break
            
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                continue
            
            # Load and process audio
            audio = load_audio_file(file_path)
            if audio is None:
                continue
            
            # Prepare inputs
            slice_tensors, full_tensor = prepare_inputs(audio)
            
            # Plot spectrograms
            plot_spectrograms([s.numpy() for s in slice_tensors], full_tensor.numpy())
            
            # Move tensors to device
            slice_tensors = [s.to(device) for s in slice_tensors]
            full_tensor = full_tensor.to(device)
            
            # Get predictions for each slice
            all_probs = []
            with torch.no_grad():
                for slice_tensor in slice_tensors:
                    outputs = model(slice_tensor.unsqueeze(0), full_tensor.unsqueeze(0))
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.append(probs)
            
            # Average probabilities across slices
            avg_probs = torch.stack(all_probs).mean(dim=0)
            predicted_idx = avg_probs.argmax(dim=1).item()
            confidence = avg_probs[0][predicted_idx].item()
            
            # Print results
            predicted_genre = idx_to_genre[predicted_idx]
            print(f"\nPredicted genre: {predicted_genre}")
            print(f"Confidence: {confidence:.2%}")
            
            # Print all genre probabilities
            print("\nAll genre probabilities:")
            for idx, prob in enumerate(avg_probs[0]):
                genre = idx_to_genre[idx]
                print(f"{genre}: {prob:.2%}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error occurred: {e}")
            continue

if __name__ == "__main__":
    main() 