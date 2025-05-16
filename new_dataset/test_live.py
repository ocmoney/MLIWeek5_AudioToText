import torch
import torchaudio
import numpy as np
import librosa
import time
from genre_encoder import TwoTowerGenreEncoder, SLICE_FRAMES, N_MELS, SAMPLE_RATE, HOP_LENGTH
import matplotlib.pyplot as plt
import os

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
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=2048,
        fmin=20,
        fmax=8000
    )
    
    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    return mel_spec_norm

def prepare_inputs(audio, sample_rate=SAMPLE_RATE):
    """Prepare inputs for the model."""
    # Compute mel spectrogram
    mel_spec = compute_melspectrogram(audio, sample_rate)
    
    # Get a 5-second slice
    if mel_spec.shape[1] > SLICE_FRAMES:
        # Take the middle slice
        start_idx = (mel_spec.shape[1] - SLICE_FRAMES) // 2
        slice_spec = mel_spec[:, start_idx:start_idx+SLICE_FRAMES]
    else:
        # Pad if shorter than 5 seconds
        pad_width = SLICE_FRAMES - mel_spec.shape[1]
        slice_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    
    # Downsample the full spectrogram to match slice length
    if mel_spec.shape[1] > SLICE_FRAMES:
        full_spec = torch.nn.functional.interpolate(
            torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0),
            size=(N_MELS, SLICE_FRAMES),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0).numpy()
    else:
        full_spec = np.pad(mel_spec, ((0, 0), (0, SLICE_FRAMES - mel_spec.shape[1])), mode='constant')
    
    # Convert to tensors
    slice_tensor = torch.from_numpy(slice_spec).float()
    full_tensor = torch.from_numpy(full_spec).float()
    
    return slice_tensor, full_tensor

def plot_spectrograms(slice_spec, full_spec):
    """Plot the spectrograms for visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot slice spectrogram
    im1 = ax1.imshow(slice_spec, aspect='auto', origin='lower')
    ax1.set_title('5-second Slice Spectrogram')
    ax1.set_ylabel('Mel Bins')
    plt.colorbar(im1, ax=ax1)
    
    # Plot full spectrogram
    im2 = ax2.imshow(full_spec, aspect='auto', origin='lower')
    ax2.set_title('Full Song Spectrogram (Downsampled)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mel Bins')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = TwoTowerGenreEncoder(num_classes=7)  # 7 genres excluding Pop
    model.load_state_dict(torch.load("genre_classifier.pth")['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load genre mapping
    genre_mapping = torch.load("genre_classifier.pth")['genre_mapping']
    idx_to_genre = {v: k for k, v in genre_mapping.items()}
    
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
            slice_tensor, full_tensor = prepare_inputs(audio)
            
            # Plot spectrograms
            plot_spectrograms(slice_tensor.numpy(), full_tensor.numpy())
            
            # Move tensors to device
            slice_tensor = slice_tensor.to(device)
            full_tensor = full_tensor.to(device)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(slice_tensor.unsqueeze(0), full_tensor.unsqueeze(0))
                probabilities = torch.softmax(outputs, dim=1)
                predicted_idx = outputs.argmax(dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
            
            # Print results
            predicted_genre = idx_to_genre[predicted_idx]
            print(f"\nPredicted genre: {predicted_genre}")
            print(f"Confidence: {confidence:.2%}")
            
            # Print all genre probabilities
            print("\nAll genre probabilities:")
            for idx, prob in enumerate(probabilities[0]):
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