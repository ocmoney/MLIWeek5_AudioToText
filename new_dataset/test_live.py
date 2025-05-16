import torch
import torchaudio
import numpy as np
import librosa
import time
from genre_encoder import GenreEncoder, SLICE_FRAMES, N_MELS, SAMPLE_RATE, HOP_LENGTH
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
        spec_slice = mel_spec[:, start_idx:start_idx+SLICE_FRAMES]
    else:
        # Pad if shorter than 5 seconds
        pad_width = SLICE_FRAMES - mel_spec.shape[1]
        spec_slice = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    
    # Convert to tensor
    spec_tensor = torch.from_numpy(spec_slice).float()
    
    return spec_tensor

def plot_spectrogram(spec):
    """Plot the spectrogram for visualization."""
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower')
    plt.title('Mel Spectrogram')
    plt.ylabel('Mel Bins')
    plt.xlabel('Time')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = GenreEncoder(num_classes=7)  # 7 genres excluding Pop
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
            spec_tensor = prepare_inputs(audio)
            
            # Plot spectrogram
            plot_spectrogram(spec_tensor.numpy())
            
            # Move tensor to device
            spec_tensor = spec_tensor.to(device)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(spec_tensor.unsqueeze(0))
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