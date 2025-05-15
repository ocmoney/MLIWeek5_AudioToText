import os
import torch
import numpy as np
from datasets import load_dataset
import csv

# Output directories
os.makedirs("spectrograms", exist_ok=True)

# Load dataset
dataset = load_dataset("teticio/audio-diffusion-instrumental-hiphop-256", split="train")

# Prepare metadata CSV
with open("metadata.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["song_id", "slice", "spectrogram_path"])

    audio_name_to_id = {}
    next_id = 0

    for i, sample in enumerate(dataset):
        # Removed the limit of 45 slices; process all slices in the dataset

        # Use 'audio_file' as the unique identifier
        audio_base_name = os.path.splitext(os.path.basename(sample["audio_file"]))[0]

        # Assign a unique id to each unique audio file name
        if audio_base_name not in audio_name_to_id:
            audio_name_to_id[audio_base_name] = next_id
            next_id += 1
        song_id = audio_name_to_id[audio_base_name]

        slice_idx = sample["slice"]

        # File name for spectrogram
        spectrogram_path = f"spectrograms/{song_id}_slice{slice_idx}.pt"

        # Convert PIL Image to numpy array, then to torch tensor
        spectrogram = torch.from_numpy(np.array(sample["image"]))
        torch.save(spectrogram, spectrogram_path)

        # Write metadata
        writer.writerow([song_id, slice_idx, spectrogram_path])

        print(f"Saved to local (remote GPU) memory: {spectrogram_path}, slice={slice_idx}")

    print("Done! All files and metadata saved.")
