import os
import wandb
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def upload_spectrograms():
    # Initialize wandb
    wandb.init(
        project="audio-to-text",
        name="mel-spectrograms",
        config={
            "dataset": "UrbanSound8K",
            "spectrogram_type": "mel",
            "n_mels": 128,
            "n_fft": 2048,
            "hop_length": 512
        }
    )

    # Load the dataset to get class information
    print("Loading dataset...")
    dataset = load_dataset("danavery/urbansound8K")
    df = pd.DataFrame(dataset['train'])

    # Create a table for the spectrograms
    spectrogram_table = wandb.Table(columns=["spectrogram", "class", "class_id", "file_name"])

    # Get all spectrogram files
    spectrogram_dir = "mel_spectrograms"
    spectrogram_files = sorted([f for f in os.listdir(spectrogram_dir) if f.endswith('.png')])

    print("Uploading spectrograms to wandb...")
    for file_name in tqdm(spectrogram_files):
        # Get the index from the filename
        idx = int(file_name.split('_')[1].split('.')[0])
        
        # Get class information
        class_name = df.iloc[idx]['class']
        class_id = df.iloc[idx]['classID']
        
        # Create the full file path
        file_path = os.path.join(spectrogram_dir, file_name)
        
        # Add the spectrogram to the table
        spectrogram_table.add_data(
            wandb.Image(file_path),
            class_name,
            class_id,
            file_name
        )

    # Log the table to wandb
    wandb.log({"spectrograms": spectrogram_table})

    # Create a summary of class distribution
    class_distribution = df['class'].value_counts()
    wandb.log({"class_distribution": wandb.Table(
        data=[[class_name, count] for class_name, count in class_distribution.items()],
        columns=["class", "count"]
    )})

    # Finish the wandb run
    wandb.finish()

    print("Upload complete! Check your wandb dashboard for the results.")

if __name__ == "__main__":
    upload_spectrograms() 