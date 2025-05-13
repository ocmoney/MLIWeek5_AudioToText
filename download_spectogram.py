import wandb
import os

# Authenticate (ensure you have run `wandb login` in your terminal previously)
wandb.login()

# Specify the artifact path
artifact_path = "olliecumming3-machine-learning-institute/audio-to-text/run-uyaqujc2-spectrograms:v0"

# Download function
def download_spectrograms():
    # Initialize the wandb run object
    run = wandb.init(project="audio-to-text", job_type="download_spectrograms")

    # Get the artifact
    artifact = run.use_artifact(artifact_path, type="run_table")

    # Download the artifact
    artifact_dir = artifact.download()

    # Create a directory for spectrograms if it doesn't exist
    spectrogram_dir = "spectrogram_images"
    if not os.path.exists(spectrogram_dir):
        os.makedirs(spectrogram_dir)

    # Get the list of files from the artifact
    artifact_files = artifact.files()

    # Keep track of downloaded files
    downloaded_files = []

    # Download files (skip printing each file if already downloaded)
    for file in artifact_files:
        # Check if the file is a spectrogram image
        if file.name.endswith('.png'):
            # Construct the path where the file will be saved in spectrogram_images
            file_path = os.path.join(spectrogram_dir, file.name.split('/')[-1])

            # Only download the file if it doesn't exist yet
            if not os.path.exists(file_path):
                file.download(root=spectrogram_dir)
                print(f"Downloaded: {file.name} to {file_path}")
                downloaded_files.append(file.name)  # Keep track of downloaded files
            else:
                print(f"File already exists: {file.name}, skipping download.")

    if not downloaded_files:
        print("No new spectrograms to download.")
    else:
        print("All new spectrograms have been downloaded!")

# Run the download function
download_spectrograms()

