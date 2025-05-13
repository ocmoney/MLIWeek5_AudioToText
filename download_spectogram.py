import wandb
import os

# Ensure you're logged in (only needs to be done once per machine/environment)
wandb.login()

def download_spectrograms():
    run = wandb.init(project="audio-to-text", job_type="download_spectrograms")

    artifact = run.use_artifact(
        "olliecumming3-machine-learning-institute/audio-to-text/run-uyaqujc2-spectrograms:v0",
        type="run_table"
    )
    
    # This will download to a default path like './artifacts/...' — avoid using this
    artifact_dir = artifact.download()

    # We now move only PNGs into spectrogram_images
    spectrogram_dir = "spectrogram_images"
    os.makedirs(spectrogram_dir, exist_ok=True)

    png_files = []
    for root, _, files in os.walk(artifact_dir):
        for file in files:
            if file.endswith(".png"):
                full_src_path = os.path.join(root, file)
                dest_path = os.path.join(spectrogram_dir, file)
                if not os.path.exists(dest_path):
                    os.link(full_src_path, dest_path)
                    png_files.append(file)

    print(f"Copied {len(png_files)} new spectrograms to '{spectrogram_dir}'.")

download_spectrograms()


