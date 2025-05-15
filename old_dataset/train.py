import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SongSlicesDataset, SpectrogramTower # Import from model.py

# --- Configuration ---
METADATA_FILE = "metadata.csv"
SPECTROGRAM_DIR = "spectrograms"
EMBEDDING_DIM = 128
MARGIN = 1.0
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
SPECTROGRAM_CHANNELS = 1 # 1 for grayscale, 3 for RGB. Adjust if your spectrograms are different.
MODEL_SAVE_PATH = "spectrogram_tower_model.pth"

# --- Training Script ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Dataset and DataLoader
    print("Initializing dataset...")
    try:
        dataset = SongSlicesDataset(METADATA_FILE, SPECTROGRAM_DIR)
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        exit()
    except FileNotFoundError as e:
        print(f"Error: Could not find {e.filename}. Please check METADATA_FILE path.")
        exit()
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 

    # Initialize Model, Loss, Optimizer
    print("Initializing model...")
    # Pass SPECTROGRAM_CHANNELS to the model constructor
    tower = SpectrogramTower(embedding_dim=EMBEDDING_DIM, in_channels=SPECTROGRAM_CHANNELS).to(device)
    triplet_loss_fn = nn.TripletMarginLoss(margin=MARGIN, p=2)
    optimizer = optim.Adam(tower.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        tower.train()
        running_loss = 0.0
        num_batches = 0
        for i, (anchor_specs, positive_specs, negative_specs) in enumerate(dataloader):
            anchor_specs = anchor_specs.to(device)
            positive_specs = positive_specs.to(device)
            negative_specs = negative_specs.to(device)

            optimizer.zero_grad()

            anchor_embeddings = tower(anchor_specs)
            positive_embeddings = tower(positive_specs)
            negative_embeddings = tower(negative_specs)

            loss = triplet_loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches +=1
            
            if (i + 1) % 10 == 0: # Print every 10 batches
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed. Average Loss: {epoch_loss:.4f}")

    print("Training finished.")

    # Save the tower model
    try:
        torch.save(tower.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}") 