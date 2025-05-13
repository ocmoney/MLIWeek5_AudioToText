import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from conv_encoder import ConvEncoder

# --- Dataset ---
class MelSpectrogramDataset(Dataset):
    def __init__(self, data_dir, files, class_to_idx, target_length=128):
        self.data_dir = data_dir
        self.files = files
        self.class_to_idx = class_to_idx
        self.target_length = target_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        
        # Load the mel spectrogram array
        mel_spec = np.load(file_path)
        
        # Ensure consistent time dimension
        if mel_spec.shape[1] > self.target_length:
            # Center crop if too long
            start = (mel_spec.shape[1] - self.target_length) // 2
            mel_spec = mel_spec[:, start:start + self.target_length]
        elif mel_spec.shape[1] < self.target_length:
            # Pad if too short
            pad_width = self.target_length - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        
        # Convert to tensor and add channel dimension
        mel_spec = torch.from_numpy(mel_spec).float()
        mel_spec = mel_spec.unsqueeze(0)  # Add channel dimension [1, n_mels, time]
        
        # Get label from filename
        label_name = file_name.split('_')[2]  # e.g., car_horn
        label = self.class_to_idx[label_name]
        
        return mel_spec, label

# --- Model ---
class ConvEncoderClassifier(nn.Module):
    def __init__(self, num_classes=10, nhead=4, num_encoder_layers=3, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.encoder = ConvEncoder(
            input_channels=1,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)    # Global average pooling over sequence
        x = self.classifier(x)
        return x

def train(config=None):
    with wandb.init(config=config):
        # If called by wandb.agent, config will be set
        config = wandb.config
        
        # --- Prepare data ---
        data_dir = "mel_spectrograms_clean"
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        random.shuffle(all_files)

        # Extract all class names
        class_names = sorted(list({f.split('_')[2] for f in all_files}))
        class_to_idx = {cls: i for i, cls in enumerate(class_names)}

        # Split 90% train, 10% test
        split_idx = int(0.9 * len(all_files))
        train_files = all_files[:split_idx]
        test_files = all_files[split_idx:]

        # Create datasets
        train_dataset = MelSpectrogramDataset(data_dir, train_files, class_to_idx, target_length=345)
        test_dataset = MelSpectrogramDataset(data_dir, test_files, class_to_idx, target_length=345)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

        # Initialize model with sweep parameters
        model = ConvEncoderClassifier(
            num_classes=len(class_names),
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Create directory for model checkpoints
        os.makedirs('checkpoints', exist_ok=True)
        best_test_acc = 0.0

        # Training loop
        for epoch in range(1, config.epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for mel_specs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}"):
                mel_specs, labels = mel_specs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(mel_specs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * mel_specs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
            train_acc = correct / total
            train_loss = running_loss / total
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Evaluation
            model.eval()
            correct = 0
            total = 0
            test_loss = 0.0
            
            with torch.no_grad():
                for mel_specs, labels in test_loader:
                    mel_specs, labels = mel_specs.to(device), labels.to(device)
                    outputs = model(mel_specs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * mel_specs.size(0)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
                    
            test_acc = correct / total
            test_loss = test_loss / total
            print(f"Epoch {epoch}: Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_accuracy': test_acc,
                    'train_accuracy': train_acc,
                    'test_loss': test_loss,
                    'train_loss': train_loss,
                    'config': dict(config)
                }
                torch.save(checkpoint, f'checkpoints/best_model.pt')
                print(f"New best model saved with test accuracy: {test_acc:.4f}")

            # Log metrics
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "best_test_accuracy": best_test_acc
            })

            # Clear GPU memory
            torch.cuda.empty_cache()

if __name__ == "__main__":
    from sweep_config import sweep_config, init_sweep, initial_params
    
    # Initialize sweep
    sweep_id, initial_params = init_sweep()
    
    # First run with known good parameters
    print("Starting with known good parameters:")
    for key, value in initial_params.items():
        print(f"{key}: {value}")
    train(initial_params)
    
    # Then start the sweep for optimization
    print("\nStarting hyperparameter sweep...")
    wandb.agent(sweep_id, function=train, count=19)  # Run 19 more trials (20 total including initial) 