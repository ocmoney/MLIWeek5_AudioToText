import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class ConvEncoder(nn.Module):
    def __init__(self, 
                 input_channels=1,
                 conv_channels=256,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        
        # 1D Convolutional layer
        self.conv = nn.Conv1d(input_channels, conv_channels, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        
        # Projection to transformer dimension
        self.projection = nn.Linear(conv_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def preprocess_spectrogram(self, spectrogram):
        """
        Preprocess the spectrogram by cropping out axes and labels
        Args:
            spectrogram: PIL Image or tensor of shape [batch_size, channels, height, width]
        Returns:
            Processed tensor of shape [batch_size, channels, height, width]
        """
        if isinstance(spectrogram, Image.Image):
            # Convert PIL image to tensor
            transform = transforms.ToTensor()
            spectrogram = transform(spectrogram)
        
        # Assuming the spectrogram has axes and labels that need to be cropped
        # You'll need to adjust these values based on your actual spectrogram format
        # This is a placeholder implementation
        batch_size, channels, height, width = spectrogram.shape
        cropped_height = height - 40  # Adjust based on your needs
        cropped_width = width - 40    # Adjust based on your needs
        
        # Center crop
        start_h = (height - cropped_height) // 2
        start_w = (width - cropped_width) // 2
        spectrogram = spectrogram[:, :, start_h:start_h + cropped_height, start_w:start_w + cropped_width]
        
        return spectrogram
    
    def forward(self, x):
        """
        Args:
            x: Input spectrogram tensor of shape [batch_size, channels, height, width]
        Returns:
            Encoded features of shape [batch_size, seq_len, d_model]
        """
        # Preprocess spectrogram
        x = self.preprocess_spectrogram(x)
        
        # Reshape for 1D convolution
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size, channels, -1)  # [batch_size, channels, height*width]
        
        # Apply 1D convolution and GELU
        x = self.conv(x)
        x = self.gelu(x)
        
        # Project to transformer dimension
        x = x.transpose(1, 2)  # [batch_size, seq_len, conv_channels]
        x = self.projection(x)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        return x

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms

    # Path to the mel_spectrograms_clean directory
    img_dir = "mel_spectrograms_clean"
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    if not img_files:
        print("No images found in mel_spectrograms_clean.")
        exit(1)
    img_path = os.path.join(img_dir, img_files[0])
    print(f"Using image: {img_path}")
    img = Image.open(img_path)
    img = img.convert("RGB")  # Ensure image is RGB, not RGBA

    # No cropping needed for these images
    img_cropped = img

    # Convert to tensor and add batch/channel dimensions
    transform = transforms.ToTensor()
    tensor = transform(img_cropped)  # [C, H, W]
    tensor = tensor.unsqueeze(0)     # [1, C, H, W]

    # Instantiate encoder
    encoder = ConvEncoder(input_channels=3)  # 3 for RGB, 1 for grayscale
    encoder.eval()

    # Run through encoder
    with torch.no_grad():
        output = encoder(tensor)
    print(f"Encoder output shape: {output.shape}")
    # Optionally, print a summary of the output
    print(f"Encoder output (first 1x5x5 block):\n{output[0, :5, :5]}")

    # Show the image for debugging
    plt.imshow(img_cropped)
    plt.title("Example Mel Spectrogram Image")
    plt.axis('off')
    plt.show()