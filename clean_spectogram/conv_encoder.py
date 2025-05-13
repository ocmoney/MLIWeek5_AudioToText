import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        # Add positional encoding to each sequence in the batch
        return x + self.pe[:x.size(1)].transpose(0, 1)  # [1, seq_len, d_model] -> [seq_len, 1, d_model]

class ConvEncoder(nn.Module):
    def __init__(self, 
                 input_channels=1,
                 conv_channels=128,
                 d_model=256,
                 nhead=4,
                 num_encoder_layers=3,
                 dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()
        
        # 1D Convolutional layer
        self.conv = nn.Conv1d(128, conv_channels, kernel_size=3, padding=1)  # Changed input_channels to 128 (n_mels)
        self.gelu = nn.GELU()
        
        # Projection to transformer dimension
        self.projection = nn.Linear(conv_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=345)  # Matches spectrogram time steps
        
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
    
    def forward(self, x):
        """
        Args:
            x: Input spectrogram tensor of shape [batch_size, channels, n_mels, time_steps]
        Returns:
            Encoded features of shape [batch_size, seq_len, d_model]
        """
        # Reshape for 1D convolution
        batch_size, channels, n_mels, time_steps = x.shape
        x = x.squeeze(1)  # Remove channel dimension since we're using n_mels as channels [batch_size, n_mels, time_steps]
        
        # Apply 1D convolution and GELU
        x = self.conv(x)
        x = self.gelu(x)
        
        # Project to transformer dimension
        x = x.transpose(1, 2)  # [batch_size, time_steps, conv_channels]
        x = self.projection(x)  # [batch_size, time_steps, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        return x