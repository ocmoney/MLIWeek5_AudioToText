import torch
import torch.nn as nn
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
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:x.size(1)].transpose(0, 1)

class ConvEncoder(nn.Module):
    def __init__(self, 
                 input_channels=1,
                 conv_channels=256,
                 d_model=512,
                 nhead=4,
                 num_encoder_layers=4,
                 dim_feedforward=1024,
                 dropout=0.1,
                 include_gru=True):
        super().__init__()
        # 1D Convolutional layer
        self.conv = nn.Conv1d(128, conv_channels, kernel_size=3, padding=1)  # n_mels=128
        self.gelu = nn.GELU()
        # Projection to transformer dimension
        self.projection = nn.Linear(conv_channels, d_model)
        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=1291)
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
        
        # GRU layer (only included if specified)
        self.include_gru = include_gru
        if include_gru:
            self.gru = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=2,
                batch_first=True,
                dropout=dropout if num_encoder_layers > 1 else 0,
                bidirectional=True
            )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gru=None):
        # x: [batch_size, channels, n_mels, time_steps]
        batch_size, channels, n_mels, time_steps = x.shape
        x = x.squeeze(1)  # [batch_size, n_mels, time_steps]
        x = self.conv(x)  # [batch_size, conv_channels, time_steps]
        x = self.gelu(x)
        x = x.transpose(1, 2)  # [batch_size, time_steps, conv_channels]
        x = self.projection(x)  # [batch_size, time_steps, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # [batch_size, time_steps, d_model]
        
        # Use provided GRU if available, otherwise use own GRU if included
        if gru is not None:
            x, _ = gru(x)  # [batch_size, time_steps, d_model*2]
            # Combine forward and backward GRU outputs
            x = x.view(batch_size, time_steps, 2, -1).mean(dim=2)  # [batch_size, time_steps, d_model]
        elif self.include_gru:
            x, _ = self.gru(x)  # [batch_size, time_steps, d_model*2]
            # Combine forward and backward GRU outputs
            x = x.view(batch_size, time_steps, 2, -1).mean(dim=2)  # [batch_size, time_steps, d_model]
        return x

class SongClassifier(nn.Module):
    def __init__(self, num_classes, nhead=4, num_encoder_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.encoder = ConvEncoder(
            input_channels=1,
            conv_channels=256,
            d_model=512,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.encoder(x)  # [batch, seq_len, d_model]
        x = x.mean(dim=1)    # Global average pooling over sequence
        x = self.classifier(x)
        return x

class SongEmbeddingNet(nn.Module):
    def __init__(self, nhead=4, num_encoder_layers=4, dim_feedforward=1024, dropout=0.1, d_model=512, temperature=0.07, normalize_embeddings=True):
        super().__init__()
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        
        # All towers have complete encoders without GRU
        self.anchor_encoder = ConvEncoder(
            input_channels=1,
            conv_channels=256,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            include_gru=False  # Disabled GRU
        )
        
        self.positive_encoder = ConvEncoder(
            input_channels=1,
            conv_channels=256,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            include_gru=False  # Disabled GRU
        )
        
        self.negative_encoder = ConvEncoder(
            input_channels=1,
            conv_channels=256,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            include_gru=False  # Disabled GRU
        )

        # Projection layers to ensure distinct representations
        self.positive_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.negative_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def normalize(self, x):
        # L2 normalize the embeddings
        return nn.functional.normalize(x, p=2, dim=1)

    def forward(self, anchor, positive=None, negative=None):
        if positive is None and negative is None:
            # Single input mode (for inference)
            x = self.anchor_encoder(anchor)
            x = x.mean(dim=1)    # Global average pooling over sequence
            if self.normalize_embeddings:
                x = self.normalize(x)  # L2 normalize
            return x  # [batch, d_model]
        else:
            # Triplet mode (for training)
            anchor_emb = self.anchor_encoder(anchor)
            positive_emb = self.positive_encoder(positive)
            negative_emb = self.negative_encoder(negative)
            
            # Apply distinct projections
            positive_emb = self.positive_projection(positive_emb)
            negative_emb = self.negative_projection(negative_emb)
            
            # Global average pooling
            anchor_emb = anchor_emb.mean(dim=1)
            positive_emb = positive_emb.mean(dim=1)
            negative_emb = negative_emb.mean(dim=1)
            
            # L2 normalize all embeddings if enabled
            if self.normalize_embeddings:
                anchor_emb = self.normalize(anchor_emb)
                positive_emb = self.normalize(positive_emb)
                negative_emb = self.normalize(negative_emb)
            
            # Scale by temperature
            anchor_emb = anchor_emb / self.temperature
            positive_emb = positive_emb / self.temperature
            negative_emb = negative_emb / self.temperature
            
            return anchor_emb, positive_emb, negative_emb  # Each [batch, d_model]
