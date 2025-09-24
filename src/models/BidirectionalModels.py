import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CNN1D(nn.Module):
    """
    1D CNN feature extractor for time series data
    (Imported from CNNHybrid for consistency)
    """
    
    def __init__(
        self,
        input_channels: int = 8,
        cnn_channels: list = [16, 32, 64],
        kernel_sizes: list = [3, 3, 3],
        dropout: float = 0.2
    ):
        super(CNN1D, self).__init__()
        
        self.input_channels = input_channels
        self.cnn_channels = cnn_channels
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        
        # CNN layers
        layers = []
        in_channels = input_channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_channels, kernel_sizes)):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*layers)
        self.output_channels = cnn_channels[-1]
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_channels)
        Returns:
            Output tensor of shape (batch_size, output_channels, reduced_time_steps)
        """
        # Transpose for 1D convolution: (batch, channels, time)
        x = x.transpose(1, 2)  # (batch_size, input_channels, time_steps)
        
        # Apply CNN
        x = self.cnn(x)  # (batch_size, output_channels, reduced_time_steps)
        
        return x


class BidirectionalGRU(nn.Module):
    """
    Bidirectional GRU model for time series classification
    """
    
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 24,
        dropout: float = 0.2,
        dense_size: int = 128
    ):
        super(BidirectionalGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.dense_size = dense_size
        
        # Bidirectional GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Bidirectional
        )
        
        # GRU output size is doubled due to bidirectional
        gru_output_size = hidden_size * 2
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Dense layers
        self.dense = nn.Linear(gru_output_size, dense_size)
        self.output_layer = nn.Linear(dense_size, num_classes)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Bidirectional GRU forward pass
        gru_out, _ = self.gru(x)  # (batch_size, time_steps, hidden_size * 2)
        
        # Take the last timestep output
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # Apply dropout
        dropped = self.dropout_layer(last_output)
        
        # Dense layers
        dense_out = self.relu(self.dense(dropped))
        output = self.output_layer(dense_out)
        
        return output


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM model for time series classification
    """
    
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 24,
        dropout: float = 0.2,
        dense_size: int = 128
    ):
        super(BidirectionalLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.dense_size = dense_size
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Bidirectional
        )
        
        # LSTM output size is doubled due to bidirectional
        lstm_output_size = hidden_size * 2
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Dense layers
        self.dense = nn.Linear(lstm_output_size, dense_size)
        self.output_layer = nn.Linear(dense_size, num_classes)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Bidirectional LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, time_steps, hidden_size * 2)
        
        # Take the last timestep output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # Apply dropout
        dropped = self.dropout_layer(last_output)
        
        # Dense layers
        dense_out = self.relu(self.dense(dropped))
        output = self.output_layer(dense_out)
        
        return output


class BidirectionalEncoder(nn.Module):
    """
    Bidirectional Transformer Encoder model for time series classification
    """
    
    def __init__(
        self,
        input_size: int = 8,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        num_classes: int = 24,
        dropout: float = 0.2,
        dense_size: int = 128
    ):
        super(BidirectionalEncoder, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.dense_size = dense_size
        
        # Input projection to d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder (inherently bidirectional)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Dense layers
        self.dense = nn.Linear(d_model, dense_size)
        self.output_layer = nn.Linear(dense_size, num_classes)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Project input to d_model
        x = self.input_projection(x)  # (batch_size, time_steps, d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoder forward pass
        transformer_out = self.transformer(x)  # (batch_size, time_steps, d_model)
        
        # Global average pooling
        pooled = torch.mean(transformer_out, dim=1)  # (batch_size, d_model)
        
        # Apply dropout
        dropped = self.dropout_layer(pooled)
        
        # Dense layers
        dense_out = self.relu(self.dense(dropped))
        output = self.output_layer(dense_out)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class MultiHeadAttentionGRU(nn.Module):
    """
    Multi-Head Attention + GRU hybrid model
    """
    
    def __init__(
        self,
        input_size: int = 8,
        d_model: int = 64,
        nhead: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 24,
        dropout: float = 0.2,
        dense_size: int = 128,
        bidirectional: bool = True
    ):
        super(MultiHeadAttentionGRU, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.dense_size = dense_size
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Projection for GRU
        self.gru_projection = nn.Linear(d_model, hidden_size)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Calculate GRU output size
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Dense layers
        self.dense = nn.Linear(gru_output_size, dense_size)
        self.output_layer = nn.Linear(dense_size, num_classes)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Project input
        x = self.input_projection(x)  # (batch_size, time_steps, d_model)
        
        # Multi-head attention
        attn_out, _ = self.multihead_attn(x, x, x)  # (batch_size, time_steps, d_model)
        
        # Residual connection and layer norm
        x = self.layer_norm(x + attn_out)
        
        # Project for GRU
        x = self.gru_projection(x)  # (batch_size, time_steps, hidden_size)
        
        # GRU forward pass
        gru_out, _ = self.gru(x)  # (batch_size, time_steps, hidden_size * 2)
        
        # Take the last timestep output
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # Apply dropout
        dropped = self.dropout_layer(last_output)
        
        # Dense layers
        dense_out = self.relu(self.dense(dropped))
        output = self.output_layer(dense_out)
        
        return output


class BidirectionalCNNGRUHybrid(nn.Module):
    """
    CNN + Bidirectional GRU Hybrid model
    최적화된 하이브리드: CNN 특징 추출 + 양방향 GRU 시계열 모델링
    """
    
    def __init__(
        self,
        input_channels: int = 8,
        cnn_channels: list = [16, 32, 64],
        kernel_sizes: list = [3, 3, 3],
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 24,
        dropout: float = 0.2,
        dense_size: int = 128
    ):
        super(BidirectionalCNNGRUHybrid, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.dense_size = dense_size
        
        # CNN feature extractor
        self.cnn = CNN1D(
            input_channels=input_channels,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout
        )
        
        # Bidirectional GRU layers
        self.gru = nn.GRU(
            input_size=self.cnn.output_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Always bidirectional
        )
        
        # GRU output size is doubled due to bidirectional
        gru_output_size = hidden_size * 2
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Dense layers
        self.dense = nn.Linear(gru_output_size, dense_size)
        self.output_layer = nn.Linear(dense_size, num_classes)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_channels)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch_size, cnn_channels, reduced_time_steps)
        
        # Transpose back for GRU: (batch, time, features)
        cnn_out = cnn_out.transpose(1, 2)  # (batch_size, reduced_time_steps, cnn_channels)
        
        # Bidirectional GRU forward pass
        gru_out, _ = self.gru(cnn_out)  # (batch_size, reduced_time_steps, hidden_size * 2)
        
        # Take the last timestep output
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # Apply dropout
        dropped = self.dropout_layer(last_output)
        
        # Dense layers
        dense_out = self.relu(self.dense(dropped))
        output = self.output_layer(dense_out)
        
        return output


# Test the models
if __name__ == "__main__":
    print("🧪 Bidirectional 모델들 테스트 시작...")
    
    # Test input
    batch_size = 16
    time_steps = 100
    input_size = 8
    x = torch.randn(batch_size, time_steps, input_size)
    
    models = [
        ("Bidirectional GRU", BidirectionalGRU(
            input_size=8,
            hidden_size=64,
            num_layers=2,
            num_classes=24,
            dropout=0.2,
            dense_size=128
        )),
        ("Bidirectional LSTM", BidirectionalLSTM(
            input_size=8,
            hidden_size=64,
            num_layers=2,
            num_classes=24,
            dropout=0.2,
            dense_size=128
        )),
        ("Bidirectional Encoder", BidirectionalEncoder(
            input_size=8,
            d_model=64,
            nhead=8,
            num_layers=2,
            num_classes=24,
            dropout=0.2,
            dense_size=128
        )),
        ("Multi-Head Attention GRU", MultiHeadAttentionGRU(
            input_size=8,
            d_model=64,
            nhead=8,
            hidden_size=64,
            num_layers=2,
            num_classes=24,
            dropout=0.2,
            dense_size=128,
            bidirectional=True
        )),
        ("🚀 Bidirectional CNN-GRU Hybrid", BidirectionalCNNGRUHybrid(
            input_channels=8,
            cnn_channels=[16, 32, 64],
            hidden_size=64,
            num_layers=2,
            num_classes=24,
            dropout=0.2,
            dense_size=128
        ))
    ]
    
    for i, (name, model) in enumerate(models, 1):
        print(f"\n{i}. {name} 테스트:")
        output = model(x)
        print(f"  입력 shape: {x.shape}")
        print(f"  출력 shape: {output.shape}")
        print(f"  모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        # CNN-GRU Hybrid 모델의 특별한 장점 강조
        if "CNN-GRU Hybrid" in name:
            print(f"  🎯 최적화된 하이브리드: CNN 특징추출 + 양방향 GRU!")
            print(f"  ⚡ 효율적: LSTM보다 빠르고 Transformer보다 가벼움!")
    
    print("\n🎯 Bidirectional CNN-GRU Hybrid 모델 추가 완료!")
