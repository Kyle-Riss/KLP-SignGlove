import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CNN1D(nn.Module):
    """
    1D CNN feature extractor for time series data
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


class CNNGRUHybrid(nn.Module):
    """
    CNN-GRU Hybrid model for time series classification
    논문 기반: CNN으로 특징 추출 후 GRU로 시계열 모델링
    
    최적화된 하이브리드 아키텍처:
    - 1D CNN: 센서 간 상관관계 및 로컬 패턴 학습
    - GRU: 시간적 의존성 및 시퀀스 모델링 (LSTM보다 효율적)
    - 양방향 옵션: 과거+미래 정보 모두 활용
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
        dense_size: int = 128,
        bidirectional: bool = False
    ):
        super(CNNGRUHybrid, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.dense_size = dense_size
        self.bidirectional = bidirectional
        
        # CNN feature extractor
        self.cnn = CNN1D(
            input_channels=input_channels,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout
        )
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=self.cnn.output_channels,
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
            x: Input tensor of shape (batch_size, time_steps, input_channels)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch_size, cnn_channels, reduced_time_steps)
        
        # Transpose back for GRU: (batch, time, features)
        cnn_out = cnn_out.transpose(1, 2)  # (batch_size, reduced_time_steps, cnn_channels)
        
        # GRU forward pass
        gru_out, _ = self.gru(cnn_out)  # (batch_size, reduced_time_steps, hidden_size)
        
        # Take the last timestep output
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        dropped = self.dropout_layer(last_output)
        
        # Dense layers
        dense_out = self.relu(self.dense(dropped))
        output = self.output_layer(dense_out)
        
        return output


class CNNLSTMHybrid(nn.Module):
    """
    CNN-LSTM Hybrid model for time series classification
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
        dense_size: int = 128,
        bidirectional: bool = False
    ):
        super(CNNLSTMHybrid, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.dense_size = dense_size
        self.bidirectional = bidirectional
        
        # CNN feature extractor
        self.cnn = CNN1D(
            input_channels=input_channels,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.cnn.output_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
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
            x: Input tensor of shape (batch_size, time_steps, input_channels)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch_size, cnn_channels, reduced_time_steps)
        
        # Transpose back for LSTM: (batch, time, features)
        cnn_out = cnn_out.transpose(1, 2)  # (batch_size, reduced_time_steps, cnn_channels)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(cnn_out)  # (batch_size, reduced_time_steps, hidden_size)
        
        # Take the last timestep output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        dropped = self.dropout_layer(last_output)
        
        # Dense layers
        dense_out = self.relu(self.dense(dropped))
        output = self.output_layer(dense_out)
        
        return output


class CNNTransformerHybrid(nn.Module):
    """
    CNN-Transformer Hybrid model for time series classification
    """
    
    def __init__(
        self,
        input_channels: int = 8,
        cnn_channels: list = [16, 32, 64],
        kernel_sizes: list = [3, 3, 3],
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        num_classes: int = 24,
        dropout: float = 0.2,
        dense_size: int = 128
    ):
        super(CNNTransformerHybrid, self).__init__()
        
        self.input_channels = input_channels
        self.d_model = d_model
        self.nhead = nhead
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
        
        # Projection to d_model
        self.projection = nn.Linear(self.cnn.output_channels, d_model)
        
        # Transformer encoder
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
            x: Input tensor of shape (batch_size, time_steps, input_channels)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch_size, cnn_channels, reduced_time_steps)
        
        # Transpose back for transformer: (batch, time, features)
        cnn_out = cnn_out.transpose(1, 2)  # (batch_size, reduced_time_steps, cnn_channels)
        
        # Project to d_model
        projected = self.projection(cnn_out)  # (batch_size, reduced_time_steps, d_model)
        
        # Transformer forward pass
        transformer_out = self.transformer(projected)  # (batch_size, reduced_time_steps, d_model)
        
        # Global average pooling
        pooled = torch.mean(transformer_out, dim=1)  # (batch_size, d_model)
        
        # Apply dropout
        dropped = self.dropout_layer(pooled)
        
        # Dense layers
        dense_out = self.relu(self.dense(dropped))
        output = self.output_layer(dense_out)
        
        return output


# Test the models
if __name__ == "__main__":
    print("🧪 CNN-Hybrid 모델들 테스트 시작...")
    
    # Test input
    batch_size = 16
    time_steps = 100
    input_channels = 8
    x = torch.randn(batch_size, time_steps, input_channels)
    
    models = [
        ("CNN-GRU Hybrid (Unidirectional)", CNNGRUHybrid(
            input_channels=8,
            cnn_channels=[16, 32, 64],
            hidden_size=64,
            num_layers=2,
            num_classes=24,
            dropout=0.2,
            dense_size=128,
            bidirectional=False
        )),
        ("CNN-GRU Hybrid (Bidirectional)", CNNGRUHybrid(
            input_channels=8,
            cnn_channels=[16, 32, 64],
            hidden_size=64,
            num_layers=2,
            num_classes=24,
            dropout=0.2,
            dense_size=128,
            bidirectional=True
        )),
        ("CNN-LSTM Hybrid (Reference)", CNNLSTMHybrid(
            input_channels=8,
            cnn_channels=[16, 32, 64],
            hidden_size=64,
            num_layers=2,
            num_classes=24,
            dropout=0.2,
            dense_size=128,
            bidirectional=False
        )),
        ("CNN-Transformer Hybrid (Reference)", CNNTransformerHybrid(
            input_channels=8,
            cnn_channels=[16, 32, 64],
            d_model=64,
            nhead=8,
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
        
        # CNN-GRU 모델의 효율성 강조
        if "CNN-GRU" in name:
            print(f"  ✅ GRU 기반 - 효율적이고 빠른 처리!")
    
    print("\n🎯 CNN-GRU Hybrid 모델 최적화 완료! (LSTM 대비 19% 적은 파라미터)")
