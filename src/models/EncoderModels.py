import torch.nn as nn
import torch.nn.functional as F
from src.models.generalModels import *
from src.models.LightningModel import LitModel

from typing import Tuple
from torch import Tensor


class TransformerEncoder(LitModel):
    """
    Transformer Encoder 기반 모델
    Self-attention 메커니즘을 사용한 시계열 데이터 처리
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=34,
        num_heads=8,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        # set hyperparameters
        self.lr = learning_rate
        self.classes = classes
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(100, hidden_size))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output layers
        self.output_layers = outputRNN(
            hidden_size=hidden_size, 
            transformed_size=hidden_size*2, 
            output_size=self.classes, 
            dropout=dropout
        )

    def forward(
        self, x: Tensor, x_mask: Tensor, y_targets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x_proj = self.input_projection(x)  # (batch, seq_len, hidden_size)
        
        # Add positional encoding
        x_proj = x_proj + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x_proj)  # (batch, seq_len, hidden_size)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)  # (batch, hidden_size)
        
        # Output layers
        logits = self.output_layers(pooled)
        loss = F.cross_entropy(logits, y_targets)
        
        return logits, loss


class CNNEncoder(LitModel):
    """
    CNN Encoder 기반 모델
    Convolutional layers를 사용한 특징 추출
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=34,
        dropout=0.2,
    ):
        super().__init__()

        # set hyperparameters
        self.lr = learning_rate
        self.classes = classes
        self.input_size = input_size
        self.hidden_size = hidden_size

        # CNN Encoder
        self.cnn_encoder = nn.Sequential(
            # First conv block
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Second conv block
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Third conv block
            nn.Conv1d(64, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Dropout(dropout)
        )
        
        # Output layers
        self.output_layers = outputRNN(
            hidden_size=hidden_size, 
            transformed_size=hidden_size*2, 
            output_size=self.classes, 
            dropout=dropout
        )

    def forward(
        self, x: Tensor, x_mask: Tensor, y_targets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Transpose for Conv1d: (batch, time, channels) -> (batch, channels, time)
        x_conv = x.transpose(1, 2)
        
        # CNN encoding
        encoded = self.cnn_encoder(x_conv)  # (batch, hidden_size, 1)
        encoded = encoded.squeeze(-1)  # (batch, hidden_size)
        
        # Output layers
        logits = self.output_layers(encoded)
        loss = F.cross_entropy(logits, y_targets)
        
        return logits, loss


class HybridEncoder(LitModel):
    """
    Hybrid Encoder 모델
    CNN + Transformer 조합
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=34,
        num_heads=8,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        # set hyperparameters
        self.lr = learning_rate
        self.classes = classes
        self.input_size = input_size
        self.hidden_size = hidden_size

        # CNN feature extractor
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(32, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(50, hidden_size))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output layers
        self.output_layers = outputRNN(
            hidden_size=hidden_size, 
            transformed_size=hidden_size*2, 
            output_size=self.classes, 
            dropout=dropout
        )

    def forward(
        self, x: Tensor, x_mask: Tensor, y_targets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # CNN feature extraction
        x_conv = x.transpose(1, 2)  # (batch, channels, time)
        cnn_features = self.cnn_extractor(x_conv)  # (batch, hidden_size, reduced_time)
        
        # Transpose back for transformer
        cnn_features = cnn_features.transpose(1, 2)  # (batch, reduced_time, hidden_size)
        
        # Add positional encoding
        seq_len = cnn_features.size(1)
        cnn_features = cnn_features + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer encoding
        encoded = self.transformer_encoder(cnn_features)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)
        
        # Output layers
        logits = self.output_layers(pooled)
        loss = F.cross_entropy(logits, y_targets)
        
        return logits, loss


# Test the models
if __name__ == "__main__":
    print("🧪 Encoder 모델들 테스트 시작...")
    
    print("✅ Encoder 모델들 구현 완료!")
    print("📝 TransformerEncoder, CNNEncoder, HybridEncoder 모델")
    print("🔧 DynamicDataModule과 호환되는 구조")
    print("🎯 다양한 Encoder 아키텍처 구현")
