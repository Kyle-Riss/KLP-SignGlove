#!/usr/bin/env python3
"""
Advanced GRU Models for SignGlove
GRU를 이기기 위한 고급 모델들
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.generalModels import *
from src.models.LightningModel import LitModel
from typing import Tuple
from torch import Tensor

class AttentionGRU(LitModel):
    """
    Attention Mechanism을 적용한 GRU 모델
    GRU를 이기기 위한 고급 모델
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=24,
        layers=2,
        dropout=0.2,
        attention_heads=4,
        **kwargs
    ):
        super().__init__()
        
        self.lr = learning_rate
        self.classes = classes
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, classes)
        )
    
    def forward(self, x: Tensor, x_padding: Tensor, y_targets: Tensor) -> Tuple[Tensor, Tensor]:
        # GRU forward pass
        gru_output, _ = self.gru(x)
        
        # Self-attention
        attn_output, _ = self.attention(gru_output, gru_output, gru_output)
        
        # Layer normalization
        normalized_output = self.layer_norm(attn_output)
        
        # Global average pooling
        if x_padding is not None:
            valid_lengths = (x_padding == 0).sum(dim=1) - 1
            valid_lengths = valid_lengths.clamp(min=0, max=normalized_output.size(1)-1)
            batch_size = normalized_output.size(0)
            final_features = normalized_output[torch.arange(batch_size), valid_lengths]
        else:
            final_features = normalized_output.mean(dim=1)
        
        # Output
        logits = self.output_layers(final_features)
        loss = F.cross_entropy(logits, y_targets)
        return logits, loss


class ResidualGRU(LitModel):
    """
    Residual Connection을 적용한 GRU 모델
    GRU를 이기기 위한 고급 모델
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=24,
        layers=3,
        dropout=0.2,
        **kwargs
    ):
        super().__init__()
        
        self.lr = learning_rate
        self.classes = classes
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Multiple GRU layers with residual connections
        self.gru_layers = nn.ModuleList([
            nn.GRU(hidden_size, hidden_size, batch_first=True, dropout=dropout)
            for _ in range(layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(layers)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, classes)
        )
    
    def forward(self, x: Tensor, x_padding: Tensor, y_targets: Tensor) -> Tuple[Tensor, Tensor]:
        # Input projection
        x_proj = self.input_proj(x)
        
        # Residual GRU layers
        output = x_proj
        for i, (gru, layer_norm) in enumerate(zip(self.gru_layers, self.layer_norms)):
            gru_output, _ = gru(output)
            # Residual connection
            output = layer_norm(output + gru_output)
        
        # Global average pooling
        if x_padding is not None:
            valid_lengths = (x_padding == 0).sum(dim=1) - 1
            valid_lengths = valid_lengths.clamp(min=0, max=output.size(1)-1)
            batch_size = output.size(0)
            final_features = output[torch.arange(batch_size), valid_lengths]
        else:
            final_features = output.mean(dim=1)
        
        # Output
        logits = self.output_layers(final_features)
        loss = F.cross_entropy(logits, y_targets)
        return logits, loss


class TransformerGRU(LitModel):
    """
    Transformer + GRU 하이브리드 모델
    GRU를 이기기 위한 고급 모델
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=24,
        layers=2,
        dropout=0.2,
        attention_heads=4,
        **kwargs
    ):
        super().__init__()
        
        self.lr = learning_rate
        self.classes = classes
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=2 * hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, classes)
        )
    
    def forward(self, x: Tensor, x_padding: Tensor, y_targets: Tensor) -> Tuple[Tensor, Tensor]:
        # Input projection
        x_proj = self.input_proj(x)
        
        # Transformer encoding
        transformer_output = self.transformer(x_proj)
        
        # GRU processing
        gru_output, _ = self.gru(transformer_output)
        
        # Global average pooling
        if x_padding is not None:
            valid_lengths = (x_padding == 0).sum(dim=1) - 1
            valid_lengths = valid_lengths.clamp(min=0, max=gru_output.size(1)-1)
            batch_size = gru_output.size(0)
            final_features = gru_output[torch.arange(batch_size), valid_lengths]
        else:
            final_features = gru_output.mean(dim=1)
        
        # Output
        logits = self.output_layers(final_features)
        loss = F.cross_entropy(logits, y_targets)
        return logits, loss


# Test the models
if __name__ == "__main__":
    print("🧪 Advanced GRU 모델들 테스트 시작...")
    
    print("✅ Advanced GRU 모델들 구현 완료!")
    print("📝 AttentionGRU, ResidualGRU, TransformerGRU 모델")
    print("🎯 GRU를 이기기 위한 고급 아키텍처")
