import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StackedGRU(nn.Module):
    """
    SignsSpeak 논문 기반 Stacked GRU 모델
    
    아키텍처:
    - 입력: (batch_size, time_steps, channels)
    - 2개의 GRU 레이어 (hidden_size=64)
    - Dropout (0.2)
    - Dense 레이어 (128)
    - 출력: (batch_size, num_classes)
    """
    
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 24,
        dropout: float = 0.2,
        dense_size: int = 128,
        bidirectional: bool = False
    ):
        super(StackedGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.dense_size = dense_size
        self.bidirectional = bidirectional
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
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
        # GRU forward pass
        gru_out, _ = self.gru(x)  # (batch_size, time_steps, hidden_size)
        
        # Take the last timestep output
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        dropped = self.dropout_layer(last_output)
        
        # Dense layers
        dense_out = self.relu(self.dense(dropped))
        output = self.output_layer(dense_out)
        
        return output


# Test the model
if __name__ == "__main__":
    print("🧪 StackedGRU 모델 테스트 시작...")
    
    # Test model creation
    model = StackedGRU(
        input_size=8,
        hidden_size=64,
        num_layers=2,
        num_classes=24,
        dropout=0.2,
        dense_size=128
    )
    
    # Test forward pass
    batch_size = 16
    time_steps = 100
    input_size = 8
    
    x = torch.randn(batch_size, time_steps, input_size)
    output = model(x)
    
    print(f"✅ 모델 테스트 성공!")
    print(f"  입력 shape: {x.shape}")
    print(f"  출력 shape: {output.shape}")
    print(f"  모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n🎯 SignsSpeak 논문 기반 Stacked GRU 모델 구현 완료!")
