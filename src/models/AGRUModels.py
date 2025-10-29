"""
Standalone A-GRU Models
Amygdala-Boosted GRU를 사용한 독자적인 분류 모델
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor

from src.models.LightningModel import LitModel
from src.models.AmygdalaGRU import StackedAGRU


class AGRUModel(LitModel):
    """단순 A-GRU 모델: 입력 → A-GRU → Classifier"""
    
    def __init__(self, learning_rate, input_size=8, hidden_size=64, classes=24, layers=2, dropout=0.3, gamma=1.0, **kwargs):
        super().__init__()
        self.lr = learning_rate
        self.classes = classes
        
        self.agru = StackedAGRU(input_size=input_size, hidden_size=hidden_size, num_layers=layers, gamma=gamma, dropout=dropout)
        self.output_layers = nn.Sequential(nn.Linear(hidden_size, 2*hidden_size), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2*hidden_size, classes))
    
    def forward(self, x: Tensor, x_padding: Tensor, y_targets: Tensor) -> Tuple[Tensor, Tensor]:
        outputs, h_n, importances = self.agru(x)
        
        if x_padding is not None:
            valid_lengths = (x_padding == 0).sum(dim=1) - 1
            valid_lengths = valid_lengths.clamp(min=0, max=outputs.size(1)-1)
            batch_size = outputs.size(0)
            final_features = outputs[torch.arange(batch_size), valid_lengths]
        else:
            final_features = outputs[:, -1, :]
        
        logits = self.output_layers(final_features)
        loss = F.cross_entropy(logits, y_targets)
        return logits, loss


if __name__ == "__main__":
    print("🧪 A-GRU Model 테스트...")
    x, x_pad, y = torch.randn(4, 87, 8), torch.zeros(4, 87), torch.randint(0, 24, (4,))
    model = AGRUModel(learning_rate=0.001)
    logits, loss = model(x, x_pad, y)
    print(f"✅ Output: {logits.shape}, Loss: {loss:.4f}, Params: {sum(p.numel() for p in model.parameters()):,}")
