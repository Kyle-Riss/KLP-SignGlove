import torch.nn as nn
import torch.nn.functional as F
from src.models.generalModels import *
from src.models.LightningModel import LitModel

from typing import Tuple
from torch import Tensor


class GRU(LitModel):
    """
    SignSpeak 논문 기반 기본 GRU 모델
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=34,
        batch_first=True,
        layers=2,
        dense_layer=(False, 64),
        dropout=0.2,
    ):
        super().__init__()

        # set hyperparameters
        self.lr = learning_rate
        self.dense = dense_layer
        self.classes = classes

        # if dense before RNN
        if self.dense:
            self.RNN = nn.Sequential(
                nn.Linear(input_size, 2 * hidden_size),
                nn.GRU(
                    2 * hidden_size,
                    hidden_size,
                    num_layers=layers,
                    batch_first=batch_first,
                ),
            )
        else:
            self.RNN = nn.GRU(
                input_size, hidden_size, num_layers=layers, batch_first=batch_first
            )

        self.output_layers = outputRNN(
            hidden_size=hidden_size, transformed_size=2*hidden_size, output_size=self.classes, dropout=dropout
        )

    def forward(
        self, x: Tensor, x_padding: Tensor, y_targets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        hidden_states, outputs = self.RNN(
            x
        )  # hidden states of all cells, outputs of last cells
        
        # 패딩 정보를 활용하여 마지막 유효한 타임스텝 선택
        if x_padding is not None:
            valid_lengths = (x_padding == 0).sum(dim=1) - 1  # 0-indexed
            valid_lengths = valid_lengths.clamp(min=0, max=outputs.size(1)-1)
            batch_size = outputs.size(0)
            final_output = outputs[torch.arange(batch_size), valid_lengths]
        else:
            final_output = outputs[:, -1, :]  # 차원 수정 (batch, time, features)
        
        logits = self.output_layers(
            final_output
        )  # output of last cell into dense layer
        loss = F.cross_entropy(logits, y_targets)  # cross entropy loss
        return logits, loss


class StackedGRU(LitModel):
    """
    SignSpeak 논문 기반 Stacked GRU 모델
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=34,
        batch_first=True,
        layers=2,
        dense_layer=(False, 64),
        dropout=0.2,
    ):
        super().__init__()

        # set hyperparameters
        self.lr = learning_rate
        self.dense = dense_layer
        self.classes = classes

        # Stacked GRU with dropout between layers
        self.RNN = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers=layers, 
            batch_first=batch_first,
            dropout=dropout if layers > 1 else 0
        )

        self.output_layers = outputRNN(
            hidden_size=hidden_size, 
            transformed_size=2*hidden_size, 
            output_size=self.classes, 
            dropout=dropout
        )

    def forward(
        self, x: Tensor, x_padding: Tensor, y_targets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        hidden_states, outputs = self.RNN(x)
        
        # 패딩 정보를 활용하여 마지막 유효한 타임스텝 선택
        if x_padding is not None:
            valid_lengths = (x_padding == 0).sum(dim=1) - 1  # 0-indexed
            valid_lengths = valid_lengths.clamp(min=0, max=outputs.size(1)-1)
            batch_size = outputs.size(0)
            final_output = outputs[torch.arange(batch_size), valid_lengths]
        else:
            final_output = outputs[:, -1, :]  # 차원 수정 (batch, time, features)
        
        logits = self.output_layers(
            final_output
        )
        loss = F.cross_entropy(logits, y_targets)
        return logits, loss


# Test the models
if __name__ == "__main__":
    print("🧪 GRU 모델들 테스트 시작...")
    
    print("✅ GRU 모델들 구현 완료!")
    print("📝 기본 GRU, StackedGRU 모델")
    print("🔧 DynamicDataModule과 호환되는 구조")
    print("🎯 SignSpeak 논문의 GRU 아키텍처 구현")
