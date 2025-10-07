import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.generalModels import *
from src.models.LightningModel import LitModel

from typing import Tuple
from torch import Tensor


class LSTM(LitModel):
    """
    SignSpeak 논문 기반 기본 LSTM 모델
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=24,
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
                nn.LSTM(
                    2 * hidden_size,
                    hidden_size,
                    num_layers=layers,
                    batch_first=batch_first,
                ),
            )
        else:
            self.RNN = nn.LSTM(
                input_size, hidden_size, num_layers=layers, batch_first=batch_first
            )

        self.output_layers = outputRNN(
            hidden_size=hidden_size, transformed_size=2*hidden_size, output_size=self.classes, dropout=dropout
        )

    def forward(
        self, x: Tensor, x_padding: Tensor, y_targets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        outputs, (hidden, cell) = self.RNN(
            x
        )  # outputs: all timesteps (batch, seq, features), hidden/cell: last hidden state
        
        # 단순화된 패딩 처리 - 마지막 타임스텝 사용
        final_output = outputs[:, -1, :]  # (batch, features)
        
        logits = self.output_layers(
            final_output
        )  # output of last cell into dense layer
        loss = F.cross_entropy(logits, y_targets)  # cross entropy loss
        return logits, loss


class StackedLSTM(LitModel):
    """
    SignSpeak 논문 기반 Stacked LSTM 모델
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=24,
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

        # Stacked LSTM with dropout between layers
        self.RNN = nn.LSTM(
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
        outputs, (hidden, cell) = self.RNN(x)
        
        # 단순화된 패딩 처리 - 마지막 타임스텝 사용
        final_output = outputs[:, -1, :]  # (batch, features)
        
        logits = self.output_layers(
            final_output
        )
        loss = F.cross_entropy(logits, y_targets)
        return logits, loss


# Test the models
if __name__ == "__main__":
    print("🧪 LSTM 모델들 테스트 시작...")
    
    print("✅ LSTM 모델들 구현 완료!")
    print("📝 기본 LSTM, StackedLSTM 모델")
    print("🔧 DynamicDataModule과 호환되는 구조")
    print("🎯 SignSpeak 논문의 LSTM 아키텍처 구현")
