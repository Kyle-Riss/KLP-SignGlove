"""
StackedGRU 추론 전용 모델
훈련 코드와 분리된 경량 추론 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedGRUInference(nn.Module):
    """
    StackedGRU 추론 전용 모델
    
    Args:
        input_size: 입력 특징 수 (default: 8)
        hidden_size: 은닉층 크기 (default: 64)
        classes: 클래스 수 (default: 24)
        layers: GRU 레이어 수 (default: 2)
        dropout: 드롭아웃 비율 (default: 0.2)
    """
    
    def __init__(
        self,
        input_size=8,
        hidden_size=64,
        classes=24,
        layers=2,
        dropout=0.2,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.classes = classes
        self.layers = layers
        
        # 훈련 모델 구조와 일치: 직접 Stacked GRU (Linear layer 없음)
        # 훈련 모델: RNN = nn.GRU(input_size -> hidden_size, layers=2)
        self.RNN = nn.GRU(
            input_size,
            hidden_size,
            num_layers=layers, 
            batch_first=True,
            dropout=dropout if layers > 1 else 0
        )
        
        # 출력 레이어
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.Dropout(dropout if self.training else 0),
            nn.Tanh(),
            nn.Linear(2 * hidden_size, classes),
        )
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch, timesteps, features)
            
        Returns:
            logits: 출력 로짓 (batch, classes)
        """
        # RNN은 직접 GRU (Linear layer 없음)
        # Stacked GRU
        outputs, hidden = self.RNN(x)
        
        # 마지막 타임스텝 사용
        final_output = outputs[:, -1, :]
        
        # 출력 레이어
        logits = self.output_layers(final_output)
        
        return logits
    
    @torch.no_grad()
    def predict(self, x):
        """
        예측 (로짓 반환) - 다른 모델과 일관성 유지
        
        Args:
            x: 입력 텐서 (batch, timesteps, features)
            
        Returns:
            logits: 클래스별 로짓 (batch, classes)
        """
        self.eval()
        return self.forward(x)
    
    @torch.no_grad()
    def predict_proba(self, x):
        """
        확률 예측
        
        Args:
            x: 입력 텐서 (batch, timesteps, features)
            
        Returns:
            probabilities: 클래스별 확률 (batch, classes)
        """
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    @torch.no_grad()
    def predict_class(self, x):
        """
        클래스 예측
        
        Args:
            x: 입력 텐서 (batch, timesteps, features)
            
        Returns:
            predicted_classes: 예측된 클래스 인덱스 (batch,)
            confidences: 예측 확률 (batch,)
        """
        self.eval()
        probabilities = self.predict_proba(x)
        confidences, predicted_classes = torch.max(probabilities, dim=-1)
        return predicted_classes, confidences
    
    def count_parameters(self):
        """파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """모델 정보 반환"""
        return {
            'model_type': 'StackedGRU',
            'architecture': 'Stacked GRU',
            'performance': '95.43% accuracy',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'classes': self.classes,
            'layers': self.layers,
            'total_parameters': self.count_parameters()
        }

