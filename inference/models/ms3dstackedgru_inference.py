"""
MS3DStackedGRU 추론 전용 모델
Multi-Scale 3D CNN + Stacked GRU 추론 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MS3DStackedGRUInference(nn.Module):
    """
    MS3DStackedGRU 추론 전용 모델 (Multi-Scale 3D CNN + Stacked GRU)
    
    성능: 98.44-98.78% Test Accuracy (데이터셋에 따라 변동)
    
    Args:
        input_size: 입력 특징 수 (default: 8)
        hidden_size: 은닉층 크기 (default: 64)
        classes: 클래스 수 (default: 24)
        cnn_filters: CNN 필터 수 (default: 32)
        dropout: 드롭아웃 비율 (default: 0.05)
    """
    
    def __init__(
        self,
        input_size=8,
        hidden_size=64,
        classes=24,
        cnn_filters=32,
        dropout=0.05,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.classes = classes
        self.cnn_filters = cnn_filters
        
        # Multi-Scale 3D CNN: 3개 타워 병렬 처리
        # Tower 1: 작은 커널
        self.tower1 = nn.Sequential(
            nn.Conv3d(1, cnn_filters, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU(),
            # 추가 Conv 레이어 (CNN 특징 추출 개선)
            nn.Conv3d(cnn_filters, cnn_filters, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU()
        )
        
        # Tower 2: 중간 커널
        self.tower2 = nn.Sequential(
            nn.Conv3d(1, cnn_filters, kernel_size=(3, 5, 3), padding=(1, 2, 1)),
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU(),
            # 추가 Conv 레이어
            nn.Conv3d(cnn_filters, cnn_filters, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU()
        )
        
        # Tower 3: 큰 커널
        self.tower3 = nn.Sequential(
            nn.Conv3d(1, cnn_filters, kernel_size=(3, 7, 3), padding=(1, 3, 1)),
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU(),
            # 추가 Conv 레이어
            nn.Conv3d(cnn_filters, cnn_filters, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU()
        )
        
        # CNN 후처리 (시간 차원 보존, 공간 차원만 풀링)
        self.cnn_post = nn.Sequential(
            nn.BatchNorm3d(cnn_filters * 3),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 4, 2)),  # 시간 보존
            nn.Dropout3d(dropout if self.training else 0)
        )
        
        # Stacked GRU 레이어
        self.gru1 = nn.GRU(cnn_filters * 3, hidden_size, 1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout if self.training else 0)
        
        # 출력 레이어
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout if self.training else 0),
            nn.Linear(2 * hidden_size, classes)
        )
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch, timesteps, features)
            
        Returns:
            logits: 출력 로짓 (batch, classes)
        """
        batch_size = x.size(0)
        
        # Reshape for 3D CNN: (batch, 1, timesteps, 4, 2)
        x_3d = x.view(batch_size, x.size(1), 4, 2)
        x_3d = x_3d.unsqueeze(1)
        x_3d = x_3d.permute(0, 1, 2, 3, 4)
        
        # Multi-Scale 3D CNN
        t1 = self.tower1(x_3d)
        t2 = self.tower2(x_3d)
        t3 = self.tower3(x_3d)
        
        # 타워 병합
        conv_out = torch.cat([t1, t2, t3], dim=1)
        conv_out = self.cnn_post(conv_out)
        
        # GRU를 위한 reshape
        conv_out = conv_out.permute(0, 2, 1, 3, 4).contiguous()
        conv_out = conv_out.view(batch_size, conv_out.size(1), -1)
        
        # Stacked GRU
        gru_out1, _ = self.gru1(conv_out)
        gru_out1 = self.dropout(gru_out1)
        gru_out2, _ = self.gru2(gru_out1)
        
        # 마지막 타임스텝 사용
        final_features = gru_out2[:, -1, :]
        final_features = self.dropout(final_features)
        
        # 출력
        logits = self.output_layers(final_features)
        
        return logits
    
    def predict(self, x):
        """
        예측 (확률 반환)
        
        Args:
            x: 입력 텐서 (batch, timesteps, features)
            
        Returns:
            probabilities: 클래스별 확률 (batch, classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def count_parameters(self):
        """파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """모델 정보 반환"""
        return {
            'model_type': 'MS3DStackedGRU',
            'architecture': 'Multi-Scale 3D CNN + Stacked GRU',
            'performance': '98.44-98.78% accuracy',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'classes': self.classes,
            'cnn_filters': self.cnn_filters,
            'total_parameters': self.count_parameters()
        }

