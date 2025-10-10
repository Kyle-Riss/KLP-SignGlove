"""
MS-CSGRU 추론 전용 모델

훈련 관련 코드가 제거된 경량화된 MSCSGRU 구현
체크포인트에서 가중치를 로드하여 추론만 수행
"""

import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor


class MSCSGRUInference(nn.Module):
    """
    MS-CSGRU: Multi-Scale CNN + Stacked GRU 추론 전용 모델
    
    훈련 관련 코드(loss, learning_rate 등)가 제거된 경량화 버전
    추론 속도와 메모리 효율성에 최적화
    """
    
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        classes: int = 24,
        cnn_filters: int = 32,
        dropout: float = 0.3
    ):
        """
        Args:
            input_size: 입력 채널 수 (기본: 8 - flex1-5 + yaw, pitch, roll)
            hidden_size: GRU 히든 사이즈 (기본: 64)
            classes: 출력 클래스 수 (기본: 24 - 자음 14개 + 모음 10개)
            cnn_filters: CNN 필터 수 (기본: 32)
            dropout: 드롭아웃 비율 (기본: 0.3)
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.classes = classes
        self.cnn_filters = cnn_filters
        
        # Multi-Scale CNN: 3개 타워 병렬 처리
        self.tower1 = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, 3, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU()
        )
        self.tower2 = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, 5, padding=2),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU()
        )
        self.tower3 = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, 7, padding=3),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU()
        )
        
        # CNN 후처리
        self.cnn_post = nn.Sequential(
            nn.BatchNorm1d(cnn_filters * 3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(dropout)
        )
        
        # Stacked GRU
        self.gru1 = nn.GRU(cnn_filters * 3, hidden_size, 1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # 분류기
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, classes)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        추론 전용 forward 메서드
        
        Args:
            x: 입력 텐서 (batch_size, time_steps, channels)
        
        Returns:
            logits: 클래스별 로짓 (batch_size, classes)
        """
        # Multi-Scale CNN
        x_conv = x.transpose(1, 2)  # (batch, channels, time)
        t1 = self.tower1(x_conv)
        t2 = self.tower2(x_conv)
        t3 = self.tower3(x_conv)
        conv_out = torch.cat([t1, t2, t3], dim=1)
        conv_out = self.cnn_post(conv_out)
        
        # Stacked GRU
        conv_out = conv_out.transpose(1, 2)  # (batch, time, channels)
        gru1_out, _ = self.gru1(conv_out)
        gru1_out = self.dropout(gru1_out)
        gru2_out, _ = self.gru2(gru1_out)
        
        # 마지막 타임스텝의 특징 사용
        final_features = gru2_out[:, -1, :]
        
        # 분류
        logits = self.output_layers(final_features)
        
        return logits
    
    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """
        추론 전용 메서드 (gradient 계산 없음)
        
        Args:
            x: 입력 텐서 (batch_size, time_steps, channels)
        
        Returns:
            logits: 클래스별 로짓 (batch_size, classes)
        """
        self.eval()
        return self.forward(x)
    
    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        """
        확률값 예측
        
        Args:
            x: 입력 텐서 (batch_size, time_steps, channels)
        
        Returns:
            probabilities: 클래스별 확률 (batch_size, classes)
        """
        self.eval()
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities
    
    @torch.no_grad()
    def predict_class(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        클래스 예측 + 확률
        
        Args:
            x: 입력 텐서 (batch_size, time_steps, channels)
        
        Returns:
            predicted_classes: 예측된 클래스 인덱스 (batch_size,)
            confidences: 예측 확률 (batch_size,)
        """
        self.eval()
        probabilities = self.predict_proba(x)
        confidences, predicted_classes = torch.max(probabilities, dim=-1)
        return predicted_classes, confidences
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, **model_kwargs):
        """
        체크포인트에서 모델 로드
        
        Args:
            checkpoint_path: 체크포인트 파일 경로 (.ckpt 또는 .pt)
            **model_kwargs: 모델 초기화 인자
        
        Returns:
            model: 로드된 모델 (평가 모드)
        """
        # 모델 초기화
        model = cls(**model_kwargs)
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Lightning 체크포인트인 경우
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # 'model.' 접두사 제거
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # 'model.' 제거
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            model.load_state_dict(new_state_dict, strict=False)
        else:
            # 일반 PyTorch 체크포인트인 경우
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        return model
    
    def count_parameters(self) -> int:
        """
        학습 가능한 파라미터 수 계산
        
        Returns:
            num_params: 파라미터 수
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """
        모델 정보 반환
        
        Returns:
            info: 모델 정보 딕셔너리
        """
        return {
            'model_type': 'MS-CSGRU',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'classes': self.classes,
            'cnn_filters': self.cnn_filters,
            'total_parameters': self.count_parameters()
        }


# 테스트 코드
if __name__ == "__main__":
    print("🧪 MSCSGRUInference 테스트...")
    
    # 모델 생성
    model = MSCSGRUInference(
        input_size=8,
        hidden_size=64,
        classes=24,
        cnn_filters=32
    )
    
    # 모델 정보 출력
    info = model.get_model_info()
    print(f"\n📊 모델 정보:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 테스트 데이터
    batch_size, time_steps, input_channels = 4, 87, 8
    x = torch.randn(batch_size, time_steps, input_channels)
    
    # 추론 테스트
    print(f"\n🔍 추론 테스트:")
    print(f"  입력 shape: {x.shape}")
    
    logits = model.predict(x)
    print(f"  로짓 shape: {logits.shape}")
    
    probabilities = model.predict_proba(x)
    print(f"  확률 shape: {probabilities.shape}")
    
    predicted_classes, confidences = model.predict_class(x)
    print(f"  예측 클래스: {predicted_classes.tolist()}")
    print(f"  확률: {confidences.tolist()}")
    
    print("\n✅ 테스트 완료!")




