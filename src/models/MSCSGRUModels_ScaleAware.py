"""
MS-CSGRU Scale-Aware Models
Multi-Scale CNN + Scale-Aware GRU 통합 모델

특징:
1. 각 CNN 타워의 특징에 독립적인 GRU 가중치 할당
2. Hard 함수 옵션으로 임베디드 최적화 지원
3. 패딩 인식 특징 추출
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor

from src.models.LightningModel import LitModel
from src.models.ScaleAwareGRU import ScaleAwareGRU


class MSCSGRU_ScaleAware(LitModel):
    """
    MS-CSGRU with Scale-Aware GRU
    
    아키텍처:
        Multi-Scale CNN (3 towers) 
        → Scale-Aware GRU Layer 1 
        → Scale-Aware GRU Layer 2
        → Padding-Aware Feature Extraction
        → Classifier
    
    개선점:
        1. 각 CNN 스케일(k=3,5,7)에 독립적인 GRU 가중치
        2. 스케일별 중요도 학습 가능
        3. 해석 가능성 향상
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=24,
        cnn_filters=32,
        dropout=0.3,
        use_hard_functions=False,  # Hard 함수 사용 여부
        **kwargs
    ):
        super().__init__()
        
        self.lr = learning_rate
        self.classes = classes
        self.use_hard = use_hard_functions
        
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
        
        # Scale-Aware Stacked GRU
        self.gru1 = ScaleAwareGRU(
            scale_sizes=(cnn_filters, cnn_filters, cnn_filters),  # (32, 32, 32)
            hidden_size=hidden_size,
            use_hard_functions=use_hard_functions
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.gru2 = ScaleAwareGRU(
            scale_sizes=(hidden_size, hidden_size, hidden_size),  # GRU1 출력을 3등분
            hidden_size=hidden_size,
            use_hard_functions=use_hard_functions
        )
        
        # 분류기
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_size, classes)
        )
    
    def forward(
        self, 
        x: Tensor, 
        x_padding: Tensor, 
        y_targets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass
        
        Args:
            x: 입력 센서 데이터 (batch, time, channels)
            x_padding: 패딩 마스크 (batch, time)
            y_targets: 타겟 레이블 (batch,)
        
        Returns:
            logits: 클래스 로짓 (batch, classes)
            loss: Cross-entropy loss
        """
        
        # Multi-Scale CNN
        x_conv = x.transpose(1, 2)  # (batch, channels, time)
        t1 = self.tower1(x_conv)
        t2 = self.tower2(x_conv) 
        t3 = self.tower3(x_conv)
        conv_out = torch.cat([t1, t2, t3], dim=1)
        conv_out = self.cnn_post(conv_out)
        
        # Scale-Aware GRU Layer 1
        conv_out = conv_out.transpose(1, 2)  # (batch, time, channels)
        gru1_out, _ = self.gru1(conv_out)
        gru1_out = self.dropout1(gru1_out)
        
        # Scale-Aware GRU Layer 2
        # GRU1 출력(64 channels)을 복제하여 3개 스케일로 사용
        # 각 스케일이 독립적으로 가중치를 학습하도록 함
        gru1_expanded = gru1_out.repeat(1, 1, 3)  # (batch, time, 192)
        gru2_out, _ = self.gru2(gru1_expanded)
        
        # 패딩 인식 특징 추출
        if x_padding is not None:
            # MaxPool로 시퀀스 길이가 절반이 되었으므로 조정
            valid_lengths = (x_padding == 0).sum(dim=1) - 1
            valid_lengths = (valid_lengths / 2).long()  # MaxPool(2) 반영
            valid_lengths = valid_lengths.clamp(min=0, max=gru2_out.size(1)-1)
            
            batch_size = gru2_out.size(0)
            final_features = gru2_out[torch.arange(batch_size), valid_lengths]
        else:
            final_features = gru2_out[:, -1, :]
        
        # 분류
        logits = self.output_layers(final_features)
        loss = F.cross_entropy(logits, y_targets)
        
        return logits, loss
    
    def get_scale_importance(self) -> dict:
        """
        각 스케일의 중요도 분석
        
        Returns:
            dict: GRU 레이어별 스케일 가중치
        """
        return {
            'gru_layer1': self.gru1.get_gate_weights(),
            'gru_layer2': self.gru2.get_gate_weights()
        }


class MSCSGRU_ScaleHard(MSCSGRU_ScaleAware):
    """
    MS-CSGRU with Scale-Aware + Hard Functions
    
    MSCSGRU_ScaleAware의 Hard 함수 버전
    임베디드 시스템 최적화를 위해 모든 활성화 함수를 Hard 버전으로 사용
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=24,
        cnn_filters=32,
        dropout=0.3,
        **kwargs
    ):
        # Hard 함수 강제 활성화
        super().__init__(
            learning_rate=learning_rate,
            input_size=input_size,
            hidden_size=hidden_size,
            classes=classes,
            cnn_filters=cnn_filters,
            dropout=dropout,
            use_hard_functions=True,  # 강제로 True
            **kwargs
        )


class MSCGRU_ScaleAware(LitModel):
    """
    MS-CGRU with Scale-Aware GRU (Single GRU)
    
    Stacked GRU 대신 단일 GRU 사용
    더 빠른 학습과 추론을 위한 경량 버전
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=24,
        cnn_filters=32,
        dropout=0.3,
        use_hard_functions=False,
        **kwargs
    ):
        super().__init__()
        
        self.lr = learning_rate
        self.classes = classes
        
        # Multi-Scale CNN
        self.tower1 = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, 3, padding=1),
            nn.BatchNorm1d(cnn_filters), nn.ReLU()
        )
        self.tower2 = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, 5, padding=2),
            nn.BatchNorm1d(cnn_filters), nn.ReLU()
        )
        self.tower3 = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, 7, padding=3),
            nn.BatchNorm1d(cnn_filters), nn.ReLU()
        )
        
        # CNN 후처리
        self.cnn_post = nn.Sequential(
            nn.BatchNorm1d(cnn_filters * 3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(dropout)
        )
        
        # Single Scale-Aware GRU
        self.gru = ScaleAwareGRU(
            scale_sizes=(cnn_filters, cnn_filters, cnn_filters),
            hidden_size=hidden_size,
            use_hard_functions=use_hard_functions
        )
        self.dropout = nn.Dropout(dropout)
        
        # 분류기
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_size, classes)
        )
    
    def forward(
        self, 
        x: Tensor, 
        x_padding: Tensor, 
        y_targets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Multi-Scale CNN
        x_conv = x.transpose(1, 2)
        t1 = self.tower1(x_conv)
        t2 = self.tower2(x_conv) 
        t3 = self.tower3(x_conv)
        conv_out = torch.cat([t1, t2, t3], dim=1)
        conv_out = self.cnn_post(conv_out)
        
        # Single GRU
        conv_out = conv_out.transpose(1, 2)
        gru_out, _ = self.gru(conv_out)
        
        # 패딩 인식 특징 추출
        if x_padding is not None:
            valid_lengths = (x_padding == 0).sum(dim=1) - 1
            valid_lengths = (valid_lengths / 2).long()
            valid_lengths = valid_lengths.clamp(min=0, max=gru_out.size(1)-1)
            batch_size = gru_out.size(0)
            final_features = gru_out[torch.arange(batch_size), valid_lengths]
        else:
            final_features = gru_out[:, -1, :]
        
        final_features = self.dropout(final_features)
        
        # 분류
        logits = self.output_layers(final_features)
        loss = F.cross_entropy(logits, y_targets)
        
        return logits, loss


# Test the models
if __name__ == "__main__":
    print("🧪 Scale-Aware 모델들 테스트 시작...")
    
    # 테스트 데이터
    batch_size, time_steps, input_channels = 4, 87, 8
    num_classes = 24
    
    x = torch.randn(batch_size, time_steps, input_channels)
    x_padding = torch.zeros(batch_size, time_steps)
    x_padding[:, 80:] = 1.0  # 80번째 이후는 패딩
    y_targets = torch.randint(0, num_classes, (batch_size,))
    
    print(f"\n입력 데이터:")
    print(f"  x: {x.shape}")
    print(f"  x_padding: {x_padding.shape}")
    print(f"  y_targets: {y_targets.shape}")
    
    models_to_test = [
        ('MSCSGRU_ScaleAware (Sigmoid/Tanh)', 
         MSCSGRU_ScaleAware(learning_rate=1e-3, input_size=8, classes=24, use_hard_functions=False)),
        
        ('MSCSGRU_ScaleHard (HardSigmoid/HardTanh)', 
         MSCSGRU_ScaleHard(learning_rate=1e-3, input_size=8, classes=24)),
        
        ('MSCGRU_ScaleAware (Single GRU)', 
         MSCGRU_ScaleAware(learning_rate=1e-3, input_size=8, classes=24, use_hard_functions=False)),
    ]
    
    print("\n" + "="*80)
    print("모델별 테스트 결과")
    print("="*80)
    
    for name, model in models_to_test:
        print(f"\n📊 {name}")
        print("-"*80)
        
        try:
            model.eval()
            with torch.no_grad():
                logits, loss = model(x, x_padding, y_targets)
            
            # 파라미터 수
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"✅ 출력 shape: {logits.shape}")
            print(f"✅ 손실: {loss.item():.4f}")
            print(f"✅ 파라미터 수: {num_params:,}")
            
            # 스케일 중요도 분석 (MSCSGRU만)
            if hasattr(model, 'get_scale_importance'):
                importance = model.get_scale_importance()
                print(f"\n📈 스케일 중요도 (GRU Layer 1 - Update Gate):")
                weights = importance['gru_layer1']['update_gate']
                print(f"  Scale 3 (kernel=3): {weights['scale_3']:.4f}")
                print(f"  Scale 5 (kernel=5): {weights['scale_5']:.4f}")
                print(f"  Scale 7 (kernel=7): {weights['scale_7']:.4f}")
                
                # 가장 중요한 스케일
                max_scale = max(weights.items(), key=lambda x: x[1])
                print(f"  → 가장 중요한 스케일: {max_scale[0]} (가중치: {max_scale[1]:.4f})")
            
        except Exception as e:
            print(f"❌ 에러: {str(e)}")
    
    # 속도 비교
    print("\n" + "="*80)
    print("추론 속도 비교")
    print("="*80)
    
    import time
    
    model_normal = MSCSGRU_ScaleAware(learning_rate=1e-3, input_size=8, classes=24, use_hard_functions=False)
    model_hard = MSCSGRU_ScaleHard(learning_rate=1e-3, input_size=8, classes=24)
    
    model_normal.eval()
    model_hard.eval()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model_normal(x, x_padding, y_targets)
            _ = model_hard(x, x_padding, y_targets)
    
    # 일반 버전
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model_normal(x, x_padding, y_targets)
    time_normal = (time.time() - start) / 100 * 1000
    
    # Hard 버전
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model_hard(x, x_padding, y_targets)
    time_hard = (time.time() - start) / 100 * 1000
    
    print(f"\nSigmoid/Tanh:      {time_normal:.2f}ms")
    print(f"HardSigmoid/Tanh:  {time_hard:.2f}ms")
    print(f"속도 향상:         {(time_normal - time_hard) / time_normal * 100:.1f}%")
    
    print("\n" + "="*80)
    print("✅ 모든 테스트 완료!")
    print("="*80)
    
    print("\n📝 다음 단계:")
    print("  1. 실제 데이터로 학습")
    print("  2. 기존 MSCSGRU와 성능 비교")
    print("  3. 스케일 중요도 분석")
    print("  4. 추론 최적화 (ONNX 변환)")

