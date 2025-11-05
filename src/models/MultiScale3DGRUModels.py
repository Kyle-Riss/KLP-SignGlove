import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor

from src.models.LightningModel import LitModel


class MS3DGRU(LitModel):
    """
    Multi-Scale 3D CNN + GRU 모델
    시간-공간 특성 학습을 위한 Multi-Scale 3D CNN 적용
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=24,
        cnn_filters=32,
        dropout=0.2,
        **kwargs
    ):
        super().__init__()
        
        self.lr = learning_rate
        self.classes = classes
        
        # Multi-Scale 3D CNN: 3개 타워 병렬 처리
        # Tower 1: 작은 커널 (3x3x3) - 세밀한 특성
        self.tower1 = nn.Sequential(
            nn.Conv3d(1, cnn_filters, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU()
        )
        
        # Tower 2: 중간 커널 (5x5x5) - 중간 특성
        self.tower2 = nn.Sequential(
            nn.Conv3d(1, cnn_filters, (5, 5, 5), padding=(2, 2, 2)),
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU()
        )
        
        # Tower 3: 큰 커널 (7x7x7) - 거시적 특성
        self.tower3 = nn.Sequential(
            nn.Conv3d(1, cnn_filters, (7, 7, 7), padding=(3, 3, 3)),
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU()
        )
        
        # CNN 후처리 - 시간과 공간 차원 모두 pooling
        self.cnn_post = nn.Sequential(
            nn.BatchNorm3d(cnn_filters * 3),
            nn.ReLU(),
            nn.MaxPool3d((2, 4, 2)),  # 시간, 높이, 너비 모두 pooling
            nn.Dropout3d(dropout)
        )
        
        # GRU - 공간 차원을 flatten한 후 GRU 입력
        self.gru = nn.GRU(cnn_filters * 3 * 1 * 1, hidden_size, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # 분류기
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_size, classes)
        )
    
    def forward(self, x: Tensor, x_padding: Tensor, y_targets: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, time_steps, input_channels = x.shape
        
        # 3D 텐서로 변환: (batch, 1, time, 4, 2) - 센서를 4x2 공간으로 재배열
        # 8개 센서를 4x2로 재배열하여 공간적 구조 생성
        x_3d = x.view(batch_size, time_steps, 4, 2)  # (batch, time, 4, 2) -> 텐서 shape 변경하는 함수 view()
        x_3d = x_3d.unsqueeze(1)  # (batch, 1, time, 4, 2) 채널 차원 추가 -> 차원 추가하는 함수 unsqueeze()
        x_3d = x_3d.transpose(1, 2)  # (batch, time, 1, 4, 2) 시간, 채널 차원 교환 -> 1은 채널 차원 -> 채널 차원 먼저 처리하고 시간 차원 처리하기 위해서 -> 차원교환 
        x_3d = x_3d.contiguous().view(batch_size, 1, time_steps, 4, 2) #3D 텐서 형태 -> 메모리 연속성 보장 -> 최종 shape 변경
        
        # Multi-Scale 3D CNN
        t1 = self.tower1(x_3d) # (batch, filters, time, 4, 2) -> 3*3*3 커널
        t2 = self.tower2(x_3d) # 5*5*5 커널
        t3 = self.tower3(x_3d) # 7*7*7 커널
        
        conv_out = torch.cat([t1, t2, t3], dim=1)  # (batch, filters*3, time, 4, 2) 3개 타워 출력 결합
        conv_out = self.cnn_post(conv_out)  # (batch, filters*3, time/2, 2, 1) MaxPool3d로 후처리
        
        # 3D → 1D 변환: (batch, time, features) 
        # conv_out shape: (batch, filters*3, time/2, 1, 1)
        # 공간 차원을 flatten: (batch, time/2, filters*3*1*1)
        conv_out = conv_out.permute(0, 2, 1, 3, 4)  # (batch, time/2, filters*3, 1, 1) 3D → 1D 변환
        conv_out = conv_out.contiguous().view(batch_size, conv_out.size(1), -1)  # (batch, time/2, filters*3*1*1) 공간 차원을 flatten -> (batch, time/2, features)) 최종결과가 GRU에 입력 가능한 형태
        
        # GRU
        gru_out, _ = self.gru(conv_out) #시퀀스 데이터 처리
        
        # 패딩 정보를 활용하여 마지막 유효한 타임스텝 선택
        if x_padding is not None: #패딩 고려해서 실제 데이터에서 마지막 타임스텝 선택 -> 패딩 있으면 유효한 타임스텝으로 계산하고, 패딩이 없으면 마지막 타임스텝 사용
            valid_lengths = (x_padding == 0).sum(dim=1) - 1 # 각 배치의 유효한 타임스텝 수 계산
            valid_lengths = valid_lengths.clamp(min=0, max=gru_out.size(1)-1) #텐서의 값을 지정된 범위로 제한하는 함수
            batch_size = gru_out.size(0)
            final_features = gru_out[torch.arange(batch_size), valid_lengths] 
        else:
            final_features = gru_out[:, -1, :]
        
        final_features = self.dropout(final_features) #최종 특징벡터 (batch, hidden_size)
        
        # 분류
        logits = self.output_layers(final_features) #분류 레이어 아까 (batch, classes=24)
        loss = F.cross_entropy(logits, y_targets) #손실 계산
        
        return logits, loss


class MS3DStackedGRU(LitModel):
    """
    Multi-Scale 3D CNN + Stacked GRU 모델 (CNN 특징 추출 개선)
    시간-공간 특성 학습 + 다층 GRU + 개선된 CNN 구조
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=24,
        cnn_filters=32,
        gru_layers=2,
        dropout=0.2,
        **kwargs
    ):
        super().__init__()
        
        self.lr = learning_rate
        self.classes = classes
        
        # 개선된 Multi-Scale 3D CNN: 더 다양한 커널 크기와 개선된 구조
        self.tower1 = nn.Sequential(
            nn.Conv3d(1, cnn_filters, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU(),
            nn.Conv3d(cnn_filters, cnn_filters, (3, 3, 3), padding=(1, 1, 1)),  # 추가 Conv 레이어
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU()
        )
        
        self.tower2 = nn.Sequential(
            nn.Conv3d(1, cnn_filters, (5, 5, 5), padding=(2, 2, 2)),
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU(),
            nn.Conv3d(cnn_filters, cnn_filters, (3, 3, 3), padding=(1, 1, 1)),  # 추가 Conv 레이어
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU()
        )
        
        self.tower3 = nn.Sequential(
            nn.Conv3d(1, cnn_filters, (7, 7, 7), padding=(3, 3, 3)),
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU(),
            nn.Conv3d(cnn_filters, cnn_filters, (3, 3, 3), padding=(1, 1, 1)),  # 추가 Conv 레이어
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU()
        )
        
        # 개선된 CNN 후처리 - 시간 차원 보존하는 pooling
        self.cnn_post = nn.Sequential(
            nn.BatchNorm3d(cnn_filters * 3),
            nn.ReLU(),
            nn.MaxPool3d((1, 4, 2)),  # 시간 차원 보존, 공간 차원만 pooling
            nn.Dropout3d(dropout)
        )
        
        # Stacked GRU - 개선된 입력 차원
        self.gru1 = nn.GRU(cnn_filters * 3 * 1 * 1, hidden_size, 1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # 분류기
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_size, classes)
        )
    
    def forward(self, x: Tensor, x_padding: Tensor, y_targets: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, time_steps, input_channels = x.shape
        
        # 3D 텐서로 변환: (batch, 1, time, 4, 2)
        x_3d = x.view(batch_size, time_steps, 4, 2)
        x_3d = x_3d.unsqueeze(1)
        x_3d = x_3d.transpose(1, 2)
        x_3d = x_3d.contiguous().view(batch_size, 1, time_steps, 4, 2)
        
        # Multi-Scale 3D CNN
        t1 = self.tower1(x_3d)
        t2 = self.tower2(x_3d)
        t3 = self.tower3(x_3d)
        
        conv_out = torch.cat([t1, t2, t3], dim=1)
        conv_out = self.cnn_post(conv_out)
        
        # 3D → 1D 변환
        # conv_out shape: (batch, filters*3, time/2, 1, 1)
        # 공간 차원을 flatten: (batch, time/2, filters*3*1*1)
        conv_out = conv_out.permute(0, 2, 1, 3, 4)  # (batch, time/2, filters*3, 1, 1)
        conv_out = conv_out.contiguous().view(batch_size, conv_out.size(1), -1)  # (batch, time/2, filters*3*1*1)
        
        # Stacked GRU
        gru1_out, _ = self.gru1(conv_out)
        gru1_out = self.dropout(gru1_out)
        gru2_out, _ = self.gru2(gru1_out)
        
        # 패딩 정보를 활용하여 마지막 유효한 타임스텝 선택
        if x_padding is not None:
            valid_lengths = (x_padding == 0).sum(dim=1) - 1
            valid_lengths = valid_lengths.clamp(min=0, max=gru2_out.size(1)-1)
            batch_size = gru2_out.size(0)
            final_features = gru2_out[torch.arange(batch_size), valid_lengths]
        else:
            final_features = gru2_out[:, -1, :]
        
        final_features = self.dropout(final_features)
        
        # 분류
        logits = self.output_layers(final_features)
        loss = F.cross_entropy(logits, y_targets)
        
        return logits, loss


class SensorAware3DGRU(LitModel):
    """
    센서 그룹별 3D CNN + GRU 모델
    Yaw/Pitch/Roll과 Flex 1-5를 별도로 처리
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=24,
        cnn_filters=32,
        dropout=0.2,
        **kwargs
    ):
        super().__init__()
        
        self.lr = learning_rate
        self.classes = classes
        
        # Yaw/Pitch/Roll 센서용 3D CNN (3개 센서)
        self.orientation_cnn = nn.Sequential(
            nn.Conv3d(1, cnn_filters, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU(),
            nn.MaxPool3d((2, 3, 1)),  # 시간, 높이, 너비 모두 pooling
            nn.Dropout3d(dropout)
        )
        
        # Flex 1-5 센서용 3D CNN (5개 센서)
        self.flex_cnn = nn.Sequential(
            nn.Conv3d(1, cnn_filters, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(cnn_filters),
            nn.ReLU(),
            nn.MaxPool3d((2, 5, 1)),  # 시간, 높이, 너비 모두 pooling
            nn.Dropout3d(dropout)
        )
        
        # 결합된 특징을 위한 GRU - 공간 차원을 flatten한 후 GRU 입력
        self.gru = nn.GRU(cnn_filters * 2 * 1 * 1, hidden_size, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # 분류기
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_size, classes)
        )
    
    def forward(self, x: Tensor, x_padding: Tensor, y_targets: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, time_steps, input_channels = x.shape
        
        # 센서 분리: Yaw/Pitch/Roll (0:3), Flex 1-5 (3:8)
        orientation_data = x[:, :, :3]  # (batch, time, 3)
        flex_data = x[:, :, 3:]  # (batch, time, 5)
        
        # Orientation 데이터 3D 변환
        ori_3d = orientation_data.view(batch_size, time_steps, 3, 1, 1)
        ori_3d = ori_3d.unsqueeze(1)  # (batch, 1, time, 3, 1)
        ori_3d = ori_3d.transpose(1, 2)  # (batch, time, 1, 3, 1)
        ori_3d = ori_3d.contiguous().view(batch_size, 1, time_steps, 3, 1)
        
        # Flex 데이터 3D 변환
        flex_3d = flex_data.view(batch_size, time_steps, 5, 1, 1)
        flex_3d = flex_3d.unsqueeze(1)  # (batch, 1, time, 5, 1)
        flex_3d = flex_3d.transpose(1, 2)  # (batch, time, 1, 5, 1)
        flex_3d = flex_3d.contiguous().view(batch_size, 1, time_steps, 5, 1)
        
        # 각 센서 그룹별 3D CNN 처리
        ori_features = self.orientation_cnn(ori_3d)  # (batch, filters, time/2, 1, 1)
        flex_features = self.flex_cnn(flex_3d)  # (batch, filters, time/2, 1, 1)
        
        # 특징 결합
        # ori_features shape: (batch, filters, time/2, 1, 1)
        # flex_features shape: (batch, filters, time/2, 1, 1)
        # 공간 차원을 flatten: (batch, time/2, filters*1*1)
        ori_features = ori_features.permute(0, 2, 1, 3, 4)  # (batch, time/2, filters, 1, 1)
        ori_features = ori_features.contiguous().view(batch_size, ori_features.size(1), -1)  # (batch, time/2, filters)
        
        flex_features = flex_features.permute(0, 2, 1, 3, 4)  # (batch, time/2, filters, 1, 1)
        flex_features = flex_features.contiguous().view(batch_size, flex_features.size(1), -1)  # (batch, time/2, filters)
        
        combined_features = torch.cat([ori_features, flex_features], dim=-1)  # (batch, time/2, filters*2)
        
        # GRU
        gru_out, _ = self.gru(combined_features)
        
        # 패딩 정보를 활용하여 마지막 유효한 타임스텝 선택
        if x_padding is not None:
            valid_lengths = (x_padding == 0).sum(dim=1) - 1
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
    print("🧪 Multi-Scale 3D CNN 모델들 테스트...")
    
    # Test data
    batch_size, time_steps, input_channels = 4, 87, 8
    num_classes = 24
    
    x = torch.randn(batch_size, time_steps, input_channels)
    x_mask = torch.ones(batch_size, time_steps)
    y_targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test MS3DGRU
    print("\n📊 MS3DGRU (Multi-Scale 3D CNN + GRU) 테스트:")
    model_ms3d = MS3DGRU(
        learning_rate=1e-3,
        input_size=input_channels,
        hidden_size=64,
        classes=num_classes
    )
    
    logits, loss = model_ms3d(x, x_mask, y_targets)
    print(f"출력 shape: {logits.shape}")
    print(f"손실: {loss.item():.4f}")
    
    # Test MS3DStackedGRU
    print("\n📊 MS3DStackedGRU (Multi-Scale 3D CNN + Stacked GRU) 테스트:")
    model_ms3d_stacked = MS3DStackedGRU(
        learning_rate=1e-3,
        input_size=input_channels,
        hidden_size=64,
        classes=num_classes
    )
    
    logits, loss = model_ms3d_stacked(x, x_mask, y_targets)
    print(f"출력 shape: {logits.shape}")
    print(f"손실: {loss.item():.4f}")
    
    # Test SensorAware3DGRU
    print("\n📊 SensorAware3DGRU (센서 그룹별 3D CNN + GRU) 테스트:")
    model_sensor_aware = SensorAware3DGRU(
        learning_rate=1e-3,
        input_size=input_channels,
        hidden_size=64,
        classes=num_classes
    )
    
    logits, loss = model_sensor_aware(x, x_mask, y_targets)
    print(f"출력 shape: {logits.shape}")
    print(f"손실: {loss.item():.4f}")
    
    print("\n✅ 모든 Multi-Scale 3D CNN 모델 테스트 완료!")
