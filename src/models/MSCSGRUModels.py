import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor


class MSCGRU(nn.Module):
    """
    MS-CGRU: Multi-Scale CNN + Single GRU 모델
    Stacked GRU 대신 단일 GRU 사용
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=34,
        cnn_filters=32,
        dropout=0.3,
        **kwargs
    ):
        super().__init__()
        
        self.lr = learning_rate
        self.classes = classes
        
        # Multi-Scale CNN: 3개 타워 병렬 처리
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
        
        # Single GRU
        self.gru = nn.GRU(cnn_filters * 3, hidden_size, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # 분류기
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_size, classes)
        )
    
    def forward(self, x: Tensor, x_mask: Tensor, y_targets: Tensor) -> Tuple[Tensor, Tensor]:
        # Multi-Scale CNN
        x_conv = x.transpose(1, 2)  # (batch, channels, time)
        t1 = self.tower1(x_conv)
        t2 = self.tower2(x_conv) 
        t3 = self.tower3(x_conv)
        conv_out = torch.cat([t1, t2, t3], dim=1)
        conv_out = self.cnn_post(conv_out)
        
        # Single GRU
        conv_out = conv_out.transpose(1, 2)  # (batch, time, channels)
        gru_out, _ = self.gru(conv_out)
        final_features = gru_out[:, -1, :]  # 마지막 시간 단계
        final_features = self.dropout(final_features)
        
        # 분류
        logits = self.output_layers(final_features)
        loss = F.cross_entropy(logits, y_targets)
        
        return logits, loss


class CNNGRU(nn.Module):
    """
    CNN-GRU: 단일 스케일 CNN + GRU 모델
    멀티스케일 대신 단일 CNN 사용
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=34,
        cnn_filters=32,
        dropout=0.3,
        **kwargs
    ):
        super().__init__()
        
        self.lr = learning_rate
        self.classes = classes
        
        # 단일 CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, 3, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(dropout)
        )
        
        # GRU
        self.gru = nn.GRU(cnn_filters, hidden_size, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # 분류기
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_size, classes)
        )
    
    def forward(self, x: Tensor, x_mask: Tensor, y_targets: Tensor) -> Tuple[Tensor, Tensor]:
        # CNN
        x_conv = x.transpose(1, 2)  # (batch, channels, time)
        conv_out = self.cnn(x_conv)
        
        # GRU
        conv_out = conv_out.transpose(1, 2)  # (batch, time, channels)
        gru_out, _ = self.gru(conv_out)
        final_features = gru_out[:, -1, :]  # 마지막 시간 단계
        final_features = self.dropout(final_features)
        
        # 분류
        logits = self.output_layers(final_features)
        loss = F.cross_entropy(logits, y_targets)
        
        return logits, loss


class CNNStackedGRU(nn.Module):
    """
    CNN-StackedGRU: 단일 스케일 CNN + 2층 GRU 모델
    단일 Conv1D 뒤에 2개의 GRU 층을 순차로 배치
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=34,
        cnn_filters=32,
        dropout=0.3,
        **kwargs
    ):
        super().__init__()
        
        self.lr = learning_rate
        self.classes = classes
        
        # 단일 CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, 3, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(dropout)
        )
        
        # Stacked GRU (2층)
        self.gru1 = nn.GRU(cnn_filters, hidden_size, 1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # 분류기
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_size, classes)
        )
    
    def forward(self, x: Tensor, x_mask: Tensor, y_targets: Tensor) -> Tuple[Tensor, Tensor]:
        # CNN
        x_conv = x.transpose(1, 2)  # (batch, channels, time)
        conv_out = self.cnn(x_conv)
        
        # GRU 2층
        conv_out = conv_out.transpose(1, 2)  # (batch, time, channels)
        gru1_out, _ = self.gru1(conv_out)
        gru1_out = self.dropout(gru1_out)
        gru2_out, _ = self.gru2(gru1_out)
        final_features = gru2_out[:, -1, :]  # 마지막 시간 단계
        final_features = self.dropout(final_features)
        
        # 분류
        logits = self.output_layers(final_features)
        loss = F.cross_entropy(logits, y_targets)
        
        return logits, loss


class MSCSGRU(nn.Module):
    """
    MS-CSGRU: Multi-Scale CNN + Stacked GRU 모델
    ASL GRU 스타일로 간결하게 구현
    """
    
    def __init__(
        self,
        learning_rate,
        input_size=8,
        hidden_size=64,
        classes=34,
        cnn_filters=32,
        gru_layers=2,
        dropout=0.3,
        **kwargs
    ):
        super().__init__()
        
        self.lr = learning_rate
        self.classes = classes
        
        # Multi-Scale CNN: 3개 타워 병렬 처리
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
        
        # Stacked GRU
        self.gru1 = nn.GRU(cnn_filters * 3, hidden_size, 1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # 분류기
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_size, classes)
        )
    
    def forward(self, x: Tensor, x_mask: Tensor, y_targets: Tensor) -> Tuple[Tensor, Tensor]:
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
        final_features = gru2_out[:, -1, :]  # 마지막 시간 단계
        
        # 분류
        logits = self.output_layers(final_features)
        loss = F.cross_entropy(logits, y_targets)
        
        return logits, loss


# Test the model
if __name__ == "__main__":
    print("🧪 CNN-GRU, MS-CGRU & MS-CSGRU 모델 테스트...")
    
    # Test data
    batch_size, time_steps, input_channels = 4, 87, 8
    num_classes = 34
    
    x = torch.randn(batch_size, time_steps, input_channels)
    x_mask = torch.ones(batch_size, time_steps)
    y_targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test CNN-GRU (Single CNN + GRU)
    print("\n📊 CNN-GRU (Single CNN + GRU) 테스트:")
    model_cnn_gru = CNNGRU(
        learning_rate=1e-3,
        input_size=input_channels,
        hidden_size=64,
        classes=num_classes
    )
    
    logits, loss = model_cnn_gru(x, x_mask, y_targets)
    print(f"출력 shape: {logits.shape}")
    print(f"손실: {loss.item():.4f}")
    
    # Test MS-CGRU (Multi-Scale CNN + Single GRU)
    print("\n📊 MS-CGRU (Multi-Scale CNN + Single GRU) 테스트:")
    model_ms_cgru = MSCGRU(
        learning_rate=1e-3,
        input_size=input_channels,
        hidden_size=64,
        classes=num_classes
    )
    
    logits, loss = model_ms_cgru(x, x_mask, y_targets)
    print(f"출력 shape: {logits.shape}")
    print(f"손실: {loss.item():.4f}")
    
    # Test MS-CSGRU (Multi-Scale CNN + Stacked GRU)
    print("\n📊 MS-CSGRU (Multi-Scale CNN + Stacked GRU) 테스트:")
    model_ms_csgru = MSCSGRU(
        learning_rate=1e-3,
        input_size=input_channels,
        hidden_size=64,
        classes=num_classes
    )
    
    logits, loss = model_ms_csgru(x, x_mask, y_targets)
    print(f"출력 shape: {logits.shape}")
    print(f"손실: {loss.item():.4f}")
    
    print("\n✅ 모든 모델 테스트 완료!")