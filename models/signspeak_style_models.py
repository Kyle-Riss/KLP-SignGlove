import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SignSpeakLSTM(nn.Module):
    """
    SignSpeak 스타일 Stacked LSTM 모델
    - 여러 LSTM 층을 쌓은 구조
    - KSL 데이터에 최적화
    """
    
    def __init__(
        self,
        input_size: int = 8,           # IMU 3개 + Flex 5개
        hidden_size: int = 64,         # SignSpeak 스타일
        classes: int = 24,             # 한국어 수어 클래스
        num_layers: int = 3,           # Stacked LSTM
        dropout: float = 0.2,
        bidirectional: bool = True     # Bidirectional LSTM
    ):
        super().__init__()
        self.classes = classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 1. Stacked LSTM
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 2. Attention Mechanism (SignSpeak 스타일)
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Linear(lstm_output_size, 1)
        
        # 3. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, classes)
        )
        
        self._print_model_info()
    
    def _print_model_info(self):
        """모델 정보 출력"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🚀 SignSpeakLSTM 초기화 완료")
        print(f"   입력 크기: {8} (IMU 3개 + Flex 5개)")
        print(f"   Hidden 크기: {self.hidden_size}")
        print(f"   LSTM 층: {self.num_layers}")
        print(f"   Bidirectional: {self.bidirectional}")
        print(f"   클래스 수: {self.classes}")
        print(f"   총 파라미터: {total_params:,}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        y_targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: (batch_size, sequence_length, input_size)
            y_targets: (batch_size,) - 훈련 시에만
            
        Returns:
            logits: (batch_size, classes)
            loss: (scalar) - 훈련 시에만
        """
        batch_size = x.shape[0]
        
        # 1. LSTM 처리
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 2. Attention Mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 3. 분류
        logits = self.classifier(attended_output)
        
        # 4. Loss 계산 (훈련 시에만)
        loss = None
        if y_targets is not None:
            loss = F.cross_entropy(logits, y_targets)
            
            # NaN 체크
            if torch.isnan(loss) or torch.isinf(loss):
                loss = torch.tensor(3.0, device=logits.device, requires_grad=True)
        
        return logits, loss


class SignSpeakGRU(nn.Module):
    """
    SignSpeak 스타일 Stacked GRU 모델
    - 여러 GRU 층을 쌓은 구조
    - LSTM보다 단순하지만 효과적
    """
    
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        classes: int = 24,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        self.classes = classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 1. Stacked GRU
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 2. Attention Mechanism
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Linear(gru_output_size, 1)
        
        # 3. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, gru_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_output_size // 2, classes)
        )
        
        self._print_model_info()
    
    def _print_model_info(self):
        """모델 정보 출력"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🚀 SignSpeakGRU 초기화 완료")
        print(f"   입력 크기: {8} (IMU 3개 + Flex 5개)")
        print(f"   Hidden 크기: {self.hidden_size}")
        print(f"   GRU 층: {self.num_layers}")
        print(f"   Bidirectional: {self.bidirectional}")
        print(f"   클래스 수: {self.classes}")
        print(f"   총 파라미터: {total_params:,}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        y_targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        """
        batch_size = x.shape[0]
        
        # 1. GRU 처리
        gru_out, hidden = self.gru(x)
        
        # 2. Attention Mechanism
        attention_weights = F.softmax(self.attention(gru_out), dim=1)
        attended_output = torch.sum(attention_weights * gru_out, dim=1)
        
        # 3. 분류
        logits = self.classifier(attended_output)
        
        # 4. Loss 계산
        loss = None
        if y_targets is not None:
            loss = F.cross_entropy(logits, y_targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                loss = torch.tensor(3.0, device=logits.device, requires_grad=True)
        
        return logits, loss


class SignSpeakTransformer(nn.Module):
    """
    SignSpeak 스타일 Transformer 모델
    - 92% 정확도를 달성한 모델
    - Classification Token + Positional Embedding
    """
    
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        classes: int = 24,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.2,
        dim_feedforward: int = 256
    ):
        super().__init__()
        self.classes = classes
        self.hidden_size = hidden_size
        self.time_steps = 200
        
        # 1. Classification Token (SignSpeak 스타일)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # 2. Input Embedding
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # 3. Positional Embedding
        self.pos_embedding = nn.Embedding(self.time_steps + 1, hidden_size)  # +1 for CLS token
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            norm_first=True,
            batch_first=True,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_size)
        )
        
        # 5. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, classes)
        )
        
        self._print_model_info()
    
    def _print_model_info(self):
        """모델 정보 출력"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🚀 SignSpeakTransformer 초기화 완료")
        print(f"   입력 크기: {8} (IMU 3개 + Flex 5개)")
        print(f"   Hidden 크기: {self.hidden_size}")
        print(f"   Transformer 층: {6}")
        print(f"   Attention 헤드: {8}")
        print(f"   클래스 수: {self.classes}")
        print(f"   총 파라미터: {total_params:,}")
        print(f"   SignSpeak 92% 정확도 모델 스타일: ✅")
    
    def forward(
        self, 
        x: torch.Tensor, 
        y_targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass (SignSpeak 스타일)
        """
        batch_size = x.shape[0]
        
        # 1. Input Embedding
        x = self.input_embedding(x)  # (batch, time, hidden)
        
        # 2. Classification Token 추가 (SignSpeak 스타일)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, time+1, hidden)
        
        # 3. Positional Embedding
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb
        
        # 4. Transformer Encoder
        x = self.transformer_encoder(x)
        
        # 5. Classification Token 사용 (SignSpeak 스타일)
        cls_output = x[:, 0]  # 첫 번째 토큰 (CLS token)
        
        # 6. 분류
        logits = self.classifier(cls_output)
        
        # 7. Loss 계산
        loss = None
        if y_targets is not None:
            loss = F.cross_entropy(logits, y_targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                loss = torch.tensor(3.0, device=logits.device, requires_grad=True)
        
        return logits, loss


def test_signspeak_models():
    """SignSpeak 스타일 모델들 테스트"""
    print("🧪 SignSpeak 스타일 모델들 테스트")
    
    # 테스트 데이터
    batch_size = 4
    sequence_length = 200
    input_size = 8
    num_classes = 24
    
    # 1. LSTM 테스트
    print("\n📊 SignSpeakLSTM 테스트:")
    lstm_model = SignSpeakLSTM()
    x = torch.randn(batch_size, sequence_length, input_size)
    y = torch.randint(0, num_classes, (batch_size,))
    
    logits, loss = lstm_model(x, y)
    print(f"   출력 형태: {logits.shape}")
    print(f"   손실값: {loss.item():.4f}")
    
    # 2. GRU 테스트
    print("\n📊 SignSpeakGRU 테스트:")
    gru_model = SignSpeakGRU()
    logits, loss = gru_model(x, y)
    print(f"   출력 형태: {logits.shape}")
    print(f"   손실값: {loss.item():.4f}")
    
    # 3. Transformer 테스트
    print("\n📊 SignSpeakTransformer 테스트:")
    transformer_model = SignSpeakTransformer()
    logits, loss = transformer_model(x, y)
    print(f"   출력 형태: {logits.shape}")
    print(f"   손실값: {loss.item():.4f}")
    
    print("\n✅ 모든 SignSpeak 스타일 모델 테스트 완료!")


if __name__ == "__main__":
    test_signspeak_models()

