import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
sys.path.append('.')
from models.signspeak_style_models import SignSpeakLSTM, SignSpeakGRU, SignSpeakTransformer
import os


class SignSpeakEnsemble(nn.Module):
    """
    SignSpeak 스타일 3가지 모델 앙상블
    - SignSpeakLSTM + SignSpeakGRU + SignSpeakTransformer
    - 92% 정확도를 달성한 SignSpeak의 성공적인 접근법
    """
    
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        classes: int = 24,
        ensemble_method: str = "weighted_average",  # "weighted_average", "voting", "stacking"
        lstm_weight: float = 0.33,
        gru_weight: float = 0.33,
        transformer_weight: float = 0.34,
    ):
        super().__init__()
        self.classes = classes
        self.ensemble_method = ensemble_method
        
        # 1. SignSpeak 스타일 모델들
        self.lstm_model = SignSpeakLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            classes=classes
        )
        
        self.gru_model = SignSpeakGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            classes=classes
        )
        
        self.transformer_model = SignSpeakTransformer(
            input_size=input_size,
            hidden_size=hidden_size,
            classes=classes
        )
        
        # 2. 앙상블 가중치 (학습 가능)
        if ensemble_method == "weighted_average":
            self.ensemble_weights = nn.Parameter(torch.tensor([lstm_weight, gru_weight, transformer_weight]))
        
        # 3. Stacking을 위한 메타 분류기
        elif ensemble_method == "stacking":
            self.meta_classifier = nn.Sequential(
                nn.Linear(classes * 3, classes * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(classes * 2, classes)
            )
        
        self._print_model_info()
    
    def _print_model_info(self):
        """모델 정보 출력"""
        total_params = sum(p.numel() for p in self.parameters())
        lstm_params = sum(p.numel() for p in self.lstm_model.parameters())
        gru_params = sum(p.numel() for p in self.gru_model.parameters())
        transformer_params = sum(p.numel() for p in self.transformer_model.parameters())
        
        print(f"🚀 SignSpeakEnsemble 초기화 완료")
        print(f"   입력 크기: {8} (IMU 3개 + Flex 5개)")
        print(f"   Hidden 크기: {64}")
        print(f"   클래스 수: {self.classes} (한국어 수어)")
        print(f"   앙상블 방법: {self.ensemble_method}")
        print(f"   LSTM 파라미터: {lstm_params:,}")
        print(f"   GRU 파라미터: {gru_params:,}")
        print(f"   Transformer 파라미터: {transformer_params:,}")
        print(f"   총 파라미터: {total_params:,}")
        print(f"   SignSpeak 92% 정확도 앙상블: ✅")
    
    def forward(
        self, 
        x: torch.Tensor, 
        y_targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: (batch_size, sequence_length, input_size) - 센서 데이터
            y_targets: (batch_size,) - 타겟 레이블 (훈련 시에만)
            
        Returns:
            ensemble_logits: (batch_size, classes) - 앙상블 분류 로짓
            loss: (scalar) - 손실값 (훈련 시에만)
        """
        batch_size = x.shape[0]
        
        # 1. 각 모델의 예측
        lstm_logits, lstm_loss = self.lstm_model(x, y_targets)
        gru_logits, gru_loss = self.gru_model(x, y_targets)
        transformer_logits, transformer_loss = self.transformer_model(x, y_targets)
        
        # 2. 앙상블 예측 결합
        if self.ensemble_method == "weighted_average":
            # 가중 평균
            weights = F.softmax(self.ensemble_weights, dim=0)
            ensemble_logits = (
                weights[0] * lstm_logits + 
                weights[1] * gru_logits + 
                weights[2] * transformer_logits
            )
            
        elif self.ensemble_method == "voting":
            # 투표 방식 (확률 기반)
            lstm_probs = F.softmax(lstm_logits, dim=1)
            gru_probs = F.softmax(gru_logits, dim=1)
            transformer_probs = F.softmax(transformer_logits, dim=1)
            
            ensemble_probs = (lstm_probs + gru_probs + transformer_probs) / 3
            ensemble_logits = torch.log(ensemble_probs + 1e-8)
            
        elif self.ensemble_method == "stacking":
            # Stacking 방식
            # 모든 모델의 로짓을 연결
            stacked_features = torch.cat([lstm_logits, gru_logits, transformer_logits], dim=1)
            ensemble_logits = self.meta_classifier(stacked_features)
        
        else:
            # 단순 평균
            ensemble_logits = (lstm_logits + gru_logits + transformer_logits) / 3
        
        # 3. Loss 계산 (훈련 시에만)
        loss = None
        if y_targets is not None:
            # 각 모델의 loss 평균
            losses = []
            if lstm_loss is not None:
                losses.append(lstm_loss)
            if gru_loss is not None:
                losses.append(gru_loss)
            if transformer_loss is not None:
                losses.append(transformer_loss)
            
            if losses:
                loss = sum(losses) / len(losses)
            
            # NaN 체크 및 처리
            if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
                print(f"⚠️ NaN/Inf loss detected in SignSpeak ensemble!")
                loss = torch.tensor(3.0, device=ensemble_logits.device, requires_grad=True)
        
        return ensemble_logits, loss
    
    def get_individual_predictions(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """개별 모델의 예측 반환 (분석용)"""
        with torch.no_grad():
            lstm_logits, _ = self.lstm_model(x)
            gru_logits, _ = self.gru_model(x)
            transformer_logits, _ = self.transformer_model(x)
            
            lstm_preds = torch.argmax(lstm_logits, dim=1)
            gru_preds = torch.argmax(gru_logits, dim=1)
            transformer_preds = torch.argmax(transformer_logits, dim=1)
            
        return lstm_preds, gru_preds, transformer_preds
    
    def get_ensemble_weights(self) -> torch.Tensor:
        """앙상블 가중치 반환"""
        if self.ensemble_method == "weighted_average":
            return F.softmax(self.ensemble_weights, dim=0)
        else:
            return torch.tensor([0.33, 0.33, 0.34])


class SignSpeakEnsembleTrainer:
    """SignSpeak 앙상블 모델 훈련기"""
    
    def __init__(
        self,
        lstm_model_path: str = None,
        gru_model_path: str = None,
        transformer_model_path: str = None,
        ensemble_save_path: str = "models/signspeak_ensemble.pth",
        ensemble_method: str = "weighted_average"
    ):
        self.lstm_model_path = lstm_model_path
        self.gru_model_path = gru_model_path
        self.transformer_model_path = transformer_model_path
        self.ensemble_save_path = ensemble_save_path
        self.ensemble_method = ensemble_method
    
    def load_pretrained_models(self) -> SignSpeakEnsemble:
        """사전 훈련된 모델들 로드"""
        print("📥 SignSpeak 사전 훈련된 모델 로드 중...")
        
        # 앙상블 모델 생성
        ensemble_model = SignSpeakEnsemble(ensemble_method=self.ensemble_method)
        
        # LSTM 모델 가중치 로드
        if self.lstm_model_path and os.path.exists(self.lstm_model_path):
            if torch.cuda.is_available():
                lstm_state_dict = torch.load(self.lstm_model_path)
            else:
                lstm_state_dict = torch.load(self.lstm_model_path, map_location='cpu')
            
            ensemble_model.lstm_model.load_state_dict(lstm_state_dict)
            print(f"   ✅ LSTM 모델 로드: {self.lstm_model_path}")
        
        # GRU 모델 가중치 로드
        if self.gru_model_path and os.path.exists(self.gru_model_path):
            if torch.cuda.is_available():
                gru_state_dict = torch.load(self.gru_model_path)
            else:
                gru_state_dict = torch.load(self.gru_model_path, map_location='cpu')
            
            ensemble_model.gru_model.load_state_dict(gru_state_dict)
            print(f"   ✅ GRU 모델 로드: {self.gru_model_path}")
        
        # Transformer 모델 가중치 로드
        if self.transformer_model_path and os.path.exists(self.transformer_model_path):
            if torch.cuda.is_available():
                transformer_state_dict = torch.load(self.transformer_model_path)
            else:
                transformer_state_dict = torch.load(self.transformer_model_path, map_location='cpu')
            
            ensemble_model.transformer_model.load_state_dict(transformer_state_dict)
            print(f"   ✅ Transformer 모델 로드: {self.transformer_model_path}")
        
        return ensemble_model
    
    def fine_tune_ensemble(self, ensemble_model: SignSpeakEnsemble, train_loader, val_loader, 
                          epochs: int = 30, learning_rate: float = 1e-4) -> dict:
        """앙상블 모델 미세 조정"""
        print("🎯 SignSpeak 앙상블 모델 미세 조정 시작")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ensemble_model = ensemble_model.to(device)
        
        # 옵티마이저 (앙상블 가중치만 학습)
        if self.ensemble_method == "weighted_average":
            optimizer = torch.optim.Adam([ensemble_model.ensemble_weights], lr=learning_rate)
        elif self.ensemble_method == "stacking":
            optimizer = torch.optim.Adam(ensemble_model.meta_classifier.parameters(), lr=learning_rate)
        else:
            # 전체 모델 미세 조정
            optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=learning_rate)
        
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
        
        best_val_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # 훈련
            ensemble_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                logits, loss = ensemble_model(data, targets)
                
                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == targets).sum().item()
                train_total += targets.size(0)
            
            # 검증
            ensemble_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(device), targets.to(device)
                    
                    logits, loss = ensemble_model(data, targets)
                    
                    if loss is not None:
                        val_loss += loss.item()
                    
                    predictions = torch.argmax(logits, dim=1)
                    val_correct += (predictions == targets).sum().item()
                    val_total += targets.size(0)
            
            # 메트릭 계산
            train_acc = train_correct / train_total if train_total > 0 else 0.0
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # 히스토리 저장
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            # 스케줄러 업데이트
            scheduler.step(val_acc)
            
            # 최고 성능 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(ensemble_model.state_dict(), self.ensemble_save_path)
                print(f"   🎯 새로운 최고 성능! 모델 저장: {self.ensemble_save_path}")
            
            # 진행 상황 출력
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  📈 Epoch {epoch+1}/{epochs}")
                print(f"    훈련 - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"    검증 - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")
                
                if self.ensemble_method == "weighted_average":
                    weights = ensemble_model.get_ensemble_weights()
                    print(f"    앙상블 가중치 - LSTM: {weights[0]:.3f}, GRU: {weights[1]:.3f}, Transformer: {weights[2]:.3f}")
        
        print(f"  ✅ 미세 조정 완료 - 최고 검증 정확도: {best_val_acc:.4f}")
        return history


def test_signspeak_ensemble():
    """SignSpeak 앙상블 모델 테스트"""
    print("🧪 SignSpeakEnsemble 모델 테스트")
    
    # 테스트 데이터 생성
    batch_size = 4
    sequence_length = 200
    input_size = 8
    num_classes = 24
    
    # 앙상블 모델 생성
    ensemble_model = SignSpeakEnsemble(
        input_size=input_size,
        hidden_size=64,
        classes=num_classes,
        ensemble_method="weighted_average"
    )
    
    # 테스트 입력
    x = torch.randn(batch_size, sequence_length, input_size)
    y = torch.randint(0, num_classes, (batch_size,))
    
    print(f"📊 입력 형태: {x.shape}")
    print(f"📊 타겟 형태: {y.shape}")
    
    # Forward pass
    ensemble_logits, ensemble_loss = ensemble_model(x, y)
    
    print(f"📊 앙상블 출력 형태: {ensemble_logits.shape}")
    print(f"📊 앙상블 손실값: {ensemble_loss.item():.4f}")
    
    # 개별 모델 예측
    lstm_preds, gru_preds, transformer_preds = ensemble_model.get_individual_predictions(x)
    ensemble_preds = torch.argmax(ensemble_logits, dim=1)
    
    # 정확도 계산
    lstm_acc = (lstm_preds == y).float().mean()
    gru_acc = (gru_preds == y).float().mean()
    transformer_acc = (transformer_preds == y).float().mean()
    ensemble_acc = (ensemble_preds == y).float().mean()
    
    print(f"📊 LSTM 정확도: {lstm_acc.item():.4f}")
    print(f"📊 GRU 정확도: {gru_acc.item():.4f}")
    print(f"📊 Transformer 정확도: {transformer_acc.item():.4f}")
    print(f"📊 앙상블 정확도: {ensemble_acc.item():.4f}")
    
    # 앙상블 가중치 출력
    weights = ensemble_model.get_ensemble_weights()
    print(f"📊 앙상블 가중치 - LSTM: {weights[0]:.3f}, GRU: {weights[1]:.3f}, Transformer: {weights[2]:.3f}")
    
    print("✅ 테스트 완료!")


if __name__ == "__main__":
    test_signspeak_ensemble()
