import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix
import numpy as np
from typing import Dict, Any, Optional


class StackedGRU(nn.Module):
    """
    SignsSpeak 논문 기반 Stacked GRU 모델
    
    아키텍처:
    - 입력: (batch_size, time_steps, channels)
    - 2개의 GRU 레이어 (hidden_size=64)
    - Dropout (0.2)
    - Dense 레이어 (128)
    - 출력: (batch_size, num_classes)
    """
    
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 24,
        dropout: float = 0.2,
        dense_size: int = 128,
        bidirectional: bool = False
    ):
        super(StackedGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.dense_size = dense_size
        self.bidirectional = bidirectional
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Calculate GRU output size
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Dense layers
        self.dense = nn.Linear(gru_output_size, dense_size)
        self.output_layer = nn.Linear(dense_size, num_classes)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)  # (batch_size, time_steps, hidden_size)
        
        # Take the last timestep output
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        dropped = self.dropout_layer(last_output)
        
        # Dense layers
        dense_out = self.relu(self.dense(dropped))
        output = self.output_layer(dense_out)
        
        return output


class StackedGRULightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for StackedGRU model
    SignsSpeak 논문 기반 훈련 설정
    """
    
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 24,
        dropout: float = 0.2,
        dense_size: int = 128,
        bidirectional: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        # AdamW hyperparameters (SignsSpeak 논문 기반)
        adamw_beta1: float = 0.9,
        adamw_beta2: float = 0.999,
        adamw_eps: float = 1e-8,
        # Learning rate scheduler hyperparameters
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 10,
        lr_scheduler_min_lr: float = 1e-6,
        lr_scheduler_threshold: float = 1e-4,
        class_names: Optional[list] = None
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Model
        self.model = StackedGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            dense_size=dense_size,
            bidirectional=bidirectional
        )
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adamw_beta1 = adamw_beta1
        self.adamw_beta2 = adamw_beta2
        self.adamw_eps = adamw_eps
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_min_lr = lr_scheduler_min_lr
        self.lr_scheduler_threshold = lr_scheduler_threshold
        
        # Class names
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        # F1-Score metrics (SignsSpeak 논문에서 중요하게 사용)
        self.train_f1_macro = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1_macro = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1_macro = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        
        self.train_f1_weighted = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.val_f1_weighted = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_f1_weighted = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        
        self.train_f1_micro = F1Score(task="multiclass", num_classes=num_classes, average="micro")
        self.val_f1_micro = F1Score(task="multiclass", num_classes=num_classes, average="micro")
        self.test_f1_micro = F1Score(task="multiclass", num_classes=num_classes, average="micro")
        
        # Precision and Recall
        self.train_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        
        self.train_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        
        # Confusion Matrix
        self.train_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.val_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        # Per-class F1-Score for detailed analysis
        self.test_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average="none")
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch["measurement"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.train_f1_macro(preds, y)
        self.train_f1_weighted(preds, y)
        self.train_f1_micro(preds, y)
        self.train_precision(preds, y)
        self.train_recall(preds, y)
        self.train_confusion_matrix(preds, y)
        
        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1_macro", self.train_f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1_weighted", self.train_f1_weighted, on_step=False, on_epoch=True)
        self.log("train/f1_micro", self.train_f1_micro, on_step=False, on_epoch=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch["measurement"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.val_f1_macro(preds, y)
        self.val_f1_weighted(preds, y)
        self.val_f1_micro(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_confusion_matrix(preds, y)
        
        # Logging
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1_macro", self.val_f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1_weighted", self.val_f1_weighted, on_step=False, on_epoch=True)
        self.log("val/f1_micro", self.val_f1_micro, on_step=False, on_epoch=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch["measurement"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, y)
        self.test_f1_macro(preds, y)
        self.test_f1_weighted(preds, y)
        self.test_f1_micro(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        self.test_confusion_matrix(preds, y)
        self.test_f1_per_class(preds, y)
        
        # Logging
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1_macro", self.test_f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1_weighted", self.test_f1_weighted, on_step=False, on_epoch=True)
        self.log("test/f1_micro", self.test_f1_micro, on_step=False, on_epoch=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        SignsSpeak 논문 기반 옵티마이저 설정
        - AdamW 옵티마이저 사용
        - Plateau learning rate decay 적용
        """
        # AdamW 옵티마이저 (SignsSpeak 논문과 동일)
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.adamw_beta1, self.adamw_beta2),
            eps=self.adamw_eps
        )
        
        # Plateau learning rate decay (SignsSpeak 논문과 동일)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",  # 정확도 기준으로 모니터링
            factor=self.lr_scheduler_factor,  # 학습률 감소 비율
            patience=self.lr_scheduler_patience,  # 대기 에포크 수
            verbose=True,
            min_lr=self.lr_scheduler_min_lr,  # 최소 학습률
            threshold=self.lr_scheduler_threshold,  # 개선 임계값
            threshold_mode='rel'  # 상대적 임계값
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/accuracy",  # 검증 정확도 모니터링
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def get_detailed_evaluation_report(self) -> str:
        """상세한 평가 결과 보고서 생성"""
        if not hasattr(self, 'test_accuracy') or self.test_accuracy.compute() is None:
            return "❌ 평가 결과가 없습니다. 먼저 테스트를 실행해주세요."
        
        # 기본 메트릭 계산
        accuracy = self.test_accuracy.compute().item()
        f1_macro = self.test_f1_macro.compute().item()
        f1_weighted = self.test_f1_weighted.compute().item()
        f1_micro = self.test_f1_micro.compute().item()
        precision = self.test_precision.compute().item()
        recall = self.test_recall.compute().item()
        
        # 클래스별 F1-Score
        f1_per_class = self.test_f1_per_class.compute()
        
        # 혼동 행렬
        confusion_matrix = self.test_confusion_matrix.compute()
        
        report = f"""
📊 SignsSpeak 논문 기반 상세 평가 보고서
{'='*60}

🎯 전체 성능 지표:
  • 정확도 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)
  • F1-Score (Macro): {f1_macro:.4f} ({f1_macro*100:.2f}%)
  • F1-Score (Weighted): {f1_weighted:.4f} ({f1_weighted*100:.2f}%)
  • F1-Score (Micro): {f1_micro:.4f} ({f1_micro*100:.2f}%)
  • Precision (Macro): {precision:.4f} ({precision*100:.2f}%)
  • Recall (Macro): {recall:.4f} ({recall*100:.2f}%)

📈 F1-Score 분석:
  • Macro F1: 클래스별 F1의 평균 (불균형 데이터에 적합)
  • Weighted F1: 클래스별 샘플 수에 따른 가중 평균
  • Micro F1: 전체 TP, FP, FN으로 계산 (정확도와 동일)

🔍 클래스별 F1-Score (상위 10개):
"""
        
        # 클래스별 F1-Score 정렬
        class_f1_scores = []
        for i, f1_score in enumerate(f1_per_class):
            class_f1_scores.append((self.class_names[i], f1_score.item()))
        
        class_f1_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (class_name, f1_score) in enumerate(class_f1_scores[:10]):
            report += f"  {i+1:2d}. {class_name}: {f1_score:.4f} ({f1_score*100:.2f}%)\n"
        
        if len(class_f1_scores) > 10:
            report += f"  ... (총 {len(class_f1_scores)}개 클래스)\n"
        
        # 성능 분석
        avg_f1 = sum([score for _, score in class_f1_scores]) / len(class_f1_scores)
        min_f1 = min([score for _, score in class_f1_scores])
        max_f1 = max([score for _, score in class_f1_scores])
        
        report += f"""
📊 성능 분석:
  • 평균 F1-Score: {avg_f1:.4f} ({avg_f1*100:.2f}%)
  • 최고 F1-Score: {max_f1:.4f} ({max_f1*100:.2f}%)
  • 최저 F1-Score: {min_f1:.4f} ({min_f1*100:.2f}%)
  • F1-Score 표준편차: {np.std([score for _, score in class_f1_scores]):.4f}

🎯 SignsSpeak 논문 비교:
  • 정확도: {accuracy*100:.2f}% (논문 대비 우수/보통/개선 필요)
  • F1-Score: {f1_macro*100:.2f}% (불균형 데이터 고려한 평가)

💡 권장사항:
"""
        
        if f1_macro < 0.7:
            report += "  • F1-Score가 낮습니다. 데이터 증강이나 모델 개선이 필요합니다.\n"
        elif f1_macro < 0.85:
            report += "  • F1-Score가 보통입니다. 하이퍼파라미터 튜닝을 고려해보세요.\n"
        else:
            report += "  • F1-Score가 우수합니다! 현재 모델이 잘 작동하고 있습니다.\n"
        
        if max_f1 - min_f1 > 0.3:
            report += "  • 클래스별 성능 차이가 큽니다. 불균형 데이터 처리를 고려해보세요.\n"
        
        report += f"{'='*60}"
        
        return report
    
    def save_evaluation_results(self, filepath: str = "evaluation_results.txt"):
        """평가 결과를 파일로 저장"""
        report = self.get_detailed_evaluation_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 평가 결과가 {filepath}에 저장되었습니다.")


# Test the model
if __name__ == "__main__":
    print("🧪 StackedGRU 모델 테스트 시작...")
    
    # Test model creation
    model = StackedGRU(
        input_size=8,
        hidden_size=64,
        num_layers=2,
        num_classes=24,
        dropout=0.2,
        dense_size=128
    )
    
    # Test forward pass
    batch_size = 16
    time_steps = 100
    input_size = 8
    
    x = torch.randn(batch_size, time_steps, input_size)
    output = model(x)
    
    print(f"✅ 모델 테스트 성공!")
    print(f"  입력 shape: {x.shape}")
    print(f"  출력 shape: {output.shape}")
    print(f"  모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test Lightning wrapper
    lightning_model = StackedGRULightning(
        input_size=8,
        hidden_size=64,
        num_layers=2,
        num_classes=24,
        dropout=0.2,
        dense_size=128,
        learning_rate=1e-3,
        class_names=['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    )
    
    print(f"✅ Lightning 모델 테스트 성공!")
    print(f"  클래스 수: {lightning_model.model.num_classes}")
    print(f"  히든 크기: {lightning_model.model.hidden_size}")
    print(f"  GRU 레이어 수: {lightning_model.model.num_layers}")
    print(f"  드롭아웃: {lightning_model.model.dropout}")
    print(f"  Dense 크기: {lightning_model.model.dense_size}")
    
    print("\n🎯 SignsSpeak 논문 기반 Stacked GRU 모델 구현 완료!")