"""
훈련 스크립트와 동일한 방식으로 실제 테스트 수행
PyTorch Lightning의 test_step을 직접 실행
"""

import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

from src.misc.DynamicDataModule import DynamicDataModule
from src.models.MultiScale3DGRUModels import MS3DGRU

def test_model_with_training_setup():
    """훈련 시와 동일한 설정으로 테스트"""
    
    print('=' * 80)
    print('🧪 훈련 스크립트 방식으로 실제 테스트')
    print('=' * 80)
    print()
    
    # 1. 데이터 모듈 설정
    print('📌 1단계: 데이터 모듈 설정')
    print('-' * 80)
    datamodule = DynamicDataModule(
        data_dir='/home/billy/25-1kp/SignGlove_HW/datasets/unified',
        batch_size=32,
        test_size=0.2,
        val_size=0.2,
        seed=42
    )
    datamodule.setup('test')
    print(f'✅ 테스트 데이터셋 준비 완료')
    print(f'   테스트 샘플 수: {len(datamodule.test_dataset)}')
    print()
    
    # 2. 모델 로드 (체크포인트에서)
    print('📌 2단계: 체크포인트에서 모델 로드')
    print('-' * 80)
    
    checkpoint_path = 'best_model/ms3dgru_best.ckpt'
    
    # 모델 생성
    model = MS3DGRU(
        learning_rate=0.001,
        input_size=8,
        hidden_size=64,
        classes=24,
        cnn_filters=32,
        dropout=0.1
    )
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # 'model.' 접두사 제거
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                cleaned_state_dict[key[6:]] = value
            else:
                cleaned_state_dict[key] = value
        model.load_state_dict(cleaned_state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    print(f'✅ 모델 로드 성공')
    print(f'   파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print()
    
    # 3. Trainer로 테스트 실행
    print('📌 3단계: PyTorch Lightning으로 테스트 실행')
    print('-' * 80)
    
    trainer = L.Trainer(
        accelerator='cpu',
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )
    
    # 테스트 실행
    test_results = trainer.test(model, datamodule=datamodule, verbose=False)
    
    print(f'✅ 테스트 완료!')
    print()
    
    # 4. 상세 결과 수집
    print('📌 4단계: 상세 결과 수집')
    print('-' * 80)
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    model.eval()
    model.to('cpu')
    
    test_loader = datamodule.test_dataloader()
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['measurement']
            y = batch['label']
            x_padding = batch.get('measurement_padding', None)
            
            # Forward pass (훈련 시와 동일)
            logits, loss = model(x, x_padding, y)
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 5. 성능 계산
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    
    CLASS_NAMES = [
        'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
        'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ'
    ]
    
    cm = confusion_matrix(all_labels, all_predictions, labels=range(24))
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    print('=' * 80)
    print('📊 실전 성능 결과')
    print('=' * 80)
    print(f'테스트 샘플 수: {len(all_labels):,}')
    print(f'정확도 (Accuracy): {accuracy * 100:.2f}%')
    print(f'F1-Score (Macro): {f1_macro:.4f}')
    print(f'F1-Score (Weighted): {f1_weighted:.4f}')
    print()
    
    wrong_count = (all_predictions != all_labels).sum()
    print(f'정확 예측: {len(all_labels) - wrong_count:,}개 ({(len(all_labels) - wrong_count) / len(all_labels) * 100:.2f}%)')
    print(f'오분류: {wrong_count:,}개 ({wrong_count / len(all_labels) * 100:.2f}%)')
    print()
    
    print('📊 클래스별 정확도:')
    print('-' * 80)
    for i, (class_name, acc) in enumerate(zip(CLASS_NAMES, class_accuracies)):
        status = '✅' if acc >= 0.98 else '⚠️' if acc >= 0.95 else '❌'
        correct = cm[i, i]
        total = cm.sum(axis=1)[i]
        print(f'{status} {class_name}: {acc * 100:.2f}% ({correct}/{total})')
    
    print()
    print('=' * 80)
    print('✅ 실전 테스트 완료!')
    print('=' * 80)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'class_accuracies': class_accuracies,
        'predictions': all_predictions,
        'labels': all_labels
    }


if __name__ == '__main__':
    try:
        results = test_model_with_training_setup()
    except Exception as e:
        print(f'❌ 오류 발생: {e}')
        import traceback
        traceback.print_exc()

