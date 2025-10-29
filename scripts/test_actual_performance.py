"""
PyTorch Lightning Trainer로 실제 테스트 수행
훈련 시와 완전히 동일한 방식
"""

import sys
sys.path.append('.')

import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

from src.misc.DynamicDataModule import DynamicDataModule
from src.models.MultiScale3DGRUModels import MS3DGRU

def test_with_trainer():
    """PyTorch Lightning Trainer로 테스트"""
    
    print('=' * 80)
    print('🧪 실제 테스트 데이터셋 성능 평가 (Trainer 사용)')
    print('=' * 80)
    print()
    
    # 데이터 모듈
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
    
    # 모델 로드
    checkpoint_path = 'best_model/ms3dgru_best.ckpt'
    
    # 체크포인트에서 하이퍼파라미터 추출
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    hyper_params = checkpoint.get('hyper_parameters', {})
    
    model = MS3DGRU(
        learning_rate=hyper_params.get('learning_rate', 0.001),
        input_size=hyper_params.get('input_size', 8),
        hidden_size=hyper_params.get('hidden_size', 64),
        classes=hyper_params.get('classes', 24),
        cnn_filters=hyper_params.get('cnn_filters', 32),
        dropout=hyper_params.get('dropout', 0.1)
    )
    
    # State dict 로드
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                cleaned_state_dict[key[6:]] = value
            else:
                cleaned_state_dict[key] = value
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        if missing_keys:
            print(f'⚠️  누락된 키: {len(missing_keys)}개')
        if unexpected_keys:
            print(f'⚠️  예상치 못한 키: {len(unexpected_keys)}개')
    
    print(f'✅ 모델 로드 완료')
    print()
    
    # Trainer 설정
    trainer = L.Trainer(
        accelerator='cpu',
        devices=1,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    
    # 테스트 실행
    print('📌 테스트 실행 중...')
    print('-' * 80)
    results = trainer.test(model, datamodule=datamodule, verbose=True)
    
    print()
    print('=' * 80)
    print('📊 테스트 결과')
    print('=' * 80)
    
    if results and len(results) > 0:
        result = results[0]
        print(f"Test Accuracy: {result.get('test/accuracy', 'N/A')}")
        print(f"Test F1-Score: {result.get('test/f1_score', 'N/A')}")
        print(f"Test Loss: {result.get('test/loss', 'N/A')}")
    else:
        print('결과가 반환되지 않았습니다.')
    
    print()
    print('=' * 80)


if __name__ == '__main__':
    try:
        test_with_trainer()
    except Exception as e:
        print(f'❌ 오류: {e}')
        import traceback
        traceback.print_exc()

