"""
3000개 데이터셋용 최적화된 훈련 스크립트
- 더 큰 데이터셋에 맞춘 하이퍼파라미터
- 예상 성능: 98.5%
"""
import sys
import os
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import json
import time

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.misc.DynamicDataModule import DynamicDataModule
from src.StackedGRUModel import StackedGRULightning
from optimal_config_3000 import OPTIMAL_CONFIG_3000, DATASET_CONFIGS, PERFORMANCE_PREDICTIONS

def train_3000_dataset_model():
    """3000개 데이터셋용 모델 훈련"""
    print("🚀 3000개 데이터셋용 최적화 모델 훈련 시작")
    print("=" * 60)
    
    # 설정 정보 출력
    config = OPTIMAL_CONFIG_3000
    print(f"📊 훈련 설정:")
    print(f"   데이터셋 크기: 3000개 (예정)")
    print(f"   모델 구조: hidden_size={config['hidden_size']}, layers={config['num_layers']}")
    print(f"   훈련 설정: lr={config['learning_rate']}, batch_size={config['batch_size']}")
    print(f"   예상 성능: {PERFORMANCE_PREDICTIONS['medium_3000']['expected_accuracy']:.1%}")
    print("=" * 60)
    
    # 결과 디렉토리 생성
    os.makedirs('results_3000', exist_ok=True)
    os.makedirs('results_3000/checkpoints', exist_ok=True)
    os.makedirs('results_3000/logs', exist_ok=True)
    
    # 데이터 모듈 설정 (3000개 데이터셋용)
    data_module = DynamicDataModule(
        data_dir='/home/billy/25-1kp/SignGlove-DataAnalysis',  # 새로운 데이터셋 경로로 변경 필요
        time_steps=config['time_steps'],
        n_channels=config['n_channels'],
        batch_size=config['batch_size'],
        use_test_split=True,
        seed=config['seed']
    )
    
    print("📁 데이터 모듈 설정 완료")
    print(f"   타임스텝: {config['time_steps']}")
    print(f"   채널 수: {config['n_channels']}")
    print(f"   배치 사이즈: {config['batch_size']}")
    
    # 모델 생성 (3000개 데이터셋용 최적 설정)
    model = StackedGRULightning(
        input_size=config['n_channels'],
        hidden_size=config['hidden_size'],
        num_classes=config['num_classes'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        dense_size=config['dense_size'],
        bidirectional=config['bidirectional'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        adamw_beta1=config['adamw_beta1'],
        adamw_beta2=config['adamw_beta2'],
        adamw_eps=config['adamw_eps'],
        lr_scheduler_factor=config['lr_scheduler_factor'],
        lr_scheduler_patience=config['lr_scheduler_patience'],
        lr_scheduler_min_lr=config['lr_scheduler_min_lr'],
        lr_scheduler_threshold=config['lr_scheduler_threshold']
    )
    
    print("🤖 모델 생성 완료")
    print(f"   총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 콜백 설정
    early_stopping = EarlyStopping(
        monitor='val/accuracy',
        patience=config['early_stopping_patience'],
        mode='max',
        verbose=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val/accuracy',
        dirpath='./results_3000/checkpoints/',
        filename='model-3000-{epoch:02d}-{val/accuracy:.4f}',
        save_top_k=3,  # 상위 3개 모델 저장
        mode='max',
        verbose=True
    )
    
    # 로거 설정
    logger = CSVLogger(
        save_dir='./results_3000/logs/',
        name='model_3000_dataset'
    )
    
    # 트레이너 설정
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=20,  # 더 큰 배치 사이즈에 맞춰 조정
        enable_progress_bar=True,
        precision=16  # 메모리 효율성을 위한 mixed precision
    )
    
    print("🏃‍♂️ 훈련 시작...")
    start_time = time.time()
    
    # 훈련
    trainer.fit(model, data_module)
    
    # 테스트
    test_results = trainer.test(model, data_module)
    
    training_time = time.time() - start_time
    
    # 결과 저장
    results = {
        'model_name': 'StackedGRU_3000',
        'dataset_size': 3000,
        'config': config,
        'test_accuracy': float(test_results[0]['test/accuracy']),
        'test_loss': float(test_results[0]['test/loss']),
        'best_val_accuracy': float(trainer.callback_metrics.get('val/accuracy', 0)),
        'best_val_loss': float(trainer.callback_metrics.get('val/loss', 0)),
        'total_epochs': trainer.current_epoch + 1,
        'training_time': training_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 결과 저장
    with open('results_3000/training_results_3000.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print("\n" + "=" * 60)
    print("🎉 3000개 데이터셋 훈련 완료!")
    print(f"✅ 테스트 정확도: {results['test_accuracy']:.4f}")
    print(f"✅ 테스트 손실: {results['test_loss']:.4f}")
    print(f"✅ 훈련 시간: {training_time:.1f}초")
    print(f"✅ 총 에포크: {results['total_epochs']}")
    
    # 성능 비교
    expected_acc = PERFORMANCE_PREDICTIONS['medium_3000']['expected_accuracy']
    actual_acc = results['test_accuracy']
    
    print(f"\n📊 성능 분석:")
    print(f"   예상 성능: {expected_acc:.1%}")
    print(f"   실제 성능: {actual_acc:.1%}")
    print(f"   성능 차이: {actual_acc - expected_acc:+.1%}")
    
    if actual_acc >= expected_acc:
        print("   🎯 목표 성능 달성!")
    else:
        print("   📈 추가 최적화 필요")
    
    print("=" * 60)
    
    return results

def compare_with_previous_model():
    """이전 모델(598개)과 성능 비교"""
    print("\n📊 이전 모델과 성능 비교")
    print("-" * 40)
    
    # 이전 모델 성능 (598개 데이터셋)
    previous_acc = 0.9750  # Improved_2 성능
    
    print(f"598개 데이터셋 (이전): {previous_acc:.1%}")
    print(f"3000개 데이터셋 (현재): 결과 확인 필요")
    print(f"예상 개선: +1.0% (데이터셋 5배 증가)")
    
    print("\n🔍 개선 요인:")
    print("   - 더 큰 훈련 데이터 (598 → 3000개)")
    print("   - 더 큰 모델 (hidden_size 48 → 64)")
    print("   - 더 깊은 네트워크 (layers 1 → 2)")
    print("   - 최적화된 하이퍼파라미터")

if __name__ == "__main__":
    try:
        # 설정 비교 출력
        from optimal_config_3000 import print_config_comparison
        print_config_comparison()
        
        print("\n" + "=" * 60)
        
        # 훈련 실행
        results = train_3000_dataset_model()
        
        # 성능 비교
        compare_with_previous_model()
        
    except Exception as e:
        print(f"❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

