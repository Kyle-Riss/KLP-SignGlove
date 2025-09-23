"""
최고 성능 모델 훈련 스크립트
- Improved_2 설정으로 훈련
- 97.5% 성능 달성
"""
import sys
import os
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.misc.DynamicDataModule import DynamicDataModule
from src.StackedGRUModel import StackedGRULightning
from optimal_config import OPTIMAL_CONFIG

def train_optimal_model():
    """최고 성능 모델 훈련"""
    print("🚀 최고 성능 모델 (Improved_2) 훈련 시작")
    print("=" * 50)
    
    # 데이터 모듈 설정
    data_module = DynamicDataModule(
        data_dir='/home/billy/25-1kp/SignGlove-DataAnalysis',
        time_steps=OPTIMAL_CONFIG['time_steps'],
        n_channels=OPTIMAL_CONFIG['n_channels'],
        batch_size=OPTIMAL_CONFIG['batch_size'],
        use_test_split=True,
        seed=OPTIMAL_CONFIG['seed']
    )
    
    # 모델 생성
    model = StackedGRULightning(
        input_size=OPTIMAL_CONFIG['n_channels'],
        hidden_size=OPTIMAL_CONFIG['hidden_size'],
        num_classes=OPTIMAL_CONFIG['num_classes'],
        num_layers=OPTIMAL_CONFIG['num_layers'],
        dropout=OPTIMAL_CONFIG['dropout'],
        dense_size=OPTIMAL_CONFIG['dense_size'],
        bidirectional=OPTIMAL_CONFIG['bidirectional'],
        learning_rate=OPTIMAL_CONFIG['learning_rate'],
        weight_decay=OPTIMAL_CONFIG['weight_decay'],
        adamw_beta1=OPTIMAL_CONFIG['adamw_beta1'],
        adamw_beta2=OPTIMAL_CONFIG['adamw_beta2'],
        adamw_eps=OPTIMAL_CONFIG['adamw_eps'],
        lr_scheduler_factor=OPTIMAL_CONFIG['lr_scheduler_factor'],
        lr_scheduler_patience=OPTIMAL_CONFIG['lr_scheduler_patience'],
        lr_scheduler_min_lr=OPTIMAL_CONFIG['lr_scheduler_min_lr'],
        lr_scheduler_threshold=OPTIMAL_CONFIG['lr_scheduler_threshold']
    )
    
    # 콜백 설정
    early_stopping = EarlyStopping(
        monitor='val/accuracy',
        patience=OPTIMAL_CONFIG['early_stopping_patience'],
        mode='max',
        verbose=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val/accuracy',
        dirpath='./checkpoints/',
        filename='optimal-model-{epoch:02d}-{val/accuracy:.4f}',
        save_top_k=1,
        mode='max',
        verbose=True
    )
    
    # 로거 설정
    logger = CSVLogger(
        save_dir='./logs/',
        name='optimal_model'
    )
    
    # 트레이너 설정
    trainer = pl.Trainer(
        max_epochs=OPTIMAL_CONFIG['max_epochs'],
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    
    # 훈련
    trainer.fit(model, data_module)
    
    # 테스트
    test_results = trainer.test(model, data_module)
    
    print("\n" + "=" * 50)
    print("🎉 훈련 완료!")
    print(f"테스트 정확도: {test_results[0]['test/accuracy']:.4f}")
    print(f"테스트 손실: {test_results[0]['test/loss']:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    train_optimal_model()
