"""
3000개 데이터셋에 최적화된 모델 설정
- 기존 598개 → 3000개 (5배 증가)
- 더 큰 모델과 적절한 정규화 필요
"""
import torch

# 3000개 데이터셋용 최적 설정
OPTIMAL_CONFIG_3000 = {
    # 모델 구조 (더 큰 모델로 복잡도 증가)
    'hidden_size': 64,           # 48 → 64 (더 큰 히든 사이즈)
    'num_layers': 2,             # 1 → 2 (더 깊은 네트워크)
    'dropout': 0.2,              # 0.15 → 0.2 (과적합 방지)
    'dense_size': 128,           # 96 → 128 (더 큰 Dense 레이어)
    'bidirectional': False,      # 단방향 유지
    
    # 훈련 설정 (더 큰 데이터셋에 맞춤)
    'learning_rate': 0.001,      # 0.0003 → 0.001 (더 큰 학습률)
    'batch_size': 32,            # 16 → 32 (더 큰 배치 사이즈)
    'weight_decay': 0.0001,      # 0.001 → 0.0001 (약한 정규화)
    
    # 옵티마이저 설정
    'adamw_beta1': 0.9,
    'adamw_beta2': 0.999,
    'adamw_eps': 1e-08,
    
    # 스케줄러 설정
    'lr_scheduler_factor': 0.5,  # 0.3 → 0.5 (더 큰 감소율)
    'lr_scheduler_patience': 20, # 15 → 20 (더 긴 patience)
    'lr_scheduler_min_lr': 1e-06,
    'lr_scheduler_threshold': 1e-04,
    
    # 데이터 설정 (동일)
    'time_steps': 100,           # 타임스텝 동일
    'n_channels': 8,             # 채널 수 동일
    'num_classes': 24,           # 클래스 수 동일
    
    # 훈련 설정
    'max_epochs': 150,           # 100 → 150 (더 많은 에포크)
    'early_stopping_patience': 40, # 30 → 40 (더 긴 patience)
    'seed': 42
}

# 데이터셋 크기별 비교 설정
DATASET_CONFIGS = {
    'small_598': {
        'hidden_size': 48,
        'num_layers': 1,
        'dropout': 0.15,
        'dense_size': 96,
        'learning_rate': 0.0003,
        'batch_size': 16,
        'weight_decay': 0.001,
        'max_epochs': 100,
        'early_stopping_patience': 30,
        'description': '598개 샘플용 설정 (현재 최고 성능)'
    },
    'medium_3000': {
        **OPTIMAL_CONFIG_3000,
        'description': '3000개 샘플용 설정 (예상 최적)'
    },
    'large_7200': {
        'hidden_size': 96,       # 더 큰 모델
        'num_layers': 2,
        'dropout': 0.3,          # 더 강한 정규화
        'dense_size': 192,
        'learning_rate': 0.002,  # 더 큰 학습률
        'batch_size': 64,        # 더 큰 배치
        'weight_decay': 0.0001,
        'max_epochs': 200,
        'early_stopping_patience': 50,
        'description': '7200개 샘플용 설정 (ASL-Sign-Research 규모)'
    }
}

# 성능 예상치
PERFORMANCE_PREDICTIONS = {
    'small_598': {
        'current_accuracy': 0.9750,
        'description': '현재 달성된 성능'
    },
    'medium_3000': {
        'expected_accuracy': 0.9850,  # 더 큰 데이터셋으로 인한 성능 향상 예상
        'description': '3000개 데이터셋 예상 성능'
    },
    'large_7200': {
        'expected_accuracy': 0.9900,  # 더 큰 데이터셋으로 인한 성능 향상 예상
        'description': '7200개 데이터셋 예상 성능'
    }
}

def get_config_for_dataset_size(dataset_size: int):
    """데이터셋 크기에 따른 최적 설정 반환"""
    if dataset_size <= 1000:
        return DATASET_CONFIGS['small_598']
    elif dataset_size <= 5000:
        return DATASET_CONFIGS['medium_3000']
    else:
        return DATASET_CONFIGS['large_7200']

def print_config_comparison():
    """설정 비교 출력"""
    print("📊 데이터셋 크기별 최적 설정 비교")
    print("=" * 60)
    
    for size, config in DATASET_CONFIGS.items():
        print(f"\n🔧 {size.upper()}:")
        print(f"   hidden_size: {config['hidden_size']}")
        print(f"   num_layers: {config['num_layers']}")
        print(f"   dropout: {config['dropout']}")
        print(f"   learning_rate: {config['learning_rate']}")
        print(f"   batch_size: {config['batch_size']}")
        print(f"   max_epochs: {config['max_epochs']}")
        print(f"   설명: {config['description']}")
    
    print(f"\n🎯 3000개 데이터셋 권장 설정:")
    print(f"   모델 복잡도: 증가 (hidden_size 48→64, layers 1→2)")
    print(f"   정규화: 조정 (dropout 0.15→0.2, weight_decay 0.001→0.0001)")
    print(f"   훈련: 확장 (batch_size 16→32, epochs 100→150)")
    print(f"   예상 성능: 98.5% (현재 97.5% 대비 향상)")

if __name__ == "__main__":
    print_config_comparison()
