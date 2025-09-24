"""
최고 성능 모델 (Improved_2: 97.5%) 설정
- hidden_size: 48
- num_layers: 1  
- dropout: 0.15
- dense_size: 96
- learning_rate: 0.0003
- batch_size: 16
- weight_decay: 0.001
- resampling_method: "padding" (ASL-Sign-Research 방식)
"""
OPTIMAL_CONFIG = {
    'hidden_size': 48,
    'num_layers': 1,
    'dropout': 0.15,
    'dense_size': 96,
    'bidirectional': False,
    'learning_rate': 0.0003,
    'batch_size': 16,
    'weight_decay': 0.001,
    'adamw_beta1': 0.9,
    'adamw_beta2': 0.999,
    'adamw_eps': 1e-08,
    'lr_scheduler_factor': 0.3,
    'lr_scheduler_patience': 15,
    'lr_scheduler_min_lr': 1e-07,
    'lr_scheduler_threshold': 1e-05,
    'time_steps': 100,
    'n_channels': 8,
    'num_classes': 24,
    'max_epochs': 100,
    'early_stopping_patience': 30,
    'seed': 42
}

# 성능 정보
PERFORMANCE_INFO = {
    'test_accuracy': 0.9750,
    'test_loss': 0.0753,
    'model_name': 'Improved_2',
    'description': 'StackedGRU with optimal hyperparameters'
}
