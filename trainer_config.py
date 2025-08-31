#!/usr/bin/env python3
"""
KLP-SignGlove: Trainer Configuration
한국 수화 인식 모델 훈련을 위한 설정 파일

이 파일은 모델 훈련에 사용된 모든 설정과 데이터 정보를 포함합니다.
다른 연구자들이 동일한 실험을 재현할 수 있도록 상세한 정보를 제공합니다.

작성자: KLP-SignGlove Team
버전: 1.0.0
날짜: 2024
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json

@dataclass
class DataConfig:
    """데이터 관련 설정"""
    
    # 데이터 경로
    data_root: str = "../SignGlove/external/SignGlove_HW/datasets/unified"
    
    # 클래스 정보
    class_names: List[str] = None
    
    # 데이터 분할 비율
    train_ratio: float = 0.6      # 훈련 데이터 60%
    val_ratio: float = 0.2        # 검증 데이터 20%
    test_ratio: float = 0.2       # 테스트 데이터 20%
    
    # 클래스별 최대 샘플 수
    max_samples_per_class: int = 25
    
    # 센서 정보
    sensor_info: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = [
                'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',  # 자음 (14개)
                'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ'  # 모음 (10개)
            ]
        
        if self.sensor_info is None:
            self.sensor_info = {
                'flex_sensors': {
                    'count': 5,
                    'names': ['flex1', 'flex2', 'flex3', 'flex4', 'flex5'],
                    'range': [0, 1023],
                    'description': '손가락 굽힘 센서 (0: 완전 펴짐, 1023: 완전 굽힘)'
                },
                'orientation_sensors': {
                    'count': 3,
                    'names': ['pitch', 'roll', 'yaw'],
                    'range': [-180, 180],
                    'description': '손목 방향 센서 (도 단위)'
                }
            }

@dataclass
class ModelConfig:
    """모델 아키텍처 설정"""
    
    # 모델 타입
    model_type: str = "ImprovedGRU"
    
    # 입력/출력 차원
    input_size: int = 8           # 5개 flex + 3개 orientation
    num_classes: int = 24         # 14개 자음 + 10개 모음
    
    # GRU 설정
    hidden_size: int = 64         # GRU 은닉층 크기
    num_layers: int = 2           # GRU 층 수
    dropout: float = 0.3          # Dropout 비율
    
    # 가중치 초기화
    weight_init: str = "xavier_uniform"
    
    # 모델 파라미터 수 (자동 계산)
    total_parameters: int = None
    
    def calculate_parameters(self) -> int:
        """모델 파라미터 수 계산"""
        # GRU 파라미터
        gru_params = 3 * (self.input_size * self.hidden_size + 
                         self.hidden_size * self.hidden_size + 
                         self.hidden_size) * self.num_layers
        
        # Dropout은 파라미터 없음
        
        # FC 레이어 파라미터
        fc_params = self.hidden_size * self.num_classes + self.num_classes
        
        return gru_params + fc_params

@dataclass
class TrainingConfig:
    """훈련 관련 설정"""
    
    # 기본 훈련 설정
    epochs: int = 100
    batch_size: int = 1           # 실시간 처리를 위해 1로 설정
    
    # 옵티마이저 설정
    optimizer: str = "Adam"
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # 스케줄러 설정
    scheduler: str = "ReduceLROnPlateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # 손실 함수
    loss_function: str = "CrossEntropyLoss"
    
    # 정규화 설정
    gradient_clip_norm: float = 1.0
    
    # 조기 종료
    early_stopping: bool = True
    patience: int = 20
    
    # 검증 설정
    validation_frequency: int = 1  # 매 에포크마다 검증
    
    # 체크포인트 설정
    save_best_model: bool = True
    checkpoint_dir: str = "./checkpoints"
    
    # 로깅 설정
    log_interval: int = 10        # 10 에포크마다 로그 출력

@dataclass
class PreprocessingConfig:
    """전처리 관련 설정"""
    
    # 정규화 방법
    normalization_method: str = "sensor_specific"
    
    # 0값 처리
    handle_zero_values: bool = True
    zero_replacement_method: str = "mean"  # 'mean', 'median', 'interpolation'
    
    # 노이즈 제거
    noise_reduction: bool = False
    noise_threshold: float = 0.1
    
    # 데이터 증강
    data_augmentation: bool = False
    augmentation_methods: List[str] = None
    
    def __post_init__(self):
        if self.augmentation_methods is None:
            self.augmentation_methods = [
                'noise_injection',
                'time_shift',
                'scaling',
                'masking'
            ]

@dataclass
class HardwareConfig:
    """하드웨어 관련 설정"""
    
    # 디바이스 설정
    device: str = "auto"          # 'auto', 'cuda', 'cpu'
    num_workers: int = 0          # 데이터 로더 워커 수
    
    # 메모리 설정
    pin_memory: bool = False
    
    # GPU 설정
    gpu_memory_fraction: float = 0.8

@dataclass
class ExperimentConfig:
    """실험 관련 설정"""
    
    # 실험 정보
    experiment_name: str = "KLP-SignGlove-GRU-Optimized"
    experiment_version: str = "1.0.0"
    
    # 시드 설정
    random_seed: int = 42
    
    # 결과 저장
    save_results: bool = True
    results_dir: str = "./results"
    
    # 시각화 설정
    create_visualizations: bool = True
    plot_training_curves: bool = True
    plot_confusion_matrix: bool = True
    
    # 모델 저장
    save_model: bool = True
    model_filename: str = "improved_preprocessing_model.pth"
    
    # 성능 메트릭
    target_accuracy: float = 0.95
    target_latency_ms: float = 50.0

class TrainerConfig:
    """통합 훈련 설정 클래스"""
    
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.preprocessing = PreprocessingConfig()
        self.hardware = HardwareConfig()
        self.experiment = ExperimentConfig()
        
        # 모델 파라미터 수 계산
        self.model.total_parameters = self.model.calculate_parameters()
    
    def to_dict(self) -> Dict:
        """설정을 딕셔너리로 변환"""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'preprocessing': self.preprocessing.__dict__,
            'hardware': self.hardware.__dict__,
            'experiment': self.experiment.__dict__
        }
    
    def save_config(self, filepath: str):
        """설정을 JSON 파일로 저장"""
        # 디렉토리가 있는 경우에만 생성
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'TrainerConfig':
        """JSON 파일에서 설정 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.data.__dict__.update(config_dict['data'])
        config.model.__dict__.update(config_dict['model'])
        config.training.__dict__.update(config_dict['training'])
        config.preprocessing.__dict__.update(config_dict['preprocessing'])
        config.hardware.__dict__.update(config_dict['hardware'])
        config.experiment.__dict__.update(config_dict['experiment'])
        
        return config
    
    def print_summary(self):
        """설정 요약 출력"""
        print("=" * 60)
        print("🎯 KLP-SignGlove Trainer Configuration Summary")
        print("=" * 60)
        
        print(f"\n📊 데이터 설정:")
        print(f"  - 데이터 경로: {self.data.data_root}")
        print(f"  - 클래스 수: {len(self.data.class_names)} (자음: 14개, 모음: 10개)")
        print(f"  - 총 데이터 파일: 600개 H5 파일")
        print(f"  - 클래스별 데이터: 25개씩 (5개 세션 × 5개 파일)")
        print(f"  - 세션 구조: 5개 세션 (1~5) / 세션당 5개 파일")
        print(f"  - 파일 형식: episode_YYYYMMDD_HHMMSS_클래스_세션.h5")
        print(f"  - 센서: {self.data.sensor_info['flex_sensors']['count']}개 flex + {self.data.sensor_info['orientation_sensors']['count']}개 orientation")
        
        print(f"\n🤖 모델 설정:")
        print(f"  - 모델 타입: {self.model.model_type}")
        print(f"  - 입력 차원: {self.model.input_size}")
        print(f"  - 출력 차원: {self.model.num_classes}")
        print(f"  - GRU 은닉층: {self.model.hidden_size}")
        print(f"  - GRU 층 수: {self.model.num_layers}")
        print(f"  - Dropout: {self.model.dropout}")
        print(f"  - 총 파라미터: {self.model.total_parameters:,}개")
        
        print(f"\n🏋️ 훈련 설정:")
        print(f"  - 에포크: {self.training.epochs}")
        print(f"  - 배치 크기: {self.training.batch_size}")
        print(f"  - 학습률: {self.training.learning_rate}")
        print(f"  - 옵티마이저: {self.training.optimizer}")
        print(f"  - 스케줄러: {self.training.scheduler}")
        print(f"  - 조기 종료: {self.training.early_stopping} (patience: {self.training.patience})")
        
        print(f"\n🔧 전처리 설정:")
        print(f"  - 정규화: {self.preprocessing.normalization_method}")
        print(f"  - 0값 처리: {self.preprocessing.handle_zero_values}")
        print(f"  - 데이터 증강: {self.preprocessing.data_augmentation}")
        
        print(f"\n💻 하드웨어 설정:")
        print(f"  - 디바이스: {self.hardware.device}")
        print(f"  - 워커 수: {self.hardware.num_workers}")
        
        print(f"\n🧪 실험 설정:")
        print(f"  - 실험명: {self.experiment.experiment_name}")
        print(f"  - 버전: {self.experiment.experiment_version}")
        print(f"  - 시드: {self.experiment.random_seed}")
        print(f"  - 목표 정확도: {self.experiment.target_accuracy*100:.1f}%")
        
        print("=" * 60)

# 기본 설정 인스턴스
DEFAULT_CONFIG = TrainerConfig()

# 성능 최적화된 설정
OPTIMIZED_CONFIG = TrainerConfig()
OPTIMIZED_CONFIG.model.hidden_size = 128
OPTIMIZED_CONFIG.model.num_layers = 3
OPTIMIZED_CONFIG.training.learning_rate = 0.0005
OPTIMIZED_CONFIG.training.epochs = 150

# 경량화 설정
LIGHTWEIGHT_CONFIG = TrainerConfig()
LIGHTWEIGHT_CONFIG.model.hidden_size = 32
LIGHTWEIGHT_CONFIG.model.num_layers = 1
LIGHTWEIGHT_CONFIG.model.dropout = 0.2
LIGHTWEIGHT_CONFIG.training.epochs = 50

if __name__ == "__main__":
    # 설정 요약 출력
    DEFAULT_CONFIG.print_summary()
    
    # 설정 파일 저장
    DEFAULT_CONFIG.save_config("trainer_config_default.json")
    OPTIMIZED_CONFIG.save_config("trainer_config_optimized.json")
    LIGHTWEIGHT_CONFIG.save_config("trainer_config_lightweight.json")
    
    print("\n✅ 설정 파일들이 생성되었습니다:")
    print("  - trainer_config_default.json")
    print("  - trainer_config_optimized.json")
    print("  - trainer_config_lightweight.json")
