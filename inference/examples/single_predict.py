"""
단일 샘플 예측 예제

훈련된 모델을 사용하여 단일 샘플에 대한 예측 수행
"""

import sys
import numpy as np
from pathlib import Path

# inference 모듈 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent))

from inference import SignGloveInference


def example_single_prediction():
    """단일 샘플 예측 예제"""
    print("=" * 60)
    print("🎯 단일 샘플 예측 예제")
    print("=" * 60)
    
    # 1. 추론 엔진 초기화
    print("\n1️⃣ 추론 엔진 초기화...")
    model_path = "src/experiments/checkpoints/best_model_epoch=57_val/loss=0.03-v2.ckpt"
    
    try:
        engine = SignGloveInference(
            model_path=model_path,
            model_type='MSCSGRU',
            input_size=8,
            hidden_size=64,
            classes=24,
            cnn_filters=32,
            target_timesteps=87,
            device='cpu'  # 또는 'cuda'
        )
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        print(f"   모델 경로: {model_path}")
        return
    
    # 2. 테스트 데이터 생성 (실제로는 센서 데이터 사용)
    print("\n2️⃣ 테스트 데이터 준비...")
    # 실제 센서 데이터 shape: (timesteps, 8)
    # - timesteps: 가변 길이 (자동으로 87로 패딩/트렁케이션)
    # - 8 channels: flex1-5 + yaw, pitch, roll
    
    raw_data = np.random.randn(87, 8)  # 더미 데이터
    print(f"   입력 데이터 shape: {raw_data.shape}")
    
    # 3. 기본 예측
    print("\n3️⃣ 기본 예측...")
    result = engine.predict_single(raw_data, top_k=5)
    engine.print_prediction(result)
    
    # 4. 상세 예측
    print("\n4️⃣ 상세 예측...")
    detailed_result = engine.predict_with_details(raw_data)
    
    print(f"\n📊 입력 정보:")
    print(f"  입력 shape: {detailed_result['input_shape']}")
    
    print(f"\n📊 예측 결과:")
    print(f"  예측 클래스: {detailed_result['predicted_class']}")
    print(f"  확률: {detailed_result['confidence']:.4f}")
    
    print(f"\n📊 상위 5개 예측:")
    for i, pred in enumerate(detailed_result['top_k_predictions'][:5], 1):
        print(f"  {i}. {pred['class']}: {pred['confidence']:.4f}")
    
    # 5. 모델 정보
    print("\n5️⃣ 모델 정보...")
    info = engine.get_model_info()
    print(f"  모델 타입: {info['model_type']}")
    print(f"  파라미터 수: {info['total_parameters']:,}")
    print(f"  클래스 수: {info['classes']}")
    print(f"  디바이스: {info['device']}")


def example_with_real_data():
    """실제 센서 데이터를 사용한 예제"""
    print("\n" + "=" * 60)
    print("🎯 실제 센서 데이터 사용 예제")
    print("=" * 60)
    
    # 실제 센서 데이터 로딩
    # 예: CSV 파일에서 로딩
    # raw_data = pd.read_csv('sensor_data.csv').values
    
    # 이 예제에서는 더미 데이터 사용
    print("\n실제 센서 데이터를 사용하려면:")
    print("""
    import pandas as pd
    
    # CSV 파일에서 센서 데이터 로딩
    sensor_data = pd.read_csv('path/to/sensor_data.csv')
    
    # 필요한 컬럼만 추출 (flex1-5, pitch, roll, yaw)
    columns = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'pitch', 'roll', 'yaw']
    raw_data = sensor_data[columns].values
    
    # 추론
    result = engine.predict_single(raw_data)
    engine.print_prediction(result)
    """)


def example_different_length_inputs():
    """다양한 길이의 입력 처리 예제"""
    print("\n" + "=" * 60)
    print("🎯 다양한 길이의 입력 처리 예제")
    print("=" * 60)
    
    print("\n추론 엔진은 자동으로 입력 길이를 조정합니다:")
    print("  - 짧은 입력 (< 87): 패딩 추가")
    print("  - 긴 입력 (> 87): 트렁케이션")
    print("  - 정확한 입력 (= 87): 그대로 사용")
    
    # 더미 엔진 (실제로는 위와 동일하게 초기화)
    print("""
    # 짧은 입력
    short_data = np.random.randn(50, 8)  # 50 timesteps
    result = engine.predict_single(short_data)
    
    # 긴 입력
    long_data = np.random.randn(100, 8)  # 100 timesteps
    result = engine.predict_single(long_data)
    
    # 정확한 길이
    exact_data = np.random.randn(87, 8)  # 87 timesteps
    result = engine.predict_single(exact_data)
    """)


if __name__ == "__main__":
    # 단일 샘플 예측 예제
    example_single_prediction()
    
    # 실제 센서 데이터 예제
    example_with_real_data()
    
    # 다양한 길이의 입력 예제
    example_different_length_inputs()
    
    print("\n" + "=" * 60)
    print("✅ 예제 실행 완료!")
    print("=" * 60)




