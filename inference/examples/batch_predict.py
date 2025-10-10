"""
배치 예측 예제

훈련된 모델을 사용하여 여러 샘플에 대한 배치 예측 수행
"""

import sys
import numpy as np
from pathlib import Path

# inference 모듈 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent))

from inference import SignGloveInference


def example_batch_prediction():
    """배치 예측 예제"""
    print("=" * 60)
    print("🎯 배치 예측 예제")
    print("=" * 60)
    
    # 1. 추론 엔진 초기화
    print("\n1️⃣ 추론 엔진 초기화...")
    model_path = "best_model/best_model.ckpt"
    
    try:
        engine = SignGloveInference(
            model_path=model_path,
            model_type='MSCSGRU',
            input_size=8,
            hidden_size=64,
            classes=24,
            cnn_filters=32,
            target_timesteps=87,
            device='cpu'
        )
    except FileNotFoundError:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    # 2. 배치 데이터 생성
    print("\n2️⃣ 배치 데이터 준비...")
    batch_size = 5
    raw_data_list = [np.random.randn(87, 8) for _ in range(batch_size)]
    print(f"   배치 크기: {batch_size}")
    print(f"   각 샘플 shape: {raw_data_list[0].shape}")
    
    # 3. 배치 예측
    print("\n3️⃣ 배치 예측 실행...")
    results = engine.predict_batch(raw_data_list, top_k=3)
    
    # 4. 결과 출력
    print("\n4️⃣ 예측 결과:")
    for i, result in enumerate(results, 1):
        print(f"\n📊 샘플 {i}:")
        print(f"  예측: {result['predicted_class']}")
        print(f"  확률: {result['confidence']:.4f}")
        print(f"  상위 3개:")
        for j, pred in enumerate(result['top_k_predictions'][:3], 1):
            print(f"    {j}. {pred['class']}: {pred['confidence']:.4f}")


def example_variable_length_batch():
    """가변 길이 배치 예측 예제"""
    print("\n" + "=" * 60)
    print("🎯 가변 길이 배치 예측 예제")
    print("=" * 60)
    
    print("\n배치 내 샘플들의 길이가 다를 수 있습니다:")
    print("자동으로 모든 샘플이 동일한 길이(87)로 조정됩니다.")
    
    print("""
    # 가변 길이 배치 데이터
    raw_data_list = [
        np.random.randn(50, 8),   # 짧은 샘플
        np.random.randn(87, 8),   # 정확한 길이
        np.random.randn(100, 8),  # 긴 샘플
        np.random.randn(70, 8),   # 중간 길이
    ]
    
    # 배치 예측 (자동으로 길이 조정)
    results = engine.predict_batch(raw_data_list)
    """)


def example_large_batch():
    """대용량 배치 처리 예제"""
    print("\n" + "=" * 60)
    print("🎯 대용량 배치 처리 예제")
    print("=" * 60)
    
    print("\n대용량 데이터는 청크로 나누어 처리하는 것이 좋습니다:")
    print("""
    def predict_large_batch(engine, raw_data_list, chunk_size=32):
        '''대용량 배치를 청크로 나누어 처리'''
        all_results = []
        
        for i in range(0, len(raw_data_list), chunk_size):
            chunk = raw_data_list[i:i + chunk_size]
            results = engine.predict_batch(chunk)
            all_results.extend(results)
            
            print(f'처리 완료: {len(all_results)}/{len(raw_data_list)}')
        
        return all_results
    
    # 사용 예제
    large_data_list = [np.random.randn(87, 8) for _ in range(1000)]
    results = predict_large_batch(engine, large_data_list, chunk_size=32)
    """)


def example_batch_statistics():
    """배치 예측 통계 예제"""
    print("\n" + "=" * 60)
    print("🎯 배치 예측 통계 예제")
    print("=" * 60)
    
    print("\n배치 예측 결과의 통계를 계산할 수 있습니다:")
    print("""
    # 배치 예측
    results = engine.predict_batch(raw_data_list)
    
    # 통계 계산
    predicted_classes = [r['predicted_class'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    # 클래스별 빈도
    from collections import Counter
    class_counts = Counter(predicted_classes)
    
    print('클래스별 예측 빈도:')
    for class_name, count in class_counts.most_common():
        print(f'  {class_name}: {count}개')
    
    # 평균 확률
    avg_confidence = np.mean(confidences)
    print(f'\\n평균 확률: {avg_confidence:.4f}')
    
    # 저신뢰도 샘플
    low_conf_samples = [i for i, c in enumerate(confidences) if c < 0.5]
    print(f'저신뢰도 샘플 ({len(low_conf_samples)}개): {low_conf_samples}')
    """)


if __name__ == "__main__":
    # 배치 예측 예제
    example_batch_prediction()
    
    # 가변 길이 배치 예제
    example_variable_length_batch()
    
    # 대용량 배치 예제
    example_large_batch()
    
    # 배치 통계 예제
    example_batch_statistics()
    
    print("\n" + "=" * 60)
    print("✅ 예제 실행 완료!")
    print("=" * 60)




