"""
MS3DGRU 모델을 사용한 추론 예제

최고 성능 모델 (98.78% accuracy)을 사용한 SignGlove 추론
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from inference import SignGloveInference


def main():
    """MS3DGRU 모델 추론 예제"""
    
    print("=" * 80)
    print("🚀 MS3DGRU 추론 예제")
    print("=" * 80)
    print()
    
    # 1. 추론 엔진 초기화
    print("📌 단계 1: MS3DGRU 추론 엔진 초기화")
    print("-" * 80)
    
    engine = SignGloveInference(
        model_path='best_model/ms3dgru_best.ckpt',  # MS3DGRU 체크포인트 경로
        model_type='MS3DGRU',  # 최고 성능 모델
        input_size=8,
        hidden_size=64,
        classes=24,
        cnn_filters=32,
        dropout=0.1,  # 최적 dropout 값
        device='cpu'  # 또는 'cuda'
    )
    
    print()
    print("✅ 초기화 완료!")
    print()
    
    # 2. 모델 정보 확인
    print("📌 단계 2: 모델 정보 확인")
    print("-" * 80)
    
    info = engine.get_model_info()
    print(f"모델 타입: {info['model_type']}")
    print(f"아키텍처: {info['architecture']}")
    print(f"성능: {info['performance']}")
    print(f"파라미터 수: {info['total_parameters']:,}")
    print(f"클래스 수: {info['classes']}")
    print(f"디바이스: {info['device']}")
    print()
    
    # 3. 단일 샘플 예측
    print("📌 단계 3: 단일 샘플 예측")
    print("-" * 80)
    
    # 테스트 데이터 생성 (실제로는 센서 데이터 사용)
    # Shape: (timesteps, 8 channels)
    # 8 channels: [flex1, flex2, flex3, flex4, flex5, yaw, pitch, roll]
    raw_data = np.random.randn(87, 8)
    
    print(f"입력 데이터 shape: {raw_data.shape}")
    print()
    
    # 예측
    result = engine.predict_single(raw_data)
    
    # 결과 출력
    engine.print_prediction(result)
    print()
    
    # 4. 배치 예측
    print("📌 단계 4: 배치 예측")
    print("-" * 80)
    
    # 여러 샘플 생성
    raw_data_list = [
        np.random.randn(87, 8),
        np.random.randn(87, 8),
        np.random.randn(87, 8)
    ]
    
    print(f"배치 크기: {len(raw_data_list)}")
    print()
    
    # 배치 예측
    results = engine.predict_batch(raw_data_list)
    
    # 결과 출력
    for i, result in enumerate(results, 1):
        print(f"샘플 {i}:")
        print(f"  예측 클래스: {result['predicted_class']}")
        print(f"  확률: {result['confidence']:.4f}")
        print(f"  상위 3개: {[p['class'] for p in result['top_k_predictions'][:3]]}")
        print()
    
    # 5. 상세 예측
    print("📌 단계 5: 상세 정보를 포함한 예측")
    print("-" * 80)
    
    detailed_result = engine.predict_with_details(raw_data)
    
    print(f"예측 클래스: {detailed_result['predicted_class']}")
    print(f"확률: {detailed_result['confidence']:.4f}")
    print(f"입력 shape: {detailed_result['input_shape']}")
    print()
    print("상위 5개 예측:")
    for i, pred in enumerate(detailed_result['top_k_predictions'], 1):
        print(f"  {i}. {pred['class']}: {pred['confidence']:.4f}")
    print()
    
    # 6. 성능 벤치마크
    print("📌 단계 6: 성능 벤치마크")
    print("-" * 80)
    
    import time
    
    # 단일 샘플 추론 시간 측정
    n_iterations = 100
    start_time = time.time()
    for _ in range(n_iterations):
        _ = engine.predict_single(raw_data, return_all_info=False)
    single_time = (time.time() - start_time) / n_iterations
    
    print(f"단일 샘플 추론 시간: {single_time*1000:.2f}ms")
    print(f"초당 추론 가능 횟수: {1/single_time:.1f} samples/sec")
    print()
    
    # 배치 추론 시간 측정
    batch_sizes = [1, 4, 8, 16, 32]
    print("배치 크기별 추론 시간:")
    for batch_size in batch_sizes:
        batch_data = [np.random.randn(87, 8) for _ in range(batch_size)]
        start_time = time.time()
        _ = engine.predict_batch(batch_data)
        batch_time = time.time() - start_time
        per_sample_time = batch_time / batch_size
        print(f"  배치 크기 {batch_size:2d}: {batch_time*1000:6.2f}ms (샘플당 {per_sample_time*1000:6.2f}ms)")
    print()
    
    print("=" * 80)
    print("✅ MS3DGRU 추론 예제 완료!")
    print("=" * 80)
    print()
    print("📊 모델 성능:")
    print("  • Test Accuracy: 98.78%")
    print("  • Test F1-Score: 0.9877")
    print("  • Test Loss: 0.052")
    print("  • Trainable Parameters: 58,840")
    print()
    print("💡 사용 팁:")
    print("  • 실제 센서 데이터는 CSV 파일에서 로드하세요")
    print("  • GPU 사용 시 device='cuda'로 설정하세요")
    print("  • 대용량 배치는 청크로 나누어 처리하세요")
    print()


if __name__ == "__main__":
    main()



