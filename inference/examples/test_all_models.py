#!/usr/bin/env python3
"""
4개 모델 추론 테스트: GRU, StackedGRU, MS3DGRU, MS3DStackedGRU
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import time
from inference import SignGloveInference


def test_all_models():
    """4개 모델 추론 테스트"""
    
    print("=" * 80)
    print("🚀 4개 모델 추론 테스트: GRU, StackedGRU, MS3DGRU, MS3DStackedGRU")
    print("=" * 80)
    print()
    
    # 모델 설정 (체크포인트 경로는 실제 경로로 확인 필요)
    models_config = {
        'GRU': {
            'path': 'archive/checkpoints_backup/checkpoints_backup/GRU_best.ckpt',
            'type': 'GRU',
            'hidden_size': 64,
            'dropout': 0.2,
            'expected_acc': '98.36%'
        },
        'StackedGRU': {
            'path': 'archive/checkpoints_backup/checkpoints_backup/GRU_best.ckpt',  # StackedGRU 체크포인트가 없으면 GRU 사용
            'type': 'StackedGRU',
            'hidden_size': 64,
            'dropout': 0.2,
            'expected_acc': '95.43%'
        },
        'MS3DGRU': {
            'path': 'best_model/ms3dgru_best.ckpt',
            'type': 'MS3DGRU',
            'cnn_filters': 32,
            'dropout': 0.1,
            'expected_acc': '98.40%'
        },
        'MS3DStackedGRU': {
            'path': 'best_model/ms3dgru_best.ckpt',  # MS3DStackedGRU 체크포인트가 없으면 MS3DGRU 사용
            'type': 'MS3DStackedGRU',
            'cnn_filters': 32,
            'dropout': 0.05,
            'expected_acc': '98.24%'
        }
    }
    
    # 테스트 데이터 생성
    print("📊 테스트 데이터 생성...")
    test_data = np.random.randn(87, 8)  # (timesteps, 8 channels)
    batch_data = [np.random.randn(87, 8) for _ in range(3)]
    print(f"  단일 샘플 shape: {test_data.shape}")
    print(f"  배치 크기: {len(batch_data)}")
    print()
    
    results = {}
    
    # 각 모델로 추론 테스트
    for model_name, config in models_config.items():
        print("=" * 80)
        print(f"🤖 모델: {model_name}")
        print(f"  예상 성능: {config['expected_acc']}")
        print("-" * 80)
        
        try:
            # 추론 엔진 초기화
            init_params = {
                'model_path': config['path'],
                'model_type': config['type'],
                'input_size': 8,
                'hidden_size': config.get('hidden_size', 64),
                'classes': 24,
                'device': 'cpu',
                'dropout': config['dropout']
            }
            
            if 'cnn_filters' in config:
                init_params['cnn_filters'] = config['cnn_filters']
            
            print(f"  초기화 중...")
            engine = SignGloveInference(**init_params)
            
            # 모델 정보
            info = engine.get_model_info()
            print(f"  ✅ 초기화 완료!")
            print(f"     - 파라미터 수: {info.get('total_parameters', 'N/A'):,}")
            print(f"     - 디바이스: {info['device']}")
            print()
            
            # 1. 단일 샘플 예측
            print(f"  📌 단일 샘플 예측:")
            start_time = time.time()
            result = engine.predict_single(test_data)
            inference_time = (time.time() - start_time) * 1000
            
            print(f"     예측 클래스: {result['predicted_class']}")
            print(f"     확률: {result['confidence']:.4f}")
            print(f"     추론 시간: {inference_time:.2f}ms")
            print(f"     상위 3개: {[p['class'] for p in result['top_k_predictions'][:3]]}")
            print()
            
            # 2. 배치 예측
            print(f"  📦 배치 예측 (3개 샘플):")
            start_time = time.time()
            batch_results = engine.predict_batch(batch_data)
            batch_time = (time.time() - start_time) * 1000
            
            for i, res in enumerate(batch_results, 1):
                print(f"     샘플 {i}: {res['predicted_class']} (확률: {res['confidence']:.4f})")
            print(f"     배치 추론 시간: {batch_time:.2f}ms (샘플당 {batch_time/len(batch_data):.2f}ms)")
            print()
            
            # 3. 성능 벤치마크
            print(f"  ⚡ 성능 벤치마크 (100회 반복):")
            n_iterations = 100
            start_time = time.time()
            for _ in range(n_iterations):
                _ = engine.predict_single(test_data, return_all_info=False)
            avg_time = ((time.time() - start_time) / n_iterations) * 1000
            
            print(f"     평균 추론 시간: {avg_time:.2f}ms")
            print(f"     초당 처리량: {1000/avg_time:.1f} samples/sec")
            print()
            
            results[model_name] = {
                'success': True,
                'single_time': inference_time,
                'batch_time': batch_time,
                'avg_time': avg_time,
                'throughput': 1000/avg_time,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence']
            }
            
        except FileNotFoundError as e:
            print(f"  ❌ 체크포인트 파일을 찾을 수 없습니다: {config['path']}")
            print(f"     실제 체크포인트 경로로 변경해주세요.")
            results[model_name] = {'success': False, 'error': 'FileNotFound'}
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'success': False, 'error': str(e)}
        
        print()
    
    # 결과 요약
    print("=" * 80)
    print("📊 추론 테스트 결과 요약")
    print("=" * 80)
    print()
    print(f"{'모델':<20} {'상태':<10} {'평균 시간(ms)':<15} {'처리량(/sec)':<15} {'예측 클래스':<10}")
    print("-" * 80)
    
    for model_name, result in results.items():
        if result['success']:
            print(f"{model_name:<20} {'✅ 성공':<10} {result['avg_time']:<15.2f} "
                  f"{result['throughput']:<15.1f} {result['predicted_class']:<10}")
        else:
            print(f"{model_name:<20} {'❌ 실패':<10} {'-':<15} {'-':<15} {'-':<10}")
    
    print()
    print("=" * 80)
    print("✅ 추론 테스트 완료!")
    print("=" * 80)


if __name__ == "__main__":
    test_all_models()

