"""
최고 성능 모델들을 사용한 추론 데모

MS3DGRU (98.78%), GRU (98.44%), MS3DStackedGRU (98.44-98.78%) 모델 사용 예제
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from inference import SignGloveInference


def demo_best_models():
    """최고 성능 모델들 데모"""
    
    print("=" * 70)
    print("🏆 최고 성능 모델들 추론 데모")
    print("=" * 70)
    
    # 체크포인트 경로 설정 (실제 경로로 변경 필요)
    models = {
        'MS3DGRU': {
            'path': 'checkpoints/ms3dgru_best.ckpt',
            'type': 'MS3DGRU',
            'accuracy': '98.78%',
            'description': '⭐ 최고 성능 - Multi-Scale 3D CNN + GRU'
        },
        'GRU': {
            'path': 'checkpoints/gru_best.ckpt',
            'type': 'GRU',
            'accuracy': '98.44%',
            'description': '안정적 성능 - 기본 GRU'
        },
        'MS3DStackedGRU': {
            'path': 'checkpoints/ms3dstackedgru_best.ckpt',
            'type': 'MS3DStackedGRU',
            'accuracy': '98.44-98.78%',
            'description': 'Multi-Scale 3D CNN + Stacked GRU'
        }
    }
    
    # 테스트 데이터 생성
    print("\n📊 테스트 데이터 생성...")
    test_data = np.random.randn(87, 8)  # (timesteps, channels)
    print(f"  Shape: {test_data.shape}")
    print(f"  데이터 타입: {test_data.dtype}")
    
    # 각 모델로 추론
    for model_name, model_config in models.items():
        print("\n" + "=" * 70)
        print(f"🤖 모델: {model_name}")
        print(f"  정확도: {model_config['accuracy']}")
        print(f"  설명: {model_config['description']}")
        print("-" * 70)
        
        try:
            # 추론 엔진 초기화
            engine = SignGloveInference(
                model_path=model_config['path'],
                model_type=model_config['type'],
                device='cpu',  # GPU 사용 시 'cuda'로 변경
                dropout=0.1 if 'MS3D' in model_name else 0.2
            )
            
            # 모델 정보 출력
            info = engine.get_model_info()
            print(f"\n  모델 정보:")
            print(f"    - 파라미터 수: {info.get('total_parameters', 'N/A'):,}")
            print(f"    - 디바이스: {info['device']}")
            
            # 단일 샘플 예측
            print(f"\n  🔍 단일 샘플 예측:")
            result = engine.predict_single(test_data, top_k=3)
            engine.print_prediction(result)
            
            # 배치 예측 테스트
            print(f"\n  📦 배치 예측 (3개 샘플):")
            batch_data = [test_data, test_data, test_data]
            batch_results = engine.predict_batch(batch_data, top_k=3)
            
            for i, res in enumerate(batch_results, 1):
                print(f"    샘플 {i}: {res['predicted_class']} "
                      f"(신뢰도: {res['confidence']:.4f})")
            
        except FileNotFoundError:
            print(f"  ⚠️  체크포인트 파일을 찾을 수 없습니다: {model_config['path']}")
            print(f"      실제 체크포인트 경로로 변경해주세요.")
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 데모 완료!")
    print("=" * 70)


def demo_model_comparison():
    """모델 성능 비교 데모"""
    
    print("\n" + "=" * 70)
    print("📊 모델 성능 비교")
    print("=" * 70)
    
    # 성능 비교 표
    comparison = {
        'MS3DGRU': {
            'accuracy': '98.78%',
            'parameters': '58,840',
            'efficiency': '1.68',
            'rank': '1위'
        },
        'GRU': {
            'accuracy': '98.44%',
            'parameters': '74,776',
            'efficiency': '1.32',
            'rank': '2위'
        },
        'MS3DStackedGRU': {
            'accuracy': '98.44-98.78%',
            'parameters': '167,032',
            'efficiency': '0.58',
            'rank': '3위'
        }
    }
    
    print("\n| 모델 | 순위 | Test Accuracy | 파라미터 수 | 효율성 |")
    print("|------|------|---------------|-------------|--------|")
    
    for model, stats in comparison.items():
        print(f"| {model:16} | {stats['rank']} | {stats['accuracy']:13} | "
              f"{stats['parameters']:11} | {stats['efficiency']:6} |")
    
    print("\n💡 권장 사항:")
    print("  1. 최고 성능 필요: MS3DGRU (98.78%)")
    print("  2. 안정적 성능: GRU (98.44%)")
    print("  3. 효율성 중시: GRU (파라미터 대비 높은 성능)")


if __name__ == "__main__":
    print("🚀 SignGlove 최고 성능 모델 추론 시스템")
    
    # 데모 실행
    demo_best_models()
    demo_model_comparison()
    
    print("\n" + "=" * 70)
    print("📝 사용 방법:")
    print("  1. 체크포인트 경로를 실제 경로로 변경하세요")
    print("  2. 실제 센서 데이터를 사용하여 테스트하세요")
    print("  3. GPU 사용 시 device='cuda'로 변경하세요")
    print("=" * 70)



