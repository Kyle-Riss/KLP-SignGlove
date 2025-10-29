"""
단일 샘플 예측 예제

훈련된 MS3DGRU 모델로 단일 센서 데이터 예측
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from pathlib import Path
from inference import SignGloveInference


def predict_from_csv(csv_path: str, model_path: str):
    """
    CSV 파일에서 센서 데이터를 로딩하여 예측
    
    Args:
        csv_path: 센서 데이터 CSV 파일 경로
        model_path: 훈련된 모델 체크포인트 경로
    """
    print(f"\n{'='*60}")
    print("📊 SignGlove 단일 샘플 예측")
    print(f"{'='*60}\n")
    
    # 1. 추론 엔진 초기화
    print("🚀 추론 엔진 초기화 중...")
    engine = SignGloveInference(
        model_path=model_path,
        model_type='MS3DGRU',
        device='cpu',  # 또는 'cuda'
        input_size=8,
        hidden_size=64,
        classes=24,
        cnn_filters=32,
        dropout=0.1
    )
    
    # 2. CSV에서 센서 데이터 로딩
    print(f"\n📁 센서 데이터 로딩: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 센서 컬럼 추출
    sensor_columns = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'pitch', 'roll', 'yaw']
    
    if not all(col in df.columns for col in sensor_columns):
        print("❌ CSV 파일에 필요한 센서 컬럼이 없습니다!")
        print(f"  필요한 컬럼: {sensor_columns}")
        print(f"  현재 컬럼: {list(df.columns)}")
        return
    
    raw_data = df[sensor_columns].values
    print(f"✅ 데이터 로딩 완료: {raw_data.shape}")
    
    # 3. 예측
    print("\n🔮 예측 수행 중...")
    result = engine.predict_single(raw_data, top_k=5)
    
    # 4. 결과 출력
    engine.print_prediction(result)
    
    # 5. 상세 정보
    print("\n📋 상세 정보:")
    print(f"  - 센서 데이터 길이: {len(raw_data)} timesteps")
    print(f"  - 예측 클래스: {result['predicted_class']}")
    print(f"  - 확률: {result['confidence']:.2%}")
    
    return result


def predict_from_numpy(model_path: str):
    """
    NumPy 배열에서 랜덤 데이터 생성하여 예측 (테스트용)
    
    Args:
        model_path: 훈련된 모델 체크포인트 경로
    """
    print(f"\n{'='*60}")
    print("🧪 SignGlove 테스트 예측 (랜덤 데이터)")
    print(f"{'='*60}\n")
    
    # 1. 추론 엔진 초기화
    print("🚀 추론 엔진 초기화 중...")
    engine = SignGloveInference(
        model_path=model_path,
        model_type='MS3DGRU',
        device='cpu',
        input_size=8,
        hidden_size=64,
        classes=24,
        cnn_filters=32,
        dropout=0.1
    )
    
    # 2. 랜덤 테스트 데이터 생성
    print("\n📊 랜덤 테스트 데이터 생성...")
    raw_data = np.random.randn(87, 8).astype(np.float32)
    print(f"✅ 데이터 생성 완료: {raw_data.shape}")
    
    # 3. 예측
    print("\n🔮 예측 수행 중...")
    result = engine.predict_single(raw_data, top_k=5)
    
    # 4. 결과 출력
    engine.print_prediction(result)
    
    return result


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SignGlove 단일 샘플 예측')
    parser.add_argument('--model', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--csv', type=str, default=None, help='센서 데이터 CSV 파일 경로')
    parser.add_argument('--test', action='store_true', help='랜덤 데이터로 테스트')
    
    args = parser.parse_args()
    
    # 모델 파일 존재 확인
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {args.model}")
        return
    
    # 예측 모드 선택
    if args.test:
        # 랜덤 데이터 테스트
        predict_from_numpy(str(model_path))
    elif args.csv:
        # CSV 파일에서 예측
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"❌ CSV 파일을 찾을 수 없습니다: {args.csv}")
            return
        predict_from_csv(str(csv_path), str(model_path))
    else:
        print("❌ --csv 또는 --test 옵션을 지정해주세요.")
        parser.print_help()


if __name__ == "__main__":
    # 사용 예시
    print("\n" + "="*60)
    print("📚 사용 예시:")
    print("="*60)
    print("\n1. CSV 파일에서 예측:")
    print("   python single_predict.py --model best_model.ckpt --csv sensor_data.csv")
    print("\n2. 랜덤 데이터로 테스트:")
    print("   python single_predict.py --model best_model.ckpt --test")
    print("\n" + "="*60 + "\n")
    
    # 실제 실행
    # main()  # 주석 해제하여 사용
