"""
배치 예측 예제

훈련된 MS3DGRU 모델로 여러 센서 데이터 한 번에 예측
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from inference import SignGloveInference


def predict_batch_from_csvs(csv_paths: List[str], model_path: str):
    """
    여러 CSV 파일에서 센서 데이터를 로딩하여 배치 예측
    
    Args:
        csv_paths: 센서 데이터 CSV 파일 경로 리스트
        model_path: 훈련된 모델 체크포인트 경로
    """
    print(f"\n{'='*60}")
    print("📊 SignGlove 배치 예측")
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
    
    # 2. 모든 CSV에서 센서 데이터 로딩
    print(f"\n📁 센서 데이터 로딩 중... ({len(csv_paths)}개 파일)")
    raw_data_list = []
    sensor_columns = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'pitch', 'roll', 'yaw']
    
    for i, csv_path in enumerate(csv_paths, 1):
        try:
            df = pd.read_csv(csv_path)
            
            if not all(col in df.columns for col in sensor_columns):
                print(f"  ⚠️  파일 {i} 스킵: 필요한 센서 컬럼 없음")
                continue
            
            raw_data = df[sensor_columns].values
            raw_data_list.append(raw_data)
            print(f"  ✅ 파일 {i} 로딩 완료: {raw_data.shape}")
            
        except Exception as e:
            print(f"  ❌ 파일 {i} 로딩 실패: {e}")
    
    if not raw_data_list:
        print("\n❌ 로딩된 데이터가 없습니다!")
        return
    
    print(f"\n✅ 총 {len(raw_data_list)}개 샘플 로딩 완료!")
    
    # 3. 배치 예측
    print("\n🔮 배치 예측 수행 중...")
    results = engine.predict_batch(raw_data_list, top_k=3)
    
    # 4. 결과 출력
    print("\n" + "="*60)
    print("📊 배치 예측 결과")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n샘플 {i}:")
        print(f"  🎯 예측: {result['predicted_class']}")
        print(f"  📈 확률: {result['confidence']:.4f}")
        print(f"  📋 상위 3개:")
        for j, pred in enumerate(result['top_k_predictions'][:3], 1):
            print(f"      {j}. {pred['class']}: {pred['confidence']:.4f}")

    print("\n" + "="*60)
    
    return results


def predict_batch_from_directory(directory: str, model_path: str):
    """
    디렉토리의 모든 CSV 파일에서 배치 예측
    
    Args:
        directory: CSV 파일이 있는 디렉토리
        model_path: 훈련된 모델 체크포인트 경로
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ 디렉토리를 찾을 수 없습니다: {directory}")
        return
    
    # 디렉토리에서 모든 CSV 파일 찾기
    csv_files = list(dir_path.glob('*.csv'))
    
    if not csv_files:
        print(f"❌ CSV 파일을 찾을 수 없습니다: {directory}")
        return
    
    print(f"\n📁 디렉토리: {directory}")
    print(f"📊 발견된 CSV 파일: {len(csv_files)}개")
    
    # 배치 예측
    csv_paths = [str(f) for f in csv_files]
    return predict_batch_from_csvs(csv_paths, model_path)


def predict_batch_random(batch_size: int, model_path: str):
    """
    랜덤 데이터로 배치 예측 (테스트용)
    
    Args:
        batch_size: 배치 크기
        model_path: 훈련된 모델 체크포인트 경로
    """
    print(f"\n{'='*60}")
    print(f"🧪 SignGlove 배치 예측 테스트 (랜덤 데이터 {batch_size}개)")
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
    print(f"\n📊 랜덤 테스트 데이터 생성... (배치 크기: {batch_size})")
    raw_data_list = [
        np.random.randn(np.random.randint(50, 120), 8).astype(np.float32)
        for _ in range(batch_size)
    ]
    print(f"✅ 데이터 생성 완료!")
    
    # 3. 배치 예측
    print("\n🔮 배치 예측 수행 중...")
    results = engine.predict_batch(raw_data_list, top_k=3)
    
    # 4. 결과 출력
    print("\n" + "="*60)
    print("📊 배치 예측 결과")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n샘플 {i}: {result['predicted_class']} ({result['confidence']:.4f})")
    
    print("\n" + "="*60)
    
    return results


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SignGlove 배치 예측')
    parser.add_argument('--model', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--csvs', type=str, nargs='+', default=None, help='센서 데이터 CSV 파일 경로들')
    parser.add_argument('--dir', type=str, default=None, help='CSV 파일이 있는 디렉토리')
    parser.add_argument('--test', type=int, default=None, help='랜덤 데이터로 테스트 (배치 크기)')
    
    args = parser.parse_args()
    
    # 모델 파일 존재 확인
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {args.model}")
        return
    
    # 예측 모드 선택
    if args.test:
        # 랜덤 데이터 테스트
        predict_batch_random(args.test, str(model_path))
    elif args.dir:
        # 디렉토리에서 모든 CSV 로딩
        predict_batch_from_directory(args.dir, str(model_path))
    elif args.csvs:
        # 지정된 CSV 파일들
        predict_batch_from_csvs(args.csvs, str(model_path))
    else:
        print("❌ --csvs, --dir, 또는 --test 옵션을 지정해주세요.")
        parser.print_help()


if __name__ == "__main__":
    # 사용 예시
    print("\n" + "="*60)
    print("📚 사용 예시:")
    print("="*60)
    print("\n1. 여러 CSV 파일 배치 예측:")
    print("   python batch_predict.py --model best_model.ckpt --csvs file1.csv file2.csv file3.csv")
    print("\n2. 디렉토리의 모든 CSV 파일 예측:")
    print("   python batch_predict.py --model best_model.ckpt --dir ./sensor_data/")
    print("\n3. 랜덤 데이터로 테스트 (배치 크기 10):")
    print("   python batch_predict.py --model best_model.ckpt --test 10")
    print("\n" + "="*60 + "\n")

    # 실제 실행
    # main()  # 주석 해제하여 사용
