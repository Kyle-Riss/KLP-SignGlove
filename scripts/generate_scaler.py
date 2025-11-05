#!/usr/bin/env python3
"""
Scaler 파일 생성 스크립트
훈련 시 사용한 동일한 데이터와 전처리 과정으로 scaler.pkl 파일을 생성합니다.
"""
import sys
import os
import pickle
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.misc.data_preprocessor import preprocess_data
from src.misc.data_loader import find_signglove_files


def generate_scaler(data_dir: str, output_path: str, target_timesteps: int = 87):
    """
    훈련 데이터로부터 scaler를 생성하고 저장합니다.
    
    Args:
        data_dir: 데이터셋 루트 디렉토리
        output_path: scaler.pkl 저장 경로
        target_timesteps: 타임스텝 길이 (기본값: 87)
    """
    print("=" * 80)
    print("📊 Scaler 파일 생성")
    print("=" * 80)
    print()
    
    print(f"📁 데이터 디렉토리: {data_dir}")
    print(f"💾 저장 경로: {output_path}")
    print()
    
    # 데이터 파일 찾기
    print("🔍 데이터 파일 검색 중...")
    all_files = find_signglove_files(data_dir)
    print(f"✅ {len(all_files)}개 파일 발견")
    print()
    
    # 데이터 전처리 (scaler 생성)
    print("🔄 데이터 전처리 및 scaler 생성 중...")
    try:
        X, y, X_padding, class_names, scaler = preprocess_data(
            files=all_files,
            target_timesteps=target_timesteps,
            n_channels=8,
            resampling_method="padding"
        )
        print(f"✅ Scaler 생성 완료!")
        print(f"   데이터 형태: {X.shape}")
        print(f"   클래스 수: {len(class_names)}")
        print()
        
        # Scaler 정보 출력
        print("📊 Scaler 통계:")
        print(f"   Mean: {scaler.mean_}")
        print(f"   Scale: {scaler.scale_}")
        print(f"   Variance: {scaler.var_}")
        print()
        
        # Scaler 저장
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"💾 Scaler 저장 완료: {output_file}")
        print(f"   파일 크기: {output_file.stat().st_size / 1024:.2f} KB")
        print()
        
        return scaler
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scaler 파일 생성')
    parser.add_argument(
        '-data_dir',
        type=str,
        default='/home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified',
        help='데이터셋 루트 디렉토리'
    )
    parser.add_argument(
        '-output',
        type=str,
        default='archive/checkpoints_backup/checkpoints_backup/scaler.pkl',
        help='Scaler 파일 저장 경로'
    )
    parser.add_argument(
        '-target_timesteps',
        type=int,
        default=87,
        help='타임스텝 길이 (기본값: 87)'
    )
    
    args = parser.parse_args()
    
    generate_scaler(
        data_dir=args.data_dir,
        output_path=args.output,
        target_timesteps=args.target_timesteps
    )
    
    print("=" * 80)
    print("✅ 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()

