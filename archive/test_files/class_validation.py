#!/usr/bin/env python3
"""
클래스 검증 스크립트
데이터셋의 클래스별 분포와 데이터 품질을 검증합니다.
"""

import os
import h5py
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def validate_classes(data_dir):
    """클래스별 데이터 검증"""
    print("🔍 클래스 검증 시작...")
    print("=" * 60)
    
    # 클래스 목록 가져오기
    classes = sorted([d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"📊 총 클래스 수: {len(classes)}")
    print(f"📁 데이터 디렉토리: {data_dir}")
    print()
    
    # 클래스별 통계
    class_stats = {}
    total_files = 0
    
    print("📈 클래스별 데이터 개수:")
    print("-" * 40)
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        
        # 하위 디렉토리에서 H5 파일 찾기
        files = []
        for root, dirs, filenames in os.walk(cls_path):
            for filename in filenames:
                if filename.endswith('.h5'):
                    files.append(os.path.join(root, filename))
        
        # 각 파일의 데이터 크기 확인
        file_sizes = []
        for file_path in files:
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'sensor_data' in f:
                        file_sizes.append(f['sensor_data'].shape[0])
            except Exception as e:
                print(f"⚠️  파일 읽기 오류 {file_path}: {e}")
        
        class_stats[cls] = {
            'file_count': len(files),
            'total_samples': sum(file_sizes) if file_sizes else 0,
            'avg_samples_per_file': np.mean(file_sizes) if file_sizes else 0,
            'min_samples': min(file_sizes) if file_sizes else 0,
            'max_samples': max(file_sizes) if file_sizes else 0
        }
        
        total_files += len(files)
        
        print(f"{cls:2s}: {len(files):3d}개 파일, "
              f"{class_stats[cls]['total_samples']:4d}개 샘플 "
              f"(평균 {class_stats[cls]['avg_samples_per_file']:5.1f})")
    
    print("-" * 40)
    print(f"📊 총 파일 수: {total_files}개")
    print(f"📊 평균 파일 수: {total_files/len(classes):.1f}개")
    print()
    
    # 데이터 품질 검증
    print("🔍 데이터 품질 검증:")
    print("-" * 40)
    
    # 샘플 수가 적은 클래스 찾기
    low_sample_classes = [cls for cls, stats in class_stats.items() 
                         if stats['total_samples'] < 1000]
    if low_sample_classes:
        print(f"⚠️  샘플 수가 적은 클래스들 (< 1000개): {low_sample_classes}")
    else:
        print("✅ 모든 클래스가 충분한 샘플을 가지고 있습니다.")
    
    # 파일 수가 적은 클래스 찾기
    low_file_classes = [cls for cls, stats in class_stats.items() 
                       if stats['file_count'] < 10]
    if low_file_classes:
        print(f"⚠️  파일 수가 적은 클래스들 (< 10개): {low_file_classes}")
    else:
        print("✅ 모든 클래스가 충분한 파일을 가지고 있습니다.")
    
    # 데이터 크기 일관성 검증
    size_variations = []
    for cls, stats in class_stats.items():
        if stats['max_samples'] > 0:
            variation = (stats['max_samples'] - stats['min_samples']) / stats['max_samples']
            size_variations.append((cls, variation))
    
    high_variation = [cls for cls, var in size_variations if var > 0.5]
    if high_variation:
        print(f"⚠️  데이터 크기 변동이 큰 클래스들 (> 50%): {high_variation}")
    else:
        print("✅ 모든 클래스의 데이터 크기가 일관적입니다.")
    
    print()
    
    # 시각화
    create_visualizations(class_stats, classes)
    
    return class_stats

def create_visualizations(class_stats, classes):
    """클래스별 통계 시각화"""
    print("📊 시각화 생성 중...")
    
    # 1. 클래스별 파일 수
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 2, 1)
    file_counts = [class_stats[cls]['file_count'] for cls in classes]
    plt.bar(range(len(classes)), file_counts, color='skyblue')
    plt.title('클래스별 파일 수')
    plt.xlabel('클래스')
    plt.ylabel('파일 수')
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 2. 클래스별 총 샘플 수
    plt.subplot(2, 2, 2)
    total_samples = [class_stats[cls]['total_samples'] for cls in classes]
    plt.bar(range(len(classes)), total_samples, color='lightgreen')
    plt.title('클래스별 총 샘플 수')
    plt.xlabel('클래스')
    plt.ylabel('샘플 수')
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 3. 클래스별 평균 샘플 수
    plt.subplot(2, 2, 3)
    avg_samples = [class_stats[cls]['avg_samples_per_file'] for cls in classes]
    plt.bar(range(len(classes)), avg_samples, color='orange')
    plt.title('클래스별 파일당 평균 샘플 수')
    plt.xlabel('클래스')
    plt.ylabel('평균 샘플 수')
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 4. 데이터 분포 히스토그램
    plt.subplot(2, 2, 4)
    plt.hist(total_samples, bins=10, color='purple', alpha=0.7)
    plt.title('클래스별 샘플 수 분포')
    plt.xlabel('총 샘플 수')
    plt.ylabel('클래스 수')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('class_validation_results.png', dpi=300, bbox_inches='tight')
    print("✅ 시각화 저장됨: class_validation_results.png")
    
    # 통계 요약
    print("\n📋 통계 요약:")
    print("-" * 40)
    print(f"총 클래스 수: {len(classes)}")
    print(f"총 파일 수: {sum(class_stats[cls]['file_count'] for cls in classes)}")
    print(f"총 샘플 수: {sum(class_stats[cls]['total_samples'] for cls in classes):,}")
    print(f"평균 파일 수: {np.mean([class_stats[cls]['file_count'] for cls in classes]):.1f}")
    print(f"평균 샘플 수: {np.mean([class_stats[cls]['total_samples'] for cls in classes]):.1f}")
    print(f"최소 샘플 수: {min([class_stats[cls]['total_samples'] for cls in classes])}")
    print(f"최대 샘플 수: {max([class_stats[cls]['total_samples'] for cls in classes])}")

def validate_data_quality(data_dir, sample_classes=3):
    """데이터 품질 상세 검증"""
    print(f"\n🔬 데이터 품질 상세 검증 (샘플 {sample_classes}개 클래스):")
    print("=" * 60)
    
    classes = sorted([d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d))])
    
    for i, cls in enumerate(classes[:sample_classes]):
        print(f"\n📁 클래스 '{cls}' 검증:")
        cls_path = os.path.join(data_dir, cls)
        
        # 하위 디렉토리에서 H5 파일 찾기
        files = []
        for root, dirs, filenames in os.walk(cls_path):
            for filename in filenames:
                if filename.endswith('.h5'):
                    files.append(os.path.join(root, filename))
                    if len(files) >= 3:  # 처음 3개 파일만
                        break
            if len(files) >= 3:
                break
        
        for j, file_path in enumerate(files):
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'sensor_data' in f:
                        data = f['sensor_data'][:]
                        print(f"  📄 {os.path.basename(file_path)}: {data.shape}, "
                              f"범위: [{data.min():.2f}, {data.max():.2f}], "
                              f"평균: {data.mean():.2f}, "
                              f"표준편차: {data.std():.2f}")
                        
                        # NaN이나 무한값 확인
                        if np.isnan(data).any():
                            print(f"    ⚠️  NaN 값 발견!")
                        if np.isinf(data).any():
                            print(f"    ⚠️  무한값 발견!")
                            
            except Exception as e:
                print(f"  ❌ {os.path.basename(file_path)}: 오류 - {e}")

if __name__ == "__main__":
    data_dir = "../SignGlove_HW/datasets/unified"
    
    if not os.path.exists(data_dir):
        print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        exit(1)
    
    # 클래스 검증 실행
    class_stats = validate_classes(data_dir)
    
    # 데이터 품질 상세 검증
    validate_data_quality(data_dir)
    
    print("\n🎉 클래스 검증 완료!")
