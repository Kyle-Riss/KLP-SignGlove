import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd

def analyze_dataset():
    """데이터셋 분석"""
    print("🔍 SignGlove 데이터셋 분석 시작")
    print("=" * 60)
    
    # 1. 데이터셋 구조 분석
    data_path = Path('/home/billy/25-1kp/SignGlove/external/SignGlove_HW/datasets/unified')
    progress_file = data_path / 'collection_progress.json'
    
    with open(progress_file, 'r', encoding='utf-8') as f:
        progress_data = json.load(f)
    
    print("📊 데이터셋 구조:")
    print(f"  - 총 에피소드: {progress_data['total_episodes']}")
    print(f"  - 클래스 수: 24개 (자음 14개 + 모음 10개)")
    print(f"  - 사용자 수: 5명 (1, 2, 3, 4, 5)")
    print(f"  - 클래스당 파일 수: 25개 (사용자당 5개)")
    print(f"  - 총 파일 수: 600개")
    
    # 2. 클래스별 데이터 분포 분석
    class_stats = progress_data['collection_stats']
    class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                   'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    
    print("\n📈 클래스별 데이터 분포:")
    for class_name in class_names:
        if class_name in class_stats:
            total_files = sum(class_stats[class_name].values())
            print(f"  {class_name}: {total_files}개 파일")
    
    # 3. 데이터 품질 분석
    print("\n🔬 데이터 품질 분석:")
    
    sequence_lengths = []
    data_quality = defaultdict(list)
    
    for class_name in class_names:
        class_dir = data_path / class_name
        if not class_dir.exists():
            continue
            
        for user_dir in class_dir.iterdir():
            if user_dir.is_dir():
                for h5_file in user_dir.glob("*.h5"):
                    try:
                        with h5py.File(h5_file, 'r') as f:
                            sensor_data = f['sensor_data'][:]
                            sequence_lengths.append(len(sensor_data))
                            
                            # 데이터 품질 체크
                            if len(sensor_data) > 0:
                                # NaN 체크
                                has_nan = np.isnan(sensor_data).any()
                                # Inf 체크
                                has_inf = np.isinf(sensor_data).any()
                                # 범위 체크
                                data_range = np.ptp(sensor_data, axis=0)
                                
                                data_quality[class_name].append({
                                    'length': len(sensor_data),
                                    'has_nan': has_nan,
                                    'has_inf': has_inf,
                                    'min_range': data_range.min(),
                                    'max_range': data_range.max(),
                                    'mean_range': data_range.mean()
                                })
                    except Exception as e:
                        print(f"⚠️ 파일 로드 오류 {h5_file}: {e}")
    
    # 4. 시퀀스 길이 분석
    sequence_lengths = np.array(sequence_lengths)
    print(f"\n📏 시퀀스 길이 통계:")
    print(f"  - 평균 길이: {sequence_lengths.mean():.1f} 샘플")
    print(f"  - 최소 길이: {sequence_lengths.min()} 샘플")
    print(f"  - 최대 길이: {sequence_lengths.max()} 샘플")
    print(f"  - 표준편차: {sequence_lengths.std():.1f} 샘플")
    
    # 5. 데이터 품질 요약
    print(f"\n✅ 데이터 품질 요약:")
    total_files = len(sequence_lengths)
    valid_files = sum(1 for lengths in data_quality.values() for data in lengths if not data['has_nan'] and not data['has_inf'])
    print(f"  - 유효한 파일: {valid_files}/{total_files} ({100*valid_files/total_files:.1f}%)")
    
    # 6. 현재 훈련에서 사용하는 데이터 분석
    print(f"\n🎯 현재 훈련 데이터 분석:")
    
    # 시퀀스 길이 20으로 분할했을 때의 시퀀스 수 계산
    total_sequences = 0
    for length in sequence_lengths:
        sequences = length // 20  # 20 샘플씩 분할
        total_sequences += sequences
    
    print(f"  - 시퀀스 길이 20으로 분할 시: {total_sequences}개 시퀀스")
    print(f"  - 클래스당 평균: {total_sequences/24:.1f}개 시퀀스")
    
    # 7. 문제점 분석
    print(f"\n⚠️ 현재 데이터셋의 문제점:")
    print(f"  1. 데이터 양 부족:")
    print(f"     - 클래스당 25개 파일 (매우 적음)")
    print(f"     - 사용자당 5개 파일 (개인별 변이 부족)")
    print(f"     - 총 600개 파일 (딥러닝에 부족)")
    
    print(f"  2. 다양성 부족:")
    print(f"     - 사용자 수: 5명 (적음)")
    print(f"     - 환경 변이 부족")
    print(f"     - 시간대별 변이 부족")
    
    print(f"  3. 시퀀스 길이 불균형:")
    print(f"     - 평균 {sequence_lengths.mean():.1f} 샘플")
    print(f"     - 표준편차 {sequence_lengths.std():.1f} 샘플")
    print(f"     - 일부 파일이 너무 짧거나 길 수 있음")
    
    # 8. 개선 방안 제시
    print(f"\n🚀 개선 방안:")
    print(f"  1. 데이터 증강 강화:")
    print(f"     - 노이즈 추가")
    print(f"     - 시간 이동")
    print(f"     - 스케일링")
    print(f"     - 마스킹")
    
    print(f"  2. 모델 아키텍처 개선:")
    print(f"     - 더 강력한 정규화")
    print(f"     - 앙상블 모델")
    print(f"     - 전이학습 적용")
    
    print(f"  3. 훈련 전략 개선:")
    print(f"     - 교차 검증 강화")
    print(f"     - 조기 종료 조정")
    print(f"     - 학습률 스케줄링")
    
    # 9. 시각화
    create_visualizations(sequence_lengths, data_quality, class_names)
    
    return {
        'total_files': total_files,
        'total_sequences': total_sequences,
        'avg_sequence_length': sequence_lengths.mean(),
        'data_quality': data_quality
    }

def create_visualizations(sequence_lengths, data_quality, class_names):
    """데이터 분석 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 시퀀스 길이 분포
    axes[0, 0].hist(sequence_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(sequence_lengths.mean(), color='red', linestyle='--', label=f'평균: {sequence_lengths.mean():.1f}')
    axes[0, 0].set_title('시퀀스 길이 분포')
    axes[0, 0].set_xlabel('시퀀스 길이 (샘플)')
    axes[0, 0].set_ylabel('파일 수')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 클래스별 평균 시퀀스 길이
    class_avg_lengths = []
    for class_name in class_names:
        if class_name in data_quality and data_quality[class_name]:
            avg_length = np.mean([data['length'] for data in data_quality[class_name]])
            class_avg_lengths.append(avg_length)
        else:
            class_avg_lengths.append(0)
    
    bars = axes[0, 1].bar(range(len(class_names)), class_avg_lengths, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('클래스별 평균 시퀀스 길이')
    axes[0, 1].set_xlabel('클래스')
    axes[0, 1].set_ylabel('평균 길이 (샘플)')
    axes[0, 1].set_xticks(range(len(class_names)))
    axes[0, 1].set_xticklabels(class_names, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 데이터 품질 히트맵
    quality_matrix = []
    for class_name in class_names:
        if class_name in data_quality and data_quality[class_name]:
            valid_ratio = sum(1 for data in data_quality[class_name] 
                            if not data['has_nan'] and not data['has_inf']) / len(data_quality[class_name])
            quality_matrix.append(valid_ratio)
        else:
            quality_matrix.append(0)
    
    quality_matrix = np.array(quality_matrix).reshape(1, -1)
    im = axes[1, 0].imshow(quality_matrix, cmap='RdYlGn', aspect='auto')
    axes[1, 0].set_title('클래스별 데이터 품질 (유효성 비율)')
    axes[1, 0].set_xticks(range(len(class_names)))
    axes[1, 0].set_xticklabels(class_names, rotation=45)
    axes[1, 0].set_yticks([])
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. 시퀀스 길이 박스플롯
    length_by_class = []
    labels = []
    for class_name in class_names:
        if class_name in data_quality and data_quality[class_name]:
            lengths = [data['length'] for data in data_quality[class_name]]
            length_by_class.append(lengths)
            labels.extend([class_name] * len(lengths))
    
    if length_by_class:
        axes[1, 1].boxplot(length_by_class, labels=class_names[:len(length_by_class)])
        axes[1, 1].set_title('클래스별 시퀀스 길이 분포')
        axes[1, 1].set_xlabel('클래스')
        axes[1, 1].set_ylabel('시퀀스 길이 (샘플)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def recommend_improvements(analysis_results):
    """개선 방안 추천"""
    print("\n🎯 구체적인 개선 방안:")
    
    total_files = analysis_results['total_files']
    total_sequences = analysis_results['total_sequences']
    
    print(f"\n1. 데이터 증강 전략:")
    print(f"   - 현재 시퀀스: {total_sequences}개")
    print(f"   - 목표 시퀀스: {total_sequences * 10}개 (10배 증강)")
    print(f"   - 증강 기법:")
    print(f"     * 노이즈 추가 (가우시안, 0.01 표준편차)")
    print(f"     * 시간 이동 (±2 샘플)")
    print(f"     * 스케일링 (0.9-1.1 배율)")
    print(f"     * 마스킹 (5-15% 랜덤 마스킹)")
    
    print(f"\n2. 모델 아키텍처 개선:")
    print(f"   - 현재 파라미터: 76,632개")
    print(f"   - 제안 파라미터: 200,000-500,000개")
    print(f"   - 개선 사항:")
    print(f"     * 더 깊은 CNN (4-5층)")
    print(f"     * 양방향 LSTM")
    print(f"     * 어텐션 메커니즘")
    print(f"     * 드롭아웃 강화 (0.3-0.5)")
    
    print(f"\n3. 훈련 전략 개선:")
    print(f"   - 배치 크기: 32 → 16 (더 안정적)")
    print(f"   - 학습률: 0.01 → 0.005 (더 정밀)")
    print(f"   - 에포크: 50 → 100 (더 긴 훈련)")
    print(f"   - 조기 종료: 10 → 20 (더 인내심)")
    
    print(f"\n4. 검증 전략 개선:")
    print(f"   - K-Fold: 5 → 10 (더 정확한 검증)")
    print(f"   - Stratified Split (클래스 균형 유지)")
    print(f"   - Hold-out Set (최종 테스트용)")
    
    print(f"\n5. 앙상블 전략:")
    print(f"   - 다중 모델 훈련 (LSTM, GRU, Transformer)")
    print(f"   - 다중 시드 훈련 (랜덤 시드 다양화)")
    print(f"   - 다중 아키텍처 (다양한 구조)")
    print(f"   - 앙상블 가중 평균 (성능 기반)")

if __name__ == "__main__":
    analysis_results = analyze_dataset()
    recommend_improvements(analysis_results)



