#!/usr/bin/env python3
"""
모음 데이터 품질 분석
모음과 자음 데이터의 차이점을 시각화하여 분석
"""

import torch
import numpy as np
import h5py
import os
from pathlib import Path
import sys
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import random
import json
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class VowelAnalysisDataset(Dataset):
    """모음 분석용 데이터셋"""
    
    def __init__(self, data_path, sequence_length=20):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
        # 클래스 분류
        self.consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        self.all_classes = self.consonants + self.vowels
        
        # 데이터 로드
        self.data, self.labels, self.class_indices = self.load_data()
        
        # 데이터 정규화
        self.normalize_data()
        
    def load_data(self):
        """데이터 로드"""
        print("📊 모음/자음 데이터 로딩 중...")
        
        data = []
        labels = []
        class_indices = defaultdict(list)
        
        for class_idx, class_name in enumerate(self.all_classes):
            print(f"  {class_name} 클래스 데이터 로딩 중...")
            
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                continue
            
            class_data_count = 0
            
            for sub_dir in class_dir.iterdir():
                if sub_dir.is_dir():
                    h5_files = list(sub_dir.glob("*.h5"))
                    
                    # 모든 사용자의 데이터 사용
                    for h5_file in h5_files:
                        try:
                            with h5py.File(h5_file, 'r') as f:
                                sensor_data = f['sensor_data'][:]
                                
                                # 시퀀스로 분할
                                for i in range(0, len(sensor_data), self.sequence_length):
                                    if i + self.sequence_length <= len(sensor_data):
                                        sequence = sensor_data[i:i+self.sequence_length]
                                        data.append(sequence)
                                        labels.append(class_idx)
                                        class_indices[class_name].append(len(data) - 1)
                                        class_data_count += 1
                        except Exception as e:
                            print(f"⚠️ 파일 로드 오류 {h5_file}: {e}")
                            continue
            
            print(f"    {class_name}: {class_data_count}개 시퀀스")
        
        return np.array(data), np.array(labels), class_indices
    
    def normalize_data(self):
        """데이터 정규화"""
        print("🔧 데이터 정규화 중...")
        
        original_shape = self.data.shape
        data_reshaped = self.data.reshape(-1, self.data.shape[-1])
        
        data_normalized = self.scaler.fit_transform(data_reshaped)
        self.data = data_normalized.reshape(original_shape)
        
        print(f"✅ 정규화 완료: 범위 [{self.data.min():.3f}, {self.data.max():.3f}]")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = self.labels[idx]
        return torch.FloatTensor(sequence), torch.LongTensor([label])

class VowelAnalyzer:
    """모음 분석기"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.consonants = dataset.consonants
        self.vowels = dataset.vowels
        self.all_classes = dataset.all_classes
        
        print(f"🔍 모음 분석기 초기화 완료")
        print(f"📊 자음: {len(self.consonants)}개, 모음: {len(self.vowels)}개")
    
    def analyze_signal_characteristics(self):
        """신호 특성 분석"""
        print("\n📊 신호 특성 분석")
        print("=" * 50)
        
        # 각 클래스별 신호 특성 계산
        signal_stats = {}
        
        for class_name in self.all_classes:
            if class_name not in self.dataset.class_indices:
                continue
            
            indices = self.dataset.class_indices[class_name]
            class_data = self.dataset.data[indices]
            
            # 신호 특성 계산
            mean_amplitude = np.mean(np.abs(class_data))
            std_amplitude = np.std(class_data)
            variance = np.var(class_data)
            energy = np.mean(class_data ** 2)
            
            # 시계열 특성
            # 각 시퀀스의 변화율 계산
            diff_data = np.diff(class_data, axis=1)
            mean_change_rate = np.mean(np.abs(diff_data))
            
            # 주파수 특성 (FFT)
            fft_data = np.fft.fft(class_data, axis=1)
            power_spectrum = np.mean(np.abs(fft_data) ** 2, axis=0)
            dominant_freq = np.argmax(power_spectrum)
            
            signal_stats[class_name] = {
                'mean_amplitude': mean_amplitude,
                'std_amplitude': std_amplitude,
                'variance': variance,
                'energy': energy,
                'mean_change_rate': mean_change_rate,
                'dominant_freq': dominant_freq,
                'sample_count': len(indices)
            }
        
        return signal_stats
    
    def visualize_signal_characteristics(self, signal_stats):
        """신호 특성 시각화"""
        print("\n📈 신호 특성 시각화")
        print("=" * 50)
        
        # 데이터 준비
        consonants_data = {k: v for k, v in signal_stats.items() if k in self.consonants}
        vowels_data = {k: v for k, v in signal_stats.items() if k in self.vowels}
        
        # 특성별 비교
        characteristics = ['mean_amplitude', 'std_amplitude', 'variance', 'energy', 'mean_change_rate']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('모음 vs 자음 신호 특성 비교', fontsize=16, fontweight='bold')
        
        for i, char in enumerate(characteristics):
            row = i // 3
            col = i % 3
            
            # 자음 데이터
            cons_values = [consonants_data[k][char] for k in consonants_data.keys()]
            cons_names = list(consonants_data.keys())
            
            # 모음 데이터
            vow_values = [vowels_data[k][char] for k in vowels_data.keys()]
            vow_names = list(vowels_data.keys())
            
            # 박스플롯
            data_to_plot = [cons_values, vow_values]
            labels = ['자음', '모음']
            
            bp = axes[row, col].boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
            if len(bp['boxes']) >= 2:
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][1].set_facecolor('lightcoral')
            
            axes[row, col].set_title(f'{char.replace("_", " ").title()}')
            axes[row, col].set_ylabel('값')
            axes[row, col].grid(True, alpha=0.3)
        
        # 샘플 수 비교
        cons_counts = [consonants_data[k]['sample_count'] for k in consonants_data.keys()]
        vow_counts = [vowels_data[k]['sample_count'] for k in vowels_data.keys()]
        
        axes[1, 2].bar(['자음', '모음'], [np.mean(cons_counts), np.mean(vow_counts)], 
                      color=['lightblue', 'lightcoral'], alpha=0.7)
        axes[1, 2].set_title('평균 샘플 수')
        axes[1, 2].set_ylabel('샘플 수')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('vowel_consonant_characteristics.png', dpi=300, bbox_inches='tight')
        print("📊 신호 특성 비교 차트가 'vowel_consonant_characteristics.png'에 저장되었습니다.")
    
    def analyze_sensor_patterns(self):
        """센서 패턴 분석"""
        print("\n🔍 센서 패턴 분석")
        print("=" * 50)
        
        # 각 클래스별 센서별 패턴 분석
        sensor_patterns = {}
        
        for class_name in self.all_classes:
            if class_name not in self.dataset.class_indices:
                continue
            
            indices = self.dataset.class_indices[class_name]
            class_data = self.dataset.data[indices]  # (samples, sequence, sensors)
            
            # 센서별 평균 패턴
            sensor_means = np.mean(class_data, axis=(0, 1))  # (sensors,)
            sensor_stds = np.std(class_data, axis=(0, 1))   # (sensors,)
            
            sensor_patterns[class_name] = {
                'sensor_means': sensor_means,
                'sensor_stds': sensor_stds
            }
        
        # 센서 패턴 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('센서별 패턴 분석', fontsize=16, fontweight='bold')
        
        # 1. 센서별 평균값 비교
        sensors = range(8)
        
        # 자음 평균
        cons_means = np.array([sensor_patterns[k]['sensor_means'] for k in self.consonants if k in sensor_patterns])
        cons_mean_avg = np.mean(cons_means, axis=0)
        cons_std_avg = np.std(cons_means, axis=0)
        
        # 모음 평균
        vow_means = np.array([sensor_patterns[k]['sensor_means'] for k in self.vowels if k in sensor_patterns])
        vow_mean_avg = np.mean(vow_means, axis=0)
        vow_std_avg = np.std(vow_means, axis=0)
        
        axes[0, 0].errorbar(sensors, cons_mean_avg, yerr=cons_std_avg, 
                           label='자음', marker='o', color='blue', capsize=5)
        axes[0, 0].errorbar(sensors, vow_mean_avg, yerr=vow_std_avg, 
                           label='모음', marker='s', color='red', capsize=5)
        axes[0, 0].set_xlabel('센서 번호')
        axes[0, 0].set_ylabel('평균값')
        axes[0, 0].set_title('센서별 평균값 비교')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 센서별 표준편차 비교
        cons_stds = np.array([sensor_patterns[k]['sensor_stds'] for k in self.consonants if k in sensor_patterns])
        cons_std_avg = np.mean(cons_stds, axis=0)
        
        vow_stds = np.array([sensor_patterns[k]['sensor_stds'] for k in self.vowels if k in sensor_patterns])
        vow_std_avg = np.mean(vow_stds, axis=0)
        
        axes[0, 1].plot(sensors, cons_std_avg, label='자음', marker='o', color='blue')
        axes[0, 1].plot(sensors, vow_std_avg, label='모음', marker='s', color='red')
        axes[0, 1].set_xlabel('센서 번호')
        axes[0, 1].set_ylabel('표준편차')
        axes[0, 1].set_title('센서별 변동성 비교')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 클래스별 센서 패턴 히트맵 (자음)
        if len(cons_means) > 0:
            im1 = axes[1, 0].imshow(cons_means.T, cmap='Blues', aspect='auto')
            axes[1, 0].set_xlabel('자음 클래스')
            axes[1, 0].set_ylabel('센서 번호')
            axes[1, 0].set_title('자음 센서 패턴 히트맵')
            axes[1, 0].set_xticks(range(len(self.consonants)))
            axes[1, 0].set_xticklabels(self.consonants)
            axes[1, 0].set_yticks(range(8))
            plt.colorbar(im1, ax=axes[1, 0])
        
        # 4. 클래스별 센서 패턴 히트맵 (모음)
        if len(vow_means) > 0:
            im2 = axes[1, 1].imshow(vow_means.T, cmap='Reds', aspect='auto')
            axes[1, 1].set_xlabel('모음 클래스')
            axes[1, 1].set_ylabel('센서 번호')
            axes[1, 1].set_title('모음 센서 패턴 히트맵')
            axes[1, 1].set_xticks(range(len(self.vowels)))
            axes[1, 1].set_xticklabels(self.vowels)
            axes[1, 1].set_yticks(range(8))
            plt.colorbar(im2, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('sensor_pattern_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 센서 패턴 분석 차트가 'sensor_pattern_analysis.png'에 저장되었습니다.")
    
    def analyze_sequence_patterns(self):
        """시퀀스 패턴 분석"""
        print("\n📈 시퀀스 패턴 분석")
        print("=" * 50)
        
        # 각 클래스별 시퀀스 패턴 분석
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('시퀀스 패턴 분석', fontsize=16, fontweight='bold')
        
        # 1. 자음 시퀀스 패턴
        axes[0, 0].set_title('자음 시퀀스 패턴')
        for i, class_name in enumerate(self.consonants[:5]):  # 처음 5개만
            if class_name in self.dataset.class_indices:
                indices = self.dataset.class_indices[class_name][:10]  # 처음 10개 샘플만
                class_data = self.dataset.data[indices]
                mean_sequence = np.mean(class_data, axis=0)  # (sequence_length, sensors)
                
                # 첫 번째 센서만 플롯
                axes[0, 0].plot(mean_sequence[:, 0], label=class_name, alpha=0.7)
        
        axes[0, 0].set_xlabel('시퀀스 시간')
        axes[0, 0].set_ylabel('센서 0 값')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 모음 시퀀스 패턴
        axes[0, 1].set_title('모음 시퀀스 패턴')
        for i, class_name in enumerate(self.vowels[:5]):  # 처음 5개만
            if class_name in self.dataset.class_indices:
                indices = self.dataset.class_indices[class_name][:10]  # 처음 10개 샘플만
                class_data = self.dataset.data[indices]
                mean_sequence = np.mean(class_data, axis=0)  # (sequence_length, sensors)
                
                # 첫 번째 센서만 플롯
                axes[0, 1].plot(mean_sequence[:, 0], label=class_name, alpha=0.7)
        
        axes[0, 1].set_xlabel('시퀀스 시간')
        axes[0, 1].set_ylabel('센서 0 값')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 자음 vs 모음 평균 시퀀스 비교
        axes[1, 0].set_title('자음 vs 모음 평균 시퀀스 비교')
        
        # 자음 평균
        cons_sequences = []
        for class_name in self.consonants:
            if class_name in self.dataset.class_indices:
                indices = self.dataset.class_indices[class_name]
                class_data = self.dataset.data[indices]
                mean_sequence = np.mean(class_data, axis=0)
                cons_sequences.append(mean_sequence)
        
        if cons_sequences:
            cons_avg = np.mean(cons_sequences, axis=0)
            axes[1, 0].plot(cons_avg[:, 0], label='자음 평균', color='blue', linewidth=2)
        
        # 모음 평균
        vow_sequences = []
        for class_name in self.vowels:
            if class_name in self.dataset.class_indices:
                indices = self.dataset.class_indices[class_name]
                class_data = self.dataset.data[indices]
                mean_sequence = np.mean(class_data, axis=0)
                vow_sequences.append(mean_sequence)
        
        if vow_sequences:
            vow_avg = np.mean(vow_sequences, axis=0)
            axes[1, 0].plot(vow_avg[:, 0], label='모음 평균', color='red', linewidth=2)
        
        axes[1, 0].set_xlabel('시퀀스 시간')
        axes[1, 0].set_ylabel('센서 0 값')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 시퀀스 변동성 비교
        axes[1, 1].set_title('시퀀스 변동성 비교')
        
        cons_variability = []
        vow_variability = []
        
        for class_name in self.consonants:
            if class_name in self.dataset.class_indices:
                indices = self.dataset.class_indices[class_name]
                class_data = self.dataset.data[indices]
                variability = np.std(class_data, axis=(0, 1))  # 전체 변동성
                cons_variability.append(np.mean(variability))
        
        for class_name in self.vowels:
            if class_name in self.dataset.class_indices:
                indices = self.dataset.class_indices[class_name]
                class_data = self.dataset.data[indices]
                variability = np.std(class_data, axis=(0, 1))  # 전체 변동성
                vow_variability.append(np.mean(variability))
        
        if cons_variability:
            axes[1, 1].hist(cons_variability, alpha=0.7, label='자음', color='blue', bins=10)
        if vow_variability:
            axes[1, 1].hist(vow_variability, alpha=0.7, label='모음', color='red', bins=10)
        
        axes[1, 1].set_xlabel('변동성')
        axes[1, 1].set_ylabel('빈도')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sequence_pattern_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 시퀀스 패턴 분석 차트가 'sequence_pattern_analysis.png'에 저장되었습니다.")
    
    def perform_statistical_tests(self, signal_stats):
        """통계적 검정"""
        print("\n📊 통계적 검정")
        print("=" * 50)
        
        # 자음과 모음 데이터 분리
        cons_data = {k: v for k, v in signal_stats.items() if k in self.consonants}
        vow_data = {k: v for k, v in signal_stats.items() if k in self.vowels}
        
        characteristics = ['mean_amplitude', 'std_amplitude', 'variance', 'energy', 'mean_change_rate']
        
        print("🔍 자음 vs 모음 통계적 검정 결과:")
        print("-" * 40)
        
        for char in characteristics:
            cons_values = [cons_data[k][char] for k in cons_data.keys()]
            vow_values = [vow_data[k][char] for k in vow_data.keys()]
            
            # t-검정 수행
            t_stat, p_value = stats.ttest_ind(cons_values, vow_values)
            
            print(f"{char.replace('_', ' ').title()}:")
            print(f"  t-통계량: {t_stat:.4f}")
            print(f"  p-값: {p_value:.4f}")
            print(f"  유의수준: {'유의함' if p_value < 0.05 else '유의하지 않음'}")
            print(f"  자음 평균: {np.mean(cons_values):.4f}")
            print(f"  모음 평균: {np.mean(vow_values):.4f}")
            print()
        
        return {
            'consonant_data': cons_data,
            'vowel_data': vow_data,
            'characteristics': characteristics
        }
    
    def generate_summary_report(self, signal_stats, test_results):
        """요약 리포트 생성"""
        print("\n📋 모음 데이터 품질 분석 요약")
        print("=" * 50)
        
        # 데이터 품질 지표 계산
        cons_data = test_results['consonant_data']
        vow_data = test_results['vowel_data']
        
        print("📊 데이터 품질 지표:")
        print("-" * 40)
        
        # 1. 샘플 수 비교
        cons_samples = [cons_data[k]['sample_count'] for k in cons_data.keys()]
        vow_samples = [vow_data[k]['sample_count'] for k in vow_data.keys()]
        
        print(f"자음 평균 샘플 수: {np.mean(cons_samples):.1f}")
        print(f"모음 평균 샘플 수: {np.mean(vow_samples):.1f}")
        print(f"샘플 수 비율 (자음/모음): {np.mean(cons_samples)/np.mean(vow_samples):.2f}")
        
        # 2. 신호 품질 비교
        characteristics = ['mean_amplitude', 'std_amplitude', 'variance', 'energy']
        
        print(f"\n📈 신호 품질 비교:")
        for char in characteristics:
            cons_values = [cons_data[k][char] for k in cons_data.keys()]
            vow_values = [vow_data[k][char] for k in vow_data.keys()]
            
            cons_mean = np.mean(cons_values)
            vow_mean = np.mean(vow_values)
            ratio = cons_mean / vow_mean if vow_mean != 0 else float('inf')
            
            print(f"{char.replace('_', ' ').title()}:")
            print(f"  자음 평균: {cons_mean:.4f}")
            print(f"  모음 평균: {vow_mean:.4f}")
            print(f"  비율 (자음/모음): {ratio:.2f}")
        
        # 3. 문제 진단
        print(f"\n🔍 문제 진단:")
        
        # 샘플 수 불균형 체크
        if np.mean(cons_samples) / np.mean(vow_samples) > 1.5:
            print("⚠️ 샘플 수 불균형: 자음 데이터가 모음보다 많음")
        
        # 신호 강도 체크
        cons_energy = np.mean([cons_data[k]['energy'] for k in cons_data.keys()])
        vow_energy = np.mean([vow_data[k]['energy'] for k in vow_data.keys()])
        
        if cons_energy / vow_energy > 2.0:
            print("⚠️ 신호 강도 차이: 자음 신호가 모음보다 훨씬 강함")
        elif vow_energy / cons_energy > 2.0:
            print("⚠️ 신호 강도 차이: 모음 신호가 자음보다 훨씬 강함")
        
        # 변동성 체크
        cons_var = np.mean([cons_data[k]['variance'] for k in cons_data.keys()])
        vow_var = np.mean([vow_data[k]['variance'] for k in vow_data.keys()])
        
        if cons_var / vow_var > 3.0:
            print("⚠️ 변동성 차이: 자음 데이터의 변동성이 모음보다 훨씬 큼")
        elif vow_var / cons_var > 3.0:
            print("⚠️ 변동성 차이: 모음 데이터의 변동성이 자음보다 훨씬 큼")
        
        # 4. 해결 방안
        print(f"\n💡 해결 방안:")
        print("1. 📊 데이터 수집 개선:")
        print("   - 모음 발음 시 센서 위치 최적화")
        print("   - 모음 데이터 수집량 증가")
        print("   - 모음 발음 방법 표준화")
        
        print("2. 🔧 전처리 개선:")
        print("   - 모음 데이터 전용 정규화")
        print("   - 노이즈 제거 강화")
        print("   - 시퀀스 길이 조정")
        
        print("3. 🎯 모델 개선:")
        print("   - 모음 클래스 가중치 증가")
        print("   - 모음 전용 데이터 증강")
        print("   - 앙상블 모델 고려")

def main():
    """메인 함수"""
    print("🔍 모음 데이터 품질 분석")
    print("=" * 50)
    
    # 데이터셋 로드
    dataset = VowelAnalysisDataset(
        data_path="../SignGlove_HW/datasets/unified",
        sequence_length=20
    )
    
    # 분석기 초기화
    analyzer = VowelAnalyzer(dataset)
    
    # 1. 신호 특성 분석
    signal_stats = analyzer.analyze_signal_characteristics()
    
    # 2. 신호 특성 시각화
    analyzer.visualize_signal_characteristics(signal_stats)
    
    # 3. 센서 패턴 분석
    analyzer.analyze_sensor_patterns()
    
    # 4. 시퀀스 패턴 분석
    analyzer.analyze_sequence_patterns()
    
    # 5. 통계적 검정
    test_results = analyzer.perform_statistical_tests(signal_stats)
    
    # 6. 요약 리포트
    analyzer.generate_summary_report(signal_stats, test_results)
    
    print(f"\n✅ 모음 데이터 품질 분석 완료!")
    print("📁 생성된 파일들:")
    print("  - vowel_consonant_characteristics.png")
    print("  - sensor_pattern_analysis.png")
    print("  - sequence_pattern_analysis.png")

if __name__ == "__main__":
    main()

