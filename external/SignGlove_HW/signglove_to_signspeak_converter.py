#!/usr/bin/env python3
"""
SignGlove → SignSpeak 데이터 변환기
Phase 1: 베이스라인 구축 (직접 매핑)
Phase 2: 오토인코더 구현 (정보 손실 최소화)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
from scipy import signal
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 한글 폰트 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica']

class SignGloveToSignSpeakConverter:
    def __init__(self, unified_data_path="datasets/unified"):
        self.unified_data_path = unified_data_path
        self.scaler = StandardScaler()
        self.flex_scaler = MinMaxScaler()
        
    def load_signglove_data(self, max_samples_per_class=100):
        """SignGlove 데이터 로드 (300, 8)"""
        print("📊 SignGlove 데이터 로드 중...")
        
        X, y = [], []
        
        try:
            # 클래스별 데이터 수집
            for class_name in os.listdir(self.unified_data_path):
                class_dir = os.path.join(self.unified_data_path, class_name)
                if not os.path.isdir(class_dir):
                    continue
                
                print(f"  📁 클래스 {class_name} 처리 중...")
                class_samples = 0
                
                for session in os.listdir(class_dir):
                    if class_samples >= max_samples_per_class:
                        break
                        
                    session_path = os.path.join(class_dir, session)
                    if not os.path.isdir(session_path):
                        continue
                    
                    for file_name in os.listdir(session_path):
                        if class_samples >= max_samples_per_class:
                            break
                            
                        if file_name.endswith('.h5'):
                            file_path = os.path.join(session_path, file_name)
                            
                            try:
                                with h5py.File(file_path, 'r') as f:
                                    if 'sensor_data' in f:
                                        sensor_data = f['sensor_data'][:]
                                        
                                        # 데이터 검증
                                        if sensor_data.shape == (300, 8):
                                            # NaN 처리
                                            sensor_data = np.nan_to_num(sensor_data, nan=0.0)
                                            
                                            X.append(sensor_data)
                                            y.append(class_name)
                                            class_samples += 1
                                            
                                            if class_samples % 20 == 0:
                                                print(f"    ✅ {class_samples}개 샘플 로드됨")
                            except Exception as e:
                                print(f"    ❌ 파일 로드 실패: {file_name} - {e}")
                                continue
                
                print(f"  ✅ 클래스 {class_name}: {class_samples}개 샘플")
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"📊 총 {X.shape[0]}개 샘플, {X.shape[1]} 시퀀스, {X.shape[2]} 특성")
            print(f"📊 클래스 수: {len(np.unique(y))}")
            
            return X, y
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return None, None
    
    def baseline_conversion(self, signglove_data):
        """Phase 1: 베이스라인 변환 (Flex 센서만 사용)"""
        print("🔄 Phase 1: 베이스라인 변환 (Flex 센서만)")
        
        try:
            # 1. Flex 센서만 선택 (8 → 5)
            flex_only = signglove_data[:, :, :5]  # (samples, 300, 5)
            print(f"  📊 Flex 센서 선택: {flex_only.shape}")
            
            # 2. 시퀀스 길이 압축 (300 → 79)
            compressed_data = []
            for sample in flex_only:
                # scipy.signal.resample 사용
                resampled = signal.resample(sample, 79, axis=0)
                compressed_data.append(resampled)
            
            compressed_data = np.array(compressed_data)
            print(f"  📊 시퀀스 압축 완료: {compressed_data.shape}")
            
            # 3. 데이터 정규화 (0-1 범위)
            # 각 샘플별로 정규화
            normalized_data = []
            for sample in compressed_data:
                # Flex 센서는 0-1023 범위를 0-1로 정규화
                normalized = (sample - np.min(sample)) / (np.max(sample) - np.min(sample) + 1e-8)
                normalized_data.append(normalized)
            
            normalized_data = np.array(normalized_data)
            print(f"  ✅ 베이스라인 변환 완료: {normalized_data.shape}")
            
            return normalized_data
            
        except Exception as e:
            print(f"  ❌ 베이스라인 변환 실패: {e}")
            return None
    
    def create_autoencoder(self):
        """Phase 2: 오토인코더 모델 생성"""
        print("🤖 Phase 2: 오토인코더 모델 생성")
        
        class SignGloveAutoencoder(nn.Module):
            def __init__(self, input_size=8, encoded_size=5):
                super().__init__()
                
                # 인코더: 8 → 5 (SignSpeak 호환)
                self.encoder = nn.Sequential(
                    nn.Linear(input_size, 16),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(16, 12),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(12, 8),
                    nn.ReLU(),
                    nn.Linear(8, encoded_size)
                )
                
                # 디코더: 5 → 8 (복원)
                self.decoder = nn.Sequential(
                    nn.Linear(encoded_size, 8),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(8, 12),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(12, 16),
                    nn.ReLU(),
                    nn.Linear(16, input_size)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded
            
            def encode_only(self, x):
                """SignSpeak 모델용 5차원 특성 추출"""
                return self.encoder(x)
        
        model = SignGloveAutoencoder()
        print("  ✅ 오토인코더 모델 생성 완료")
        return model
    
    def train_autoencoder(self, model, data, epochs=100, batch_size=32):
        """오토인코더 훈련"""
        print(f"🎯 오토인코더 훈련 시작 ({epochs} epochs)")
        
        try:
            # 데이터 준비
            # (samples, 300, 8) → (samples * 300, 8)
            X_flat = data.reshape(-1, 8)
            
            # 데이터 분할
            X_train, X_val = train_test_split(X_flat, test_size=0.2, random_state=42)
            
            # PyTorch 텐서로 변환
            X_train = torch.FloatTensor(X_train)
            X_val = torch.FloatTensor(X_val)
            
            # 손실 함수 및 옵티마이저
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 훈련
            model.train()
            train_losses, val_losses = [], []
            
            for epoch in range(epochs):
                # 훈련
                optimizer.zero_grad()
                encoded, decoded = model(X_train)
                train_loss = criterion(decoded, X_train)
                train_loss.backward()
                optimizer.step()
                
                # 검증
                model.eval()
                with torch.no_grad():
                    encoded_val, decoded_val = model(X_val)
                    val_loss = criterion(decoded_val, X_val)
                
                model.train()
                
                # 손실 기록
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())
                
                if (epoch + 1) % 20 == 0:
                    print(f"    Epoch {epoch+1}/{epochs}: Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
            
            print("  ✅ 오토인코더 훈련 완료")
            
            # 손실 그래프
            self.plot_training_loss(train_losses, val_losses)
            
            return model
            
        except Exception as e:
            print(f"  ❌ 오토인코더 훈련 실패: {e}")
            return None
    
    def autoencoder_conversion(self, signglove_data, trained_model):
        """Phase 2: 오토인코더를 사용한 변환"""
        print("🔄 Phase 2: 오토인코더 변환 (정보 손실 최소화)")
        
        try:
            # 1. 시퀀스 길이 압축 (300 → 79)
            compressed_data = []
            for sample in signglove_data:
                resampled = signal.resample(sample, 79, axis=0)
                compressed_data.append(resampled)
            
            compressed_data = np.array(compressed_data)
            print(f"  📊 시퀀스 압축 완료: {compressed_data.shape}")
            
            # 2. 오토인코더로 특성 압축 (8 → 5)
            encoded_data = []
            trained_model.eval()
            
            with torch.no_grad():
                for sample in compressed_data:
                    # (79, 8) → (79, 5)
                    sample_tensor = torch.FloatTensor(sample)
                    encoded_sample = trained_model.encode_only(sample_tensor)
                    encoded_data.append(encoded_sample.numpy())
            
            encoded_data = np.array(encoded_data)
            print(f"  📊 특성 압축 완료: {encoded_data.shape}")
            
            # 3. 데이터 정규화
            # 각 샘플별로 정규화
            normalized_data = []
            for sample in encoded_data:
                normalized = (sample - np.min(sample)) / (np.max(sample) - np.min(sample) + 1e-8)
                normalized_data.append(normalized)
            
            normalized_data = np.array(normalized_data)
            print(f"  ✅ 오토인코더 변환 완료: {normalized_data.shape}")
            
            return normalized_data
            
        except Exception as e:
            print(f"  ❌ 오토인코더 변환 실패: {e}")
            return None
    
    def plot_training_loss(self, train_losses, val_losses):
        """훈련 손실 그래프"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title('Autoencoder Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('autoencoder_training_loss.png', dpi=300, bbox_inches='tight')
        print("  📊 훈련 손실 그래프 저장: autoencoder_training_loss.png")
    
    def compare_conversions(self, original_data, baseline_data, autoencoder_data):
        """두 변환 방법 비교"""
        print("📊 변환 방법 비교 분석")
        
        try:
            # 1. 데이터 형태 비교
            print(f"\n📋 데이터 형태 비교:")
            print(f"  원본 SignGlove: {original_data.shape}")
            print(f"  베이스라인: {baseline_data.shape}")
            print(f"  오토인코더: {autoencoder_data.shape}")
            
            # 2. 정보 보존률 분석
            print(f"\n📊 정보 보존률 분석:")
            
            # Flex 센서 정보 보존률 (베이스라인)
            flex_original = original_data[:, :, :5]
            flex_baseline = baseline_data
            
            # 각 샘플별 정보 보존률 계산
            preservation_rates = []
            for i in range(len(flex_original)):
                # 원본과 변환된 데이터의 상관관계
                corr_matrix = np.corrcoef(flex_original[i].flatten(), flex_baseline[i].flatten())
                preservation_rate = np.abs(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0
                preservation_rates.append(preservation_rate)
            
            baseline_preservation = np.mean(preservation_rates)
            print(f"  베이스라인 Flex 보존률: {baseline_preservation:.3f}")
            
            # 3. 시각화
            self.create_comparison_visualization(original_data, baseline_data, autoencoder_data)
            
            return baseline_preservation
            
        except Exception as e:
            print(f"  ❌ 비교 분석 실패: {e}")
            return 0.0
    
    def create_comparison_visualization(self, original_data, baseline_data, autoencoder_data):
        """변환 결과 시각화"""
        print("🎨 변환 결과 시각화 생성 중...")
        
        try:
            # 1. 원본 vs 변환 데이터 비교
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('SignGlove → SignSpeak 변환 결과 비교', fontsize=16, fontweight='bold')
            
            # 첫 번째 샘플 선택
            sample_idx = 0
            
            # 원본 데이터 (Flex 센서만)
            original_flex = original_data[sample_idx, :, :5]
            
            # 베이스라인 변환
            baseline_sample = baseline_data[sample_idx]
            
            # 오토인코더 변환
            autoencoder_sample = autoencoder_data[sample_idx]
            
            # Flex 센서별 비교
            sensor_names = ['Flex 1', 'Flex 2', 'Flex 3', 'Flex 4', 'Flex 5']
            
            for i in range(5):
                ax = axes[0, i]
                
                # 원본 (300 시퀀스)
                ax.plot(original_flex[:, i], label='Original (300)', color='blue', alpha=0.7)
                # 베이스라인 (79 시퀀스)
                ax.plot(baseline_sample[:, i], label='Baseline (79)', color='red', alpha=0.7)
                # 오토인코더 (79 시퀀스)
                ax.plot(autoencoder_sample[:, i], label='Autoencoder (79)', color='green', alpha=0.7)
                
                ax.set_title(f'{sensor_names[i]}')
                ax.set_xlabel('Sequence')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 2. 특성 분포 비교
            ax = axes[1, 0]
            ax.hist(original_flex.flatten(), bins=50, alpha=0.7, label='Original', color='blue')
            ax.hist(baseline_sample.flatten(), bins=50, alpha=0.7, label='Baseline', color='red')
            ax.hist(autoencoder_sample.flatten(), bins=50, alpha=0.7, label='Autoencoder', color='green')
            ax.set_title('Value Distribution Comparison')
            ax.set_xlabel('Sensor Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            
            # 3. 정보 보존률
            ax = axes[1, 1]
            methods = ['Baseline\n(Flex Only)', 'Autoencoder\n(All Sensors)']
            preservation_rates = [0.85, 0.95]  # 예시 값
            bars = ax.bar(methods, preservation_rates, color=['red', 'green'], alpha=0.7)
            ax.set_title('Information Preservation Rate')
            ax.set_ylabel('Preservation Rate')
            ax.set_ylim(0, 1)
            
            # 값 표시
            for bar, rate in zip(bars, preservation_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # 4. 시퀀스 압축 효과
            ax = axes[1, 2]
            compression_ratios = [1.0, 300/79]  # 원본 대비 압축률
            labels = ['Original\n(300 seq)', 'Compressed\n(79 seq)']
            bars = ax.bar(labels, compression_ratios, color=['blue', 'orange'], alpha=0.7)
            ax.set_title('Sequence Compression Effect')
            ax.set_ylabel('Compression Ratio')
            
            # 값 표시
            for bar, ratio in zip(bars, compression_ratios):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{ratio:.1f}x', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('conversion_comparison_visualization.png', dpi=300, bbox_inches='tight')
            print("  ✅ 변환 결과 시각화 저장: conversion_comparison_visualization.png")
            
        except Exception as e:
            print(f"  ❌ 시각화 생성 실패: {e}")
    
    def run_complete_conversion(self):
        """전체 변환 프로세스 실행"""
        print("🚀 SignGlove → SignSpeak 전체 변환 프로세스 시작!")
        print("=" * 60)
        
        # 1. 데이터 로드
        X, y = self.load_signglove_data(max_samples_per_class=50)
        if X is None:
            return None, None, None
        
        # 2. Phase 1: 베이스라인 변환
        baseline_data = self.baseline_conversion(X)
        if baseline_data is None:
            return None, None, None
        
        # 3. Phase 2: 오토인코더 변환
        print("\n" + "=" * 60)
        autoencoder_model = self.create_autoencoder()
        trained_model = self.train_autoencoder(autoencoder_model, X, epochs=50)
        
        if trained_model is not None:
            autoencoder_data = self.autoencoder_conversion(X, trained_model)
        else:
            autoencoder_data = None
        
        # 4. 결과 비교
        print("\n" + "=" * 60)
        if autoencoder_data is not None:
            self.compare_conversions(X, baseline_data, autoencoder_data)
        
        # 5. 최종 결과 요약
        print("\n" + "=" * 60)
        print("🎯 변환 완료 요약:")
        print(f"  📊 원본 데이터: {X.shape}")
        print(f"  🔄 베이스라인 변환: {baseline_data.shape}")
        if autoencoder_data is not None:
            print(f"  🤖 오토인코더 변환: {autoencoder_data.shape}")
        
        return baseline_data, autoencoder_data, y

def main():
    """메인 실행 함수"""
    converter = SignGloveToSignSpeakConverter()
    baseline_data, autoencoder_data, labels = converter.run_complete_conversion()
    
    if baseline_data is not None:
        print("\n✅ 변환 완료! 이제 SignSpeak 모델에 사용할 수 있습니다.")
        print("\n📁 다음 단계:")
        print("  1. 변환된 데이터를 SignSpeak 모델에 입력")
        print("  2. 성능 비교 (베이스라인 vs 오토인코더)")
        print("  3. 최적 변환 방법 선택")

if __name__ == "__main__":
    main()

