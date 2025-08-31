#!/usr/bin/env python3
"""
실제 학습 곡선과 평가 곡선
개선된 모델의 실제 학습 히스토리 사용
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class RealTrainingCurveGenerator:
    """실제 학습 곡선 생성기"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        print('📊 실제 학습 곡선 생성기 초기화 완료')
    
    def load_real_training_history(self, model_path='../models/improved_regularized_model.pth'):
        """실제 학습 히스토리 로드"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            history = checkpoint['history']
            print(f'✅ 실제 학습 히스토리 로드 완료: {model_path}')
            print(f'📊 총 에포크: {len(history["epochs"])}')
            print(f'🏆 최고 검증 정확도: {max(history["val_acc"]):.4f}')
            print(f'📉 최저 검증 손실: {min(history["val_loss"]):.4f}')
            return history
        except FileNotFoundError:
            print(f'⚠️ 모델 파일을 찾을 수 없습니다: {model_path}')
            return None
        except Exception as e:
            print(f'⚠️ 히스토리 로드 실패: {e}')
            return None
    
    def plot_real_training_curves(self, history, save_path='real_training_curves.png'):
        """실제 학습 곡선 플롯"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Real SignGlove Model Training Curves (Improved RegularizedModel)', fontsize=16, fontweight='bold')
        
        epochs = history['epochs']
        
        # 1. 손실 곡선
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, alpha=0.8)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
        ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(max(history['train_loss']), max(history['val_loss'])) * 1.1)
        
        # 최고 성능 지점 표시
        best_val_loss_idx = np.argmin(history['val_loss'])
        ax1.axvline(x=epochs[best_val_loss_idx], color='g', linestyle='--', alpha=0.7, label=f'Best Val Loss (Epoch {epochs[best_val_loss_idx]})')
        ax1.legend()
        
        # 2. 정확도 곡선
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2, alpha=0.8)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2, alpha=0.8)
        ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.0)
        
        # 최고 성능 지점 표시
        best_val_acc_idx = np.argmax(history['val_acc'])
        ax2.axvline(x=epochs[best_val_acc_idx], color='g', linestyle='--', alpha=0.7, label=f'Best Val Acc (Epoch {epochs[best_val_acc_idx]})')
        ax2.legend()
        
        # 3. 손실 비교 (로그 스케일)
        ax3.semilogy(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, alpha=0.8)
        ax3.semilogy(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
        ax3.set_title('Loss Curves (Log Scale)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss (Log Scale)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 과적합 분석
        overfitting = np.array(history['val_loss']) - np.array(history['train_loss'])
        ax4.plot(epochs, overfitting, 'g-', label='Overfitting Gap', linewidth=2, alpha=0.8)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation Loss - Training Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 과적합 경고선
        ax4.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Moderate Overfitting')
        ax4.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='High Overfitting')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✅ 실제 학습 곡선 저장: {save_path}')
        plt.show()
    
    def plot_detailed_real_analysis(self, history, save_path='detailed_real_analysis.png'):
        """실제 상세 분석"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Detailed Real Training Analysis (Improved RegularizedModel)', fontsize=16, fontweight='bold')
        
        epochs = history['epochs']
        
        # 1. 손실 변화율
        train_loss_diff = np.diff(history['train_loss'])
        val_loss_diff = np.diff(history['val_loss'])
        
        axes[0, 0].plot(epochs[1:], train_loss_diff, 'b-', label='Training Loss Change', linewidth=2, alpha=0.8)
        axes[0, 0].plot(epochs[1:], val_loss_diff, 'r-', label='Validation Loss Change', linewidth=2, alpha=0.8)
        axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Loss Change Rate')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss Change')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 정확도 변화율
        train_acc_diff = np.diff(history['train_acc'])
        val_acc_diff = np.diff(history['val_acc'])
        
        axes[0, 1].plot(epochs[1:], train_acc_diff, 'b-', label='Training Acc Change', linewidth=2, alpha=0.8)
        axes[0, 1].plot(epochs[1:], val_acc_diff, 'r-', label='Validation Acc Change', linewidth=2, alpha=0.8)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Accuracy Change Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy Change')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 학습 안정성 (이동 평균)
        window_size = 5
        train_loss_smooth = np.convolve(history['train_loss'], np.ones(window_size)/window_size, mode='valid')
        val_loss_smooth = np.convolve(history['val_loss'], np.ones(window_size)/window_size, mode='valid')
        smooth_epochs = epochs[window_size-1:]
        
        axes[0, 2].plot(smooth_epochs, train_loss_smooth, 'b-', label='Training Loss (Smoothed)', linewidth=2, alpha=0.8)
        axes[0, 2].plot(smooth_epochs, val_loss_smooth, 'r-', label='Validation Loss (Smoothed)', linewidth=2, alpha=0.8)
        axes[0, 2].set_title('Smoothed Loss Curves')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 손실 분포
        axes[1, 0].hist(history['train_loss'], bins=20, alpha=0.7, label='Training Loss', color='blue', density=True)
        axes[1, 0].hist(history['val_loss'], bins=20, alpha=0.7, label='Validation Loss', color='red', density=True)
        axes[1, 0].set_title('Loss Distribution')
        axes[1, 0].set_xlabel('Loss Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 정확도 분포
        axes[1, 1].hist(history['train_acc'], bins=20, alpha=0.7, label='Training Accuracy', color='blue', density=True)
        axes[1, 1].hist(history['val_acc'], bins=20, alpha=0.7, label='Validation Accuracy', color='red', density=True)
        axes[1, 1].set_title('Accuracy Distribution')
        axes[1, 1].set_xlabel('Accuracy Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 성능 지표 요약
        metrics = {
            'Final Train Loss': f"{history['train_loss'][-1]:.4f}",
            'Final Val Loss': f"{history['val_loss'][-1]:.4f}",
            'Final Train Acc': f"{history['train_acc'][-1]:.4f}",
            'Final Val Acc': f"{history['val_acc'][-1]:.4f}",
            'Best Val Acc': f"{max(history['val_acc']):.4f}",
            'Best Val Loss': f"{min(history['val_loss']):.4f}",
            'Overfitting Gap': f"{history['val_loss'][-1] - history['train_loss'][-1]:.4f}",
            'Total Epochs': f"{len(history['epochs'])}"
        }
        
        axes[1, 2].axis('off')
        y_pos = 0.95
        for metric, value in metrics.items():
            axes[1, 2].text(0.05, y_pos, f'{metric}: {value}', fontsize=11, fontweight='bold')
            y_pos -= 0.11
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✅ 실제 상세 분석 저장: {save_path}')
        plt.show()
    
    def plot_performance_comparison(self, history, save_path='performance_comparison.png'):
        """성능 비교 분석"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Comparison Analysis', fontsize=16, fontweight='bold')
        
        epochs = history['epochs']
        
        # 1. 학습 vs 검증 성능 비교
        axes[0, 0].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2, alpha=0.8)
        axes[0, 0].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2, alpha=0.8)
        axes[0, 0].fill_between(epochs, history['train_acc'], history['val_acc'], alpha=0.2, color='gray')
        axes[0, 0].set_title('Training vs Validation Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1.0)
        
        # 2. 손실 수렴 분석
        axes[0, 1].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, alpha=0.8)
        axes[0, 1].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
        axes[0, 1].set_title('Loss Convergence Analysis')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 수렴 지점 표시
        convergence_epoch = np.where(np.array(history['val_loss']) < 0.3)[0]
        if len(convergence_epoch) > 0:
            convergence_epoch = convergence_epoch[0]
            axes[0, 1].axvline(x=epochs[convergence_epoch], color='g', linestyle='--', alpha=0.7, label=f'Convergence (Epoch {epochs[convergence_epoch]})')
            axes[0, 1].legend()
        
        # 3. 과적합 지수
        overfitting_index = np.array(history['val_loss']) / np.array(history['train_loss'])
        axes[1, 0].plot(epochs, overfitting_index, 'purple', linewidth=2, alpha=0.8)
        axes[1, 0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No Overfitting')
        axes[1, 0].axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='Moderate Overfitting')
        axes[1, 0].axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='High Overfitting')
        axes[1, 0].set_title('Overfitting Index (Val Loss / Train Loss)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Overfitting Index')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 성능 향상률
        improvement_rate = np.diff(history['val_acc'])
        axes[1, 1].plot(epochs[1:], improvement_rate, 'g-', linewidth=2, alpha=0.8)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Validation Accuracy Improvement Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Improvement Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 성능 개선 구간 표시
        positive_improvement = improvement_rate > 0
        if np.any(positive_improvement):
            axes[1, 1].fill_between(epochs[1:], improvement_rate, 0, where=positive_improvement, alpha=0.3, color='green', label='Improvement')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✅ 성능 비교 분석 저장: {save_path}')
        plt.show()
    
    def generate_real_training_report(self, history, save_path='real_training_report.txt'):
        """실제 학습 보고서 생성"""
        report = []
        report.append("=" * 70)
        report.append("REAL SIGNGLOVE MODEL TRAINING REPORT (Improved RegularizedModel)")
        report.append("=" * 70)
        report.append("")
        
        # 기본 통계
        report.append("📊 BASIC STATISTICS:")
        report.append(f"  Total Epochs: {len(history['epochs'])}")
        report.append(f"  Final Training Loss: {history['train_loss'][-1]:.4f}")
        report.append(f"  Final Validation Loss: {history['val_loss'][-1]:.4f}")
        report.append(f"  Final Training Accuracy: {history['train_acc'][-1]:.4f}")
        report.append(f"  Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
        report.append("")
        
        # 최고 성능
        best_val_acc = max(history['val_acc'])
        best_val_acc_epoch = history['val_acc'].index(best_val_acc) + 1
        best_val_loss = min(history['val_loss'])
        best_val_loss_epoch = history['val_loss'].index(best_val_loss) + 1
        
        report.append("🏆 BEST PERFORMANCE:")
        report.append(f"  Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_val_acc_epoch})")
        report.append(f"  Best Validation Loss: {best_val_loss:.4f} (Epoch {best_val_loss_epoch})")
        report.append("")
        
        # 과적합 분석
        overfitting_gap = history['val_loss'][-1] - history['train_loss'][-1]
        overfitting_index = history['val_loss'][-1] / history['train_loss'][-1]
        
        report.append("🔍 OVERFITTING ANALYSIS:")
        report.append(f"  Final Overfitting Gap: {overfitting_gap:.4f}")
        report.append(f"  Overfitting Index: {overfitting_index:.4f}")
        
        if overfitting_index < 1.2:
            report.append("  ✅ Excellent generalization! Very low overfitting.")
        elif overfitting_index < 1.5:
            report.append("  ✅ Good generalization! Low overfitting.")
        elif overfitting_index < 2.0:
            report.append("  ⚠️ Moderate overfitting detected!")
        else:
            report.append("  ❌ High overfitting detected!")
        report.append("")
        
        # 학습 안정성
        train_loss_std = np.std(history['train_loss'][-10:])
        val_loss_std = np.std(history['val_loss'][-10:])
        
        report.append("📈 LEARNING STABILITY:")
        report.append(f"  Training Loss Stability: {train_loss_std:.4f}")
        report.append(f"  Validation Loss Stability: {val_loss_std:.4f}")
        
        if train_loss_std < 0.01 and val_loss_std < 0.01:
            report.append("  ✅ Very stable learning!")
        elif train_loss_std < 0.02 and val_loss_std < 0.02:
            report.append("  ✅ Stable learning!")
        else:
            report.append("  ⚠️ Unstable learning detected!")
        report.append("")
        
        # 수렴 분석
        convergence_epoch = np.where(np.array(history['val_loss']) < 0.3)[0]
        if len(convergence_epoch) > 0:
            convergence_epoch = convergence_epoch[0] + 1
            report.append("🎯 CONVERGENCE ANALYSIS:")
            report.append(f"  Model converged at epoch: {convergence_epoch}")
            report.append(f"  Convergence speed: {convergence_epoch}/{len(history['epochs'])} epochs")
        report.append("")
        
        # 성능 평가
        report.append("📊 PERFORMANCE EVALUATION:")
        if best_val_acc > 0.95:
            report.append("  🏆 Outstanding performance! (>95% accuracy)")
        elif best_val_acc > 0.90:
            report.append("  🎯 Excellent performance! (>90% accuracy)")
        elif best_val_acc > 0.85:
            report.append("  ✅ Good performance! (>85% accuracy)")
        elif best_val_acc > 0.80:
            report.append("  ⚠️ Acceptable performance! (>80% accuracy)")
        else:
            report.append("  ❌ Poor performance! (<80% accuracy)")
        report.append("")
        
        # 권장사항
        report.append("💡 RECOMMENDATIONS:")
        if overfitting_index > 1.5:
            report.append("  - Consider increasing dropout rate")
            report.append("  - Add more regularization")
            report.append("  - Implement early stopping")
        
        if best_val_acc < 0.90:
            report.append("  - Consider model architecture improvements")
            report.append("  - Check data quality and preprocessing")
            report.append("  - Try different learning rates")
        
        if convergence_epoch > len(history['epochs']) * 0.8:
            report.append("  - Model may need more training epochs")
        
        report.append("")
        report.append("=" * 70)
        
        # 파일 저장
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f'✅ 실제 학습 보고서 저장: {save_path}')
        
        # 콘솔 출력
        for line in report:
            print(line)

def main():
    """메인 실행 함수"""
    print('📊 실제 학습 곡선 생성 시스템')
    print('=' * 60)
    
    # 실제 학습 곡선 생성기 초기화
    generator = RealTrainingCurveGenerator()
    
    # 실제 학습 히스토리 로드
    history = generator.load_real_training_history('../models/improved_regularized_model.pth')
    
    if history is None:
        print('❌ 실제 학습 히스토리를 로드할 수 없습니다.')
        return
    
    # 1. 실제 학습 곡선
    print('\n📈 실제 학습 곡선 생성 중...')
    generator.plot_real_training_curves(history, 'real_training_curves.png')
    
    # 2. 실제 상세 분석
    print('\n🔍 실제 상세 분석 생성 중...')
    generator.plot_detailed_real_analysis(history, 'detailed_real_analysis.png')
    
    # 3. 성능 비교 분석
    print('\n📊 성능 비교 분석 생성 중...')
    generator.plot_performance_comparison(history, 'performance_comparison.png')
    
    # 4. 실제 학습 보고서
    print('\n📋 실제 학습 보고서 생성 중...')
    generator.generate_real_training_report(history, 'real_training_report.txt')
    
    print('\n🎉 실제 학습 곡선과 분석이 완료되었습니다!')
    print('📁 생성된 파일들:')
    print('  - real_training_curves.png')
    print('  - detailed_real_analysis.png')
    print('  - performance_comparison.png')
    print('  - real_training_report.txt')

if __name__ == "__main__":
    main()
