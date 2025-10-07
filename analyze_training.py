#!/usr/bin/env python3
"""
KLP-SignGlove Training Analysis
ASL 스타일의 시각화 스크립트
"""
import os
import re
import sys
import argparse
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np


def parse_training_log(log_path: str) -> Dict[str, List[float]]:
    """
    PyTorch Lightning 로그에서 메트릭을 추출합니다.
    
    예상 로그 형식:
    Epoch 0: 100%|██████████| 14/14 [00:00<00:00, 35.61it/s, v_num=0, val/loss=2.45, val/accuracy=0.234, val/f1_score=0.189, learning_rate=0.001, train/loss=2.67, train/accuracy=0.156]
    """
    
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    val_f1 = []
    learning_rate = []
    
    # 정규표현식 패턴 (더 유연하게)
    train_loss_re = re.compile(r"train/loss=([0-9.eE+-]+)")
    train_acc_re = re.compile(r"train/accuracy=([0-9.eE+-]+)")
    val_loss_re = re.compile(r"val/loss=([0-9.eE+-]+)")
    val_acc_re = re.compile(r"val/accuracy=([0-9.eE+-]+)")
    val_f1_re = re.compile(r"val/f1_score=([0-9.eE+-]+)")
    lr_re = re.compile(r"learning_rate=([0-9.eE+-]+)")
    
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # train/loss
            m = train_loss_re.search(line)
            if m:
                try:
                    val = float(m.group(1))
                    if not np.isnan(val) and not np.isinf(val):
                        train_loss.append(val)
                except:
                    pass
            
            # train/accuracy
            m = train_acc_re.search(line)
            if m:
                try:
                    val = float(m.group(1))
                    if not np.isnan(val) and not np.isinf(val):
                        train_acc.append(val)
                except:
                    pass
            
            # val/loss
            m = val_loss_re.search(line)
            if m:
                try:
                    val = float(m.group(1))
                    if not np.isnan(val) and not np.isinf(val):
                        val_loss.append(val)
                except:
                    pass
            
            # val/accuracy
            m = val_acc_re.search(line)
            if m:
                try:
                    val = float(m.group(1))
                    if not np.isnan(val) and not np.isinf(val):
                        val_acc.append(val)
                except:
                    pass
            
            # val/f1_score
            m = val_f1_re.search(line)
            if m:
                try:
                    val = float(m.group(1))
                    if not np.isnan(val) and not np.isinf(val):
                        val_f1.append(val)
                except:
                    pass
            
            # learning_rate
            m = lr_re.search(line)
            if m:
                try:
                    val = float(m.group(1))
                    if not np.isnan(val) and not np.isinf(val):
                        learning_rate.append(val)
                except:
                    pass
    
    # 중복 제거 (progress bar redraw 때문에 같은 값이 여러 번 나올 수 있음)
    def deduplicate(lst: List[float]) -> List[float]:
        result = []
        prev = None
        for val in lst:
            if prev is None or abs(val - prev) > 1e-9:
                result.append(val)
                prev = val
        return result
    
    metrics = {
        'train_loss': deduplicate(train_loss),
        'train_acc': deduplicate(train_acc),
        'val_loss': deduplicate(val_loss),
        'val_acc': deduplicate(val_acc),
        'val_f1': deduplicate(val_f1),
        'learning_rate': deduplicate(learning_rate),
    }
    
    return metrics


def plot_training_curves(metrics: Dict[str, List[float]], save_dir: str) -> None:
    """ASL 스타일의 시각화 곡선을 생성합니다."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 에포크 수 결정
    n_epochs = max(
        len(metrics['train_loss']),
        len(metrics['val_loss']),
        len(metrics['train_acc']),
        len(metrics['val_acc'])
    )
    
    if n_epochs == 0:
        raise ValueError("❌ 로그 파일에서 메트릭을 찾을 수 없습니다.")
    
    print(f"📊 총 {n_epochs} 에포크의 데이터를 찾았습니다.")
    
    # 1) Train vs Val Loss
    plt.figure(figsize=(10, 5))
    if metrics['train_loss']:
        epochs_train = list(range(1, len(metrics['train_loss']) + 1))
        plt.plot(epochs_train, metrics['train_loss'], 
                label='Train Loss', color='tab:blue', linewidth=2, alpha=0.8)
    if metrics['val_loss']:
        epochs_val = list(range(1, len(metrics['val_loss']) + 1))
        plt.plot(epochs_val, metrics['val_loss'], 
                label='Val Loss', color='tab:red', linewidth=2, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    # x축을 실제 데이터 길이에 맞게 제한
    max_epochs = max(len(metrics['train_loss']) if metrics['train_loss'] else 0,
                     len(metrics['val_loss']) if metrics['val_loss'] else 0)
    plt.xlim(1, max_epochs)
    plt.title('Train vs Validation Loss (KLP)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_val_loss.png'), dpi=200)
    plt.close()
    print("✅ train_val_loss.png 생성 완료")
    
    # 2) Train vs Val Accuracy
    plt.figure(figsize=(10, 5))
    if metrics['train_acc']:
        epochs_train = list(range(1, len(metrics['train_acc']) + 1))
        plt.plot(epochs_train, metrics['train_acc'], 
                label='Train Accuracy', color='tab:blue', linewidth=2, alpha=0.8)
    if metrics['val_acc']:
        epochs_val = list(range(1, len(metrics['val_acc']) + 1))
        plt.plot(epochs_val, metrics['val_acc'], 
                label='Val Accuracy', color='tab:red', linewidth=2, alpha=0.8)
    else:
        # Val Accuracy가 없으면 F1 Score로 대체
        if metrics['val_f1']:
            epochs_val = list(range(1, len(metrics['val_f1']) + 1))
            plt.plot(epochs_val, metrics['val_f1'], 
                    label='Val F1 Score', color='tab:red', linewidth=2, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.05)
    # x축을 실제 데이터 길이에 맞게 제한
    max_epochs = max(len(metrics['train_acc']) if metrics['train_acc'] else 0,
                     len(metrics['val_acc']) if metrics['val_acc'] else 0,
                     len(metrics['val_f1']) if metrics['val_f1'] else 0)
    plt.xlim(1, max_epochs)
    plt.title('Train vs Validation Accuracy (KLP)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_val_accuracy.png'), dpi=200)
    plt.close()
    print("✅ train_val_accuracy.png 생성 완료")
    
    # 3) Overfitting Analysis
    if metrics['train_loss'] and metrics['val_loss'] and metrics['train_acc'] and metrics['val_acc']:
        n_common = min(len(metrics['train_loss']), len(metrics['val_loss']),
                      len(metrics['train_acc']), len(metrics['val_acc']))
        
        if n_common > 0:
            epochs_common = list(range(1, n_common + 1))
            loss_gap = [metrics['val_loss'][i] - metrics['train_loss'][i] for i in range(n_common)]
            acc_gap = [metrics['train_acc'][i] - metrics['val_acc'][i] for i in range(n_common)]
            
            fig, axs = plt.subplots(1, 2, figsize=(14, 5))
            
            axs[0].plot(epochs_common, loss_gap, label='Loss Gap (Val - Train)', 
                       color='tab:purple', linewidth=2, alpha=0.8)
            axs[0].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            axs[0].set_xlabel('Epoch', fontsize=12)
            axs[0].set_ylabel('Gap', fontsize=12)
            axs[0].set_title('Loss Gap (Val - Train)', fontsize=13, fontweight='bold')
            axs[0].grid(True, alpha=0.3)
            axs[0].legend(fontsize=10)
            
            axs[1].plot(epochs_common, acc_gap, label='Acc Gap (Train - Val)', 
                       color='tab:orange', linewidth=2, alpha=0.8)
            axs[1].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            axs[1].set_xlabel('Epoch', fontsize=12)
            axs[1].set_ylabel('Gap', fontsize=12)
            axs[1].set_title('Accuracy Gap (Train - Val)', fontsize=13, fontweight='bold')
            axs[1].grid(True, alpha=0.3)
            axs[1].legend(fontsize=10)
            
            fig.suptitle('Overfitting Analysis (KLP)', fontsize=15, fontweight='bold')
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, 'overfitting_analysis.png'), dpi=200)
            plt.close(fig)
            print("✅ overfitting_analysis.png 생성 완료")
    
    # 4) Detailed Metrics (F1 Score + Learning Rate)
    fig = plt.figure(figsize=(10, 8))
    
    # F1 Score
    plt.subplot(3, 1, 1)
    if metrics['val_f1']:
        epochs_f1 = list(range(1, len(metrics['val_f1']) + 1))
        plt.plot(epochs_f1, metrics['val_f1'], 
                label='Val F1 Score', color='tab:green', linewidth=2, alpha=0.8)
        plt.ylabel('F1 Score', fontsize=11)
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.title('Validation F1 Score', fontsize=12, fontweight='bold')
    
    # Val Loss (log scale)
    plt.subplot(3, 1, 2)
    if metrics['val_loss']:
        epochs_val = list(range(1, len(metrics['val_loss']) + 1))
        plt.semilogy(epochs_val, metrics['val_loss'], 
                    label='Val Loss', color='tab:red', linewidth=2, alpha=0.8)
        plt.ylabel('Loss (log)', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.title('Validation Loss (Log Scale)', fontsize=12, fontweight='bold')
    
    # Learning Rate
    plt.subplot(3, 1, 3)
    if metrics['learning_rate']:
        epochs_lr = list(range(1, len(metrics['learning_rate']) + 1))
        plt.plot(epochs_lr, metrics['learning_rate'], 
                label='Learning Rate', color='tab:cyan', linewidth=2, alpha=0.8)
        plt.xlabel('Epoch', fontsize=11)
        plt.ylabel('Learning Rate', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    
    fig.suptitle('Detailed Metrics (KLP)', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'detailed_metrics.png'), dpi=200)
    plt.close(fig)
    print("✅ detailed_metrics.png 생성 완료")


def print_summary(metrics: Dict[str, List[float]]) -> None:
    """학습 결과 요약을 출력합니다."""
    print("\n" + "="*70)
    print("📊 학습 결과 요약")
    print("="*70)
    
    if metrics['train_loss']:
        print(f"🔵 Train Loss: {metrics['train_loss'][0]:.4f} → {metrics['train_loss'][-1]:.4f}")
    if metrics['train_acc']:
        print(f"🔵 Train Acc:  {metrics['train_acc'][0]:.4f} → {metrics['train_acc'][-1]:.4f}")
    
    if metrics['val_loss']:
        print(f"🔴 Val Loss:   {metrics['val_loss'][0]:.4f} → {metrics['val_loss'][-1]:.4f}")
    if metrics['val_acc']:
        print(f"🔴 Val Acc:    {metrics['val_acc'][0]:.4f} → {metrics['val_acc'][-1]:.4f}")
    if metrics['val_f1']:
        print(f"🟢 Val F1:     {metrics['val_f1'][0]:.4f} → {metrics['val_f1'][-1]:.4f}")
    
    # 최고 성능
    if metrics['val_acc']:
        best_val_acc = max(metrics['val_acc'])
        best_epoch = metrics['val_acc'].index(best_val_acc) + 1
        print(f"\n🏆 Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    if metrics['val_f1']:
        best_val_f1 = max(metrics['val_f1'])
        best_epoch_f1 = metrics['val_f1'].index(best_val_f1) + 1
        print(f"🏆 Best Val F1:  {best_val_f1:.4f} (Epoch {best_epoch_f1})")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KLP-SignGlove Training Analysis')
    parser.add_argument('-model', '--model_name', type=str, default='GRU', 
                       help='Model name (GRU, LSTM, MSCSGRU, etc.)')
    parser.add_argument('-log', '--log_file', type=str, 
                       help='Log file path (default: training_output_{model_name}.log)')
    
    args = parser.parse_args()
    
    repo_root = os.path.dirname(os.path.abspath(__file__))
    
    # 로그 파일 경로 설정
    if args.log_file:
        log_path = args.log_file
    else:
        log_path = os.path.join(repo_root, f"training_output_{args.model_name}.log")
    
    # 시각화 저장 폴더 설정
    save_dir = os.path.join(repo_root, "visualizations", args.model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"❌ 로그 파일을 찾을 수 없습니다: {log_path}")
    
    print(f"🔍 {args.model_name} 모델 로그 파싱 중...")
    metrics = parse_training_log(log_path)
    
    print(f"📈 {args.model_name} 모델 시각화 생성 중...")
    plot_training_curves(metrics, save_dir)
    
    print_summary(metrics)
    
    print(f"\n✅ {args.model_name} 모델 시각화가 완료되었습니다: {save_dir}/")

