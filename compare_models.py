#!/usr/bin/env python3
"""
모델별 성능 비교 시각화 스크립트
여러 모델의 학습 결과를 비교하여 시각화합니다.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import glob
from typing import Dict, List, Tuple, Optional
import argparse

def parse_log_file(log_file: str) -> Dict[str, List[float]]:
    """로그 파일에서 메트릭을 파싱합니다."""
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'learning_rate': []
    }
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Epoch-averaged 로그 형식 파싱
        epoch_pattern = r'Epoch (\d+):.*?train/loss=([\d.]+).*?train/accuracy=([\d.]+).*?val/loss=([\d.]+).*?val/accuracy=([\d.]+|nan\.0).*?val/f1_score=([\d.]+).*?learning_rate=([\d.]+)'
        matches = re.findall(epoch_pattern, content)
        
        for match in matches:
            epoch, train_loss, train_acc, val_loss, val_acc, val_f1, lr = match
            
            # NaN 값 처리
            val_acc = 0.0 if val_acc == 'nan.0' else float(val_acc)
            
            metrics['train_loss'].append(float(train_loss))
            metrics['train_acc'].append(float(train_acc))
            metrics['val_loss'].append(float(val_loss))
            metrics['val_acc'].append(val_acc)
            metrics['val_f1'].append(float(val_f1))
            metrics['learning_rate'].append(float(lr))
            
    except Exception as e:
        print(f"❌ 로그 파일 파싱 실패: {log_file} - {e}")
        return {}
    
    return metrics

def plot_model_comparison(model_results: Dict[str, Dict], save_dir: str = "model_comparison"):
    """모델별 성능 비교 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 색상 팔레트
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # 1. Loss 비교
    plt.figure(figsize=(15, 10))
    
    # Train Loss
    plt.subplot(2, 3, 1)
    for i, (model_name, metrics) in enumerate(model_results.items()):
        if metrics['train_loss']:
            epochs = list(range(1, len(metrics['train_loss']) + 1))
            plt.plot(epochs, metrics['train_loss'], 
                    label=f'{model_name} (Train)', 
                    color=colors[i % len(colors)], 
                    linewidth=2, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation Loss
    plt.subplot(2, 3, 2)
    for i, (model_name, metrics) in enumerate(model_results.items()):
        if metrics['val_loss']:
            epochs = list(range(1, len(metrics['val_loss']) + 1))
            plt.plot(epochs, metrics['val_loss'], 
                    label=f'{model_name} (Val)', 
                    color=colors[i % len(colors)], 
                    linewidth=2, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Train Accuracy
    plt.subplot(2, 3, 3)
    for i, (model_name, metrics) in enumerate(model_results.items()):
        if metrics['train_acc']:
            epochs = list(range(1, len(metrics['train_acc']) + 1))
            plt.plot(epochs, metrics['train_acc'], 
                    label=f'{model_name} (Train)', 
                    color=colors[i % len(colors)], 
                    linewidth=2, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation Accuracy/F1
    plt.subplot(2, 3, 4)
    for i, (model_name, metrics) in enumerate(model_results.items()):
        if metrics['val_acc'] and any(acc > 0 for acc in metrics['val_acc']):
            epochs = list(range(1, len(metrics['val_acc']) + 1))
            plt.plot(epochs, metrics['val_acc'], 
                    label=f'{model_name} (Val Acc)', 
                    color=colors[i % len(colors)], 
                    linewidth=2, alpha=0.8)
        elif metrics['val_f1']:
            epochs = list(range(1, len(metrics['val_f1']) + 1))
            plt.plot(epochs, metrics['val_f1'], 
                    label=f'{model_name} (Val F1)', 
                    color=colors[i % len(colors)], 
                    linewidth=2, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/F1 Score')
    plt.title('Validation Performance Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # F1 Score
    plt.subplot(2, 3, 5)
    for i, (model_name, metrics) in enumerate(model_results.items()):
        if metrics['val_f1']:
            epochs = list(range(1, len(metrics['val_f1']) + 1))
            plt.plot(epochs, metrics['val_f1'], 
                    label=f'{model_name}', 
                    color=colors[i % len(colors)], 
                    linewidth=2, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning Rate
    plt.subplot(2, 3, 6)
    for i, (model_name, metrics) in enumerate(model_results.items()):
        if metrics['learning_rate']:
            epochs = list(range(1, len(metrics['learning_rate']) + 1))
            plt.plot(epochs, metrics['learning_rate'], 
                    label=f'{model_name}', 
                    color=colors[i % len(colors)], 
                    linewidth=2, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 최종 성능 요약 테이블
    create_performance_summary(model_results, save_dir)
    
    print(f"✅ 모델 비교 시각화 완료: {save_dir}/")

def create_performance_summary(model_results: Dict[str, Dict], save_dir: str):
    """최종 성능 요약 테이블 생성"""
    summary_data = []
    
    for model_name, metrics in model_results.items():
        if not metrics:
            continue
            
        # 최종 성능 추출
        final_train_loss = metrics['train_loss'][-1] if metrics['train_loss'] else 0
        final_train_acc = metrics['train_acc'][-1] if metrics['train_acc'] else 0
        final_val_loss = metrics['val_loss'][-1] if metrics['val_loss'] else 0
        final_val_acc = metrics['val_acc'][-1] if metrics['val_acc'] else 0
        final_val_f1 = metrics['val_f1'][-1] if metrics['val_f1'] else 0
        
        # 최고 성능 추출
        best_val_loss = min(metrics['val_loss']) if metrics['val_loss'] else 0
        best_val_f1 = max(metrics['val_f1']) if metrics['val_f1'] else 0
        
        summary_data.append({
            'Model': model_name,
            'Final Train Loss': f"{final_train_loss:.4f}",
            'Final Train Acc': f"{final_train_acc:.4f}",
            'Final Val Loss': f"{final_val_loss:.4f}",
            'Final Val Acc': f"{final_val_acc:.4f}",
            'Final Val F1': f"{final_val_f1:.4f}",
            'Best Val Loss': f"{best_val_loss:.4f}",
            'Best Val F1': f"{best_val_f1:.4f}",
            'Epochs': len(metrics['train_loss'])
        })
    
    # 테이블 시각화
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    if summary_data:
        # 데이터 준비
        headers = list(summary_data[0].keys())
        rows = [[row[header] for header in headers] for row in summary_data]
        
        # 테이블 생성
        table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # 스타일링
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(rows) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
    
    plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'{save_dir}/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Model Comparison Visualization')
    parser.add_argument('--log_dir', type=str, default='.', help='Directory containing log files')
    parser.add_argument('--save_dir', type=str, default='model_comparison', help='Directory to save comparison plots')
    parser.add_argument('--models', nargs='+', help='Specific models to compare (e.g., GRU LSTM MSCSGRU)')
    
    args = parser.parse_args()
    
    # 로그 파일 찾기
    log_files = glob.glob(f"{args.log_dir}/training_output_*.log")
    
    if not log_files:
        print("❌ 로그 파일을 찾을 수 없습니다.")
        return
    
    print(f"🔍 {len(log_files)}개의 로그 파일을 찾았습니다.")
    
    # 모델별 결과 수집
    model_results = {}
    
    for log_file in log_files:
        # 파일명에서 모델명 추출
        filename = os.path.basename(log_file)
        if 'training_output_' in filename:
            model_name = filename.replace('training_output_', '').replace('.log', '')
        else:
            model_name = os.path.splitext(filename)[0]
        
        # 특정 모델만 비교하는 경우 필터링
        if args.models and model_name not in args.models:
            continue
        
        print(f"📊 {model_name} 모델 파싱 중...")
        metrics = parse_log_file(log_file)
        
        if metrics:
            model_results[model_name] = metrics
            print(f"  ✅ {len(metrics['train_loss'])} 에포크 데이터 파싱 완료")
        else:
            print(f"  ❌ 파싱 실패")
    
    if not model_results:
        print("❌ 파싱된 모델 결과가 없습니다.")
        return
    
    print(f"\n📈 {len(model_results)}개 모델 비교 시각화 시작...")
    plot_model_comparison(model_results, args.save_dir)
    print("✅ 모델 비교 완료!")

if __name__ == "__main__":
    main()

