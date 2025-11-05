#!/usr/bin/env python3
"""
4개 모델에 대한 혼동 행렬 생성: GRU, StackedGRU, MS3DGRU, MS3DStackedGRU
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score, classification_report
)
from tqdm import tqdm
import torch

from src.misc.DynamicDataModule import DynamicDataModule
from inference import SignGloveInference

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정"""
    import matplotlib.font_manager as fm
    available_fonts = [f.name for f in fm.fontManager.ttflist if 'nanum' in f.name.lower()]
    if available_fonts:
        preferred_fonts = ['NanumGothic', 'NanumBarunGothic', 'NanumSquare']
        for font in preferred_fonts:
            if font in available_fonts:
                plt.rcParams['font.family'] = font
                print(f"✅ Korean font setup complete: {font}")
                return True
        plt.rcParams['font.family'] = available_fonts[0]
        print(f"✅ Korean font setup complete: {available_fonts[0]}")
        return True
    print("⚠️  Korean font not found. Using English labels.")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    return False

# 한글 클래스명
CLASS_NAMES = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ'
]

def generate_confusion_matrix_for_model(model_name, config, test_loader, output_dir):
    """단일 모델에 대한 혼동 행렬 생성"""
    
    print(f"\n🤖 모델: {model_name} 혼동 행렬 생성 중...")
    
    try:
        # 추론 엔진 초기화
        init_params = {
            'model_path': config['path'],
            'model_type': config['type'],
            'input_size': 8,
            'hidden_size': config.get('hidden_size', 64),
            'classes': 24,
            'device': 'cpu',
            'dropout': config.get('dropout', 0.2),
            'scaler_path': config.get('scaler_path', None)  # Scaler 경로 추가
        }
        if 'cnn_filters' in config:
            init_params['cnn_filters'] = config['cnn_filters']
        # GRU 모델은 layers=1로 설정 (체크포인트와 일치)
        if config['type'] == 'GRU':
            init_params['layers'] = config.get('layers', 1)
        
        engine = SignGloveInference(**init_params)
        
        # 테스트 데이터셋에서 예측 수행
        all_predictions = []
        all_labels = []
        
        print(f"  📊 테스트 데이터셋 예측 중...")
        for batch in tqdm(test_loader, desc=f"  {model_name}"):
            measurements = batch['measurement'].numpy()  # (batch_size, timesteps, channels)
            labels = batch['label'].numpy()  # (batch_size,)
            
            # 배치별로 예측
            for i in range(len(measurements)):
                sample = measurements[i]  # (timesteps, channels)
                label = labels[i]
                
                # 예측 수행 (DynamicDataModule의 데이터는 이미 정규화되어 있으므로 normalize=False)
                result = engine.predict_single(sample, return_all_info=True, normalize=False)
                predicted_class_idx = result['predicted_class_idx']  # 인덱스 사용
                
                all_predictions.append(predicted_class_idx)
                all_labels.append(int(label))  # 명시적으로 int로 변환
        
        all_predictions = np.array(all_predictions, dtype=np.int32)
        all_labels = np.array(all_labels, dtype=np.int32)
        
        # 성능 지표 계산
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        print(f"  ✅ 완료!")
        print(f"     Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"     F1-Score: {f1:.4f}")
        print(f"     Precision: {precision:.4f}")
        print(f"     Recall: {recall:.4f}")
        
        # 혼동 행렬 생성
        cm = confusion_matrix(all_labels, all_predictions, labels=range(24))
        
        # 혼동 행렬 시각화
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # 정규화된 혼동 행렬 (퍼센트)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # NaN 처리
        
        # 히트맵 생성
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('True Label', fontsize=14)
        ax.set_title(
            f'Confusion Matrix - {model_name}\n'
            f'Accuracy: {accuracy*100:.2f}% | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}',
            fontsize=16,
            fontweight='bold'
        )
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 저장
        output_file = output_dir / f'confusion_matrix_{model_name.lower()}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 저장 완료: {output_file}")
        
        # 정규화되지 않은 혼동 행렬도 저장 (원본 숫자)
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('True Label', fontsize=14)
        ax.set_title(
            f'Confusion Matrix (Raw Counts) - {model_name}\n'
            f'Total Samples: {len(all_labels)}',
            fontsize=16,
            fontweight='bold'
        )
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        output_file_raw = output_dir / f'confusion_matrix_{model_name.lower()}_raw.png'
        plt.savefig(output_file_raw, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'model': model_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'total_samples': len(all_labels)
        }
        
    except Exception as e:
        print(f"  ❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_comparison_visualization(results, output_dir):
    """모든 모델의 혼동 행렬을 비교하는 시각화"""
    
    if not results or len([r for r in results if r is not None]) == 0:
        print("⚠️  비교 시각화를 생성할 결과가 없습니다.")
        return
    
    valid_results = [r for r in results if r is not None]
    
    # 1. 성능 지표 비교 (Bar Plot)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = [r['model'] for r in valid_results]
    accuracies = [r['accuracy'] * 100 for r in valid_results]
    f1_scores = [r['f1_score'] for r in valid_results]
    precisions = [r['precision'] for r in valid_results]
    recalls = [r['recall'] for r in valid_results]
    
    # Accuracy
    axes[0, 0].bar(models, accuracies, color='skyblue')
    axes[0, 0].set_title('Accuracy Comparison (%)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].set_ylim([min(accuracies) - 2, 100])
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # F1-Score
    axes[0, 1].bar(models, f1_scores, color='lightgreen')
    axes[0, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('F1-Score', fontsize=12)
    axes[0, 1].set_ylim([min(f1_scores) - 0.02, 1.0])
    for i, v in enumerate(f1_scores):
        axes[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Precision
    axes[1, 0].bar(models, precisions, color='lightcoral')
    axes[1, 0].set_title('Precision Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Precision', fontsize=12)
    axes[1, 0].set_ylim([min(precisions) - 0.02, 1.0])
    for i, v in enumerate(precisions):
        axes[1, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Recall
    axes[1, 1].bar(models, recalls, color='gold')
    axes[1, 1].set_title('Recall Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Recall', fontsize=12)
    axes[1, 1].set_ylim([min(recalls) - 0.02, 1.0])
    for i, v in enumerate(recalls):
        axes[1, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    comparison_file = output_dir / 'confusion_matrix_comparison.png'
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 비교 시각화 저장 완료: {comparison_file}")
    
    # 2. 모든 모델의 혼동 행렬을 한 번에 보기 (2x2 그리드)
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()
    
    for idx, result in enumerate(valid_results):
        if idx >= 4:
            break
        ax = axes[idx]
        
        cm_norm = result['confusion_matrix_normalized']
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            cbar_kws={'label': 'Normalized Count'}
        )
        ax.set_title(
            f"{result['model']}\nAcc: {result['accuracy']*100:.2f}% | F1: {result['f1_score']:.4f}",
            fontsize=12,
            fontweight='bold'
        )
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    grid_file = output_dir / 'confusion_matrix_grid_all_models.png'
    plt.savefig(grid_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 그리드 시각화 저장 완료: {grid_file}")

def main():
    """메인 함수"""
    setup_korean_font()
    plt.rcParams['axes.unicode_minus'] = False
    
    output_dir = Path('visualizations/confusion_matrices')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("📊 4개 모델 혼동 행렬 생성: GRU, StackedGRU, MS3DGRU, MS3DStackedGRU")
    print("=" * 80)
    print()
    
    # 테스트 데이터셋 로드
    print("📌 테스트 데이터셋 로드 중...")
    data_dir = "/home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified"
    datamodule = DynamicDataModule(
        data_dir=data_dir,
        time_steps=87,
        n_channels=8,
        batch_size=32,
        seed=42,
        use_test_split=True
    )
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()
    
    print(f"✅ 테스트 데이터셋 로드 완료")
    print(f"   테스트 샘플 수: {len(datamodule.test_dataset)}")
    print()
    
    # Scaler 파일 경로 (모든 모델에 공통)
    scaler_path = 'archive/checkpoints_backup/checkpoints_backup/scaler.pkl'
    
    # 모델 설정 (원본 체크포인트 직접 사용)
    models_config = {
        'GRU': {
            'path': 'checkpoints/best_model_epoch=epoch=92_val/loss=val/loss=0.04.ckpt',  # 원본 체크포인트 직접 사용
            'type': 'GRU',
            'hidden_size': 64,
            'layers': 1,  # 체크포인트는 layers=1로 학습됨
            'dropout': 0.2,
            'scaler_path': scaler_path
        },
        'StackedGRU': {
            'path': 'checkpoints/best_model_epoch=epoch=68_val/loss=val/loss=0.19.ckpt',  # 새로 재훈련된 StackedGRU 체크포인트 (최고 성능: 94.30%)
            'type': 'StackedGRU',
            'hidden_size': 64,
            'dropout': 0.2,
            'scaler_path': scaler_path
        },
        'MS3DGRU': {
            'path': 'best_model/ms3dgru_best.ckpt',
            'type': 'MS3DGRU',
            'cnn_filters': 32,
            'dropout': 0.1,
            'scaler_path': scaler_path
        },
        'MS3DStackedGRU': {
            'path': 'checkpoints/best_model_epoch=epoch=82_val/loss=val/loss=0.05.ckpt',  # 원본 체크포인트 직접 사용
            'type': 'MS3DStackedGRU',
            'cnn_filters': 32,
            'dropout': 0.05,
            'scaler_path': scaler_path
        }
    }
    
    # 각 모델에 대해 혼동 행렬 생성
    all_results = []
    for model_name, config in models_config.items():
        result = generate_confusion_matrix_for_model(model_name, config, test_loader, output_dir)
        all_results.append(result)
    
    # 비교 시각화 생성
    print("\n🎨 비교 시각화 생성 중...")
    create_comparison_visualization(all_results, output_dir)
    
    # 요약 정보 저장
    summary_path = output_dir / 'confusion_matrix_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("📊 혼동 행렬 결과 요약\n")
        f.write("=" * 80 + "\n\n")
        
        for result in all_results:
            if result is None:
                continue
            f.write(f"모델: {result['model']}\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)\n")
            f.write(f"  F1-Score: {result['f1_score']:.4f}\n")
            f.write(f"  Precision: {result['precision']:.4f}\n")
            f.write(f"  Recall: {result['recall']:.4f}\n")
            f.write(f"  Total Samples: {result['total_samples']}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("✅ 완료!\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ 요약 정보 저장 완료: {summary_path}")
    
    print("\n" + "=" * 80)
    print("✅ 모든 혼동 행렬 생성 완료!")
    print(f"📁 결과 저장 위치: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()

