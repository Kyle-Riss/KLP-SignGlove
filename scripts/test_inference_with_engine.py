"""
추론 엔진을 사용한 실제 테스트
"""

import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.misc.DynamicDataModule import DynamicDataModule
from inference import SignGloveInference

# 한글 클래스명
CLASS_NAMES = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ'
]

def test_with_inference_engine():
    """추론 엔진을 사용한 실제 테스트"""
    
    print('=' * 80)
    print('🧪 추론 엔진을 사용한 MS3DGRU 성능 테스트')
    print('=' * 80)
    print()
    
    # 1. 추론 엔진 초기화
    print('📌 1단계: 추론 엔진 초기화')
    print('-' * 80)
    try:
        engine = SignGloveInference(
            model_path='inference/best_models/ms3dgru_best.ckpt',
            model_type='MS3DGRU',
            device='cpu'
        )
        print(f'✅ 추론 엔진 초기화 성공')
        print(f'   모델 파라미터: {engine.model.count_parameters():,}')
    except Exception as e:
        print(f'❌ 초기화 실패: {e}')
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # 2. 테스트 데이터셋 로드
    print('📌 2단계: 테스트 데이터셋 로드')
    print('-' * 80)
    datamodule = DynamicDataModule(
        data_dir='/home/billy/25-1kp/SignGlove_HW/datasets/unified',
        batch_size=32,
        test_size=0.2,
        val_size=0.2,
        seed=42
    )
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()
    print(f'✅ 테스트 데이터셋 준비 완료')
    print(f'   테스트 샘플 수: {len(datamodule.test_dataset):,}')
    print()
    
    # 3. 예측 수행
    print('📌 3단계: 예측 수행 중...')
    print('-' * 80)
    
    all_predictions = []
    all_labels = []
    
    engine.model.eval()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='예측 진행'):
            x = batch['measurement']
            y = batch['label']
            x_padding = batch.get('measurement_padding', None)
            
            # 데이터를 numpy로 변환 (추론 엔진 입력 형식)
            batch_size = x.shape[0]
            for i in range(batch_size):
                sample = x[i].cpu().numpy()  # (time, channels)
                label = y[i].item()
                
                # 추론 엔진으로 예측
                result = engine.predict_single(sample, top_k=1, return_all_info=False)
                pred_class = result['predicted_class']
                pred_idx = CLASS_NAMES.index(pred_class)
                
                all_predictions.append(pred_idx)
                all_labels.append(label)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    print(f'✅ 예측 완료! 총 {len(all_labels):,}개 샘플')
    print()
    
    # 4. 성능 계산
    print('📌 4단계: 성능 지표 계산')
    print('-' * 80)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    f1_per_class = f1_score(all_labels, all_predictions, average=None)
    
    print(f'✅ 성능 지표:')
    print(f'   정확도 (Accuracy): {accuracy * 100:.2f}%')
    print(f'   F1-Score (Macro): {f1_macro:.4f}')
    print(f'   F1-Score (Weighted): {f1_weighted:.4f}')
    print()
    
    # 5. Confusion Matrix
    print('📌 5단계: Confusion Matrix 생성')
    print('-' * 80)
    
    cm = confusion_matrix(all_labels, all_predictions, labels=range(24))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], cbar_kws={'label': 'Count'}, annot_kws={'size': 8})
    axes[0].set_title('Confusion Matrix (Count)', fontsize=16, fontweight='bold', pad=20)
    axes[0].set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Class', fontsize=12, fontweight='bold')
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], cbar_kws={'label': 'Normalized'}, annot_kws={'size': 8})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    axes[1].set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Class', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_dir = Path('inference/performance_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'inference_engine_confusion_matrix.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✅ 저장: {output_file}')
    print()
    
    # 6. 클래스별 성능
    print('📌 6단계: 클래스별 성능')
    print('-' * 80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    sorted_indices = np.argsort(class_accuracies)
    colors = ['#e74c3c' if acc < 0.95 else '#f39c12' if acc < 0.98 else '#2ecc71' 
              for acc in class_accuracies[sorted_indices]]
    
    bars = ax1.barh(range(len(CLASS_NAMES)), class_accuracies[sorted_indices] * 100, 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_yticks(range(len(CLASS_NAMES)))
    ax1.set_yticklabels([CLASS_NAMES[i] for i in sorted_indices], fontsize=11)
    ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Class-wise Accuracy', fontsize=14, fontweight='bold', pad=20)
    ax1.axvline(x=98, color='green', linestyle='--', alpha=0.5, label='98% threshold')
    ax1.axvline(x=95, color='orange', linestyle='--', alpha=0.5, label='95% threshold')
    ax1.set_xlim([90, 100])
    ax1.legend(fontsize=10)
    ax1.grid(axis='x', alpha=0.3)
    
    for i, (idx, acc) in enumerate(zip(sorted_indices, class_accuracies[sorted_indices])):
        ax1.text(acc * 100 + 0.3, i, f'{acc*100:.1f}%', va='center', fontsize=9)
    
    bars2 = ax2.barh(range(len(CLASS_NAMES)), f1_per_class[sorted_indices] * 100, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_yticks(range(len(CLASS_NAMES)))
    ax2.set_yticklabels([CLASS_NAMES[i] for i in sorted_indices], fontsize=11)
    ax2.set_xlabel('F1-Score (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Class-wise F1-Score', fontsize=14, fontweight='bold', pad=20)
    ax2.axvline(x=98, color='green', linestyle='--', alpha=0.5, label='98% threshold')
    ax2.axvline(x=95, color='orange', linestyle='--', alpha=0.5, label='95% threshold')
    ax2.set_xlim([90, 100])
    ax2.legend(fontsize=10)
    ax2.grid(axis='x', alpha=0.3)
    
    for i, (idx, f1) in enumerate(zip(sorted_indices, f1_per_class[sorted_indices])):
        ax2.text(f1 * 100 + 0.3, i, f'{f1*100:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    accuracy_file = output_dir / 'inference_engine_class_accuracy.png'
    plt.savefig(accuracy_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✅ 저장: {accuracy_file}')
    print()
    
    # 7. 최종 요약
    print('=' * 80)
    print('📊 최종 성능 요약')
    print('=' * 80)
    print(f'모델: MS3DGRU (추론 엔진 사용)')
    print(f'체크포인트: inference/best_models/ms3dgru_best.ckpt')
    print(f'테스트 샘플 수: {len(all_labels):,}')
    print(f'정확도: {accuracy * 100:.2f}%')
    print(f'F1-Score (Macro): {f1_macro:.4f}')
    print(f'F1-Score (Weighted): {f1_weighted:.4f}')
    print()
    
    wrong_count = (all_predictions != all_labels).sum()
    print(f'✅ 정확 예측: {len(all_labels) - wrong_count:,}개 ({(len(all_labels) - wrong_count) / len(all_labels) * 100:.2f}%)')
    print(f'❌ 오분류: {wrong_count:,}개 ({wrong_count / len(all_labels) * 100:.2f}%)')
    print()
    
    print('📊 클래스별 성능:')
    print('-' * 80)
    for i, (class_name, acc) in enumerate(zip(CLASS_NAMES, class_accuracies)):
        status = '✅' if acc >= 0.98 else '⚠️' if acc >= 0.95 else '❌'
        correct = cm[i, i]
        total = cm.sum(axis=1)[i]
        print(f'{status} {class_name}: {acc * 100:.2f}% ({correct}/{total})')
    
    print()
    print('생성된 파일:')
    print(f'  • {output_file}')
    print(f'  • {accuracy_file}')
    print()
    print('=' * 80)


if __name__ == '__main__':
    try:
        test_with_inference_engine()
    except Exception as e:
        print(f'❌ 오류 발생: {e}')
        import traceback
        traceback.print_exc()

