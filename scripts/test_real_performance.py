"""
실제 테스트 데이터셋으로 모델 성능 평가
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

def test_model_performance(
    model_path: str,
    model_type: str = 'MS3DGRU',
    data_dir: str = '/home/billy/25-1kp/SignGlove_HW/datasets/unified',
    batch_size: int = 32,
    device: str = 'cpu'
):
    """
    실제 테스트 데이터셋으로 모델 성능 평가
    
    Args:
        model_path: 체크포인트 경로
        model_type: 모델 타입
        data_dir: 데이터셋 디렉토리
        batch_size: 배치 크기
        device: 디바이스
    """
    print('=' * 80)
    print(f'🧪 실전 모델 성능 테스트: {model_type}')
    print('=' * 80)
    print()
    
    # 1. 추론 엔진 초기화
    print('📌 단계 1: 추론 엔진 초기화')
    print('-' * 80)
    try:
        engine = SignGloveInference(
            model_path=model_path,
            model_type=model_type,
            device=device
        )
        print(f'✅ 모델 로드 성공: {model_type}')
        print(f'   파라미터 수: {engine.model.count_parameters():,}')
    except Exception as e:
        print(f'❌ 모델 로드 실패: {e}')
        return
    
    print()
    
    # 2. 테스트 데이터셋 로드
    print('📌 단계 2: 테스트 데이터셋 로드')
    print('-' * 80)
    try:
        datamodule = DynamicDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            test_size=0.2,
            val_size=0.2,
            seed=42
        )
        datamodule.setup('test')
        test_loader = datamodule.test_dataloader()
        
        print(f'✅ 데이터셋 로드 성공')
        print(f'   데이터 디렉토리: {data_dir}')
        print(f'   배치 크기: {batch_size}')
        print(f'   테스트 배치 수: {len(test_loader)}')
    except Exception as e:
        print(f'❌ 데이터셋 로드 실패: {e}')
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # 3. 예측 수행
    print('📌 단계 3: 예측 수행 중...')
    print('-' * 80)
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    engine.model.eval()
    engine.model.to(device)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='예측 진행')):
            x = batch['measurement']
            y = batch['label']
            x_padding = batch.get('measurement_padding', None)
            
            # 디바이스로 이동
            x = x.to(device)
            y = y.to(device)
            
            # 예측
            logits = engine.model.predict(x)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    print(f'✅ 예측 완료!')
    print(f'   총 샘플 수: {len(all_labels)}')
    print()
    
    # 4. 성능 지표 계산
    print('📌 단계 4: 성능 지표 계산')
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
    
    # 클래스별 정확도
    cm = confusion_matrix(all_labels, all_predictions, labels=range(24))
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    print('📊 클래스별 정확도 (상위 10개):')
    class_acc_dict = {CLASS_NAMES[i]: acc for i, acc in enumerate(class_accuracies)}
    sorted_classes = sorted(class_acc_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    for class_name, acc in sorted_classes:
        print(f'   {class_name}: {acc * 100:.2f}%')
    print()
    
    # 오분류 분석
    print('📊 오분류 분석:')
    wrong_predictions = all_predictions != all_labels
    num_wrong = wrong_predictions.sum()
    print(f'   잘못 예측: {num_wrong}개 ({num_wrong / len(all_labels) * 100:.2f}%)')
    print(f'   정확 예측: {len(all_labels) - num_wrong}개 ({(len(all_labels) - num_wrong) / len(all_labels) * 100:.2f}%)')
    print()
    
    # 5. Confusion Matrix 생성
    print('📌 단계 5: Confusion Matrix 생성')
    print('-' * 80)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 정규화된 Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Count)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], cbar_kws={'label': 'Normalized'})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)
    
    plt.tight_layout()
    
    output_dir = Path('inference/performance_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'real_test_confusion_matrix_{model_type.lower()}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✅ 저장: {output_file}')
    print()
    
    # 6. 오분류 샘플 분석
    print('📌 단계 6: 오분류 샘플 상세 분석')
    print('-' * 80)
    
    wrong_indices = np.where(wrong_predictions)[0]
    
    if len(wrong_indices) > 0:
        print(f'📋 오분류 샘플 상세 (최대 10개):')
        for i, idx in enumerate(wrong_indices[:10]):
            true_label = all_labels[idx]
            pred_label = all_predictions[idx]
            true_prob = all_probs[idx][true_label]
            pred_prob = all_probs[idx][pred_label]
            
            print(f'   샘플 {i+1}:')
            print(f'     정답: {CLASS_NAMES[true_label]} (확률: {true_prob:.4f})')
            print(f'     예측: {CLASS_NAMES[pred_label]} (확률: {pred_prob:.4f})')
            print(f'     차이: {pred_prob - true_prob:.4f}')
    else:
        print('   ✅ 모든 샘플 정확히 예측!')
    
    print()
    
    # 7. 클래스별 상세 리포트
    print('📌 단계 7: 클래스별 상세 리포트 생성')
    print('-' * 80)
    
    report = classification_report(
        all_labels, all_predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    
    # 리포트를 파일로 저장
    report_file = output_dir / f'real_test_report_{model_type.lower()}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f'실제 테스트 데이터셋 성능 리포트\n')
        f.write(f'모델: {model_type}\n')
        f.write(f'총 샘플 수: {len(all_labels)}\n')
        f.write(f'정확도: {accuracy * 100:.2f}%\n')
        f.write(f'F1-Score (Macro): {f1_macro:.4f}\n')
        f.write(f'F1-Score (Weighted): {f1_weighted:.4f}\n')
        f.write('\n')
        f.write(classification_report(all_labels, all_predictions, target_names=CLASS_NAMES))
    
    print(f'✅ 리포트 저장: {report_file}')
    print()
    
    # 8. 클래스별 정확도 시각화
    print('📌 단계 8: 클래스별 정확도 시각화')
    print('-' * 80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # 클래스별 정확도
    sorted_indices = np.argsort(class_accuracies)
    colors = ['red' if acc < 0.95 else 'green' if acc >= 0.98 else 'orange' 
              for acc in class_accuracies[sorted_indices]]
    
    bars = ax1.barh(range(len(CLASS_NAMES)), class_accuracies[sorted_indices], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(CLASS_NAMES)))
    ax1.set_yticklabels([CLASS_NAMES[i] for i in sorted_indices], fontsize=10)
    ax1.set_xlabel('Accuracy', fontsize=12)
    ax1.set_title('Class-wise Accuracy', fontsize=14, fontweight='bold')
    ax1.axvline(x=0.95, color='orange', linestyle='--', alpha=0.5, label='95% threshold')
    ax1.axvline(x=0.98, color='green', linestyle='--', alpha=0.5, label='98% threshold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # 값 표시
    for i, (idx, acc) in enumerate(zip(sorted_indices, class_accuracies[sorted_indices])):
        ax1.text(acc + 0.01, i, f'{acc*100:.1f}%', va='center', fontsize=8)
    
    # F1-Score per class
    ax2.barh(range(len(CLASS_NAMES)), f1_per_class[sorted_indices], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(CLASS_NAMES)))
    ax2.set_yticklabels([CLASS_NAMES[i] for i in sorted_indices], fontsize=10)
    ax2.set_xlabel('F1-Score', fontsize=12)
    ax2.set_title('Class-wise F1-Score', fontsize=14, fontweight='bold')
    ax2.axvline(x=0.95, color='orange', linestyle='--', alpha=0.5, label='95% threshold')
    ax2.axvline(x=0.98, color='green', linestyle='--', alpha=0.5, label='98% threshold')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    # 값 표시
    for i, (idx, f1) in enumerate(zip(sorted_indices, f1_per_class[sorted_indices])):
        ax2.text(f1 + 0.01, i, f'{f1:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    
    accuracy_file = output_dir / f'real_test_class_accuracy_{model_type.lower()}.png'
    plt.savefig(accuracy_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✅ 저장: {accuracy_file}')
    print()
    
    # 9. 최종 요약
    print('=' * 80)
    print('📊 최종 성능 요약')
    print('=' * 80)
    print(f'모델: {model_type}')
    print(f'테스트 샘플 수: {len(all_labels):,}')
    print(f'정확도: {accuracy * 100:.2f}%')
    print(f'F1-Score (Macro): {f1_macro:.4f}')
    print(f'F1-Score (Weighted): {f1_weighted:.4f}')
    print(f'오분류: {num_wrong}개 ({num_wrong / len(all_labels) * 100:.2f}%)')
    print()
    print(f'✅ 95% 이상 정확도 클래스: {(class_accuracies >= 0.95).sum()}/24')
    print(f'✅ 98% 이상 정확도 클래스: {(class_accuracies >= 0.98).sum()}/24')
    print()
    print('생성된 파일:')
    print(f'  • {output_file}')
    print(f'  • {accuracy_file}')
    print(f'  • {report_file}')
    print()
    print('=' * 80)


if __name__ == '__main__':
    # MS3DGRU 모델 테스트
    test_model_performance(
        model_path='best_model/ms3dgru_best.ckpt',
        model_type='MS3DGRU',
        data_dir='/home/billy/25-1kp/SignGlove_HW/datasets/unified',
        batch_size=32,
        device='cpu'
    )

