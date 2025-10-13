"""
추론 기반 시각화 스크립트
학습된 모델의 추론 결과를 다양한 방법으로 시각화
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

from src.models.GRUModels import GRU
from src.misc.DynamicDataModule import DynamicDataModule

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("📊 추론 기반 시각화 생성")
print("=" * 80)

# 1. 모델 로딩
print("\n1️⃣ 모델 로딩...")
checkpoint_path = "src/experiments/checkpoints/best_model_epoch=57_val/loss=0.03-v2.ckpt"
device = torch.device('cpu')

model = GRU.load_from_checkpoint(
    checkpoint_path,
    input_size=8,
    hidden_size=64,
    classes=24,
    learning_rate=1e-3,
    map_location=device
)
model.to(device)
model.eval()
print(f"✅ 모델 로드 완료")

# 2. 데이터 로딩
print("\n2️⃣ 테스트 데이터 로딩...")
data_module = DynamicDataModule(
    data_dir="/home/billy/25-1kp/SignGlove_HW/datasets/unified",
    time_steps=87,
    batch_size=1
)
data_module.setup(stage='test')
test_loader = data_module.test_dataloader()

# 클래스 이름
class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 
               'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 
               'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']

# 3. 추론 수행
print("\n3️⃣ 전체 테스트 세트 추론 중...")
all_true_labels = []
all_predictions = []
all_confidences = []
all_logits = []

for batch in test_loader:
    x = batch['measurement'].to(device)
    x_padding = batch['measurement_padding'].to(device)
    true_labels = batch['label'].to(device)
    
    with torch.no_grad():
        logits, _ = model(x, x_padding, true_labels)
        predictions = torch.argmax(logits, dim=1)
        confidences = torch.softmax(logits, dim=1)
        max_confidences = confidences.max(dim=1)[0]
    
    all_true_labels.extend(true_labels.cpu().numpy())
    all_predictions.extend(predictions.cpu().numpy())
    all_confidences.extend(max_confidences.cpu().numpy())
    all_logits.append(confidences.cpu().numpy())

all_true_labels = np.array(all_true_labels)
all_predictions = np.array(all_predictions)
all_confidences = np.array(all_confidences)
all_logits = np.vstack(all_logits)

print(f"✅ 추론 완료: {len(all_true_labels)}개 샘플")

# 시각화 저장 폴더 생성
save_dir = Path("inference/inference_visualizations")
save_dir.mkdir(exist_ok=True)
print(f"\n💾 시각화 저장 폴더: {save_dir}")

# ============================================================================
# 4. 혼동 행렬 (Confusion Matrix)
# ============================================================================
print("\n4️⃣ 혼동 행렬 생성 중...")
cm = confusion_matrix(all_true_labels, all_predictions)

plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - GRU Model', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.xticks(rotation=0, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 저장: confusion_matrix.png")

# ============================================================================
# 5. 정규화된 혼동 행렬 (Normalized Confusion Matrix)
# ============================================================================
print("\n5️⃣ 정규화된 혼동 행렬 생성 중...")
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(16, 14))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn', 
            xticklabels=class_names, yticklabels=class_names,
            vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})
plt.title('Normalized Confusion Matrix - GRU Model', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.xticks(rotation=0, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig(save_dir / 'confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 저장: confusion_matrix_normalized.png")

# ============================================================================
# 6. 클래스별 정확도 (Per-Class Accuracy)
# ============================================================================
print("\n6️⃣ 클래스별 정확도 생성 중...")
class_accuracy = []
for i in range(len(class_names)):
    mask = all_true_labels == i
    if mask.sum() > 0:
        acc = (all_predictions[mask] == i).sum() / mask.sum()
        class_accuracy.append(acc)
    else:
        class_accuracy.append(0)

class_accuracy = np.array(class_accuracy)

plt.figure(figsize=(14, 8))
colors = ['#2ecc71' if acc >= 0.95 else '#f39c12' if acc >= 0.90 else '#e74c3c' 
          for acc in class_accuracy]
bars = plt.bar(range(len(class_names)), class_accuracy, color=colors, alpha=0.8, edgecolor='black')
plt.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% Threshold', alpha=0.7)
plt.axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='90% Threshold', alpha=0.7)
plt.xticks(range(len(class_names)), class_names, fontsize=14, fontweight='bold')
plt.yticks(fontsize=12)
plt.ylim(0, 1.05)
plt.xlabel('Class', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
plt.title('Per-Class Accuracy - GRU Model', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)

# 정확도 값 표시
for i, (bar, acc) in enumerate(zip(bars, class_accuracy)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(save_dir / 'per_class_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 저장: per_class_accuracy.png")

# ============================================================================
# 7. 신뢰도 분포 (Confidence Distribution)
# ============================================================================
print("\n7️⃣ 신뢰도 분포 생성 중...")
correct_mask = all_predictions == all_true_labels
correct_confidences = all_confidences[correct_mask]
incorrect_confidences = all_confidences[~correct_mask]

plt.figure(figsize=(12, 7))
plt.hist(correct_confidences, bins=50, alpha=0.7, color='green', 
         label=f'Correct ({len(correct_confidences)} samples)', edgecolor='black')
plt.hist(incorrect_confidences, bins=50, alpha=0.7, color='red', 
         label=f'Incorrect ({len(incorrect_confidences)} samples)', edgecolor='black')
plt.axvline(x=0.8, color='orange', linestyle='--', linewidth=2, 
            label='80% Threshold', alpha=0.7)
plt.xlabel('Confidence', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=14, fontweight='bold')
plt.title('Confidence Distribution - GRU Model', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(save_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 저장: confidence_distribution.png")

# ============================================================================
# 8. 클래스별 신뢰도 박스플롯 (Per-Class Confidence)
# ============================================================================
print("\n8️⃣ 클래스별 신뢰도 박스플롯 생성 중...")
class_confidences = [all_confidences[all_true_labels == i] for i in range(len(class_names))]

plt.figure(figsize=(16, 8))
bp = plt.boxplot(class_confidences, labels=class_names, patch_artist=True,
                 showmeans=True, meanline=True)

# 박스 색상 설정
for patch in bp['boxes']:
    patch.set_facecolor('#3498db')
    patch.set_alpha(0.7)

plt.axhline(y=0.95, color='green', linestyle='--', linewidth=2, 
            label='95% Threshold', alpha=0.7)
plt.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, 
            label='80% Threshold', alpha=0.7)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=12)
plt.ylim(0, 1.05)
plt.xlabel('Class', fontsize=14, fontweight='bold')
plt.ylabel('Confidence', fontsize=14, fontweight='bold')
plt.title('Per-Class Confidence Distribution - GRU Model', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(save_dir / 'per_class_confidence.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 저장: per_class_confidence.png")

# ============================================================================
# 9. 정확도 vs 신뢰도 산점도 (Accuracy vs Confidence)
# ============================================================================
print("\n9️⃣ 정확도 vs 신뢰도 산점도 생성 중...")
correct_colors = ['green' if c else 'red' for c in correct_mask]

plt.figure(figsize=(12, 8))
plt.scatter(all_confidences, range(len(all_confidences)), 
           c=correct_colors, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
plt.axvline(x=0.8, color='orange', linestyle='--', linewidth=2, 
            label='80% Threshold', alpha=0.7)
plt.xlabel('Confidence', fontsize=14, fontweight='bold')
plt.ylabel('Sample Index', fontsize=14, fontweight='bold')
plt.title('Prediction Confidence per Sample - GRU Model', fontsize=16, fontweight='bold', pad=20)

# 범례
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', alpha=0.6, label='Correct'),
                   Patch(facecolor='red', alpha=0.6, label='Incorrect')]
plt.legend(handles=legend_elements, fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(save_dir / 'confidence_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 저장: confidence_scatter.png")

# ============================================================================
# 10. Top-5 정확도 (Top-5 Accuracy)
# ============================================================================
print("\n🔟 Top-5 정확도 분석 중...")
top5_correct = 0
for i in range(len(all_true_labels)):
    top5_indices = np.argsort(all_logits[i])[-5:][::-1]
    if all_true_labels[i] in top5_indices:
        top5_correct += 1

top5_accuracy = top5_correct / len(all_true_labels)

# ============================================================================
# 11. 오분류 분석 (Misclassification Analysis)
# ============================================================================
print("\n1️⃣1️⃣ 오분류 분석 중...")
misclassified_indices = np.where(~correct_mask)[0]
misclassified_data = []

for idx in misclassified_indices:
    misclassified_data.append({
        'sample_idx': idx,
        'true_label': class_names[all_true_labels[idx]],
        'predicted_label': class_names[all_predictions[idx]],
        'confidence': all_confidences[idx]
    })

# ============================================================================
# 12. 종합 결과 요약
# ============================================================================
print("\n" + "=" * 80)
print("📊 추론 결과 요약")
print("=" * 80)

overall_accuracy = (all_predictions == all_true_labels).sum() / len(all_true_labels)
print(f"\n🎯 Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print(f"🎯 Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")
print(f"\n📊 총 샘플 수: {len(all_true_labels)}")
print(f"✅ 정답: {correct_mask.sum()}")
print(f"❌ 오답: {(~correct_mask).sum()}")
print(f"\n💯 평균 신뢰도: {all_confidences.mean():.4f} ({all_confidences.mean()*100:.2f}%)")
print(f"💯 정답 샘플 평균 신뢰도: {correct_confidences.mean():.4f} ({correct_confidences.mean()*100:.2f}%)")
if len(incorrect_confidences) > 0:
    print(f"💯 오답 샘플 평균 신뢰도: {incorrect_confidences.mean():.4f} ({incorrect_confidences.mean()*100:.2f}%)")

print(f"\n📉 저신뢰도 샘플 (< 80%): {(all_confidences < 0.8).sum()}")
print(f"📉 고신뢰도 샘플 (>= 95%): {(all_confidences >= 0.95).sum()}")

# 최악의 클래스 출력
worst_classes = np.argsort(class_accuracy)[:5]
print(f"\n⚠️  정확도가 낮은 클래스 Top 5:")
for i, idx in enumerate(worst_classes, 1):
    print(f"  {i}. {class_names[idx]}: {class_accuracy[idx]:.2%}")

# 가장 많이 혼동되는 쌍
print(f"\n🔄 가장 많이 혼동되는 클래스 쌍:")
confusion_pairs = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append((class_names[i], class_names[j], cm[i, j]))
confusion_pairs.sort(key=lambda x: x[2], reverse=True)

for i, (true_label, pred_label, count) in enumerate(confusion_pairs[:5], 1):
    print(f"  {i}. {true_label} → {pred_label}: {count}회")

print("\n" + "=" * 80)
print(f"✅ 시각화 생성 완료!")
print(f"📁 저장 위치: {save_dir.absolute()}")
print("=" * 80)

