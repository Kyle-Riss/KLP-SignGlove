"""
올바른 MS3DGRU 체크포인트로 Confusion Matrix 생성
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
import matplotlib.font_manager as fm
import seaborn as sns
from tqdm import tqdm

from src.misc.DynamicDataModule import DynamicDataModule
from src.models.MultiScale3DGRUModels import MS3DGRU

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정"""
    # 사용 가능한 한글 폰트 찾기
    korean_fonts = [
        'NanumGothic',
        'NanumBarunGothic',
        'Malgun Gothic',
        'AppleGothic',
        'Noto Sans CJK KR',
        'DejaVu Sans'  # fallback
    ]
    
    # 폰트 경로 직접 지정 (Linux)
    font_paths = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    
    # 폰트 찾기
    font_found = False
    for font_name in korean_fonts:
        try:
            plt.rcParams['font.family'] = font_name
            # 테스트
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '한글', fontsize=12)
            plt.close(fig)
            font_found = True
            print(f'✅ 한글 폰트 설정: {font_name}')
            break
        except:
            continue
    
    # 폰트 경로로 직접 설정
    if not font_found:
        for font_path in font_paths:
            if Path(font_path).exists():
                try:
                    font_prop = fm.FontProperties(fname=font_path)
                    plt.rcParams['font.family'] = font_prop.get_name()
                    font_found = True
                    print(f'✅ 한글 폰트 설정: {font_path}')
                    break
                except:
                    continue
    
    if not font_found:
        # 마지막 fallback: 폰트 없이도 작동하도록
        plt.rcParams['font.family'] = 'DejaVu Sans'
        print('⚠️  한글 폰트를 찾을 수 없습니다. 기본 폰트 사용 (한글이 깨질 수 있음)')
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

# 폰트 설정 실행
setup_korean_font()

# 한글 클래스명
CLASS_NAMES = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ'
]

def generate_confusion_matrix():
    """Confusion Matrix 생성"""
    
    print('=' * 80)
    print('📊 MS3DGRU Confusion Matrix 생성')
    print('=' * 80)
    print()
    
    # 1. 데이터 모듈 설정
    print('📌 1단계: 테스트 데이터셋 로드')
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
    
    # 2. 모델 로드
    print('📌 2단계: 모델 로드')
    print('-' * 80)
    checkpoint_path = 'inference/best_models/ms3dgru_best.ckpt'
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 하이퍼파라미터 추출 (model_config 또는 hyper_parameters 사용)
    if 'model_config' in checkpoint:
        hyper_params = checkpoint['model_config']
    elif 'hyper_parameters' in checkpoint:
        hyper_params = checkpoint['hyper_parameters']
    else:
        hyper_params = {}
    
    model = MS3DGRU(
        learning_rate=hyper_params.get('learning_rate', 0.001),
        input_size=hyper_params.get('input_size', 8),
        hidden_size=hyper_params.get('hidden_size', 64),
        classes=hyper_params.get('classes', 24),
        cnn_filters=hyper_params.get('cnn_filters', 32),
        dropout=hyper_params.get('dropout', 0.1)
    )
    
    # State dict 로드 (model_config 형태는 직접 로드 가능)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # model_config 형태는 이미 clean한 상태
        # model. 접두사가 있을 수도 있으니 확인
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                cleaned_state_dict[key[6:]] = value
            else:
                cleaned_state_dict[key] = value
        
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        if missing_keys:
            print(f'⚠️  누락된 키: {len(missing_keys)}개')
            if missing_keys:
                print(f'   예시: {list(missing_keys)[:3]}')
        if unexpected_keys:
            print(f'⚠️  예상치 못한 키: {len(unexpected_keys)}개')
    
    model.eval()
    print(f'✅ 모델 로드 완료')
    print(f'   모델 타입: {checkpoint.get("model_type", "N/A")}')
    if 'model_info' in checkpoint:
        mi = checkpoint['model_info']
        if 'performance' in mi:
            perf = mi['performance']
            print(f'   예상 성능: {perf.get("test_accuracy", 0)*100:.2f}%')
    print()
    
    # 3. 예측 수행
    print('📌 3단계: 예측 수행 중...')
    print('-' * 80)
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='예측 진행'):
            x = batch['measurement']
            y = batch['label']
            x_padding = batch.get('measurement_padding', None)
            
            # Forward pass
            logits, loss = model(x, x_padding, y)
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
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
    
    # 5. Confusion Matrix 생성
    print('📌 5단계: Confusion Matrix 생성')
    print('-' * 80)
    
    cm = confusion_matrix(all_labels, all_predictions, labels=range(24))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Count Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], cbar_kws={'label': 'Count'}, annot_kws={'size': 8})
    axes[0].set_title('Confusion Matrix (Count)', fontsize=16, fontweight='bold', pad=20)
    axes[0].set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Class', fontsize=12, fontweight='bold')
    
    # Normalized Confusion Matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], cbar_kws={'label': 'Normalized'}, annot_kws={'size': 8})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    axes[1].set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Class', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_dir = Path('inference/performance_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'real_test_confusion_matrix_ms3dgru_final.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✅ 저장: {output_file}')
    print()
    
    # 6. 클래스별 성능 시각화
    print('📌 6단계: 클래스별 성능 시각화')
    print('-' * 80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 클래스별 정확도
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
    
    # 값 표시
    for i, (idx, acc) in enumerate(zip(sorted_indices, class_accuracies[sorted_indices])):
        ax1.text(acc * 100 + 0.3, i, f'{acc*100:.1f}%', va='center', fontsize=9)
    
    # F1-Score per class
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
    
    # 값 표시
    for i, (idx, f1) in enumerate(zip(sorted_indices, f1_per_class[sorted_indices])):
        ax2.text(f1 * 100 + 0.3, i, f'{f1*100:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    accuracy_file = output_dir / 'real_test_class_accuracy_ms3dgru_final.png'
    plt.savefig(accuracy_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✅ 저장: {accuracy_file}')
    print()
    
    # 7. 상세 리포트 생성
    print('📌 7단계: 상세 리포트 생성')
    print('-' * 80)
    
    report = classification_report(
        all_labels, all_predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    
    report_file = output_dir / 'real_test_report_ms3dgru_final.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f'MS3DGRU 실전 테스트 성능 리포트\n')
        f.write(f'=' * 60 + '\n\n')
        f.write(f'체크포인트: {checkpoint_path}\n')
        f.write(f'테스트 샘플 수: {len(all_labels):,}\n')
        f.write(f'정확도: {accuracy * 100:.2f}%\n')
        f.write(f'F1-Score (Macro): {f1_macro:.4f}\n')
        f.write(f'F1-Score (Weighted): {f1_weighted:.4f}\n')
        f.write(f'\n클래스별 상세 성능:\n')
        f.write('-' * 60 + '\n')
        
        for i, class_name in enumerate(CLASS_NAMES):
            class_report = report.get(class_name, {})
            precision = class_report.get('precision', 0)
            recall = class_report.get('recall', 0)
            f1 = class_report.get('f1-score', 0)
            support = class_report.get('support', 0)
            
            correct = cm[i, i]
            total = cm.sum(axis=1)[i]
            acc = class_accuracies[i]
            
            f.write(f'{class_name:3s}: Acc={acc*100:6.2f}%, '
                   f'Prec={precision*100:6.2f}%, Rec={recall*100:6.2f}%, '
                   f'F1={f1*100:6.2f}%, Correct={correct:2d}/{total:2d}\n')
        
        f.write('\n' + '=' * 60 + '\n')
        f.write(classification_report(all_labels, all_predictions, target_names=CLASS_NAMES))
    
    print(f'✅ 저장: {report_file}')
    print()
    
    # 8. 최종 요약
    print('=' * 80)
    print('📊 최종 성능 요약')
    print('=' * 80)
    print(f'모델: MS3DGRU')
    print(f'체크포인트: {checkpoint_path}')
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
    print(f'  • {report_file}')
    print()
    print('=' * 80)


if __name__ == '__main__':
    try:
        generate_confusion_matrix()
    except Exception as e:
        print(f'❌ 오류 발생: {e}')
        import traceback
        traceback.print_exc()

