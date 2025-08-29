import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# MLP 모델 정의 (더 간단한 버전)
class SimpleMLPModel(nn.Module):
    def __init__(self, input_size=160, hidden_size=64, num_classes=24, dropout=0.6):
        super(SimpleMLPModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.network(x)

# 기존 데이터셋 클래스 재사용
from vowel_improvement import VowelImprovedDataset

print('🔧 교차 검증으로 과적합 확인 시작')

# 데이터셋 로드
dataset = VowelImprovedDataset(
    data_path='../SignGlove_HW/datasets/unified',
    sequence_length=20,
    apply_vowel_enhancement=False
)

print(f'📊 전체 데이터셋 크기: {len(dataset)}')
print(f'📊 클래스당 평균 샘플 수: {len(dataset) // len(dataset.all_classes)}')

# 5-fold 교차 검증
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

fold_scores = []
fold_reports = []

for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset)), dataset.labels)):
    print(f'\n🔄 Fold {fold + 1}/{n_folds} 시작')
    
    # 데이터 분할
    train_indices, val_indices = train_test_split(
        train_idx, test_size=0.2, random_state=42, 
        stratify=[dataset.labels[i] for i in train_idx]
    )
    
    # 데이터로더
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_indices), 
                            batch_size=16, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_indices), 
                          batch_size=16, shuffle=False)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), 
                           batch_size=16, shuffle=False)
    
    # 모델 생성 (더 간단한 모델)
    model = SimpleMLPModel(input_size=160, hidden_size=64, num_classes=24, dropout=0.6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 훈련 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    # 훈련 (더 짧게)
    best_val_loss = float('inf')
    patience = 0
    max_patience = 5
    
    for epoch in range(15):  # 더 짧은 훈련
        # 훈련
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets.squeeze()).sum().item()
        
        # 검증
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets.squeeze())
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets.squeeze()).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
        
        if patience >= max_patience:
            break
    
    # 테스트 성능 평가
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.squeeze().cpu().numpy())
    
    # 성능 계산
    test_accuracy = 100 * sum(1 for p, t in zip(all_predictions, all_targets) if p == t) / len(all_targets)
    fold_scores.append(test_accuracy)
    
    # 리포트 생성
    report = classification_report(all_targets, all_predictions, 
                                 target_names=dataset.all_classes, output_dict=True, zero_division=0)
    fold_reports.append(report)
    
    print(f'  Fold {fold + 1} 테스트 정확도: {test_accuracy:.2f}%')

# 결과 분석
print(f'\n📊 교차 검증 결과:')
print('=' * 40)
print(f'평균 정확도: {np.mean(fold_scores):.2f}% ± {np.std(fold_scores):.2f}%')
print(f'최고 정확도: {np.max(fold_scores):.2f}%')
print(f'최저 정확도: {np.min(fold_scores):.2f}%')

# 각 fold별 성능
print(f'\n📈 Fold별 성능:')
for i, score in enumerate(fold_scores):
    print(f'  Fold {i+1}: {score:.2f}%')

# 과적합 분석
print(f'\n🔍 과적합 분석:')
if np.std(fold_scores) > 5.0:
    print(f'  ⚠️  높은 표준편차 ({np.std(fold_scores):.2f}%) - 과적합 가능성 높음')
else:
    print(f'  ✅ 안정적인 성능 - 과적합 가능성 낮음')

if np.max(fold_scores) - np.min(fold_scores) > 10.0:
    print(f'  ⚠️  큰 성능 차이 ({np.max(fold_scores) - np.min(fold_scores):.2f}%) - 데이터 불균형 가능성')
else:
    print(f'  ✅ 일관된 성능 - 데이터 균형 양호')

# 전체 평균 리포트
print(f'\n📊 전체 평균 성능 (모든 fold):')
if 'weighted avg' in fold_reports[0]:
    avg_precision = np.mean([r['weighted avg']['precision'] for r in fold_reports]) * 100
    avg_recall = np.mean([r['weighted avg']['recall'] for r in fold_reports]) * 100
    avg_f1 = np.mean([r['weighted avg']['f1-score'] for r in fold_reports]) * 100
    
    print(f'  평균 Precision: {avg_precision:.1f}%')
    print(f'  평균 Recall: {avg_recall:.1f}%')
    print(f'  평균 F1-Score: {avg_f1:.1f}%')

print('\n✅ 교차 검증 완료!')
