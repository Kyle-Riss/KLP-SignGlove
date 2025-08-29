import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# MLP 모델 정의
class MLPModel(nn.Module):
    def __init__(self, input_size=160, hidden_sizes=[256, 128, 64], num_classes=24, dropout=0.5):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.network(x)

from vowel_improvement import VowelImprovedDataset

print('🔍 개별 클래스 100% 성능 상세 분석')

# 데이터셋 로드
dataset = VowelImprovedDataset(
    data_path='../SignGlove_HW/datasets/unified',
    sequence_length=20,
    apply_vowel_enhancement=False
)

# 데이터 분할
train_indices, test_indices = train_test_split(
    range(len(dataset)), test_size=0.3, random_state=42, stratify=dataset.labels
)
train_indices, val_indices = train_test_split(
    train_indices, test_size=0.2, random_state=42, 
    stratify=[dataset.labels[i] for i in train_indices]
)

# 데이터로더
train_loader = DataLoader(torch.utils.data.Subset(dataset, train_indices), batch_size=16, shuffle=True)
val_loader = DataLoader(torch.utils.data.Subset(dataset, val_indices), batch_size=16, shuffle=False)
test_loader = DataLoader(torch.utils.data.Subset(dataset, test_indices), batch_size=16, shuffle=False)

# 모델 생성
model = MLPModel(input_size=160, hidden_sizes=[256, 128, 64], num_classes=24, dropout=0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 훈련 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

print(f'🚀 모델 훈련 시작 (디바이스: {device})')

# 훈련
best_val_loss = float('inf')
patience = 0
max_patience = 8

for epoch in range(30):
    model.train()
    train_loss = 0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets.squeeze())
            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 0
        torch.save(model.state_dict(), 'analysis_model.pth')
    else:
        patience += 1
    
    if patience >= max_patience:
        break

# 최고 모델 로드
model.load_state_dict(torch.load('analysis_model.pth'))

# 상세 분석
model.eval()
all_predictions = []
all_targets = []
all_probabilities = []

with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.squeeze().cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

# 혼동 행렬 생성
cm = confusion_matrix(all_targets, all_predictions)

print('\n📊 혼동 행렬 (전체):')
print('=' * 50)
print('실제\예측', end='')
for i, class_name in enumerate(dataset.all_classes):
    print(f'\t{class_name}', end='')
print()

for i, class_name in enumerate(dataset.all_classes):
    print(f'{class_name}', end='')
    for j in range(len(dataset.all_classes)):
        print(f'\t{cm[i][j]}', end='')
    print()

# 100% 성능 클래스들 분석
perfect_classes = ['ㄱ', 'ㄴ', 'ㄹ', 'ㅇ', 'ㅓ', 'ㅣ']
print(f'\n🔍 100% 성능 클래스 상세 분석:')
print('=' * 50)

for class_name in perfect_classes:
    class_idx = dataset.all_classes.index(class_name)
    
    # 해당 클래스의 실제 샘플 수
    actual_samples = cm[class_idx].sum()
    
    # 정확히 분류된 샘플 수
    correct_predictions = cm[class_idx][class_idx]
    
    # 다른 클래스로 잘못 분류된 샘플 수
    wrong_predictions = actual_samples - correct_predictions
    
    # 다른 클래스에서 이 클래스로 잘못 분류된 샘플 수
    false_positives = cm[:, class_idx].sum() - correct_predictions
    
    print(f'\n{class_name} 클래스:')
    print(f'  실제 샘플 수: {actual_samples}')
    print(f'  정확히 분류된 수: {correct_predictions}')
    print(f'  잘못 분류된 수: {wrong_predictions}')
    print(f'  다른 클래스에서 잘못 분류된 수: {false_positives}')
    
    if wrong_predictions > 0:
        print(f'  ⚠️  실제로는 100%가 아님!')
        # 어떤 클래스로 잘못 분류되었는지 확인
        for j, other_class in enumerate(dataset.all_classes):
            if j != class_idx and cm[class_idx][j] > 0:
                print(f'    → {other_class}로 {cm[class_idx][j]}개 잘못 분류')
    else:
        print(f'  ✅ 진짜 100% 정확도!')

# 신뢰도 분석
print(f'\n🎯 예측 신뢰도 분석:')
print('=' * 50)

for class_name in perfect_classes:
    class_idx = dataset.all_classes.index(class_name)
    
    # 해당 클래스의 샘플들만 추출
    class_samples = []
    class_probs = []
    
    for i, (pred, target, prob) in enumerate(zip(all_predictions, all_targets, all_probabilities)):
        if target == class_idx:
            class_samples.append(i)
            class_probs.append(prob[class_idx])
    
    if class_samples:
        avg_confidence = np.mean(class_probs)
        min_confidence = np.min(class_probs)
        max_confidence = np.max(class_probs)
        
        print(f'{class_name}: 평균 신뢰도 {avg_confidence:.3f} (최소: {min_confidence:.3f}, 최대: {max_confidence:.3f})')

# 데이터 품질 확인
print(f'\n📈 데이터 품질 확인:')
print('=' * 50)

for class_name in perfect_classes:
    class_idx = dataset.all_classes.index(class_name)
    
    # 훈련/검증/테스트에서 해당 클래스 샘플 수 확인
    train_count = sum(1 for i in train_indices if dataset.labels[i] == class_idx)
    val_count = sum(1 for i in val_indices if dataset.labels[i] == class_idx)
    test_count = sum(1 for i in test_indices if dataset.labels[i] == class_idx)
    
    print(f'{class_name}: 훈련 {train_count}개, 검증 {val_count}개, 테스트 {test_count}개')

print('\n✅ 상세 분석 완료!')

