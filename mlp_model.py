import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# MLP 모델 정의
class MLPModel(nn.Module):
    def __init__(self, input_size=160, hidden_sizes=[256, 128, 64], num_classes=24, dropout=0.5):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 은닉층 구성
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # 출력층
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # 입력 형태 변환: (batch, seq_len, features) -> (batch, seq_len * features)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.network(x)

# 기존 데이터셋 클래스 재사용
from vowel_improvement import VowelImprovedDataset

print('🔧 MLP 모델 훈련 시작')

# 데이터셋 로드 (향상 전처리 없이)
dataset = VowelImprovedDataset(
    data_path='../SignGlove_HW/datasets/unified',
    sequence_length=20,
    apply_vowel_enhancement=False  # 전처리 없이
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

# 모델 생성 (20 * 8 = 160 입력 크기)
model = MLPModel(input_size=160, hidden_sizes=[256, 128, 64], num_classes=24, dropout=0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 훈련 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

print(f'🚀 MLP 모델 훈련 시작 (디바이스: {device})')
print(f'📊 모델 구조: {model}')

# 훈련
best_val_loss = float('inf')
patience = 0
max_patience = 8

for epoch in range(30):
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
        
        # 그래디언트 클리핑
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
    
    # 학습률 스케줄러
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 0
        torch.save(model.state_dict(), 'mlp_model.pth')
    else:
        patience += 1
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/30] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'  Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    if patience >= max_patience:
        print(f'🛑 Early stopping at epoch {epoch+1}')
        break

# 최고 모델 로드
model.load_state_dict(torch.load('mlp_model.pth'))

# 모음 성능 평가
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

# 모음 클래스만 추출
vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
vowel_indices = [dataset.all_classes.index(v) for v in vowels]

vowel_predictions = []
vowel_targets = []

for pred, target in zip(all_predictions, all_targets):
    if target in vowel_indices:
        vowel_predictions.append(pred)
        vowel_targets.append(target)

# 성능 리포트
print(f'모음 예측된 클래스 수: {len(set(vowel_predictions))}')
print(f'모음 타겟 클래스 수: {len(set(vowel_targets))}')
print(f'예측된 클래스: {sorted(set(vowel_predictions))}')
print(f'타겟 클래스: {sorted(set(vowel_targets))}')

# 실제 예측된 클래스에 맞춰 리포트 생성
unique_classes = sorted(set(vowel_targets + vowel_predictions))
vowel_class_names = [dataset.all_classes[i] for i in unique_classes]

report = classification_report(vowel_targets, vowel_predictions, 
                             target_names=vowel_class_names, output_dict=True, zero_division=0)

print('\n📊 MLP 모델 모음 성능:')
print('=' * 40)

for vowel in vowels:
    if vowel in report:
        precision = report[vowel]['precision'] * 100
        recall = report[vowel]['recall'] * 100
        f1 = report[vowel]['f1-score'] * 100
        print(f'  {vowel}: Precision={precision:.1f}%, Recall={recall:.1f}%, F1={f1:.1f}%')
    else:
        print(f'  {vowel}: 예측되지 않음')

if 'weighted avg' in report:
    overall_precision = report['weighted avg']['precision'] * 100
    overall_recall = report['weighted avg']['recall'] * 100
    overall_f1 = report['weighted avg']['f1-score'] * 100
    print(f'\n📈 모음 전체 성능:')
    print(f'  Precision: {overall_precision:.1f}%')
    print(f'  Recall: {overall_recall:.1f}%')
    print(f'  F1-Score: {overall_f1:.1f}%')

# 전체 클래스 성능도 확인
print(f'\n🎯 전체 클래스 성능:')
print(f'  전체 정확도: {100 * sum(1 for p, t in zip(all_predictions, all_targets) if p == t) / len(all_targets):.1f}%')

print('\n✅ MLP 모델 훈련 완료!')
