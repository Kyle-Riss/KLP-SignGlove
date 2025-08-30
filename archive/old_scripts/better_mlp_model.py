import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# 더 나은 MLP 모델
class BetterMLPModel(nn.Module):
    def __init__(self, input_size=160, hidden_sizes=[128, 64, 32], num_classes=24, dropout=0.5):
        super(BetterMLPModel, self).__init__()
        
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

# 강화된 데이터 증강
class EnhancedH5Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, sequence_length=20, use_augmentation=True):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.use_augmentation = use_augmentation
        self.data = []
        self.labels = []
        self.class_names = []
        
        # 클래스별 데이터 로드
        for class_name in sorted(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_names.append(class_name)
                class_idx = len(self.class_names) - 1
                
                # 각 세션의 파일들 로드
                for session in ['1', '2', '3', '4', '5']:
                    session_dir = os.path.join(class_dir, session)
                    if os.path.exists(session_dir):
                        for file_name in os.listdir(session_dir):
                            if file_name.endswith('.h5'):
                                file_path = os.path.join(session_dir, file_name)
                                try:
                                    with h5py.File(file_path, 'r') as f:
                                        sensor_data = f['sensor_data'][:]  # (300, 8)
                                        
                                        # 시퀀스 길이에 맞게 조정
                                        if len(sensor_data) >= sequence_length:
                                            # 마지막 sequence_length 프레임 사용
                                            data = sensor_data[-sequence_length:]
                                            self.data.append(data)
                                            self.labels.append(class_idx)
                                            
                                            # 강화된 데이터 증강
                                            if self.use_augmentation:
                                                # 1. 노이즈 추가 (3가지 강도)
                                                for noise_level in [0.005, 0.01, 0.02]:
                                                    noise_data = data + np.random.normal(0, noise_level, data.shape)
                                                    self.data.append(noise_data)
                                                    self.labels.append(class_idx)
                                                
                                                # 2. 시간 이동 (3가지 변형)
                                                for shift in [2, 4, 6]:
                                                    if len(sensor_data) >= sequence_length + shift:
                                                        shifted_data = sensor_data[-(sequence_length+shift):-shift]
                                                        self.data.append(shifted_data)
                                                        self.labels.append(class_idx)
                                                
                                                # 3. 스케일링 변형 (약간만)
                                                for scale in [0.98, 1.02]:
                                                    scaled_data = data * scale
                                                    self.data.append(scaled_data)
                                                    self.labels.append(class_idx)
                                                
                                                # 4. 마스킹 (일부 센서 값 제거)
                                                masked_data = data.copy()
                                                mask_indices = np.random.choice(8, 2, replace=False)
                                                masked_data[:, mask_indices] = 0
                                                self.data.append(masked_data)
                                                self.labels.append(class_idx)
                                                
                                except Exception as e:
                                    print(f"Error loading {file_path}: {e}")
        
        print(f"📊 강화된 데이터셋 로드 완료: {len(self.data)} 샘플, {len(self.class_names)} 클래스")
        print(f"📋 클래스: {self.class_names}")
        
        # 클래스별 샘플 수 출력
        for i, class_name in enumerate(self.class_names):
            count = sum(1 for label in self.labels if label == i)
            print(f"  {class_name}: {count} 샘플")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])

print('🔧 더 나은 MLP 모델 훈련 시작')

# 강화된 데이터셋 로드
dataset = EnhancedH5Dataset(
    data_dir='../SignGlove/external/SignGlove_HW/datasets/unified',
    sequence_length=20,
    use_augmentation=True
)

# 데이터 분할 (stratified)
train_indices, test_indices = train_test_split(
    range(len(dataset)), test_size=0.3, random_state=42, 
    stratify=dataset.labels
)
train_indices, val_indices = train_test_split(
    train_indices, test_size=0.2, random_state=42, 
    stratify=[dataset.labels[i] for i in train_indices]
)

# 데이터로더
train_loader = DataLoader(torch.utils.data.Subset(dataset, train_indices), 
                         batch_size=32, shuffle=True)  # 배치 크기 증가
val_loader = DataLoader(torch.utils.data.Subset(dataset, val_indices), 
                       batch_size=32, shuffle=False)
test_loader = DataLoader(torch.utils.data.Subset(dataset, test_indices), 
                        batch_size=32, shuffle=False)

# 더 나은 모델 생성
model = BetterMLPModel(input_size=160, hidden_sizes=[128, 64, 32], num_classes=24, dropout=0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
print(f'📊 모델 파라미터 수: {total_params:,}')

# 훈련 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)  # AdamW 사용
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

print(f'🚀 더 나은 MLP 모델 훈련 시작 (디바이스: {device})')

# 훈련
best_val_loss = float('inf')
patience = 0
max_patience = 20

for epoch in range(150):
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
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets.squeeze()).sum().item()
    
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
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets.squeeze()).sum().item()
    
    # 학습률 조정
    scheduler.step(val_loss)
    
    # 조기 종료
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 0
        # 모델 저장
        torch.save(model.state_dict(), 'better_mlp_model.pth')
    else:
        patience += 1
    
    if patience >= max_patience:
        print(f'🛑 조기 종료 (에포크 {epoch+1})')
        break
    
    # 결과 출력
    if (epoch + 1) % 15 == 0:
        print(f'에포크 {epoch+1:3d}: '
              f'훈련 손실={train_loss/len(train_loader):.4f}, '
              f'훈련 정확도={100.*train_correct/train_total:.2f}%, '
              f'검증 손실={val_loss/len(val_loader):.4f}, '
              f'검증 정확도={100.*val_correct/val_total:.2f}%')

# 최종 테스트
model.load_state_dict(torch.load('better_mlp_model.pth'))
model.eval()
test_correct = 0
test_total = 0
all_predictions = []
all_targets = []

with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        _, predicted = outputs.max(1)
        test_total += targets.size(0)
        test_correct += predicted.eq(targets.squeeze()).sum().item()
        
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.squeeze().cpu().numpy())

print(f'\n🎯 최종 테스트 결과:')
print(f'정확도: {100.*test_correct/test_total:.2f}%')

# 분류 리포트
print(f'\n📊 분류 리포트:')
print(classification_report(all_targets, all_predictions, 
                          target_names=dataset.class_names, zero_division=0))
