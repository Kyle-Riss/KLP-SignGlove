import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import os
from pathlib import Path
import sys
sys.path.append('models')
from deep_learning import DeepLearningPipeline

def load_simple_dataset():
    """간단한 데이터셋 로드"""
    data_path = Path('/home/billy/25-1kp/SignGlove/external/SignGlove_HW/datasets/unified')
    
    data = []
    labels = []
    
    # 각 클래스별로 몇 개 파일만 로드
    class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                   'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = data_path / class_name
        if not class_dir.exists():
            continue
            
        # 각 클래스에서 최대 10개 파일만 로드
        file_count = 0
        for sub_dir in class_dir.iterdir():
            if sub_dir.is_dir() and file_count < 10:
                for h5_file in sub_dir.glob("*.h5"):
                    try:
                        with h5py.File(h5_file, 'r') as f:
                            sensor_data = f['sensor_data'][:]
                            
                            # 시퀀스 길이 20으로 분할
                            for i in range(0, len(sensor_data), 20):
                                if i + 20 <= len(sensor_data):
                                    sequence = sensor_data[i:i+20]
                                    data.append(sequence)
                                    labels.append(class_idx)
                                    file_count += 1
                                    break  # 파일당 하나의 시퀀스만
                    except Exception as e:
                        print(f"파일 로드 오류 {h5_file}: {e}")
                        continue
                    
                    if file_count >= 10:
                        break
            if file_count >= 10:
                break
    
    print(f"로드된 데이터: {len(data)}개 시퀀스")
    print(f"클래스별 샘플 수:")
    for i in range(24):
        count = labels.count(i)
        if count > 0:
            print(f"  클래스 {i}: {count}개")
    
    return np.array(data), np.array(labels)

def train_simple_model():
    """간단한 모델 훈련"""
    print("🚀 간단한 모델 훈련 시작...")
    
    # 데이터 로드
    data, labels = load_simple_dataset()
    
    if len(data) == 0:
        print("❌ 데이터를 로드할 수 없습니다!")
        return
    
    # 데이터 정규화
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    
    # 텐서 변환
    data_tensor = torch.FloatTensor(data)
    labels_tensor = torch.LongTensor(labels)
    
    # 간단한 분할 (80% 훈련, 20% 검증)
    n_samples = len(data)
    train_size = int(0.8 * n_samples)
    indices = torch.randperm(n_samples)
    
    train_data = data_tensor[indices[:train_size]]
    train_labels = labels_tensor[indices[:train_size]]
    val_data = data_tensor[indices[train_size:]]
    val_labels = labels_tensor[indices[train_size:]]
    
    print(f"훈련 데이터: {len(train_data)}개")
    print(f"검증 데이터: {len(val_data)}개")
    
    # 모델 생성
    model = DeepLearningPipeline(
        input_features=8,
        sequence_length=20,
        num_classes=24,
        hidden_dim=64,  # 더 큰 hidden_dim
        num_layers=2,   # 더 많은 레이어
        dropout=0.2
    )
    
    # 옵티마이저 및 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}개")
    
    # 훈련
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(50):
        # 훈련
        model.train()
        optimizer.zero_grad()
        
        outputs = model(train_data)
        loss = criterion(outputs['class_logits'], train_labels)
        loss.backward()
        optimizer.step()
        
        # 정확도 계산
        with torch.no_grad():
            train_pred = outputs['class_logits'].argmax(dim=1)
            train_acc = (train_pred == train_labels).float().mean().item()
            
            # 검증
            model.eval()
            val_outputs = model(val_data)
            val_pred = val_outputs['class_logits'].argmax(dim=1)
            val_acc = (val_pred == val_labels).float().mean().item()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1:3d}/50 | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    print(f"\n🎯 최고 검증 정확도: {best_val_acc:.4f}")
    
    # 최고 모델 저장
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save({
            'model_state_dict': best_model_state,
            'val_accuracy': best_val_acc
        }, 'simple_test_model.pth')
        print("✅ 모델 저장 완료: simple_test_model.pth")
    
    # 테스트 예측
    print("\n🔍 모델 예측 테스트:")
    model.eval()
    with torch.no_grad():
        test_outputs = model(val_data[:5])  # 처음 5개 샘플 테스트
        test_pred = test_outputs['class_logits'].argmax(dim=1)
        test_probs = torch.softmax(test_outputs['class_logits'], dim=1)
        
        for i in range(5):
            true_label = val_labels[i].item()
            pred_label = test_pred[i].item()
            confidence = test_probs[i][pred_label].item()
            print(f"  샘플 {i+1}: 실제={true_label}, 예측={pred_label}, 신뢰도={confidence:.4f}")

if __name__ == "__main__":
    train_simple_model()



