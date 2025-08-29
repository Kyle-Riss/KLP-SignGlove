import torch
import numpy as np
import sys
sys.path.append('models')
from deep_learning import DeepLearningPipeline

def test_model():
    print("🔍 모델 예측 테스트 시작...")
    
    # 모델 로드
    checkpoint = torch.load('cross_validation_model.pth', map_location='cpu', weights_only=False)
    model = DeepLearningPipeline(
        input_features=8, 
        sequence_length=20, 
        num_classes=24, 
        hidden_dim=48, 
        num_layers=1, 
        dropout=0.3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ 모델 로드 완료")
    
    # 여러 테스트 입력으로 예측
    for i in range(5):
        # 랜덤 입력 생성
        test_input = torch.randn(1, 20, 8)
        
        with torch.no_grad():
            output = model(test_input)
            logits = output['class_logits']
            probs = torch.softmax(logits, dim=1)
            
            max_prob_idx = torch.argmax(probs, dim=1).item()
            max_prob_val = torch.max(probs, dim=1)[0].item()
            
            print(f"\n테스트 {i+1}:")
            print(f"  최대 확률 인덱스: {max_prob_idx}")
            print(f"  최대 확률 값: {max_prob_val:.4f}")
            print(f"  전체 확률 분포: {probs[0].numpy()}")
    
    # 실제 센서 데이터와 유사한 입력 테스트
    print("\n🔬 실제 센서 데이터 유사 입력 테스트:")
    
    # flex 센서 데이터 (0-1023 범위)
    flex_data = np.random.randint(0, 1024, (20, 5))
    # IMU 데이터 (-180~180 범위)
    imu_data = np.random.uniform(-180, 180, (20, 3))
    
    # 결합
    sensor_data = np.concatenate([flex_data, imu_data], axis=1)
    
    # 정규화 (0-1 범위)
    sensor_data = (sensor_data - sensor_data.min()) / (sensor_data.max() - sensor_data.min() + 1e-8)
    
    test_input = torch.FloatTensor(sensor_data).unsqueeze(0)
    
    with torch.no_grad():
        output = model(test_input)
        logits = output['class_logits']
        probs = torch.softmax(logits, dim=1)
        
        max_prob_idx = torch.argmax(probs, dim=1).item()
        max_prob_val = torch.max(probs, dim=1)[0].item()
        
        print(f"  최대 확률 인덱스: {max_prob_idx}")
        print(f"  최대 확률 값: {max_prob_val:.4f}")
        print(f"  상위 5개 확률:")
        top5_probs, top5_indices = torch.topk(probs[0], 5)
        for j, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
            print(f"    {j+1}. 인덱스 {idx}: {prob:.4f}")

if __name__ == "__main__":
    test_model()
