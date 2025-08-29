"""
ONNX 최적화 단독 실행
"""

import torch
import torch.nn as nn
import time
import os
import json
import numpy as np
from pathlib import Path
import sys

# 프로젝트 모듈 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.deep_learning import DeepLearningPipeline
from training.dataset import KSLCsvDataset
from torch.utils.data import DataLoader, Subset

def run_onnx_optimization():
    """ONNX 최적화 실행"""
    print("🚀 ONNX 최적화 시작")
    
    # 모델 설정
    model_config = {
        'input_features': 8,
        'sequence_length': 20,
        'num_classes': 5,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3
    }
    
    device = torch.device('cpu')  # ONNX는 CPU에서 실행
    
    # 모델 로드
    model = DeepLearningPipeline(**model_config)
    model_path = 'best_dl_model.pth'
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅ 모델 로드 성공: {model_path}")
    else:
        print(f"⚠️  모델 파일 없음: {model_path}")
        return
    
    model.to(device)
    model.eval()
    
    # 데이터 준비
    csv_dir = 'integrations/SignGlove_HW'
    dataset = KSLCsvDataset(csv_dir, window_size=20, stride=10)
    total_size = len(dataset)
    test_size = int(0.2 * total_size)
    test_indices = list(range(total_size - test_size, total_size))
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"📊 테스트 데이터: {len(test_dataset)}개 샘플")
    
    # ONNX 내보내기
    output_dir = Path('optimization/compressed_models')
    dummy_input = torch.randn(1, 20, 8)
    onnx_path = output_dir / 'optimized_model.onnx'
    
    print("📦 ONNX 모델 내보내기 중...")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['sensor_data'],
        output_names=['class_logits', 'features', 'attention_weights'],
        dynamic_axes={
            'sensor_data': {0: 'batch_size'},
            'class_logits': {0: 'batch_size'},
            'features': {0: 'batch_size'},
            'attention_weights': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNX 모델 생성: {onnx_path}")
    
    # ONNX Runtime으로 성능 평가
    import onnxruntime as ort
    
    print("🔍 ONNX Runtime 성능 평가 중...")
    ort_session = ort.InferenceSession(str(onnx_path))
    
    correct = 0
    total = 0
    inference_times = []
    
    for data, targets in test_loader:
        data_np = data.numpy().astype(np.float32)
        
        start_time = time.time()
        ort_inputs = {'sensor_data': data_np}
        ort_outputs = ort_session.run(None, ort_inputs)
        inference_time = (time.time() - start_time) * 1000
        
        # class_logits만 사용 (첫 번째 출력)
        logits = torch.from_numpy(ort_outputs[0])
        _, predicted = torch.max(logits, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        inference_times.append(inference_time)
    
    accuracy = 100 * correct / total
    avg_inference_time = np.mean(inference_times)
    fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
    
    # 모델 크기 확인
    onnx_size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    
    print(f"\n📊 ONNX 성능 결과:")
    print(f"  정확도: {accuracy:.2f}%")
    print(f"  모델 크기: {onnx_size_mb:.2f} MB")
    print(f"  추론 속도: {avg_inference_time:.2f} ms")
    print(f"  FPS: {fps:.1f}")
    
    # 결과 저장
    onnx_results = {
        'accuracy': accuracy,
        'model_size_mb': onnx_size_mb,
        'inference_time_ms': avg_inference_time,
        'fps': fps
    }
    
    results_path = output_dir / 'onnx_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(onnx_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 ONNX 결과 저장: {results_path}")
    
    return onnx_results

if __name__ == "__main__":
    run_onnx_optimization()
