#!/usr/bin/env python3
"""
모델 로딩 문제 해결
체크포인트 구조 확인 및 수정
"""

import torch
import os
import sys
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 개선된 모델 아키텍처 import
sys.path.append('.')
from improved_model_architecture import RegularizedModel

def check_model_checkpoint(model_path='models/improved_regularized_model.pth'):
    """모델 체크포인트 구조 확인"""
    print('🔍 모델 체크포인트 구조 확인 중...')
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f'✅ 체크포인트 로드 성공: {model_path}')
        
        print('\n📊 체크포인트 키 목록:')
        for key in checkpoint.keys():
            print(f'  - {key}')
        
        print('\n📊 체크포인트 상세 정보:')
        for key, value in checkpoint.items():
            if isinstance(value, dict):
                print(f'  {key}: {type(value)} (keys: {list(value.keys())})')
            elif isinstance(value, list):
                print(f'  {key}: {type(value)} (length: {len(value)})')
            else:
                print(f'  {key}: {type(value)} = {value}')
        
        return checkpoint
        
    except Exception as e:
        print(f'❌ 체크포인트 로드 실패: {e}')
        return None

def fix_model_loading(checkpoint):
    """모델 로딩 수정"""
    print('\n🔧 모델 로딩 수정 중...')
    
    try:
        # 모델 초기화
        model = RegularizedModel(input_size=8, hidden_size=96, num_classes=24, dropout=0.5)
        
        # 체크포인트에서 모델 상태 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print('✅ model_state_dict 로드 성공')
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print('✅ state_dict 로드 성공')
        else:
            print('❌ 모델 상태를 찾을 수 없습니다')
            return None
        
        # 디바이스 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print(f'✅ 모델 로드 완료 (디바이스: {device})')
        
        # 테스트 추론
        print('\n🧪 모델 테스트 추론...')
        test_input = torch.randn(1, 300, 8).to(device)  # 배치 크기 1, 시퀀스 길이 300, 특성 8
        
        with torch.no_grad():
            output = model(test_input)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            print(f'✅ 테스트 추론 성공!')
            print(f'  - 출력 형태: {output.shape}')
            print(f'  - 예측 클래스: {predicted_idx.item()}')
            print(f'  - 신뢰도: {confidence.item():.4f}')
        
        return model
        
    except Exception as e:
        print(f'❌ 모델 로딩 실패: {e}')
        return None

def test_single_inference(model, data_dir='real_data_filtered'):
    """단일 파일 추론 테스트"""
    print(f'\n🧪 단일 파일 추론 테스트: {data_dir}')
    
    # 첫 번째 CSV 파일 찾기
    csv_files = list(Path(data_dir).rglob("*.csv"))
    if not csv_files:
        print('❌ CSV 파일을 찾을 수 없습니다')
        return
    
    test_file = csv_files[0]
    print(f'📁 테스트 파일: {test_file}')
    
    try:
        # 데이터 로드
        df = pd.read_csv(test_file)
        data = df.iloc[:, :8].values.astype(np.float32)
        
        print(f'📊 데이터 형태: {data.shape}')
        print(f'📊 데이터 범위: {data.min():.4f} ~ {data.max():.4f}')
        
        # 데이터 전처리
        target_length = 300
        if len(data) < target_length:
            padding = np.zeros((target_length - len(data), data.shape[1]))
            data = np.vstack([data, padding])
        elif len(data) > target_length:
            data = data[:target_length]
        
        # 텐서 변환
        device = next(model.parameters()).device
        input_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
        
        print(f'📊 입력 텐서 형태: {input_tensor.shape}')
        
        # 추론
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                          'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
            
            predicted_class = class_names[predicted_idx.item()]
            
            print(f'✅ 추론 성공!')
            print(f'  - 예측 클래스: {predicted_class}')
            print(f'  - 신뢰도: {confidence.item():.4f}')
            print(f'  - 클래스 인덱스: {predicted_idx.item()}')
        
        return True
        
    except Exception as e:
        print(f'❌ 단일 추론 실패: {e}')
        return False

def main():
    """메인 실행 함수"""
    print('🔧 모델 로딩 문제 해결')
    print('=' * 60)
    
    # 1. 체크포인트 구조 확인
    checkpoint = check_model_checkpoint()
    
    if checkpoint is None:
        print('❌ 체크포인트를 확인할 수 없습니다')
        return
    
    # 2. 모델 로딩 수정
    model = fix_model_loading(checkpoint)
    
    if model is None:
        print('❌ 모델 로딩에 실패했습니다')
        return
    
    # 3. 단일 추론 테스트
    success = test_single_inference(model)
    
    if success:
        print('\n🎉 모델 로딩 문제 해결 완료!')
        print('✅ 이제 정상적으로 추론이 가능합니다')
    else:
        print('\n❌ 단일 추론 테스트 실패')

if __name__ == "__main__":
    main()
