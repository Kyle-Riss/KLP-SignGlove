#!/usr/bin/env python3
"""
수정된 정규화된 데이터 추론
올바른 모델 로딩으로 정확한 추론 수행
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# 개선된 모델 아키텍처 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from improved_model_architecture import RegularizedModel
from class_discriminator import ClassDiscriminator

class CorrectedFilteredInference:
    """수정된 정규화된 데이터 추론 시스템"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        print('🔍 수정된 정규화된 데이터 추론 시스템 초기화 중...')
        
        # 수정된 모델 로드
        self.model = self._load_corrected_model()
        
        # 차별화기 로드
        self.discriminator = self._load_discriminator()
        
        print('✅ 수정된 정규화된 데이터 추론 시스템 초기화 완료')
        print(f'  - 개선된 모델: RegularizedModel')
        print(f'  - 차별화기: ㄹ/ㅕ 구분')
        print(f'  - 디바이스: {self.device}')
    
    def _load_corrected_model(self, model_path='../models/improved_regularized_model.pth'):
        """수정된 모델 로드"""
        try:
            # 체크포인트 로드
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 모델 초기화
            model = RegularizedModel(input_size=8, hidden_size=96, num_classes=24, dropout=0.5)
            
            # 모델 상태 로드
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            print(f'✅ 수정된 모델 로드 완료')
            print(f'  - 모델 타입: {checkpoint["model_type"]}')
            print(f'  - 최종 검증 손실: {min(checkpoint["history"]["val_loss"]):.4f}')
            return model
        except Exception as e:
            print(f'❌ 모델 로드 실패: {e}')
            return None
    
    def _load_discriminator(self):
        """ㄹ/ㅕ 차별화기 로드"""
        try:
            discriminator = ClassDiscriminator()
            
            # real_data_filtered에서 ㄹ과 ㅕ 데이터 수집
            ㄹ_data = []
            ㅕ_data = []
            
            # ㄹ 데이터 수집
            ㄹ_dir = Path('../real_data_filtered/ㄹ')
            if ㄹ_dir.exists():
                for file_path in ㄹ_dir.rglob('*.csv'):
                    try:
                        df = pd.read_csv(file_path)
                        data = df.iloc[:, :8].values.astype(np.float32)
                        # Flex5_mean, Flex3_mean 특성 추출
                        flex5_mean = np.mean(data[:, 4])  # Flex5
                        flex3_mean = np.mean(data[:, 2])  # Flex3
                        ㄹ_data.append([flex5_mean, flex3_mean])
                    except Exception as e:
                        continue
            
            # ㅕ 데이터 수집
            ㅕ_dir = Path('../real_data_filtered/ㅕ')
            if ㅕ_dir.exists():
                for file_path in ㅕ_dir.rglob('*.csv'):
                    try:
                        df = pd.read_csv(file_path)
                        data = df.iloc[:, :8].values.astype(np.float32)
                        # Flex5_mean, Flex3_mean 특성 추출
                        flex5_mean = np.mean(data[:, 4])  # Flex5
                        flex3_mean = np.mean(data[:, 2])  # Flex3
                        ㅕ_data.append([flex5_mean, flex3_mean])
                    except Exception as e:
                        continue
            
            if len(ㄹ_data) > 0 and len(ㅕ_data) > 0:
                discriminator.train_ml_model(ㄹ_data, ㅕ_data)
                print(f'✅ 차별화기 학습 완료 (ㄹ: {len(ㄹ_data)}개, ㅕ: {len(ㅕ_data)}개)')
            else:
                print('⚠️ 차별화기 학습 데이터 부족')
            
            return discriminator
        except Exception as e:
            print(f'❌ 차별화기 로드 실패: {e}')
            return None
    
    def preprocess_data(self, data):
        """데이터 전처리 (정규화된 데이터용)"""
        # 정규화된 데이터는 이미 전처리되어 있음
        # 시퀀스 길이 맞추기 (패딩)
        target_length = 300
        if len(data) < target_length:
            # 패딩
            padding = np.zeros((target_length - len(data), data.shape[1]))
            data = np.vstack([data, padding])
        elif len(data) > target_length:
            # 자르기
            data = data[:target_length]
        
        return torch.FloatTensor(data).unsqueeze(0).to(self.device)
    
    def predict_with_enhanced_filter(self, data):
        """향상된 필터와 함께 예측"""
        try:
            # 데이터 전처리
            processed_data = self.preprocess_data(data)
            
            # 모델 예측
            with torch.no_grad():
                outputs = self.model(processed_data)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_class = self.class_names[predicted_idx.item()]
                confidence = confidence.item()
            
            # ㄹ/ㅕ 후처리 필터 적용
            if predicted_class in ['ㄹ', 'ㅕ'] and self.discriminator is not None:
                try:
                    # Flex5_mean, Flex3_mean 특성 추출
                    flex5_mean = np.mean(data[:, 4])
                    flex3_mean = np.mean(data[:, 2])
                    features = [[flex5_mean, flex3_mean]]
                    
                    # 차별화기 예측
                    discriminator_pred, discriminator_conf = self.discriminator.predict(features)
                    
                    if discriminator_pred == 'ㅕ' and predicted_class == 'ㄹ':
                        print(f'🔄 후처리 필터 적용: ㄹ → ㅕ (신뢰도: {discriminator_conf:.3f})')
                        predicted_class = 'ㅕ'
                        confidence = discriminator_conf
                    elif discriminator_pred == 'ㄹ' and predicted_class == 'ㅕ':
                        print(f'🔄 후처리 필터 적용: ㅕ → ㄹ (신뢰도: {discriminator_conf:.3f})')
                        predicted_class = 'ㄹ'
                        confidence = discriminator_conf
                except Exception as e:
                    print(f'⚠️ 차별화기 오류: {e}, 기본 결과 유지')
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f'❌ 예측 오류: {e}')
            return None, 0.0
    
    def batch_inference_corrected(self, data_dir, output_file='corrected_filtered_results.json'):
        """수정된 정규화된 데이터 배치 추론"""
        print(f'🚀 수정된 정규화된 데이터 배치 추론 시작: {data_dir}')
        
        results = []
        total_files = 0
        processed_files = 0
        
        # 모든 CSV 파일 찾기
        csv_files = list(Path(data_dir).rglob("*.csv"))
        total_files = len(csv_files)
        
        print(f'📁 총 {total_files}개 파일 발견')
        
        for file_path in csv_files:
            try:
                # 데이터 로드
                df = pd.read_csv(file_path)
                data = df.iloc[:, :8].values.astype(np.float32)
                
                # 향상된 추론 실행
                start_time = time.time()
                predicted_class, confidence = self.predict_with_enhanced_filter(data)
                inference_time = time.time() - start_time
                
                if predicted_class is not None:
                    result = {
                        'file': str(file_path),
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'inference_time': inference_time,
                        'enhanced': True,
                        'improved_model': True,
                        'filtered_data': True,
                        'corrected_loading': True
                    }
                    results.append(result)
                    processed_files += 1
                    
                    # 진행 상황 출력
                    if processed_files % 50 == 0:
                        print(f'  진행률: {processed_files}/{total_files} ({processed_files/total_files*100:.1f}%)')
                
            except Exception as e:
                print(f'⚠️ 파일 처리 실패: {file_path} - {e}')
        
        # 결과 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f'✅ 수정된 정규화된 데이터 추론 완료: {processed_files}/{total_files} 파일 처리')
        print(f'📊 결과 저장: {output_file}')
        
        return results

def analyze_corrected_results(results_file='corrected_filtered_results.json'):
    """수정된 정규화된 데이터 결과 분석"""
    print('📊 수정된 정규화된 데이터 추론 결과 분석')
    print('=' * 50)
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 기본 통계
    total_files = len(results)
    avg_confidence = np.mean([r['confidence'] for r in results])
    avg_inference_time = np.mean([r['inference_time'] for r in results])
    
    print(f'📁 총 파일 수: {total_files}')
    print(f'📊 평균 신뢰도: {avg_confidence:.3f}')
    print(f'⏱️ 평균 추론 시간: {avg_inference_time*1000:.1f}ms')
    
    # 클래스별 통계
    class_stats = {}
    for result in results:
        class_name = result['predicted_class']
        if class_name not in class_stats:
            class_stats[class_name] = {'count': 0, 'confidences': []}
        
        class_stats[class_name]['count'] += 1
        class_stats[class_name]['confidences'].append(result['confidence'])
    
    print('\n📊 클래스별 통계:')
    print(f"{'Class':<4} {'Count':<6} {'Avg_Conf':<10} {'Min_Conf':<10} {'Max_Conf':<10}")
    print('-' * 50)
    
    for class_name in sorted(class_stats.keys()):
        stats = class_stats[class_name]
        avg_conf = np.mean(stats['confidences'])
        min_conf = np.min(stats['confidences'])
        max_conf = np.max(stats['confidences'])
        
        print(f"{class_name:<4} {stats['count']:<6} {avg_conf:<10.3f} {min_conf:<10.3f} {max_conf:<10.3f}")
    
    # 성능 개선 분석
    print('\n📈 성능 개선 분석:')
    high_confidence_count = sum(1 for r in results if r['confidence'] > 0.8)
    medium_confidence_count = sum(1 for r in results if 0.6 <= r['confidence'] <= 0.8)
    low_confidence_count = sum(1 for r in results if r['confidence'] < 0.6)
    
    print(f'  높은 신뢰도 (>0.8): {high_confidence_count}개 ({high_confidence_count/total_files*100:.1f}%)')
    print(f'  중간 신뢰도 (0.6-0.8): {medium_confidence_count}개 ({medium_confidence_count/total_files*100:.1f}%)')
    print(f'  낮은 신뢰도 (<0.6): {low_confidence_count}개 ({low_confidence_count/total_files*100:.1f}%)')

def main():
    """메인 실행 함수"""
    print('🔍 수정된 정규화된 데이터 추론 테스트')
    print('=' * 60)
    
    # 수정된 정규화된 데이터 추론 시스템 초기화
    corrected_inference = CorrectedFilteredInference()
    
    # real_data_filtered로 테스트
    data_dir = '../real_data_filtered'
    
    if os.path.exists(data_dir):
        print(f'\n🧪 수정된 정규화된 데이터로 추론 테스트')
        results = corrected_inference.batch_inference_corrected(data_dir, 'corrected_filtered_results.json')
        
        # 결과 분석
        analyze_corrected_results('corrected_filtered_results.json')
        
        print('\n🎉 수정된 정규화된 데이터 추론 테스트 완료!')
        print('✅ 올바른 모델 로딩으로 높은 정확도를 기대합니다.')
    else:
        print(f'⚠️ 정규화된 데이터 디렉토리를 찾을 수 없습니다: {data_dir}')
        print('real_data_filtered 폴더가 있는지 확인해주세요!')

if __name__ == "__main__":
    main()
