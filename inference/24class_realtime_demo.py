"""
24개 클래스 (14개 자음 + 10개 모음) 실시간 추론 데모
새로 학습된 24개 클래스 모델을 사용한 정확한 추론 시스템
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import glob
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.deep_learning import DeepLearningPipeline
from training.label_mapping import KSLLabelMapper

class TwentyFourClassInference:
    """24개 클래스 실시간 추론 시스템"""
    
    def __init__(self, model_path: str = "best_24class_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_mapper = KSLLabelMapper()  # 24개 클래스 지원
        
        # 모델 로드
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"🎯 24개 클래스 추론 시스템 초기화 완료")
        print(f"🖥️  장치: {self.device}")
        print(f"📊 지원 클래스: {self.label_mapper.get_num_classes()}개")
        print(f"  🔤 자음: {len(self.label_mapper.get_consonants())}개")
        print(f"  🅰️ 모음: {len(self.label_mapper.get_vowels())}개")
    
    def _load_model(self, model_path: str) -> DeepLearningPipeline:
        """24개 클래스 모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        # 모델 초기화 (24개 클래스)
        model = DeepLearningPipeline(
            input_features=8,  # pitch, roll, yaw + 5 flex sensors
            sequence_length=20,
            num_classes=24,    # 24개 클래스
            hidden_dim=256,
            num_layers=3,
            dropout=0.3
        ).to(self.device)
        
        # 가중치 로드
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"✅ 모델 로드 완료: {model_path}")
        
        return model
    
    def predict_single_window(self, window_data: np.ndarray) -> Tuple[str, float]:
        """단일 윈도우 데이터에 대한 예측"""
        # 데이터 전처리
        if window_data.shape != (20, 8):
            raise ValueError(f"윈도우 데이터 형태가 잘못되었습니다. 예상: (20, 8), 실제: {window_data.shape}")
        
        # 텐서 변환
        input_tensor = torch.FloatTensor(window_data).unsqueeze(0).to(self.device)
        
        # 예측
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # 모델 출력이 딕셔너리인 경우 처리
            if isinstance(output, dict):
                output = output['class_logits']
            
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # 클래스명 변환
            class_name = self.label_mapper.get_class_name(predicted_class)
            
            return class_name, confidence
    
    def load_test_data(self, data_dir: str = "integrations/SignGlove_HW") -> List[Tuple[str, np.ndarray, str]]:
        """테스트 데이터 로드"""
        print("📁 테스트 데이터 로드 중...")
        
        test_data = []
        unified_files = glob.glob(os.path.join(data_dir, "*_unified_data_*.csv"))
        
        for file_path in unified_files:
            filename = os.path.basename(file_path)
            
            # 실제 클래스 추출
            actual_class = None
            for class_name in self.label_mapper.class_to_id.keys():
                if class_name in filename:
                    actual_class = class_name
                    break
            
            if actual_class is None:
                continue
            
            try:
                # CSV 파일 로드
                df = pd.read_csv(file_path, encoding='latin1')
                
                # 컬럼명 확인 및 매핑
                available_cols = df.columns.tolist()
                flex_cols = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5']
                
                # 각 방향의 컬럼을 찾기
                pitch_col = [col for col in available_cols if 'pitch' in col.lower()][0]
                roll_col = [col for col in available_cols if 'roll' in col.lower()][0]
                yaw_col = [col for col in available_cols if 'yaw' in col.lower()][0]
                
                target_cols = flex_cols + [pitch_col, roll_col, yaw_col]
                data = df[target_cols].values
                
                # 윈도우 생성 (20개 샘플, 10개 스트라이드)
                window_size = 20
                stride = 10
                
                for start in range(0, len(data) - window_size + 1, stride):
                    window = data[start:start + window_size]
                    test_data.append((filename, window, actual_class))
                
            except Exception as e:
                print(f"파일 처리 오류 {filename}: {e}")
                continue
        
        print(f"✅ 테스트 데이터 로드 완료: {len(test_data)}개 윈도우")
        return test_data
    
    def run_comprehensive_test(self, test_data: List[Tuple[str, np.ndarray, str]]):
        """포괄적인 테스트 실행"""
        print("🚀 24개 클래스 포괄적 테스트 시작...")
        
        # 결과 저장
        results = []
        class_stats = {}
        
        # 각 클래스별 통계 초기화
        for class_name in self.label_mapper.class_to_id.keys():
            class_stats[class_name] = {
                'total': 0,
                'correct': 0,
                'predictions': []
            }
        
        # 테스트 실행
        for i, (filename, window_data, actual_class) in enumerate(test_data):
            predicted_class, confidence = self.predict_single_window(window_data)
            
            # 결과 저장
            is_correct = predicted_class == actual_class
            results.append({
                'filename': filename,
                'actual': actual_class,
                'predicted': predicted_class,
                'confidence': confidence,
                'correct': is_correct
            })
            
            # 클래스별 통계 업데이트
            class_stats[actual_class]['total'] += 1
            if is_correct:
                class_stats[actual_class]['correct'] += 1
            class_stats[actual_class]['predictions'].append(predicted_class)
            
            # 진행상황 출력
            if (i + 1) % 100 == 0:
                print(f"  진행률: {i + 1}/{len(test_data)} ({100 * (i + 1) / len(test_data):.1f}%)")
        
        # 결과 분석
        self._analyze_results(results, class_stats)
    
    def _analyze_results(self, results: List[Dict], class_stats: Dict):
        """결과 분석 및 출력"""
        print("\n" + "=" * 80)
        print("📊 24개 클래스 추론 결과 분석")
        print("=" * 80)
        
        # 전체 통계
        total_tests = len(results)
        correct_predictions = sum(1 for r in results if r['correct'])
        overall_accuracy = 100 * correct_predictions / total_tests
        
        print(f"📈 전체 성능:")
        print(f"  총 테스트: {total_tests}개")
        print(f"  정확한 예측: {correct_predictions}개")
        print(f"  전체 정확도: {overall_accuracy:.2f}%")
        
        # 자음/모음별 성능
        consonant_correct = 0
        consonant_total = 0
        vowel_correct = 0
        vowel_total = 0
        
        for class_name, stats in class_stats.items():
            if stats['total'] > 0:
                if self.label_mapper.is_consonant(class_name):
                    consonant_total += stats['total']
                    consonant_correct += stats['correct']
                else:
                    vowel_total += stats['total']
                    vowel_correct += stats['correct']
        
        consonant_accuracy = 100 * consonant_correct / consonant_total if consonant_total > 0 else 0
        vowel_accuracy = 100 * vowel_correct / vowel_total if vowel_total > 0 else 0
        
        print(f"\n🔤 자음 성능:")
        print(f"  총 테스트: {consonant_total}개")
        print(f"  정확한 예측: {consonant_correct}개")
        print(f"  자음 정확도: {consonant_accuracy:.2f}%")
        
        print(f"\n🅰️ 모음 성능:")
        print(f"  총 테스트: {vowel_total}개")
        print(f"  정확한 예측: {vowel_correct}개")
        print(f"  모음 정확도: {vowel_accuracy:.2f}%")
        
        # 클래스별 상세 성능
        print(f"\n📋 클래스별 상세 성능:")
        print("-" * 60)
        print(f"{'클래스':<4} {'타입':<4} {'총수':<6} {'정확':<6} {'정확도':<8} {'주요 오류'}")
        print("-" * 60)
        
        for class_name in sorted(class_stats.keys()):
            stats = class_stats[class_name]
            if stats['total'] > 0:
                accuracy = 100 * stats['correct'] / stats['total']
                char_type = "자음" if self.label_mapper.is_consonant(class_name) else "모음"
                
                # 주요 오류 분석
                error_predictions = [p for p in stats['predictions'] if p != class_name]
                if error_predictions:
                    from collections import Counter
                    most_common_error = Counter(error_predictions).most_common(1)[0]
                    error_info = f"{most_common_error[0]}({most_common_error[1]}회)"
                else:
                    error_info = "없음"
                
                print(f"{class_name:<4} {char_type:<4} {stats['total']:<6} {stats['correct']:<6} {accuracy:<7.1f}% {error_info}")
        
        # 오류 케이스 분석
        print(f"\n❌ 주요 오류 케이스:")
        error_cases = [r for r in results if not r['correct']]
        if error_cases:
            error_summary = {}
            for case in error_cases:
                key = f"{case['actual']} → {case['predicted']}"
                if key not in error_summary:
                    error_summary[key] = 0
                error_summary[key] += 1
            
            for error_pattern, count in sorted(error_summary.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {error_pattern}: {count}회")
        else:
            print("  오류 없음! 🎉")
        
        # 신뢰도 분석
        confidences = [r['confidence'] for r in results]
        correct_confidences = [r['confidence'] for r in results if r['correct']]
        incorrect_confidences = [r['confidence'] for r in results if not r['correct']]
        
        print(f"\n🎯 신뢰도 분석:")
        print(f"  전체 평균 신뢰도: {np.mean(confidences):.3f}")
        print(f"  정확한 예측 평균 신뢰도: {np.mean(correct_confidences):.3f}")
        if incorrect_confidences:
            print(f"  오류 예측 평균 신뢰도: {np.mean(incorrect_confidences):.3f}")
        
        print("=" * 80)
    
    def run_realtime_simulation(self, test_data: List[Tuple[str, np.ndarray, str]], 
                               samples_per_class: int = 5):
        """실시간 추론 시뮬레이션"""
        print(f"\n🎮 실시간 추론 시뮬레이션 (클래스당 {samples_per_class}개 샘플)")
        print("=" * 60)
        
        # 클래스별로 샘플 선택
        class_samples = {}
        for filename, window_data, actual_class in test_data:
            if actual_class not in class_samples:
                class_samples[actual_class] = []
            if len(class_samples[actual_class]) < samples_per_class:
                class_samples[actual_class].append((filename, window_data, actual_class))
        
        # 실시간 시뮬레이션
        total_correct = 0
        total_tests = 0
        
        for class_name in sorted(class_samples.keys()):
            char_type = "자음" if self.label_mapper.is_consonant(class_name) else "모음"
            print(f"\n🔍 {class_name} ({char_type}) 테스트:")
            
            class_correct = 0
            for i, (filename, window_data, actual_class) in enumerate(class_samples[class_name]):
                # 실시간 처리 시뮬레이션
                start_time = time.time()
                predicted_class, confidence = self.predict_single_window(window_data)
                processing_time = (time.time() - start_time) * 1000  # ms
                
                is_correct = predicted_class == actual_class
                if is_correct:
                    class_correct += 1
                    total_correct += 1
                
                total_tests += 1
                
                status = "✅" if is_correct else "❌"
                print(f"  {status} 샘플 {i+1}: {actual_class} → {predicted_class} "
                      f"(신뢰도: {confidence:.3f}, 처리시간: {processing_time:.1f}ms)")
            
            class_accuracy = 100 * class_correct / len(class_samples[class_name])
            print(f"  📊 {class_name} 정확도: {class_accuracy:.1f}% ({class_correct}/{len(class_samples[class_name])})")
        
        overall_accuracy = 100 * total_correct / total_tests
        print(f"\n🎯 전체 실시간 시뮬레이션 결과:")
        print(f"  총 테스트: {total_tests}개")
        print(f"  정확한 예측: {total_correct}개")
        print(f"  전체 정확도: {overall_accuracy:.2f}%")

def main():
    """메인 실행 함수"""
    print("🎯 24개 클래스 (14개 자음 + 10개 모음) 실시간 추론 데모")
    print("=" * 80)
    
    # 추론 시스템 초기화
    try:
        inference = TwentyFourClassInference("best_24class_model.pth")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("먼저 24개 클래스 모델을 학습시켜주세요: python training/train_24classes.py")
        return
    
    # 테스트 데이터 로드
    test_data = inference.load_test_data()
    
    if not test_data:
        print("❌ 테스트 데이터가 없습니다.")
        return
    
    # 1. 포괄적 테스트
    inference.run_comprehensive_test(test_data)
    
    # 2. 실시간 시뮬레이션
    inference.run_realtime_simulation(test_data, samples_per_class=3)
    
    print("\n🎉 24개 클래스 추론 데모 완료!")

if __name__ == "__main__":
    main()
