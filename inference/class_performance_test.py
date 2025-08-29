import torch
import numpy as np
import h5py
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# 추론 시스템 import
from signglove_inference import SignGloveInference

class ClassPerformanceTester:
    """클래스별 인식 성능 테스터"""
    
    def __init__(self, model_path: str, data_path: str):
        """
        Args:
            model_path: 훈련된 모델 파일 경로
            data_path: 테스트 데이터 경로
        """
        self.inference_system = SignGloveInference('../simple_robust_model.pth')
        self.data_path = data_path
        self.label_mapper = self.inference_system.label_mapper
        
        # 결과 저장용
        self.class_results = defaultdict(list)
        self.confusion_matrix = np.zeros((24, 24))
        
    def load_test_data(self) -> Dict[str, List[np.ndarray]]:
        """테스트 데이터 로드"""
        print("📊 테스트 데이터 로딩 중...")
        
        class_data = defaultdict(list)
        
        # 데이터 경로에서 각 클래스별 파일 로드
        data_dir = Path(self.data_path)
        
        for class_dir in data_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                
                # 하위 디렉토리들 (1, 2, 3, 4, 5)에서 H5 파일들 로드
                for sub_dir in class_dir.iterdir():
                    if sub_dir.is_dir():
                        for h5_file in sub_dir.glob("*.h5"):
                            try:
                                with h5py.File(h5_file, 'r') as f:
                                    sensor_data = f['sensor_data'][:]
                                    
                                    # 시퀀스 길이에 맞게 분할
                                    sequence_length = self.inference_system.config['sequence_length']
                                    for i in range(0, len(sensor_data), sequence_length):
                                        if i + sequence_length <= len(sensor_data):
                                            sequence = sensor_data[i:i+sequence_length]
                                            class_data[class_name].append(sequence)
                            except Exception as e:
                                print(f"⚠️ 파일 로드 오류 {h5_file}: {e}")
        
        print(f"📊 로드된 클래스: {list(class_data.keys())}")
        for class_name, sequences in class_data.items():
            print(f"  {class_name}: {len(sequences)}개 시퀀스")
        
        return class_data
    
    def test_single_class(self, class_name: str, test_sequences: List[np.ndarray]) -> Dict:
        """단일 클래스 테스트"""
        print(f"🧪 {class_name} 클래스 테스트 중... ({len(test_sequences)}개 시퀀스)")
        
        correct = 0
        total = len(test_sequences)
        predictions = []
        confidences = []
        
        for i, sequence in enumerate(test_sequences):
            try:
                # 예측 수행
                result = self.inference_system.predict_single(sequence)
                predicted_label = result['predicted_label']
                confidence = result['confidence']
                
                predictions.append(predicted_label)
                confidences.append(confidence)
                
                # 정확도 계산
                if predicted_label == class_name:
                    correct += 1
                
                # 혼동 행렬 업데이트
                true_idx = list(self.label_mapper.values()).index(class_name)
                pred_idx = list(self.label_mapper.values()).index(predicted_label)
                self.confusion_matrix[true_idx][pred_idx] += 1
                
                # 진행률 표시
                if (i + 1) % 10 == 0:
                    print(f"  진행률: {i+1}/{total} ({((i+1)/total)*100:.1f}%)")
                    
            except Exception as e:
                print(f"⚠️ 예측 오류: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'class_name': class_name,
            'total_samples': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'predictions': predictions,
            'confidences': confidences
        }
    
    def test_all_classes(self) -> Dict:
        """모든 클래스 테스트"""
        print("🚀 전체 클래스 성능 테스트 시작!")
        
        # 테스트 데이터 로드
        class_data = self.load_test_data()
        
        if not class_data:
            print("❌ 테스트 데이터를 찾을 수 없습니다!")
            return {}
        
        # 각 클래스별 테스트
        results = {}
        total_correct = 0
        total_samples = 0
        
        for class_name, sequences in class_data.items():
            if len(sequences) > 0:
                result = self.test_single_class(class_name, sequences)
                results[class_name] = result
                
                total_correct += result['correct_predictions']
                total_samples += result['total_samples']
                
                print(f"✅ {class_name}: {result['accuracy']:.3f} ({result['correct_predictions']}/{result['total_samples']})")
        
        # 전체 성능 계산
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return {
            'class_results': results,
            'overall_accuracy': overall_accuracy,
            'total_samples': total_samples,
            'total_correct': total_correct,
            'confusion_matrix': self.confusion_matrix
        }
    
    def generate_performance_report(self, results: Dict) -> str:
        """성능 리포트 생성"""
        if not results:
            return "❌ 테스트 결과가 없습니다."
        
        report = []
        report.append("=" * 60)
        report.append("🎯 SignGlove 클래스별 인식 성능 리포트")
        report.append("=" * 60)
        
        # 전체 성능
        report.append(f"\n📊 전체 성능:")
        report.append(f"  전체 정확도: {results['overall_accuracy']:.3f} ({results['total_correct']}/{results['total_samples']})")
        report.append(f"  총 샘플 수: {results['total_samples']:,}")
        
        # 클래스별 성능
        report.append(f"\n📈 클래스별 성능:")
        report.append("-" * 40)
        
        class_results = results['class_results']
        sorted_classes = sorted(class_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for class_name, result in sorted_classes:
            accuracy = result['accuracy']
            confidence = result['avg_confidence']
            samples = result['total_samples']
            
            # 성능 등급
            if accuracy >= 0.9:
                grade = "🟢 우수"
            elif accuracy >= 0.7:
                grade = "🟡 양호"
            elif accuracy >= 0.5:
                grade = "🟠 보통"
            else:
                grade = "🔴 개선 필요"
            
            report.append(f"  {class_name}: {accuracy:.3f} ({confidence:.3f}) [{samples}개] {grade}")
        
        # 성능 통계
        accuracies = [r['accuracy'] for r in class_results.values()]
        confidences = [r['avg_confidence'] for r in class_results.values()]
        
        report.append(f"\n📊 성능 통계:")
        report.append(f"  평균 정확도: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        report.append(f"  최고 정확도: {np.max(accuracies):.3f}")
        report.append(f"  최저 정확도: {np.min(accuracies):.3f}")
        report.append(f"  평균 신뢰도: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}")
        
        # 문제가 있는 클래스들
        low_accuracy_classes = [name for name, result in class_results.items() if result['accuracy'] < 0.7]
        if low_accuracy_classes:
            report.append(f"\n⚠️ 개선이 필요한 클래스들:")
            for class_name in low_accuracy_classes:
                result = class_results[class_name]
                report.append(f"  {class_name}: {result['accuracy']:.3f} (샘플: {result['total_samples']}개)")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict, output_dir: str = "performance_results"):
        """결과 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # JSON 결과 저장 (numpy 배열을 리스트로 변환)
        json_results = {
            'class_results': results['class_results'],
            'overall_accuracy': float(results['overall_accuracy']),
            'total_samples': int(results['total_samples']),
            'total_correct': int(results['total_correct']),
            'confusion_matrix': results['confusion_matrix'].tolist()
        }
        json_path = output_path / "class_performance_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        # 혼동 행렬 저장
        confusion_path = output_path / "confusion_matrix.npy"
        np.save(confusion_path, results['confusion_matrix'])
        
        # 성능 리포트 저장
        report = self.generate_performance_report(results)
        report_path = output_path / "performance_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"💾 결과 저장 완료: {output_path}")
        return output_path
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, output_path: str = "performance_results"):
        """혼동 행렬 시각화"""
        plt.figure(figsize=(12, 10))
        
        # 정규화된 혼동 행렬
        normalized_cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        normalized_cm = np.nan_to_num(normalized_cm)
        
        # 히트맵 생성
        labels = list(self.label_mapper.values())
        sns.heatmap(normalized_cm, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   xticklabels=labels,
                   yticklabels=labels)
        
        plt.title('SignGlove 클래스별 혼동 행렬', fontsize=16, pad=20)
        plt.xlabel('예측 라벨', fontsize=12)
        plt.ylabel('실제 라벨', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 저장
        plot_path = Path(output_path) / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 혼동 행렬 저장: {plot_path}")
    
    def plot_class_accuracy(self, results: Dict, output_path: str = "performance_results"):
        """클래스별 정확도 시각화"""
        class_results = results['class_results']
        
        # 데이터 준비
        classes = list(class_results.keys())
        accuracies = [class_results[c]['accuracy'] for c in classes]
        confidences = [class_results[c]['avg_confidence'] for c in classes]
        
        # 색상 설정 (정확도에 따라)
        colors = []
        for acc in accuracies:
            if acc >= 0.9:
                colors.append('green')
            elif acc >= 0.7:
                colors.append('orange')
            elif acc >= 0.5:
                colors.append('yellow')
            else:
                colors.append('red')
        
        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 정확도 막대 그래프
        bars1 = ax1.bar(classes, accuracies, color=colors, alpha=0.7)
        ax1.set_title('클래스별 인식 정확도', fontsize=14, pad=20)
        ax1.set_ylabel('정확도', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # 신뢰도 막대 그래프
        bars2 = ax2.bar(classes, confidences, color='skyblue', alpha=0.7)
        ax2.set_title('클래스별 평균 신뢰도', fontsize=14, pad=20)
        ax2.set_ylabel('신뢰도', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # x축 라벨 회전
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 저장
        plot_path = Path(output_path) / "class_accuracy.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 클래스별 정확도 그래프 저장: {plot_path}")

def main():
    """메인 함수"""
    # 경로 설정
    model_path = "../cross_validation_model.pth"
    data_path = "/home/billy/25-1kp/SignGlove/external/SignGlove_HW/datasets/unified"
    
    print("🎯 SignGlove 클래스별 인식 성능 테스트")
    print("=" * 50)
    
    # 테스터 초기화
    tester = ClassPerformanceTester(model_path, data_path)
    
    # 전체 클래스 테스트
    start_time = time.time()
    results = tester.test_all_classes()
    end_time = time.time()
    
    if not results:
        print("❌ 테스트를 완료할 수 없습니다.")
        return
    
    # 결과 출력
    print("\n" + "=" * 50)
    print(tester.generate_performance_report(results))
    print(f"\n⏱️ 테스트 소요 시간: {end_time - start_time:.2f}초")
    
    # 결과 저장
    output_path = tester.save_results(results)
    
    # 시각화
    tester.plot_confusion_matrix(results['confusion_matrix'], str(output_path))
    tester.plot_class_accuracy(results, str(output_path))
    
    print(f"\n🎉 클래스별 성능 테스트 완료!")
    print(f"📁 결과 저장 위치: {output_path}")

if __name__ == "__main__":
    main()
