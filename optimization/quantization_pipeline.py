"""
KLP-SignGlove 모델 양자화 파이프라인
제공된 워크플로우 다이어그램 기반 구현

Phase 1: 모델 준비 및 기준 성능 측정
Phase 2: 양자화 실행 (PTQ)
Phase 3: 양자화 모델 평가 및 비교
Phase 4: 배포
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.utils.data import DataLoader, Subset
import torch.onnx
import time
import os
import json
import numpy as np
from collections import OrderedDict
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# 프로젝트 모듈 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.deep_learning import DeepLearningPipeline
from training.dataset import KSLCsvDataset
from training.train_deep_learning import DeepLearningTrainer

class ModelQuantizationPipeline:
    """모델 양자화 파이프라인"""
    
    def __init__(self, 
                 model_path: str = 'best_dl_model.pth',
                 csv_dir: str = 'integrations/SignGlove_HW',
                 output_dir: str = 'optimization/quantized_models'):
        """
        Args:
            model_path: FP32 모델 경로
            csv_dir: 보정용 데이터셋 경로
            output_dir: 양자화된 모델 저장 경로
        """
        self.model_path = model_path
        self.csv_dir = csv_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 장치 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  사용 장치: {self.device}")
        
        # 모델 설정
        self.model_config = {
            'input_features': 8,
            'sequence_length': 20,
            'num_classes': 5,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.3
        }
        
        # 양자화 백엔드 설정 (M1 Mac 호환)
        try:
            torch.backends.quantized.engine = 'fbgemm'  # x86 CPU 최적화
        except RuntimeError:
            torch.backends.quantized.engine = 'qnnpack'  # ARM/Mobile 최적화
        
        print(f"🔧 양자화 백엔드: {torch.backends.quantized.engine}")
        
        # 결과 저장용
        self.results = {
            'fp32': {},
            'int8_ptq': {},
            'int8_qat': {}
        }
        
    def phase1_model_preparation(self) -> Tuple[nn.Module, Dict]:
        """
        Phase 1: 모델 준비 및 기준 성능 측정
        """
        print("\n" + "="*60)
        print("📀 Phase 1: 모델 준비 및 기준 성능 측정")
        print("="*60)
        
        # 1. FP32 모델 로드
        print("📥 FP32 모델 로드 중...")
        fp32_model = DeepLearningPipeline(**self.model_config)
        
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            fp32_model.load_state_dict(state_dict)
            print(f"✅ 모델 로드 성공: {self.model_path}")
        else:
            print(f"⚠️  사전 훈련된 모델이 없습니다. 기본 초기화 모델 사용: {self.model_path}")
        
        fp32_model.to(self.device)
        fp32_model.eval()
        
        # 2. 데이터 준비
        print("📊 보정 데이터셋 준비 중...")
        dataset = KSLCsvDataset(
            self.csv_dir,
            window_size=20,
            stride=10
        )
        
        # 데이터셋을 train/val/test로 분할 (8:1:1)
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        
        # 보정용 데이터 (validation set 사용)
        calibration_indices = list(range(train_size, train_size + val_size))
        calibration_dataset = Subset(dataset, calibration_indices)
        
        # 테스트용 데이터
        test_indices = list(range(train_size + val_size, total_size))
        test_dataset = Subset(dataset, test_indices)
        
        self.calibration_loader = DataLoader(
            calibration_dataset, batch_size=16, shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=16, shuffle=False
        )
        
        print(f"📈 데이터셋 정보:")
        print(f"  전체: {total_size}개")
        print(f"  보정용: {len(calibration_dataset)}개")
        print(f"  테스트: {len(test_dataset)}개")
        
        # 3. 기준 성능 평가
        print("🎯 FP32 모델 기준 성능 평가 중...")
        fp32_metrics = self._evaluate_model(fp32_model, "FP32")
        
        self.results['fp32'] = fp32_metrics
        
        print(f"\n📊 FP32 기준 성능:")
        print(f"  정확도: {fp32_metrics['accuracy']:.2f}%")
        print(f"  모델 크기: {fp32_metrics['model_size_mb']:.2f} MB")
        print(f"  추론 속도: {fp32_metrics['inference_time_ms']:.2f} ms")
        print(f"  FPS: {fp32_metrics['fps']:.1f}")
        
        return fp32_model, fp32_metrics
    
    def phase2_post_training_quantization(self, fp32_model: nn.Module) -> nn.Module:
        """
        Phase 2: Post-Training Quantization (PTQ) 실행
        """
        print("\n" + "="*60)
        print("🔬 Phase 2: Post-Training Quantization (PTQ)")
        print("="*60)
        
        # 1. 양자화 설정 적용
        print("⚙️  양자화 설정 적용 중...")
        
        # 모델 복사 (원본 보존)
        ptq_model = DeepLearningPipeline(**self.model_config)
        ptq_model.load_state_dict(fp32_model.state_dict())
        ptq_model.eval()
        
        # 양자화 설정 (백엔드에 맞게)
        backend = torch.backends.quantized.engine
        qconfig = torch.quantization.get_default_qconfig(backend)
        ptq_model.qconfig = qconfig
        
        # 양자화 준비
        torch.quantization.prepare(ptq_model, inplace=True)
        
        print(f"✅ 양자화 설정 완료")
        print(f"  QConfig: {qconfig}")
        
        # 2. 모델 보정 (Calibration)
        print("🔍 모델 보정 중...")
        print("  대표 데이터셋으로 활성화 값 통계 정보 수집...")
        
        calibration_samples = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(self.calibration_loader):
                data = data.to(torch.float32)
                
                # 보정 실행
                _ = ptq_model(data)
                calibration_samples += data.size(0)
                
                # 진행률 표시
                if batch_idx % 10 == 0:
                    print(f"    보정 진행: {calibration_samples}개 샘플")
        
        print(f"✅ 보정 완료: 총 {calibration_samples}개 샘플 사용")
        
        # 3. INT8 모델 변환
        print("🔄 INT8 모델 변환 중...")
        int8_model = torch.quantization.convert(ptq_model, inplace=False)
        
        print("✅ INT8 변환 완료")
        
        # 4. 양자화된 모델 저장
        quantized_model_path = self.output_dir / 'int8_ptq_model.pth'
        torch.save(int8_model.state_dict(), quantized_model_path)
        print(f"💾 양자화 모델 저장: {quantized_model_path}")
        
        return int8_model
    
    def phase3_evaluation_and_comparison(self, 
                                       fp32_model: nn.Module, 
                                       int8_model: nn.Module) -> Dict:
        """
        Phase 3: 양자화 모델 평가 및 비교
        """
        print("\n" + "="*60)
        print("📊 Phase 3: 양자화 모델 평가 및 비교")
        print("="*60)
        
        # 1. INT8 모델 평가
        print("🎯 INT8 모델 성능 평가 중...")
        int8_metrics = self._evaluate_model(int8_model, "INT8-PTQ")
        
        self.results['int8_ptq'] = int8_metrics
        
        # 2. 성능 비교 분석
        print("⚖️  성능 비교 분석:")
        
        fp32_acc = self.results['fp32']['accuracy']
        int8_acc = self.results['int8_ptq']['accuracy']
        accuracy_drop = fp32_acc - int8_acc
        
        fp32_size = self.results['fp32']['model_size_mb']
        int8_size = self.results['int8_ptq']['model_size_mb']
        size_reduction = (fp32_size - int8_size) / fp32_size * 100
        
        fp32_time = self.results['fp32']['inference_time_ms']
        int8_time = self.results['int8_ptq']['inference_time_ms']
        speed_improvement = (fp32_time - int8_time) / fp32_time * 100
        
        comparison = {
            'accuracy_drop_percent': accuracy_drop,
            'size_reduction_percent': size_reduction,
            'speed_improvement_percent': speed_improvement,
            'compression_ratio': fp32_size / int8_size,
            'needs_qat': accuracy_drop > 5.0  # 5% 이상 정확도 하락시 QAT 필요
        }
        
        print(f"\n📈 비교 결과:")
        print(f"  정확도 변화: {accuracy_drop:+.2f}% ({fp32_acc:.2f}% → {int8_acc:.2f}%)")
        print(f"  모델 크기 감소: {size_reduction:.1f}% ({fp32_size:.2f}MB → {int8_size:.2f}MB)")
        print(f"  속도 개선: {speed_improvement:+.1f}% ({fp32_time:.2f}ms → {int8_time:.2f}ms)")
        print(f"  압축 비율: {comparison['compression_ratio']:.1f}x")
        
        if comparison['needs_qat']:
            print(f"⚠️  정확도 하락이 {accuracy_drop:.1f}%로 큽니다. QAT 고려 필요")
        else:
            print(f"✅ 정확도 하락이 허용 범위({accuracy_drop:.1f}%) 내입니다.")
        
        self.results['comparison'] = comparison
        
        return comparison
    
    def phase4_deployment_export(self, int8_model: nn.Module):
        """
        Phase 4: 배포를 위한 모델 내보내기
        """
        print("\n" + "="*60)
        print("🚀 Phase 4: 배포용 모델 내보내기")
        print("="*60)
        
        try:
            # 1. ONNX 변환
            print("📦 ONNX 형식으로 변환 중...")
            
            # 더미 입력 생성
            dummy_input = torch.randn(1, 20, 8)  # (batch, sequence, features)
            
            onnx_path = self.output_dir / 'int8_model.onnx'
            
            # ONNX 내보내기 (양자화된 모델)
            torch.onnx.export(
                int8_model,
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
            
            print(f"✅ ONNX 변환 완료: {onnx_path}")
            print(f"📱 엣지 배포 준비:")
            print(f"  - Raspberry Pi: ONNX Runtime 사용")
            print(f"  - Arduino/MCU: TensorFlow Lite Micro 변환 필요")
            
        except Exception as e:
            print(f"❌ ONNX 변환 실패: {e}")
            print("🔄 대안: PyTorch Mobile 형식으로 저장")
            
            # PyTorch Mobile 형식으로 저장
            mobile_model = torch.jit.script(int8_model)
            mobile_path = self.output_dir / 'int8_model_mobile.pt'
            mobile_model.save(str(mobile_path))
            print(f"✅ PyTorch Mobile 모델 저장: {mobile_path}")
        
        # 2. 배포 정보 파일 생성
        deployment_info = {
            'model_info': {
                'type': 'INT8-PTQ',
                'input_shape': [1, 20, 8],
                'output_classes': ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ'],
                'preprocessing': {
                    'flex_normalization': '0-1024 → 0-1',
                    'orientation_normalization': '-180~180 → -1~1'
                }
            },
            'performance': self.results['int8_ptq'],
            'comparison': self.results.get('comparison', {}),
            'deployment_targets': {
                'raspberry_pi': {
                    'runtime': 'ONNX Runtime',
                    'expected_fps': f"{self.results['int8_ptq']['fps']:.1f}",
                    'memory_usage': f"{self.results['int8_ptq']['model_size_mb']:.1f} MB"
                },
                'mobile': {
                    'runtime': 'PyTorch Mobile',
                    'file': 'int8_model_mobile.pt'
                }
            }
        }
        
        info_path = self.output_dir / 'deployment_info.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(deployment_info, f, indent=2, ensure_ascii=False)
        
        print(f"📋 배포 정보 저장: {info_path}")
    
    def _evaluate_model(self, model: nn.Module, model_name: str) -> Dict:
        """모델 성능 평가"""
        model.eval()
        
        correct = 0
        total = 0
        inference_times = []
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data = data.to(torch.float32)
                
                # 추론 시간 측정
                start_time = time.time()
                outputs = model(data)
                inference_time = (time.time() - start_time) * 1000  # ms
                
                if isinstance(outputs, dict):
                    logits = outputs['class_logits']
                else:
                    logits = outputs
                
                _, predicted = torch.max(logits.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                inference_times.append(inference_time)
        
        accuracy = 100 * correct / total
        avg_inference_time = np.mean(inference_times)
        fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
        
        # 모델 크기 계산
        if hasattr(model, 'state_dict'):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            model_size_mb = (param_size + buffer_size) / 1024 / 1024
        else:
            model_size_mb = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'model_size_mb': model_size_mb,
            'inference_time_ms': avg_inference_time,
            'fps': fps,
            'total_samples': total,
            'correct_predictions': correct
        }
        
        return metrics
    
    def optional_quantization_aware_training(self, fp32_model: nn.Module) -> Optional[nn.Module]:
        """
        Optional: Quantization-Aware Training (QAT)
        정확도 하락이 큰 경우에만 실행
        """
        print("\n" + "="*60)
        print("⚙️  Optional: Quantization-Aware Training (QAT)")
        print("="*60)
        
        if not self.results.get('comparison', {}).get('needs_qat', False):
            print("✅ PTQ 결과가 만족스러워 QAT를 건너뜁니다.")
            return None
        
        print("🔄 QAT 적용 중...")
        
        # QAT 모델 준비
        qat_model = DeepLearningPipeline(**self.model_config)
        qat_model.load_state_dict(fp32_model.state_dict())
        qat_model.train()
        
        # QAT 설정 (백엔드에 맞게)
        backend = torch.backends.quantized.engine
        qat_model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        torch.quantization.prepare_qat(qat_model, inplace=True)
        
        # 간단한 파인튜닝 (실제로는 더 정교한 학습 필요)
        optimizer = torch.optim.Adam(qat_model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        
        print("🎓 QAT 파인튜닝 중...")
        qat_model.train()
        for epoch in range(5):  # 적은 에포크로 파인튜닝
            for batch_idx, (data, targets) in enumerate(self.calibration_loader):
                data = data.to(torch.float32)
                
                optimizer.zero_grad()
                outputs = qat_model(data)
                loss = criterion(outputs['class_logits'], targets)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 10 == 0:
                    print(f"  에포크 {epoch+1}/5, 배치 {batch_idx}, 손실: {loss.item():.4f}")
        
        # QAT 모델을 INT8로 변환
        qat_model.eval()
        int8_qat_model = torch.quantization.convert(qat_model, inplace=False)
        
        # QAT 모델 평가
        qat_metrics = self._evaluate_model(int8_qat_model, "INT8-QAT")
        self.results['int8_qat'] = qat_metrics
        
        print(f"✅ QAT 완료")
        print(f"  QAT 정확도: {qat_metrics['accuracy']:.2f}%")
        
        # QAT 모델 저장
        qat_model_path = self.output_dir / 'int8_qat_model.pth'
        torch.save(int8_qat_model.state_dict(), qat_model_path)
        print(f"💾 QAT 모델 저장: {qat_model_path}")
        
        return int8_qat_model
    
    def generate_summary_report(self):
        """종합 보고서 생성"""
        print("\n" + "="*60)
        print("📋 양자화 파이프라인 종합 보고서")
        print("="*60)
        
        # 결과 시각화
        self._plot_comparison_results()
        
        # 텍스트 보고서
        report_path = self.output_dir / 'quantization_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("KLP-SignGlove 모델 양자화 보고서\n")
            f.write("="*50 + "\n\n")
            
            f.write("📊 성능 비교:\n")
            for model_type, metrics in self.results.items():
                if model_type != 'comparison' and metrics:
                    f.write(f"\n{model_type.upper()}:\n")
                    f.write(f"  정확도: {metrics['accuracy']:.2f}%\n")
                    f.write(f"  모델 크기: {metrics['model_size_mb']:.2f} MB\n")
                    f.write(f"  추론 속도: {metrics['inference_time_ms']:.2f} ms\n")
                    f.write(f"  FPS: {metrics['fps']:.1f}\n")
            
            if 'comparison' in self.results:
                comp = self.results['comparison']
                f.write(f"\n📈 개선 효과:\n")
                f.write(f"  모델 크기 감소: {comp['size_reduction_percent']:.1f}%\n")
                f.write(f"  속도 개선: {comp['speed_improvement_percent']:+.1f}%\n")
                f.write(f"  압축 비율: {comp['compression_ratio']:.1f}x\n")
                f.write(f"  정확도 변화: {comp['accuracy_drop_percent']:+.2f}%\n")
        
        print(f"📄 보고서 저장: {report_path}")
        
        # JSON 결과 저장
        results_path = self.output_dir / 'quantization_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 결과 데이터 저장: {results_path}")
    
    def _plot_comparison_results(self):
        """결과 비교 시각화"""
        if len(self.results) < 2:
            return
        
        # 데이터 준비
        models = []
        accuracies = []
        sizes = []
        speeds = []
        
        for model_type, metrics in self.results.items():
            if model_type != 'comparison' and metrics:
                models.append(model_type.upper())
                accuracies.append(metrics['accuracy'])
                sizes.append(metrics['model_size_mb'])
                speeds.append(metrics['fps'])
        
        # 시각화
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 정확도 비교
        axes[0].bar(models, accuracies, color=['blue', 'orange', 'green'][:len(models)])
        axes[0].set_title('모델 정확도 비교')
        axes[0].set_ylabel('정확도 (%)')
        for i, v in enumerate(accuracies):
            axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center')
        
        # 모델 크기 비교
        axes[1].bar(models, sizes, color=['blue', 'orange', 'green'][:len(models)])
        axes[1].set_title('모델 크기 비교')
        axes[1].set_ylabel('크기 (MB)')
        for i, v in enumerate(sizes):
            axes[1].text(i, v + 0.05, f'{v:.2f}MB', ha='center')
        
        # FPS 비교
        axes[2].bar(models, speeds, color=['blue', 'orange', 'green'][:len(models)])
        axes[2].set_title('추론 속도 비교')
        axes[2].set_ylabel('FPS')
        for i, v in enumerate(speeds):
            axes[2].text(i, v + 5, f'{v:.1f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quantization_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_full_pipeline(self):
        """전체 양자화 파이프라인 실행"""
        print("🚀 KLP-SignGlove 모델 양자화 파이프라인 시작")
        print("📋 워크플로우: Phase 1 → Phase 2 → Phase 3 → Phase 4")
        
        try:
            # Phase 1: 모델 준비 및 기준 성능 측정
            fp32_model, fp32_metrics = self.phase1_model_preparation()
            
            # Phase 2: Post-Training Quantization
            int8_ptq_model = self.phase2_post_training_quantization(fp32_model)
            
            # Phase 3: 양자화 모델 평가 및 비교
            comparison = self.phase3_evaluation_and_comparison(fp32_model, int8_ptq_model)
            
            # Optional: QAT (필요한 경우에만)
            int8_qat_model = self.optional_quantization_aware_training(fp32_model)
            
            # Phase 4: 배포용 모델 내보내기
            final_model = int8_qat_model if int8_qat_model is not None else int8_ptq_model
            self.phase4_deployment_export(final_model)
            
            # 종합 보고서 생성
            self.generate_summary_report()
            
            print("\n🎉 양자화 파이프라인 완료!")
            print(f"📁 결과 저장 위치: {self.output_dir}")
            
            return self.results
            
        except Exception as e:
            print(f"❌ 파이프라인 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    # 양자화 파이프라인 실행
    pipeline = ModelQuantizationPipeline(
        model_path='best_dl_model.pth',
        csv_dir='integrations/SignGlove_HW'
    )
    
    results = pipeline.run_full_pipeline()
    
    if results:
        print("\n✅ 양자화 파이프라인 성공적으로 완료!")
        
        # 주요 결과 요약
        if 'comparison' in results:
            comp = results['comparison']
            print(f"📈 주요 개선 효과:")
            print(f"  🗜️  모델 크기: {comp['size_reduction_percent']:.1f}% 감소")
            print(f"  ⚡ 추론 속도: {comp['speed_improvement_percent']:+.1f}% 개선")
            print(f"  🎯 정확도 변화: {comp['accuracy_drop_percent']:+.2f}%")
    else:
        print("❌ 양자화 파이프라인 실패")
