"""
추론 후처리 유틸리티

모델 출력을 사람이 읽을 수 있는 형식으로 변환
"""

import torch
import numpy as np
from typing import List, Dict, Union


class InferencePostprocessor:
    """
    추론용 후처리기
    
    모델 출력(logits)을 클래스 이름, 확률 등으로 변환
    """
    
    # 한국어 수화 자모 클래스 (24개)
    DEFAULT_CLASS_NAMES = [
        'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 
        'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',  # 자음 14개
        'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ'  # 모음 10개
    ]
    
    def __init__(self, class_names: List[str] = None):
        """
        Args:
            class_names: 클래스 이름 리스트 (None이면 기본값 사용)
        """
        self.class_names = class_names if class_names is not None else self.DEFAULT_CLASS_NAMES
        self.num_classes = len(self.class_names)
    
    def logits_to_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        로짓을 확률로 변환
        
        Args:
            logits: 모델 출력 로짓 (batch_size, classes)
        
        Returns:
            probabilities: 확률 (batch_size, classes)
        """
        return torch.softmax(logits, dim=-1)
    
    def logits_to_class(
        self,
        logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        로짓을 클래스 인덱스와 확률로 변환
        
        Args:
            logits: 모델 출력 로짓 (batch_size, classes)
        
        Returns:
            predicted_classes: 예측 클래스 인덱스 (batch_size,)
            confidences: 예측 확률 (batch_size,)
        """
        probabilities = self.logits_to_probabilities(logits)
        confidences, predicted_classes = torch.max(probabilities, dim=-1)
        return predicted_classes, confidences
    
    def format_single_prediction(
        self,
        logits: torch.Tensor,
        top_k: int = 5
    ) -> Dict[str, Union[str, float, List[Dict]]]:
        """
        단일 샘플의 예측 결과를 포맷팅
        
        Args:
            logits: 모델 출력 로짓 (1, classes) 또는 (classes,)
            top_k: 상위 K개 클래스 반환
        
        Returns:
            result: 포맷팅된 예측 결과 딕셔너리
        """
        # 차원 확인
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        # 확률 계산
        probabilities = self.logits_to_probabilities(logits)[0]
        
        # Top-1 예측
        top1_conf, top1_idx = torch.max(probabilities, dim=-1)
        
        # Top-K 예측
        top_k_values, top_k_indices = torch.topk(probabilities, min(top_k, self.num_classes))
        
        top_k_predictions = []
        for i in range(len(top_k_indices)):
            idx = top_k_indices[i].item()
            conf = top_k_values[i].item()
            top_k_predictions.append({
                'class': self.class_names[idx],
                'class_idx': idx,
                'confidence': float(conf)
            })
        
        result = {
            'predicted_class': self.class_names[top1_idx.item()],
            'predicted_class_idx': top1_idx.item(),
            'confidence': float(top1_conf.item()),
            'top_k_predictions': top_k_predictions
        }
        
        return result
    
    def format_batch_predictions(
        self,
        logits: torch.Tensor,
        top_k: int = 5
    ) -> List[Dict[str, Union[str, float, List[Dict]]]]:
        """
        배치 예측 결과를 포맷팅
        
        Args:
            logits: 모델 출력 로짓 (batch_size, classes)
            top_k: 상위 K개 클래스 반환
        
        Returns:
            results: 포맷팅된 예측 결과 리스트
        """
        results = []
        for i in range(logits.shape[0]):
            result = self.format_single_prediction(logits[i], top_k=top_k)
            results.append(result)
        
        return results
    
    def get_class_probabilities(
        self,
        logits: torch.Tensor
    ) -> Dict[str, float]:
        """
        모든 클래스의 확률을 딕셔너리로 반환
        
        Args:
            logits: 모델 출력 로짓 (1, classes) 또는 (classes,)
        
        Returns:
            class_probs: 클래스별 확률 딕셔너리
        """
        # 차원 확인
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        # 확률 계산
        probabilities = self.logits_to_probabilities(logits)[0]
        
        # 딕셔너리 생성
        class_probs = {}
        for idx, class_name in enumerate(self.class_names):
            class_probs[class_name] = float(probabilities[idx].item())
        
        return class_probs
    
    def print_prediction(self, prediction: Dict):
        """
        예측 결과를 보기 좋게 출력
        
        Args:
            prediction: format_single_prediction의 반환값
        """
        print(f"\n🎯 예측 결과:")
        print(f"  예측 클래스: {prediction['predicted_class']}")
        print(f"  확률: {prediction['confidence']:.4f}")
        
        print(f"\n📊 상위 {len(prediction['top_k_predictions'])}개 예측:")
        for i, pred in enumerate(prediction['top_k_predictions'], 1):
            print(f"  {i}. {pred['class']}: {pred['confidence']:.4f}")


# 테스트 코드
if __name__ == "__main__":
    print("🧪 InferencePostprocessor 테스트...")
    
    # 후처리기 생성
    postprocessor = InferencePostprocessor()
    
    # 테스트 로짓
    logits = torch.randn(1, 24)
    
    print("\n📊 단일 샘플 후처리:")
    result = postprocessor.format_single_prediction(logits, top_k=5)
    postprocessor.print_prediction(result)
    
    # 배치 후처리
    print("\n📊 배치 후처리:")
    batch_logits = torch.randn(3, 24)
    batch_results = postprocessor.format_batch_predictions(batch_logits, top_k=3)
    
    for i, result in enumerate(batch_results, 1):
        print(f"\n샘플 {i}:")
        print(f"  예측: {result['predicted_class']} ({result['confidence']:.4f})")
    
    print("\n✅ 테스트 완료!")




