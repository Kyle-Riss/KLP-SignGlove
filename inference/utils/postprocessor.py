"""
추론 후처리 유틸리티

로짓을 클래스 예측으로 변환, 확률 계산 등
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


# 한국어 수화 자모 클래스
DEFAULT_CLASS_NAMES = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ'
]


class InferencePostprocessor:
    """
    추론 후처리기
    
    로짓을 사람이 읽을 수 있는 예측 결과로 변환
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Args:
            class_names: 클래스 이름 리스트 (None이면 기본값 사용)
        """
        self.class_names = class_names or DEFAULT_CLASS_NAMES
    
    def logits_to_probabilities(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        로짓을 확률로 변환
        
        Args:
            logits: (batch_size, num_classes) 로짓
        
        Returns:
            probabilities: (batch_size, num_classes) 확률
        """
        return F.softmax(logits, dim=-1)
    
    def logits_to_class(
        self,
        logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        로짓에서 최고 확률 클래스와 확률 추출
        
        Args:
            logits: (batch_size, num_classes) 로짓
        
        Returns:
            predicted_class: (batch_size,) 예측 클래스 인덱스
            confidence: (batch_size,) 예측 확률
        """
        probabilities = self.logits_to_probabilities(logits)
        confidence, predicted_class = torch.max(probabilities, dim=-1)
        
        return predicted_class, confidence
    
    def get_top_k_predictions(
        self,
        logits: torch.Tensor,
        k: int = 5
    ) -> List[Dict]:
        """
        상위 K개 예측 반환
        
        Args:
            logits: (batch_size, num_classes) 로짓
            k: 상위 K개
        
        Returns:
            top_k_list: 배치별 상위 K개 예측 리스트
        """
        probabilities = self.logits_to_probabilities(logits)
        batch_size = probabilities.size(0)
        
        top_k_list = []
        
        for i in range(batch_size):
            probs = probabilities[i]
            top_k_probs, top_k_indices = torch.topk(probs, k=min(k, len(probs)))
            
            predictions = []
            for prob, idx in zip(top_k_probs, top_k_indices):
                predictions.append({
                    'class': self.class_names[idx.item()],
                    'class_idx': idx.item(),
                    'confidence': float(prob.item())
                })
            
            top_k_list.append(predictions)
        
        return top_k_list
    
    def format_single_prediction(
        self,
        logits: torch.Tensor,
        top_k: int = 5
    ) -> Dict:
        """
        단일 샘플 예측 결과 포맷팅
        
        Args:
            logits: (1, num_classes) 로짓
            top_k: 상위 K개
        
        Returns:
            result: 예측 결과 딕셔너리
        """
        predicted_class, confidence = self.logits_to_class(logits)
        top_k_predictions = self.get_top_k_predictions(logits, k=top_k)
        
        result = {
            'predicted_class': self.class_names[predicted_class.item()],
            'predicted_class_idx': predicted_class.item(),
            'confidence': float(confidence.item()),
            'top_k_predictions': top_k_predictions[0]
        }
        
        return result
    
    def format_batch_predictions(
        self,
        logits: torch.Tensor,
        top_k: int = 5
    ) -> List[Dict]:
        """
        배치 예측 결과 포맷팅
        
        Args:
            logits: (batch_size, num_classes) 로짓
            top_k: 상위 K개
        
        Returns:
            results: 예측 결과 리스트
        """
        predicted_classes, confidences = self.logits_to_class(logits)
        top_k_predictions_list = self.get_top_k_predictions(logits, k=top_k)
        
        results = []
        for i in range(logits.size(0)):
            result = {
                'predicted_class': self.class_names[predicted_classes[i].item()],
                'predicted_class_idx': predicted_classes[i].item(),
                'confidence': float(confidences[i].item()),
                'top_k_predictions': top_k_predictions_list[i]
            }
            results.append(result)
        
        return results
    
    def get_class_probabilities(
        self,
        logits: torch.Tensor
    ) -> Dict[str, float]:
        """
        모든 클래스의 확률 반환
        
        Args:
            logits: (1, num_classes) 로짓
        
        Returns:
            class_probs: {클래스명: 확률} 딕셔너리
        """
        probabilities = self.logits_to_probabilities(logits)
        probs = probabilities[0]  # 단일 샘플 가정
        
        class_probs = {}
        for i, class_name in enumerate(self.class_names):
            class_probs[class_name] = float(probs[i].item())
        
        return class_probs
    
    def print_prediction(self, prediction: Dict):
        """
        예측 결과를 보기 좋게 출력
        
        Args:
            prediction: format_single_prediction 또는 format_batch_predictions의 반환값
        """
        print("\n" + "="*50)
        print("📊 예측 결과")
        print("="*50)
        print(f"\n🎯 예측 클래스: {prediction['predicted_class']}")
        print(f"📈 확률: {prediction['confidence']:.4f}")
        
        if 'top_k_predictions' in prediction:
            print(f"\n📋 상위 {len(prediction['top_k_predictions'])}개 예측:")
            for i, pred in enumerate(prediction['top_k_predictions'], 1):
                print(f"  {i}. {pred['class']}: {pred['confidence']:.4f}")
        
        if 'input_shape' in prediction:
            print(f"\n📏 입력 shape: {prediction['input_shape']}")
        
        print("="*50 + "\n")


# 테스트 코드
if __name__ == "__main__":
    print("🧪 InferencePostprocessor 테스트...")
    
    # 후처리기 초기화
    postprocessor = InferencePostprocessor()
    
    # 테스트 로짓 (단일 샘플)
    print("\n1️⃣ 단일 샘플 후처리:")
    logits = torch.randn(1, 24)
    result = postprocessor.format_single_prediction(logits, top_k=5)
    postprocessor.print_prediction(result)
    
    # 배치 후처리
    print("\n2️⃣ 배치 후처리:")
    batch_logits = torch.randn(3, 24)
    batch_results = postprocessor.format_batch_predictions(batch_logits, top_k=3)
    
    for i, result in enumerate(batch_results, 1):
        print(f"샘플 {i}: {result['predicted_class']} ({result['confidence']:.4f})")
    
    print("\n✅ 후처리 테스트 완료!")
