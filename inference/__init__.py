"""
추론 전용 모듈

이 패키지는 훈련된 SignGlove 모델을 사용한 추론 기능을 제공합니다.
훈련 코드와 완전히 분리되어 있으며, 경량화되고 배포에 최적화되어 있습니다.

지원 모델:
- GRU: 기본 GRU 모델 (98.44% accuracy)
- MS3DGRU: Multi-Scale 3D CNN + GRU (98.78% accuracy) ⭐ 최고 성능
- MS3DStackedGRU: Multi-Scale 3D CNN + Stacked GRU (98.44-98.78% accuracy)
- MSCSGRU: Multi-Scale CNN + Stacked GRU (기존 모델)
"""

from .engine import SignGloveInference, load_inference_engine

__all__ = ['SignGloveInference', 'load_inference_engine']
__version__ = '2.0.0'




