"""
추론 전용 모듈

이 패키지는 훈련된 SignGlove 모델을 사용한 추론 기능을 제공합니다.
훈련 코드와 완전히 분리되어 있으며, 경량화되고 배포에 최적화되어 있습니다.
"""

from .engine import SignGloveInference

__all__ = ['SignGloveInference']
__version__ = '1.0.0'



