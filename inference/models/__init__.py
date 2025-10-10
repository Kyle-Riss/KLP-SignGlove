"""
추론 전용 모델들

훈련 관련 코드가 제거된 경량화된 모델 구현
"""

from .mscsgru_inference import MSCSGRUInference

__all__ = ['MSCSGRUInference']




