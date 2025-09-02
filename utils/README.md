# 유틸리티 (Utils)

이 폴더는 SignGlove 프로젝트의 유틸리티 도구들을 포함합니다.

## 파일 목록

### `class_discriminator.py`
- **기능**: ㄹ/ㅕ 후처리 필터
- **모델**: Random Forest
- **특성**: Flex5_mean, Flex3_mean
- **목적**: ㄹ과 ㅕ의 혼동 해결
- **정확도**: 100% (훈련 데이터 기준)

### `fix_model_loading.py`
- **기능**: 모델 로딩 문제 해결
- **용도**: 체크포인트 구조 확인 및 수정
- **사용법**: `python3 fix_model_loading.py`

## 사용 방법

### ClassDiscriminator 사용
```python
from utils.class_discriminator import ClassDiscriminator

# 차별화기 초기화
discriminator = ClassDiscriminator()

# 데이터로 훈련
discriminator.train_ml_model(ㄹ_data, ㅕ_data)

# 예측
prediction, confidence = discriminator.predict(features)
```

### 모델 로딩 수정
```bash
# 모델 로딩 문제 진단 및 수정
python3 utils/fix_model_loading.py
```

## 주요 특징

### ClassDiscriminator
- **Random Forest**: 안정적인 분류 성능
- **특성 추출**: Flex 센서 데이터의 평균값 활용
- **후처리 필터**: 메인 모델의 예측 결과 개선
- **높은 정확도**: 100% 훈련 정확도

### fix_model_loading
- **체크포인트 분석**: 모델 파일 구조 확인
- **로딩 수정**: 다양한 체크포인트 형식 지원
- **테스트 추론**: 모델 정상 작동 확인
- **오류 진단**: 상세한 오류 메시지 제공

## 의존성

- **scikit-learn**: Random Forest 분류기
- **torch**: PyTorch 모델 로딩
- **numpy**: 수치 계산
- **pandas**: 데이터 처리



