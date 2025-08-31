# SignGlove 프로젝트 요약

## 🎯 프로젝트 목표
SignGlove는 데이터 장갑을 사용하여 한글 자음/모음을 실시간으로 인식하는 딥러닝 시스템입니다.

## 📊 최종 성과
- **평균 신뢰도**: 86.3%
- **처리 속도**: 0.7ms
- **인식 문자**: 24개 (ㄱ,ㄴ,ㄷ,ㄹ,ㅁ,ㅂ,ㅅ,ㅇ,ㅈ,ㅊ,ㅋ,ㅌ,ㅍ,ㅎ,ㅏ,ㅑ,ㅓ,ㅕ,ㅗ,ㅛ,ㅜ,ㅠ,ㅡ,ㅣ)
- **모델 정확도**: 94.78%

## 🏗️ 핵심 구성 요소

### 1. 모델 아키텍처
- **RGRU**: 메인 분류 모델
- **ClassDiscriminator**: ㄹ/ㅕ 후처리 필터
- **입력**: 8차원 센서 데이터 (5개 Flex + 3개 IMU)
- **출력**: 24개 클래스 확률 분포

### 2. 데이터 처리
- **정규화**: Flex 센서 (0-1023 → 0-1), IMU 센서 (절댓값 후 정규화)
- **시퀀스 길이**: 300 (패딩/자르기)
- **데이터 분할**: 훈련 60%, 검증 20%, 테스트 20%

### 3. 핵심 파일
- `improved_model_architecture.py`: 모델 정의
- `improved_training.py`: 훈련 시스템
- `corrected_filtered_inference.py`: 추론 시스템
- `class_discriminator.py`: 후처리 필터
- `analyze_inference_accuracy.py`: 정확도 분석

## 🚀 사용 방법

### 환경 설정
```bash
pip install -r requirements.txt
```

### 모델 훈련
```bash
python3 improved_training.py
```

### 추론 실행
```bash
python3 corrected_filtered_inference.py
```

### 결과 분석
```bash
python3 analyze_inference_accuracy.py
```

## 📈 성능 분석

### 클래스별 성능
- **높은 정확도 (>90%)**: 11개 클래스
- **중간 정확도 (80-90%)**: 6개 클래스
- **낮은 정확도 (<80%)**: 7개 클래스

### 전체 성능
- **높은 신뢰도 (>80%)**: 73.7%
- **중간 신뢰도 (60-80%)**: 16.0%
- **낮은 신뢰도 (<60%)**: 10.3%

## 🔧 기술적 특징

### 정규화 기법
- **LayerNorm**: 입력 및 중간 레이어 정규화
- **Dropout**: 과적합 방지 (0.5)
- **Weight Decay**: L2 정규화

### 모델 구조
- **GRU**: 시퀀스 데이터 처리
- **Attention**: 중요 시점 집중
- **Multi-layer Classifier**: 복잡한 패턴 학습

## 📁 파일 구조

### 핵심 파일
```
KLP-SignGlove-Clean/
├── README.md                           # 프로젝트 문서
├── improved_model_architecture.py      # 모델 아키텍처
├── improved_training.py                # 훈련 시스템
├── corrected_filtered_inference.py     # 추론 시스템
├── class_discriminator.py              # 후처리 필터
├── analyze_inference_accuracy.py       # 정확도 분석
├── real_training_curves.py             # 학습 곡선
├── fix_model_loading.py                # 모델 로딩 수정
├── requirements.txt                    # 의존성 목록
└── models/
    └── improved_regularized_model.pth  # 훈련된 모델
```

### 데이터 폴더
- `real_data/`: 원시 센서 데이터
- `real_data_filtered/`: 정규화된 데이터

### 결과 파일
- `corrected_filtered_results.json`: 추론 결과
- `real_training_curves.png`: 학습 곡선
- `inference_accuracy_analysis.png`: 정확도 분석
- `real_training_report.txt`: 훈련 보고서

## 🎉 주요 성과

### 1. 과적합 완전 해결
- **과적합 지수**: 0.1766 (매우 낮음)
- **일반화 성능**: 우수
- **안정적인 학습**: 일관된 성능

### 2. 실용적인 성능
- **빠른 추론**: 0.7ms 평균 처리 시간
- **높은 정확도**: 86.3% 평균 신뢰도
- **안정적인 결과**: 모든 파일에서 예측 성공

### 3. 확장 가능한 구조
- **모듈화**: 각 구성 요소 독립적
- **재사용성**: 다양한 데이터셋 적용 가능
- **유지보수성**: 명확한 코드 구조

## 🔮 향후 개선 방향

### 1. 성능 개선
- **낮은 정확도 클래스**: 추가 데이터 수집
- **후처리 필터**: 더 많은 클래스 쌍 적용
- **앙상블 모델**: 여러 모델 조합

### 2. 기능 확장
- **실시간 추론**: 웹 인터페이스
- **모바일 지원**: 경량화 모델
- **다국어 지원**: 다른 언어 확장

### 3. 사용성 개선
- **API 서버**: RESTful API 제공
- **문서화**: 상세한 사용 가이드
- **테스트**: 자동화된 테스트 시스템

---

**SignGlove: 한글 자음/모음 인식의 새로운 패러다임** 🚀
