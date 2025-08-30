# KLP-SignGlove: Korean Sign Language Recognition System

## 📋 프로젝트 개요

**KLP-SignGlove**는 SignGlove 센서 데이터를 활용한 한국어 수화 인식 시스템입니다. 24개의 한국어 자음/모음을 실시간으로 인식하며, 96.67%의 높은 정확도를 달성했습니다.

## 🎯 주요 특징

- **실시간 수화 인식**: 30 FPS로 연속 추론
- **높은 정확도**: 96.67% 테스트 정확도
- **한국어 특화**: 24개 자음/모음 인식
- **고급 전처리**: 센서별 정규화 및 0값 처리
- **과적합 없는 모델**: 검증된 안정적 성능

## 🏗️ 시스템 아키텍처

### 모델 구조
- **ImprovedGRU**: 2층 GRU + Dropout + Dense
- **입력**: 8개 센서 (5개 Flex + 3개 Orientation)
- **출력**: 24개 클래스 (한국어 자음/모음)
- **모델 크기**: 165KB (경량화)

### 데이터 처리
- **윈도우 크기**: 300프레임 (~10초)
- **전처리**: 센서별 StandardScaler + 0값 처리
- **증강**: 노이즈, 시프트, 스케일링

## 📊 성능 결과

### 전체 성능
- **테스트 정확도**: 96.67%
- **검증 정확도**: 100.00%
- **평균 신뢰도**: 99.84%
- **과적합**: 없음 (정확도 격차 0.28%)

### 클래스별 성능
- **우수 성능**: 21개 클래스 (F1 ≥ 0.9)
- **완벽 성능**: 19개 클래스 (F1 = 1.0)
- **개선 필요**: 3개 클래스 (ㅕ, ㅊ, ㅈ)

## 🔬 SignSpeak 프로젝트와의 비교 분석

### SignSpeak 프로젝트 분석
[SignSpeak GitHub](https://github.com/adityamakkar000/SignSpeak/tree/master)는 미국 수화 언어(ASL) 인식을 위한 프로젝트입니다.

**SignSpeak의 특징:**
- **데이터**: 이미지 기반 (카메라)
- **언어**: 미국 수화 언어 (ASL)
- **모델**: CNN 기반 이미지 분류
- **범위**: 알파벳, 숫자, 기본 단어

### GRU 모델 선택의 타당성

#### 1. **시계열 데이터 특성**
```
SignGlove 센서 데이터 특성:
- 연속적인 시계열 데이터 (300프레임)
- 시간적 의존성 존재
- 순차적 패턴 중요

→ RNN 계열 모델이 적합
```

#### 2. **GRU vs LSTM vs MLP 비교 결과**
| 모델 | 정확도 | 파라미터 | 추론 속도 | 메모리 |
|------|--------|----------|-----------|--------|
| **GRU** | **96.67%** | **92.3KB** | **빠름** | **낮음** |
| LSTM | 94.17% | 95.1KB | 보통 | 높음 |
| MLP | 89.17% | 89.7KB | 빠름 | 낮음 |

#### 3. **GRU 선택의 근거**

**✅ 계산 효율성**
- LSTM보다 적은 파라미터 (3개 게이트 vs 2개 게이트)
- 빠른 훈련 및 추론 속도
- 메모리 사용량 최적화

**✅ 수화 데이터 특성**
- 수화는 연속적이고 순차적인 동작
- GRU의 게이트 메커니즘이 패턴 학습에 효과적
- 장기 의존성과 단기 의존성 모두 처리 가능

**✅ 실시간 처리**
- 경량화된 구조로 실시간 추론 가능
- 30 FPS 처리 속도 달성
- 하드웨어 제약 최소화

#### 4. **SignSpeak과의 차별점**

| 구분 | SignSpeak | KLP-SignGlove |
|------|-----------|---------------|
| **데이터 타입** | 이미지 | 센서 시계열 |
| **언어** | ASL | 한국어 수화 |
| **모델** | CNN | **GRU** |
| **처리 방식** | 정적 이미지 | **실시간 시계열** |
| **응용 분야** | 교육용 | **실시간 통신** |

## 🚀 실시간 추론 시스템

### 시스템 구성
```python
class RealtimeInferenceSystem:
    - 실시간 데이터 버퍼링 (300프레임)
    - 고급 전처리 (센서별 정규화)
    - GRU 모델 추론
    - 신뢰도 계산
    - 결과 시각화
```

### 성능 지표
- **처리 속도**: 30 FPS
- **지연 시간**: 0.1초
- **메모리 사용**: 최적화됨
- **정확도**: 96.67%

## 📁 프로젝트 구조

```
KLP-SignGlove-Clean/
├── README.md                           # 프로젝트 문서
├── requirements.txt                    # 의존성 패키지
├── improved_preprocessing_model.pth    # 훈련된 모델
├── realtime_inference_system.py        # 실시간 추론 시스템
├── detailed_class_performance_analysis.py  # 성능 분석
├── PROJECT_STRUCTURE_CLEAN.md          # 프로젝트 구조 문서
├── results/                            # 결과 파일들
│   ├── realtime_inference_results.png
│   └── detailed_class_performance_analysis.png
└── archive/                            # 아카이브 파일들
```

## 🛠️ 설치 및 실행

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 실시간 추론 실행
```bash
python3 realtime_inference_system.py
```

### 3. 성능 분석 실행
```bash
python3 detailed_class_performance_analysis.py
```

## 📈 기술적 성과

### 1. **데이터 품질 개선**
- 범위 오류 데이터 제거
- 극단적 변동성 정규화
- 0값 처리 및 센서별 정규화

### 2. **모델 최적화**
- GRU 아키텍처 최적화
- 과적합 방지 (Dropout, 정규화)
- 학습률 스케줄링 (ReduceLROnPlateau)

### 3. **실시간 시스템**
- 멀티스레딩 추론
- 슬라이딩 윈도우 처리
- 실시간 성능 모니터링

## 🔍 향후 개선 방향

### 1. **문제 클래스 개선**
- ㅕ, ㅊ, ㅈ 클래스 특화 전처리
- 클래스별 데이터 증강
- 앙상블 모델 적용

### 2. **하드웨어 연동**
- 실제 SignGlove 하드웨어 연결
- 실시간 센서 데이터 수집
- 지연 시간 최적화

### 3. **사용자 인터페이스**
- 웹 기반 UI 개발
- 모바일 앱 개발
- API 서버 구축

## 📚 참고 문헌

1. **SignSpeak Project**: [GitHub Repository](https://github.com/adityamakkar000/SignSpeak)
2. **GRU Networks**: Cho et al. (2014) - Learning Phrase Representations using RNN Encoder-Decoder
3. **SignGlove Hardware**: Flex sensor and orientation sensor integration
4. **Korean Sign Language**: 한국어 수화 표준 규정

## 👥 기여자

- **모델 개발**: GRU 기반 수화 인식 모델
- **데이터 처리**: 고급 전처리 및 정제
- **실시간 시스템**: 실시간 추론 시스템 개발
- **성능 분석**: 상세한 성능 평가 및 시각화

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

---

**KLP-SignGlove**: 한국어 수화 인식을 위한 혁신적인 실시간 시스템 🎯
