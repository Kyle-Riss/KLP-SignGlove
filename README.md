# 🤟 SignGlove_HW - KSL 실시간 인식 시스템

**한국수어(KSL) 인식을 위한 완전한 End-to-End 딥러닝 시스템**

SignGlove 하드웨어와 연동하여 손가락 움직임과 손목 방향을 실시간으로 인식하고, 딥러닝/머신러닝을 통해 한국수어를 분류하는 완전한 솔루션입니다. 🚀

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🚀 주요 기능

### ✨ 핵심 성능
- **실시간 추론**: 562 FPS (1.8ms 평균 추론 시간)
- **높은 정확도**: 테스트 정확도 91.67%
- **다중 모델 지원**: 머신러닝, 딥러닝, 규칙 기반 분류기
- **실시간 TTS**: macOS 한국어 음성 합성 지원

### 🎯 인식 가능한 한국수어
**현재 버전**: 한국수어 자음 5개 + 확장 가능한 34개 클래스 아키텍처
- **ㄱ (기역)**: precision=1.00, recall=1.00, f1=1.00
- **ㄴ (니은)**: precision=0.88, recall=1.00, f1=0.93
- **ㄷ (디귿)**: precision=0.80, recall=0.89, f1=0.84
- **ㄹ (리을)**: precision=1.00, recall=0.69, f1=0.81
- **ㅁ (미음)**: precision=0.94, recall=1.00, f1=0.97

**확장 지원**: 자음 14개 + 모음 10개 + 숫자 10개 (총 34클래스 설계)

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SignGlove     │    │   전처리 &      │    │   딥러닝 모델   │
│   하드웨어      │────│   특징 추출     │────│   (CNN+LSTM)    │
│                 │    │                 │    │                 │
│ • Flex 센서 x5  │    │ • 정규화        │    │ • Attention     │
│ • IMU (3축)     │    │ • 필터링        │    │ • 91.67% 정확도 │
│ • 실시간 스트림 │    │ • 윈도우링      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│      TTS        │    │   실시간 추론   │◄────────────┘
│     엔진        │◄───│   파이프라인    │
│                 │    │                 │
│ • 한국어 발음   │    │ • 안정성 체크   │
│ • 신뢰도 기반   │    │ • 성능 모니터링 │
│ • 비동기 출력   │    │ • 콜백 시스템   │
└─────────────────┘    └─────────────────┘
```

## 🚀 빠른 시작

### 1. 저장소 클론 및 환경 설정
```bash
git clone https://github.com/Kyle-Riss/KLP-SignGlove.git
cd KLP-SignGlove/ksl_project
pip install -r requirements.txt
```

### 2. 🎮 즉시 데모 실행 (시뮬레이션 모드)
```bash
# 실시간 딥러닝 데모 실행
PYTHONPATH=/path/to/ksl_project python realtime_demo.py \
    --model best_dl_model.pth \
    --connection simulation \
    --confidence 0.7

# 통합 시스템 테스트
PYTHONPATH=/path/to/ksl_project python test_realtime.py
```

### 3. 🔌 실제 하드웨어 연결
```python
# UART/USB 시리얼 연결
python realtime_demo.py --connection uart --model best_dl_model.pth

# WiFi/TCP 연결  
python realtime_demo.py --connection wifi --model best_dl_model.pth

# 시뮬레이션 모드 (개발/테스트)
python realtime_demo.py --connection simulation --model best_dl_model.pth
```

## 🎮 사용법

### 1. 모델 학습
```bash
# 딥러닝 모델 학습 (권장)
python training/train_deep_learning.py --csv_dir integrations/SignGlove_HW --epochs 50

# 머신러닝 모델 학습
python training/train_classical_ml.py --csv_dir integrations/SignGlove_HW
```

### 2. 실시간 데모 실행
```bash
# 시뮬레이션 모드로 데모 실행
python realtime_demo.py --model best_dl_model.pth --connection simulation

# 실제 하드웨어 연결
python realtime_demo.py --model best_dl_model.pth --connection uart
```

### 3. 시스템 테스트
```bash
# 전체 시스템 통합 테스트
python test_realtime.py
```

## 📊 성능 벤치마크 & 실험 결과

### 🏆 모델 성능 비교 (완전 구현 완료)
| 모델 | 테스트 정확도 | 추론 시간 | 특징 | 상태 |
|------|--------------|-----------|------|------|
| **🥇 딥러닝 (CNN+LSTM+Attention)** | **91.67%** | **1.8ms** | 시계열 학습, Attention | ✅ **완료** |
| 🥈 XGBoost | 88.89% | 0.5ms | 빠른 추론, 해석 가능 | ✅ 완료 |
| 🥉 Random Forest | 86.11% | 0.3ms | 안정적, 경량 | ✅ 완료 |
| Rule-based | 75.00% | 0.1ms | 단순, 확장성 제한 | ✅ 완료 |

### ⚡ 실시간 성능 (실제 측정값)
- **🚀 추론 속도**: 562 FPS (CPU 기준)
- **⏱️ 총 지연시간**: < 20ms (센서 → TTS 음성 출력)
- **💾 메모리 사용량**: ~100MB
- **🖥️ CPU 사용률**: ~15% (멀티스레드 실시간 모드)
- **🎯 안정성**: 예측 일관성 체크 (5-window 기준)

## 🔧 주요 컴포넌트

### 1. 데이터 수집 (`data_collection/`)
- **라벨 매핑**: 34개 KSL 클래스 지원
- **서버**: 실시간 데이터 수집 웹 인터페이스

### 2. 전처리 (`preprocessing/`)
- **정규화**: Min-Max, Z-score, Robust 정규화
- **필터링**: Butterworth 저역통과 필터
- **윈도우링**: 시계열 데이터 분할

### 3. 모델 (`models/`)
- **딥러닝**: CNN+LSTM+Attention 모델
- **머신러닝**: XGBoost, Random Forest
- **규칙 기반**: 임계값 기반 분류기

### 4. 실시간 추론 (`inference/`)
- **파이프라인**: 멀티스레드 실시간 처리
- **안정성 체크**: 연속 예측 일관성 검증
- **성능 모니터링**: FPS, 지연시간 추적

### 5. 하드웨어 인터페이스 (`hardware/`)
- **UART/USB**: 시리얼 통신 지원
- **WiFi/TCP**: 무선 연결 지원
- **시뮬레이션**: CSV 기반 가상 하드웨어

### 6. TTS 엔진 (`tts/`)
- **한국어 지원**: 자음 발음 매핑
- **신뢰도 기반**: 예측 신뢰도에 따른 출력
- **비동기 처리**: 논블로킹 음성 합성

## 📈 학습 결과

### 딥러닝 모델 학습 곡선
- **30 에포크** 학습
- **Early Stopping** (patience=10)
- **학습률 스케줄링** (ReduceLROnPlateau)

```
Epoch 24: Best Model ✓
- Train Loss: 0.3238, Train Acc: 87.97%
- Val Loss: 0.1603, Val Acc: 93.98%
- Test Acc: 91.67%
```

### 클래스별 성능 (F1-Score)
```
           precision    recall  f1-score   support
       ㄱ       1.00      1.00      1.00        19
       ㄴ       0.88      1.00      0.93        14
       ㄷ       0.80      0.89      0.84        18
       ㄹ       1.00      0.69      0.81        16
       ㅁ       0.94      1.00      0.97        17
```

## 🛠️ 개발 및 확장

### 프로젝트 구조
```
ksl_project/
├── configs/           # 설정 파일
├── data_collection/   # 데이터 수집
├── deployment/        # 배포 스크립트
├── evaluation/        # 모델 평가
├── features/          # 특징 추출
├── hardware/          # 하드웨어 인터페이스
├── inference/         # 실시간 추론
├── integrations/      # 샘플 데이터
├── models/            # 모델 정의
├── preprocessing/     # 전처리
├── training/          # 모델 학습
├── tts/              # 음성 합성
├── realtime_demo.py  # 실시간 데모
└── test_realtime.py  # 통합 테스트
```

### 새로운 수어 추가
1. 라벨 매핑 업데이트 (`data_collection/label_mapping.py`)
2. 학습 데이터 수집
3. 모델 재학습
4. TTS 발음 매핑 추가

### 새로운 센서 지원
1. 하드웨어 인터페이스 확장 (`hardware/signglove_hw.py`)
2. 전처리 파이프라인 수정 (`preprocessing/`)
3. 모델 입력 차원 조정

## 🎯 실제 실행 예시

### 🎮 실시간 데모 실행 결과
```bash
$ PYTHONPATH=/path/to/ksl_project python realtime_demo.py --connection simulation

🚀 실시간 수어 인식 데모 시작!
============================================================
1. 하드웨어 초기화...
✅ 시뮬레이션 모드 연결 성공: 7045개 샘플

2. 추론 파이프라인 초기화...
✅ 모델 로드 성공: best_dl_model.pth

3. TTS 엔진 초기화...
✅ macOS 한국어 음성 설정: Yuna

📊 실시간 상태 (시간: 14:08:03)
   하드웨어: simulation (큐: 0)
   FPS: 562.3 | 추론 시간: 1.8ms
   총 예측: 195 | 성공: 195
   최근 예측: ㄱ (1.00)

🎯 안정적인 예측: ㄱ (신뢰도: 1.00, 안정성: 1.00)
🔊 TTS: 기역

📋 사용법:
- 'q' + Enter: 종료
- 's' + Enter: 통계 보기  
- 'c' + Enter: 신뢰도 임계값 변경
- 't' + Enter: TTS 테스트
```

### 🧪 통합 테스트 결과
```bash
$ python test_realtime.py

🧪 실시간 수어 인식 시스템 컴포넌트 테스트
============================================================
1. 하드웨어 시뮬레이션 테스트...
✅ 하드웨어 연결 성공 (84개 센서 데이터 수신)

2. 추론 파이프라인 테스트...  
✅ 추론 테스트 성공 (예측: ㄱ, 신뢰도: 0.997)

3. TTS 엔진 테스트...
✅ TTS 테스트 성공 (macOS 한국어 음성)

🎉 모든 컴포넌트 테스트 성공!

📊 성능 통계:
   총 예측 수: 195
   안정적 예측: 2
   평균 추론 시간: 1.8ms  
   추론 FPS: 562.3

🎉 통합 테스트 성공!
```

## 📚 참고 자료

### 논문 및 연구
- CNN-LSTM 기반 수어 인식 연구
- Attention 메커니즘을 활용한 시계열 분류
- 실시간 센서 데이터 처리 최적화

### 기술 스택
- **딥러닝**: PyTorch, CNN, LSTM, Attention
- **머신러닝**: scikit-learn, XGBoost
- **신호처리**: scipy, numpy
- **하드웨어**: 시리얼 통신, TCP/IP
- **음성합성**: macOS TTS, pyttsx3

## 🔄 프로젝트 현황 & 로드맵

### ✅ 현재 완료된 기능 (v1.0)
- [x] **딥러닝 모델**: CNN+LSTM+Attention (91.67% 정확도)
- [x] **실시간 추론**: 562 FPS 멀티스레드 파이프라인
- [x] **하드웨어 연동**: UART/WiFi/시뮬레이션 모드
- [x] **전체 통합 테스트**: 모든 컴포넌트 검증 완료
- [x] **다중 모델 지원**: 딥러닝/XGBoost/RF/Rule-based

### 🚧 다음 단계 (v2.0)
- [ ] **확장된 수어**: 34개 클래스 (자음+모음+숫자) 학습 데이터 수집
- [ ] **TTS 엔진**: 한국어 음성 합성 (macOS 지원)
- [ ] **모바일 앱**: React Native/Flutter 기반 모바일 인터페이스
- [ ] **웹 서비스**: 브라우저 기반 실시간 인식 서비스
- [ ] **엣지 최적화**: TensorRT/CoreML을 통한 모바일 최적화
- [ ] **클라우드 배포**: Docker/Kubernetes 기반 확장 가능한 서비스

## 🤝 기여하기

1. 저장소 포크: `Fork` 버튼 클릭
2. 기능 브랜치 생성: `git checkout -b feature/amazing-feature`
3. 변경사항 커밋: `git commit -m 'Add amazing feature'`
4. 브랜치 푸시: `git push origin feature/amazing-feature`  
5. Pull Request 생성

### 🎯 기여 가능한 영역
- **데이터 수집**: 추가 수어 동작 데이터 기여
- **모델 개선**: 새로운 딥러닝 아키텍처 제안
- **하드웨어 지원**: 다양한 센서/장치 연동
- **UI/UX**: 사용자 인터페이스 개선
- **문서화**: 코드 문서화 및 튜토리얼 작성

## 📄 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다.

## 🔗 링크

- **GitHub**: [KLP-SignGlove 저장소](https://github.com/Kyle-Riss/KLP-SignGlove)
- **Issues**: 버그 리포트 및 기능 요청
- **Discussions**: 기술적 토론 및 아이디어 공유

## 📈 프로젝트 통계

![GitHub stars](https://img.shields.io/github/stars/Kyle-Riss/KLP-SignGlove)
![GitHub forks](https://img.shields.io/github/forks/Kyle-Riss/KLP-SignGlove)
![GitHub issues](https://img.shields.io/github/issues/Kyle-Riss/KLP-SignGlove)
![GitHub last commit](https://img.shields.io/github/last-commit/Kyle-Riss/KLP-SignGlove)

---

**🎯 미션**: 한국수어 접근성 향상을 통한 포용적 소통 문화 구현

**© 2025 SignGlove_HW Team. 모든 사람이 자유롭게 소통할 수 있는 세상을 꿈꿉니다. 🤟**
