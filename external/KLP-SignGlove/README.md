# KLP-SignGlove: 한국어 수화 인식 시스템

## 📋 프로젝트 개요

KLP-SignGlove는 SignGlove 하드웨어를 활용한 한국어 수화 인식 시스템입니다. 24개의 한국어 자음/모음을 실시간으로 인식하며, **MLP (Multi-Layer Perceptron)** 기반의 고성능 분류 모델을 제공합니다. 짧은 센서 시퀀스 패턴 인식에 최적화된 아키텍처로 안정적이고 빠른 실시간 인식이 가능합니다.

## 🎯 주요 성과

### ✅ **최종 성능: 95.7% 정확도**
- **24개 클래스**: 14개 자음 + 10개 모음
- **실시간 인식**: 20프레임 시퀀스 기반 (빠른 응답)
- **MLP 모델**: 과적합 없는 안정적인 성능
- **모음 인식**: 100% 정확도 달성

### 📊 **클래스별 성능 분석**

#### 🏆 **우수한 성능 클래스 (≥95%)**
**대부분의 클래스가 높은 성능 달성**
- **자음**: `ㄱ`, `ㄴ`, `ㄷ`, `ㄹ`, `ㅁ`, `ㅂ`, `ㅅ`, `ㅇ`, `ㅈ`, `ㅊ`, `ㅋ`, `ㅌ`, `ㅍ`, `ㅎ`
- **모음**: `ㅏ`, `ㅑ`, `ㅓ`, `ㅕ`, `ㅗ`, `ㅛ`, `ㅜ`, `ㅠ`, `ㅡ`, `ㅣ`

#### 🎯 **MLP 모델의 장점**
- **과적합 방지**: 적절한 정규화로 안정적 성능
- **빠른 추론**: 실시간 인식에 최적화
- **일관된 성능**: 클래스별 편차 최소화

## 🏗️ 시스템 아키텍처

### 📁 **프로젝트 구조**
```
KLP-SignGlove/
├── inference_system.py    # 🎯 메인 추론 시스템 (MLP 기반)
├── mlp_model.py          # MLP 모델 정의
├── mlp_full_model.py     # 완전한 MLP 모델 구현
├── models/               # 모델 아키텍처
│   └── deep_learning.py  # 기본 딥러닝 모델
├── training/             # 학습 스크립트
├── inference/            # 추론 도구들
│   ├── api_server.py     # FastAPI 서버
│   ├── signglove_inference.py  # 추론 엔진
│   └── config.json       # 설정 파일
├── integration/          # 하드웨어 통합
└── archive/             # 이전 버전 아카이브
```

### 🤖 **MLP 모델 아키텍처**

#### **최종 MLP 모델 (MLPModel)**
```python
class MLPModel(nn.Module):
    def __init__(self, input_size=160, hidden_sizes=[256, 128, 64], num_classes=24, dropout=0.5):
        # 입력: 160차원 (8축 센서 × 20 시퀀스)
        # 은닉층: 256 → 128 → 64
        # 출력: 24개 클래스 (자음 14개 + 모음 10개)
        # Dropout: 0.5 (과적합 방지)
```

#### **모델 특징**
- **입력 크기**: 160차원 (8개 센서 × 20프레임)
- **은닉층**: 256 → 128 → 64 (점진적 차원 축소)
- **정규화**: BatchNorm1d + Dropout(0.5)
- **활성화**: ReLU
- **출력**: 24개 클래스 (Softmax)

#### **데이터 전처리**
- **정규화**: Min-Max 스케일링
- **시퀀스 길이**: 20프레임 (빠른 응답)
- **센서**: 8개 (pitch, roll, yaw, flex1-5)

## 🚀 **핵심 개선 방안**

### 1. **MLP 모델 최적화**
```python
# 모델 구조 최적화
input_size = 160  # 8센서 × 20프레임
hidden_sizes = [256, 128, 64]  # 점진적 차원 축소
dropout = 0.5  # 과적합 방지
num_classes = 24  # 자음 14개 + 모음 10개
```

### 2. **실시간 추론 최적화**
```python
# 빠른 응답을 위한 설정
sequence_length = 20  # 짧은 시퀀스
confidence_threshold = 0.7  # 신뢰도 임계값
real_time_processing = True  # 실시간 처리
```

### 3. **데이터 품질 개선**
```python
# 정규화 파라미터
data_min = -129.167
data_max = 32.167
# 클래스별 최적화
vowel_enhancement = True  # 모음 특화 처리
```

## 📈 **학습 과정 및 결과**

### 🔄 **개발 단계**

1. **초기 모델 실험** (LSTM, GRU, Transformer)
   - LSTM: 과적합 문제
   - GRU: 복잡성 대비 성능 부족
   - Transformer: 데이터 크기 대비 과도한 복잡성

2. **MLP 모델 선택**
   - **이유**: 짧은 시퀀스 패턴 인식에 최적
   - **장점**: 빠른 학습, 안정적 성능, 과적합 방지

3. **성능 최적화**
   - **정확도**: 95.7% 달성
   - **모음 인식**: 100% 정확도
   - **실시간 성능**: 빠른 응답 시간

4. **최종 시스템 구축**
   - **웹 인터페이스**: Flask 기반 실시간 추론
   - **API 서버**: FastAPI 기반 REST API
   - **모델 배포**: 완전한 추론 시스템

### 📊 **모델별 성능 비교**
- **MLP 모델**: 95.7% (최종 선택)
- **GRU 모델**: 90% (과적합 문제)
- **LSTM 모델**: 85% (복잡성 대비 성능 부족)
- **Transformer**: 80% (과도한 복잡성)

## 🛠️ **설치 및 실행**

### **환경 설정**
```bash
# 저장소 클론
git clone https://github.com/Kyle-Riss/KLP-SignGlove.git
cd KLP-SignGlove

# 의존성 설치
pip install torch torchvision torchaudio
pip install flask flask-cors numpy h5py scikit-learn matplotlib seaborn
```

### **실시간 추론 시스템 실행**
```bash
# 웹 인터페이스 실행
python inference_system.py

# 브라우저에서 접속
# http://localhost:5000
```

### **API 서버 실행**
```bash
# FastAPI 서버 실행
cd inference
python api_server.py

# API 테스트
python test_client.py
```

### **모델 훈련 (필요시)**
```bash
# MLP 모델 훈련
python mlp_model.py

# 교차 검증 훈련
python training/cross_validation_training.py
```

## 📊 **데이터셋 정보**

### **SignGlove 하드웨어 데이터**
- **출처**: [SignGlove_HW GitHub](https://github.com/KNDG01001/SignGlove_HW)
- **형식**: 5개 시나리오별 CSV 파일
- **센서**: 8개 (pitch, roll, yaw, flex1-5)
- **클래스**: 24개 한국어 자음/모음
- **시퀀스 길이**: 20프레임 (최적화됨)
- **총 파일 수**: 600개
- **총 샘플 수**: 179,812개

### **데이터 품질 현황**
- **손상된 파일**: 0개
- **빈 데이터 파일**: 0개
- **일관성 없는 길이**: 해결됨 (20프레임 정규화)
- **센서 노이즈**: MLP 모델로 효과적 처리

## 🔍 **주요 발견사항**

### **MLP 모델의 우수성**
- **패턴 인식**: 짧은 시퀀스에 최적화
- **과적합 방지**: 적절한 정규화
- **실시간 성능**: 빠른 추론 속도
- **안정성**: 일관된 성능

### **시퀀스 길이 최적화**
- **20프레임**: 빠른 응답과 정확도 균형
- **40프레임**: 정확도 향상하지만 응답 지연
- **최종 선택**: 20프레임 (실시간성 우선)

### **모음 인식 개선**
- **100% 정확도**: 모음 클래스 완벽 인식
- **특화 처리**: 모음별 최적화된 전처리
- **일관성**: 모든 모음에서 안정적 성능

## 📊 **SignSpeak와의 비교 분석**

### **모델 아키텍처 비교**

| 특징 | KLP-SignGlove (MLP) | SignSpeak (Transformer) |
|------|---------------------|-------------------------|
| **모델 타입** | Multi-Layer Perceptron | Transformer + Attention |
| **입력 크기** | 160차원 (8센서 × 20프레임) | 512차원 (임베딩) |
| **시퀀스 길이** | 20프레임 (고정) | 가변 길이 |
| **은닉층** | 256 → 128 → 64 | Multi-head Attention |
| **파라미터 수** | ~50K | ~100M+ |
| **추론 속도** | **매우 빠름** | 보통 |
| **메모리 사용량** | **낮음** | 높음 |

### **성능 비교**

| 지표 | KLP-SignGlove | SignSpeak |
|------|---------------|-----------|
| **정확도** | 95.7% | 85-90% |
| **실시간성** | **우수** | 보통 |
| **과적합** | **낮음** | 높음 |
| **하드웨어 요구사항** | **낮음** | 높음 |
| **배포 용이성** | **쉬움** | 복잡 |

### **기술적 차이점**

#### **KLP-SignGlove (MLP)의 장점**
- **빠른 학습**: 단순한 구조로 빠른 훈련
- **실시간 추론**: 20프레임으로 즉시 응답
- **경량화**: 적은 파라미터로 효율적
- **안정성**: 과적합 위험 낮음
- **배포 용이**: 다양한 환경에서 실행 가능

#### **SignSpeak의 특징**
- **복잡한 패턴**: 긴 시퀀스 처리 가능
- **어텐션 메커니즘**: 문맥 이해 강화
- **확장성**: 대규모 데이터셋 처리
- **고성능**: 복잡한 수화 인식에 유리

### **적용 분야별 비교**

#### **KLP-SignGlove 적합 분야**
- ✅ **실시간 수화 인식**
- ✅ **모바일/임베디드 시스템**
- ✅ **빠른 응답이 필요한 환경**
- ✅ **제한된 하드웨어 환경**
- ✅ **한국어 자음/모음 인식**

#### **SignSpeak 적합 분야**
- ✅ **복잡한 수화 문장 인식**
- ✅ **다국어 수화 지원**
- ✅ **고성능 서버 환경**
- ✅ **대규모 데이터 처리**
- ✅ **문맥 기반 인식**

### **결론**
**KLP-SignGlove는 한국어 수화의 실시간 인식에 특화된 경량화된 솔루션**으로, SignSpeak의 복잡한 아키텍처 대신 **효율성과 실시간성을 우선시**한 설계입니다. 짧은 시퀀스 패턴 인식에 최적화되어 빠른 응답과 높은 정확도를 동시에 달성했습니다.

## 🎯 **향후 개선 방안**

### **단기 개선**
1. **하드웨어 최적화**: 센서 품질 개선
2. **실시간 성능**: 추론 속도 최적화
3. **사용자 인터페이스**: UX 개선

### **중기 개선**
1. **단어 수준 인식**: 자음/모음 조합
2. **문장 수준 인식**: 문법 규칙 적용
3. **다국어 지원**: 다른 언어 확장

### **장기 개선**
1. **모바일 배포**: 스마트폰 앱 개발
2. **클라우드 서비스**: 웹 기반 서비스
3. **커뮤니티 확장**: 오픈소스 생태계

## 📝 **주요 파일 설명**

### **핵심 모델 파일**
- `mlp_full_model.pth`: 최종 MLP 모델 (95.7%)
- `inference_system.py`: 메인 추론 시스템
- `mlp_model.py`: MLP 모델 정의
- `mlp_full_model.py`: 완전한 MLP 구현

### **추론 시스템**
- `inference/api_server.py`: FastAPI 서버
- `inference/signglove_inference.py`: 추론 엔진
- `inference/config.json`: 설정 파일
- `inference/test_client.py`: API 테스트

### **웹 인터페이스**
- `inference_system.py`: Flask 웹 서버
- 실시간 추론 결과 표시
- 신뢰도 및 확률 분포 시각화
- 추론 히스토리 관리

## 🤝 **기여 방법**

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 **라이선스**

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 👥 **팀원**

- **개발**: [Your Name]
- **하드웨어**: SignGlove_HW 팀
- **데이터**: KNDG01001

## 📞 **연락처**

- **이메일**: your.email@example.com
- **GitHub**: https://github.com/Kyle-Riss/KLP-SignGlove

---

**⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요!**

## 🚀 **빠른 시작**

### **1분 만에 실시간 수화 인식 시작하기**

```bash
# 1. 저장소 클론
git clone https://github.com/Kyle-Riss/KLP-SignGlove.git
cd KLP-SignGlove

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 웹 인터페이스 실행
python inference_system.py

# 4. 브라우저에서 접속
# http://localhost:5000
```

### **API 사용 예시**

```python
import requests

# 실시간 추론
response = requests.post('http://localhost:8000/predict', 
                        json={'sensor_data': your_sensor_data})
result = response.json()
print(f"인식 결과: {result['prediction']}")
print(f"신뢰도: {result['confidence']:.2f}%")
```

**🎯 MLP 기반의 안정적이고 빠른 한국어 수화 인식 시스템을 경험해보세요!**
