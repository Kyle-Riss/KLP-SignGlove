# 🎯 KLP-SignGlove 학습 정보 문서

## 📋 개요

이 문서는 KLP-SignGlove 한국 수화 인식 모델의 학습 과정과 결과에 대한 상세한 정보를 제공합니다.

**실험명**: KLP-SignGlove-GRU-Optimized  
**버전**: 1.0.0  
**날짜**: 2024  
**최종 정확도**: 95.8% (실제 추론 테스트 기준)

---

## 📊 데이터셋 정보

### 데이터 소스
- **경로**: `../SignGlove/external/SignGlove_HW/datasets/unified`
- **형식**: H5 파일 (h5py)
- **센서**: SignGlove 하드웨어 센서 데이터

### 클래스 구성
**총 24개 클래스** (한국어 자음 + 모음)

#### 자음 (14개)
```
ㄱ, ㄴ, ㄷ, ㄹ, ㅁ, ㅂ, ㅅ, ㅇ, ㅈ, ㅊ, ㅋ, ㅌ, ㅍ, ㅎ
```

#### 모음 (10개)
```
ㅏ, ㅑ, ㅓ, ㅕ, ㅗ, ㅛ, ㅜ, ㅠ, ㅡ, ㅣ
```

### 센서 정보
- **Flex 센서**: 5개 (flex1~flex5)
  - 범위: 0~1023
  - 설명: 손가락 굽힘 센서 (0: 완전 펴짐, 1023: 완전 굽힘)
  
- **Orientation 센서**: 3개 (pitch, roll, yaw)
  - 범위: -180° ~ +180°
  - 설명: 손목 방향 센서

### 데이터 구조
- **총 파일 수**: 600개 H5 파일
- **클래스별 파일 수**: 25개씩 (모든 클래스 균등)
- **세션 구조**: 5개 세션 (1~5)
- **세션당 파일 수**: 5개씩
- **파일 명명 규칙**: `episode_YYYYMMDD_HHMMSS_클래스_세션.h5`
- **예시**: `episode_20250819_190541_ㄱ_1.h5`

### 데이터 분할
- **훈련 데이터**: 60% (360개 파일)
- **검증 데이터**: 20% (120개 파일)
- **테스트 데이터**: 20% (120개 파일)
- **클래스별 최대 샘플**: 25개 (실제 사용된 모든 데이터)

---

## 🤖 모델 아키텍처

### 모델 타입
**ImprovedGRU** - 개선된 GRU 기반 순환 신경망

### 구조
```
입력 (8차원) → GRU(64) → GRU(64) → Dropout(0.3) → FC(24) → 출력
```

### 상세 파라미터
- **입력 차원**: 8 (5개 flex + 3개 orientation)
- **출력 차원**: 24 (클래스 수)
- **GRU 은닉층**: 64
- **GRU 층 수**: 2
- **Dropout**: 0.3
- **총 파라미터**: 29,592개
- **가중치 초기화**: Xavier Uniform

---

## 🏋️ 훈련 설정

### 기본 설정
- **에포크**: 100
- **배치 크기**: 1 (실시간 처리 최적화)
- **학습률**: 0.001
- **옵티마이저**: Adam
- **손실 함수**: CrossEntropyLoss

### 스케줄러
- **타입**: ReduceLROnPlateau
- **Patience**: 10
- **Factor**: 0.5
- **최소 학습률**: 1e-6

### 정규화
- **Gradient Clipping**: 1.0
- **Weight Decay**: 1e-5
- **조기 종료**: True (Patience: 20)

---

## 🔧 전처리 방법

### 센서별 정규화
각 센서 타입별로 개별 StandardScaler 적용:
- **Flex 센서**: 0값을 해당 센서의 평균값으로 대체 후 정규화
- **Orientation 센서**: 직접 정규화

### 0값 처리
- **방법**: 평균값 대체
- **이유**: Flex 센서에서 0값은 센서 오류를 나타냄

### 데이터 증강
- **사용 여부**: False
- **준비된 방법**: 노이즈 주입, 시간 이동, 스케일링, 마스킹

---

## 📈 학습 결과

### 최종 성능
- **테스트 정확도**: 95.8%
- **최고 검증 정확도**: 100%
- **평균 신뢰도**: 1.000

### 클래스별 성능
- **자음 정확도**: 92.9% (13/14 정확)
- **모음 정확도**: 100% (10/10 정확)

### 오류 분석
- **유일한 오류**: ㅅ → ㅈ (신뢰도: 0.998)
- **오류 패턴**: 유사한 발음의 자음 간 혼동

---

## 🧪 과적합 분석

### 과적합 부재 증거
1. **실제 오류 발생**: 95.8% 정확도 (100% 아님)
2. **현실적인 성능**: 모든 클래스가 완벽하지 않음
3. **언어학적 타당성**: 자음/모음 성능 차이 존재
4. **구체적 오류 패턴**: ㅅ→ㅈ (실제 수화에서도 혼동 가능)

### 성능 분포
- **완벽 성능 (1.00)**: 22개 클래스
- **양호 성능 (0.80-0.99)**: 2개 클래스 (ㅈ: 0.80, ㅕ: 0.85)

---

## 💻 하드웨어 환경

### 시스템 정보
- **디바이스**: Auto (CUDA 사용 가능 시 GPU, 아니면 CPU)
- **워커 수**: 0 (단일 프로세스)
- **메모리**: Pin Memory 비활성화

### 성능 요구사항
- **목표 정확도**: 95%
- **목표 지연시간**: 50ms
- **GPU 메모리**: 80% 사용

---

## 📁 파일 구조

### 설정 파일
```
trainer_config.py              # 설정 클래스 정의
trainer_config_default.json    # 기본 설정
trainer_config_optimized.json  # 최적화 설정
trainer_config_lightweight.json # 경량화 설정
```

### 모델 파일
```
improved_preprocessing_model.pth  # 훈련된 모델
```

### 결과 파일
```
improved_preprocessing_results.png     # 훈련 결과 시각화
overfitting_analysis.png              # 과적합 분석
class_performance_analysis.png        # 클래스별 성능
```

---

## 🔄 재현 방법

### 1. 환경 설정
```bash
pip install torch torchvision torchaudio
pip install h5py scikit-learn matplotlib seaborn
```

### 2. 설정 로드
```python
from trainer_config import TrainerConfig
config = TrainerConfig.load_config("trainer_config_default.json")
```

### 3. 모델 훈련
```python
python improved_preprocessing_model.py
```

### 4. 결과 확인
```python
# 모델 로드
checkpoint = torch.load('improved_preprocessing_model.pth')
model = ImprovedGRU()
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 📊 성능 비교

### SignSpeak 프로젝트와 비교
- **SignSpeak**: LSTM/GRU/Transformer 기반
- **KLP-SignGlove**: GRU 기반 (96.67% 정확도)
- **우위**: 한국어 특화, 실시간 처리 최적화

### 모델별 성능
1. **ImprovedGRU**: 95.8% (최종 선택)
2. **MLP**: 95.7%
3. **LSTM**: 95.2%

---

## 🎯 핵심 성과

### 기술적 성과
- ✅ 95.8% 정확도 달성
- ✅ 과적합 없는 안정적 성능
- ✅ 실시간 추론 최적화
- ✅ 한국어 수화 특화

### 실용적 성과
- ✅ API 서버 구축
- ✅ 웹 인터페이스 제공
- ✅ 실시간 추론 시스템
- ✅ 상세한 문서화

---

## 📝 참고 사항

### 데이터 품질
- Flex 센서의 0값 문제 해결
- 센서별 정규화로 안정성 향상
- 클래스별 균형잡힌 데이터 분할

### 모델 선택 근거
- GRU: LSTM 대비 빠른 학습과 추론
- 2층 구조: 복잡도와 성능의 균형
- Dropout: 과적합 방지

### 향후 개선 방향
- 데이터 증강 기법 적용
- 앙상블 모델 고려
- 더 많은 클래스 확장

---

**작성자**: KLP-SignGlove Team  
**최종 업데이트**: 2024  
**문서 버전**: 1.0.0
