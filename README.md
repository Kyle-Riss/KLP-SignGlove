# KLP-SignGlove: 한국어 수화 인식 시스템

## 📋 프로젝트 개요

KLP-SignGlove는 SignGlove 하드웨어를 활용한 한국어 수화 인식 시스템입니다. 24개의 한국어 자음/모음을 실시간으로 인식하며, 딥러닝 기반의 고성능 분류 모델을 제공합니다. 교차 검증, 특화 모델, 앙상블 모델 등 다양한 접근법을 통해 최적의 성능을 달성했습니다.

## 🎯 주요 성과

### ✅ **최종 성능: 77.33% 정확도**
- **24개 클래스**: 14개 자음 + 10개 모음
- **실시간 인식**: 200프레임 시퀀스 기반
- **교차 검증**: 안정적인 모델 성능
- **클래스별 최적화**: 문제 클래스 특화 처리

### 📊 **클래스별 성능 분석**

#### 🏆 **우수한 성능 클래스 (≥95%)**
**14개 클래스가 높은 성능 달성**
- `ㄱ`, `ㄴ`, `ㄷ`, `ㄹ`, `ㅁ`, `ㅂ`, `ㅇ`, `ㅎ`, `ㅏ`, `ㅑ`, `ㅓ`, `ㅗ`, `ㅛ`, `ㅜ` - **95% 이상**

#### 🟡 **양호한 성능 클래스 (80-95%)**
**2개 클래스가 안정적인 성능**
- `ㅠ`, `ㅍ` - **80-95%**

#### 🔴 **개선 필요 클래스 (<80%)**
**8개 클래스 추가 개선 필요**
- `ㅅ`, `ㅈ`, `ㅊ`, `ㅋ`, `ㅌ`, `ㅕ`, `ㅡ`, `ㅣ` - **80% 미만**
- **특히 `ㅊ`**: 0% 정확도 (가장 심각한 문제)

### ⚠️ **문제 클래스 분석**
- **공통 문제**: 높은 Yaw 분산
- **특화 필요**: 클래스별 맞춤 전처리
- **센서 의존도**: Flex 센서 vs IMU 센서 차이

## 🏗️ 시스템 아키텍처

### 📁 **프로젝트 구조**
```
KLP-SignGlove/
├── models/                 # 딥러닝 모델
│   └── deep_learning.py   # 메인 모델 (DeepLearningPipeline)
├── training/              # 학습 스크립트
│   ├── cross_validation_training.py      # 🎯 교차 검증 훈련
│   ├── specialized_model_trainer.py      # 특화 모델 훈련
│   ├── ensemble_model_trainer.py         # 앙상블 모델 훈련
│   ├── optimized_cv_trainer.py           # 최적화 모델 훈련
│   ├── final_model_analysis.py           # 최종 모델 분석
│   ├── specialized_model_analysis.py     # 특화 모델 분석
│   ├── label_mapping.py                  # 라벨 매핑
│   ├── dataset.py                        # 데이터셋 클래스
│   ├── project_complete_summary.md       # 전체 프로젝트 요약
│   ├── optimization_summary.md           # 최적화 작업 요약
│   └── cleanup_summary.md                # 정리 작업 요약
├── inference/             # 추론 스크립트
├── integrations/          # 하드웨어 통합
│   └── SignGlove_HW/     # SignGlove 하드웨어 데이터
└── archive/              # 이전 버전 아카이브
```

### 🤖 **모델 아키텍처**

#### **1. 교차 검증 모델 (DeepLearningPipeline)**
- **입력**: 8개 센서 (pitch, roll, yaw, flex1-5)
- **시퀀스 길이**: 200프레임 (정규화됨)
- **은닉층**: 48차원, 1레이어
- **출력**: 24개 클래스 (자음 14개 + 모음 10개)
- **정규화**: Dropout 0.5, Weight Decay 1e-3

#### **2. 특화 모델 (SpecializedModel)**
- **구조**: Conv1D + BiLSTM + Multi-head Attention
- **특징**: 문제 클래스 특화 전처리
- **파라미터**: 122,312개
- **성능**: 9.50% (개선 필요)

#### **3. 앙상블 모델**
- **구성**: 교차 검증 모델 + 특화 모델
- **동적 가중치**: 클래스별 성능에 따른 조정
- **성능**: 76.50%

#### **4. 최적화 모델 (OptimizedModel)**
- **구조**: Conv1D + BatchNorm + BiLSTM + Attention
- **특징**: 클래스별 특화 강화
- **목표**: 80% 이상 성능
- **상태**: 훈련 진행 중

## 🚀 **핵심 개선 방안**

### 1. **교차 검증 전략**
```python
# K-Fold 교차 검증
n_folds = 5
# 강화된 정규화
dropout = 0.5
weight_decay = 1e-3
# 데이터 증강
label_smoothing = 0.2
```

### 2. **클래스별 특화 처리**
```python
# 문제 클래스 식별
problematic_classes = ['ㅊ', 'ㅌ', 'ㅅ', 'ㅈ', 'ㅋ', 'ㅕ', 'ㅡ', 'ㅣ']

# ㅊ 클래스: Yaw 영향도 감소
yaw_weight = 0.7  # 30% 감소
flex_amplification = 1.3  # 30% 증폭

# 모음 클래스: Pitch/Roll 강조
pitch_roll_weight = 1.15  # 15% 증폭
```

### 3. **향상된 전처리 파이프라인**
```python
# 1. 다단계 Yaw 보정
# 2. Flex 센서 최적화
# 3. IMU 센서 강화
# 4. 클래스별 특화 강화
# 5. 데이터 길이 정규화 (200프레임)
# 6. 클래스별 차별화 증강
```

## 📈 **학습 과정 및 결과**

### 🔄 **개발 단계**

1. **초기 과적합 문제** (1.000 정확도)
   - 원인: 데이터 누수 (같은 시나리오 내 중복)
   - 해결: 시나리오 단위 데이터 분할

2. **데이터 분할 전략 진화**
   - 시나리오 기반 분할 → 파일 기반 분할 → 계층적 샘플링 → 교차 검증
   - 각 단계별 성능 향상 및 문제 해결

3. **모델 아키텍처 발전**
   - 기본 모델 → 특화 모델 → 앙상블 모델 → 최적화 모델
   - 클래스별 특화 처리 및 정규화 강화

4. **최종 성능 달성**
   - **77.33% 정확도** 달성 (교차 검증 모델)
   - 14개 클래스가 95% 이상 성능
   - 8개 클래스 개선 필요

### 📊 **모델별 성능 비교**
- **교차 검증 모델**: 77.33% (최고 성능)
- **앙상블 모델**: 76.50% (성능 하락)
- **특화 모델**: 9.50% (개선 필요)
- **최적화 모델**: 훈련 진행 중 (목표 >80%)

## 🛠️ **설치 및 실행**

### **환경 설정**
```bash
# 저장소 클론
git clone https://github.com/your-repo/KLP-SignGlove.git
cd KLP-SignGlove

# 의존성 설치
pip install -r requirements.txt
```

### **모델 훈련**
```bash
# 교차 검증 모델 훈련
python training/cross_validation_training.py

# 특화 모델 훈련
python training/specialized_model_trainer.py

# 앙상블 모델 훈련
python training/ensemble_model_trainer.py

# 최적화 모델 훈련 (진행 중)
python training/optimized_cv_trainer.py
```

### **성능 분석**
```bash
# 최종 모델 성능 분석
python training/final_model_analysis.py

# 특화 모델 분석
python training/specialized_model_analysis.py
```

## 📊 **데이터셋 정보**

### **SignGlove 하드웨어 데이터**
- **출처**: [SignGlove_HW GitHub](https://github.com/KNDG01001/SignGlove_HW)
- **형식**: 5개 시나리오별 CSV 파일
- **센서**: 8개 (pitch, roll, yaw, flex1-5)
- **클래스**: 24개 한국어 자음/모음
- **총 파일 수**: 600개
- **총 샘플 수**: 179,812개

### **데이터 품질 현황**
- **손상된 파일**: 0개
- **빈 데이터 파일**: 0개
- **일관성 없는 길이**: 해결됨 (200프레임 정규화)
- **Yaw 센서 노이즈**: 클래스별 특화 처리로 개선

## 🔍 **주요 발견사항**

### **교차 검증 효과**
- **안정적 성능**: 89.67% ± 1.35% (검증)
- **과적합 방지**: Train-Val Gap 최소화
- **일반화 성능**: 실제 테스트에서 77.33%

### **문제 클래스 특성**
- **Yaw 분산**: 모든 문제 클래스에서 높음
- **센서 의존도**: 클래스별 차이 존재
- **해결 방안**: 클래스별 특화 전처리 필요

### **앙상블 모델 한계**
- **성능 하락**: 약한 모델의 영향
- **동적 가중치**: 효과적이지 않음
- **개선 방향**: 강한 모델 조합 필요

## 🎯 **향후 개선 방안**

### **단기 개선 (최적화 모델 완성)**
1. **최적화 모델 훈련 완료**: 현재 진행 중
2. **성능 분석**: 클래스별 개선 효과 확인
3. **하이퍼파라미터 튜닝**: 필요시 조정

### **중기 개선**
1. **문제 클래스 특화**: 8개 클래스 전용 모델
2. **센서 품질 개선**: 하드웨어 보정
3. **데이터 증강**: 클래스별 특화 증강

### **장기 개선**
1. **실시간 시스템**: 추론 최적화
2. **단어 수준 인식**: 자음/모음 조합
3. **문장 수준 인식**: 문법 규칙 적용

## 📝 **주요 파일 설명**

### **핵심 모델 파일**
- `cross_validation_model.pth`: 최고 성능 모델 (77.33%)
- `specialized_model.pth`: 특화 모델
- `ensemble_model.pth`: 앙상블 모델
- `optimized_cv_model.pth`: 최적화 모델 (진행 중)

### **분석 결과 파일**
- `final_model_analysis.png`: 최종 모델 분석 차트
- `specialized_model_analysis.png`: 특화 모델 분석 차트
- `ensemble_model_analysis.png`: 앙상블 모델 분석 차트
- `cross_validation_analysis.png`: 교차 검증 분석 차트

### **보고서 파일**
- `final_model_performance_report.json`: 최종 성능 분석
- `specialized_model_analysis_report.json`: 특화 모델 분석
- `ensemble_model_report.json`: 앙상블 모델 분석
- `cross_validation_results.json`: 교차 검증 결과

### **문서 파일**
- `project_complete_summary.md`: 전체 프로젝트 요약
- `optimization_summary.md`: 최적화 작업 요약
- `cleanup_summary.md`: 정리 작업 요약

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
- **GitHub**: https://github.com/your-username

---

**⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요!**
