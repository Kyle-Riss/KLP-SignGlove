# KLP-SignGlove: 한국어 수화 인식 시스템

## 📋 프로젝트 개요

KLP-SignGlove는 SignGlove 하드웨어를 활용한 한국어 수화 인식 시스템입니다. 24개의 한국어 자음/모음을 실시간으로 인식하며, 딥러닝 기반의 고성능 분류 모델을 제공합니다. 데이터 품질 개선과 전처리 파이프라인 최적화를 통해 높은 인식률을 달성했습니다.

## 🎯 주요 성과

### ✅ **최종 성능: 86.67% 정확도**
- **24개 클래스**: 14개 자음 + 10개 모음
- **실시간 인식**: 300프레임 시퀀스 기반
- **데이터 품질 개선**: 상보 필터 및 전처리 최적화
- **클래스별 최적화**: 과적합 문제 해결 및 성능 균형

### 📊 **클래스별 성능 분석**

#### 🏆 **우수한 성능 클래스 (F1-Score ≥ 0.9)**
**13개 클래스가 완벽에 가까운 성능 달성**
- `ㄱ`, `ㄷ`, `ㄹ`, `ㅁ`, `ㅅ`, `ㅇ`, `ㅋ`, `ㅌ`, `ㅑ`, `ㅓ`, `ㅜ`, `ㅠ` - **F1-Score: 1.000**
- `ㄴ` - **F1-Score: 0.909**

#### 🟡 **양호한 성능 클래스 (F1-Score 0.8-0.9)**
**5개 클래스가 안정적인 성능**
- `ㅣ` - **F1-Score: 0.889**
- `ㅊ`, `ㅍ` - **F1-Score: 0.857**
- `ㅈ`, `ㅡ` - **F1-Score: 0.800**

#### 🔴 **개선 필요 클래스 (F1-Score < 0.7)**
**4개 클래스 추가 개선 필요**
- `ㅗ` - **F1-Score: 0.667**
- `ㅛ` - **F1-Score: 0.571**
- `ㅏ`, `ㅂ` - **F1-Score: 0.500**

### ⚠️ **과적합 분석 결과**
- **과적합 위험도**: 3/4 (높음)
- **주요 위험 요인**:
  - 6개 클래스가 3개 이하 샘플로 100% 정확도
  - 12개 클래스가 Precision=1.0, Recall=1.0 달성
  - 성능 격차가 큼 (완벽 클래스 vs 일반 클래스)

## 🏗️ 시스템 아키텍처

### 📁 **프로젝트 구조**
```
KLP-SignGlove/
├── models/                 # 딥러닝 모델
│   └── deep_learning.py   # 메인 모델 (DeepLearningPipeline)
├── training/              # 학습 스크립트
│   ├── improved_preprocessing_pipeline.py    # 🎯 개선된 전처리 파이프라인
│   ├── train_with_improved_preprocessing.py  # 최종 학습 스크립트
│   ├── data_quality_improvement.py           # 데이터 품질 분석
│   ├── class_accuracy_analysis.py            # 클래스별 정확도 분석
│   ├── overfitting_analysis.py               # 과적합 분석
│   └── label_mapping.py   # 라벨 매핑
├── inference/             # 추론 스크립트
├── integrations/          # 하드웨어 통합
│   └── SignGlove_HW/     # SignGlove 하드웨어 데이터
└── archive/              # 이전 버전 아카이브
```

### 🤖 **모델 아키텍처**

#### **DeepLearningPipeline**
- **입력**: 8개 센서 (pitch, roll, yaw, flex1-5)
- **시퀀스 길이**: 300프레임 (정규화됨)
- **은닉층**: 128차원, 2레이어
- **출력**: 24개 클래스 (자음 14개 + 모음 10개)
- **정규화**: Dropout 0.3, Weight Decay 1e-4

## 🚀 **핵심 개선 방안**

### 1. **데이터 품질 개선**
```python
# 상보 필터 적용 (IMU 센서 안정화)
complementary_filter_alpha = 0.96

# 노이즈 감소 (이동 평균 필터)
noise_reduction_window = 5

# 이상치 제거 (3σ 기준)
outlier_threshold = 3.0
```

### 2. **클래스별 최적화**
```python
# 낮은 일관성 클래스: 강한 정규화
low_consistency_classes = ['ㄱ', 'ㄴ', 'ㄷ', 'ㅁ', 'ㅕ']

# 높은 일관성 클래스: 표준 가중치
high_consistency_classes = ['ㄹ', 'ㅌ', 'ㅅ', 'ㅇ']

# yaw 노이즈가 높은 클래스: 특별 처리
noisy_yaw_classes = ['ㄱ', 'ㅁ', 'ㅍ', 'ㅓ']
```

### 3. **전처리 파이프라인**
```python
# 1. 기본 정리 (이상치 제거)
# 2. 상보 필터 적용 (IMU 센서)
# 3. 노이즈 감소 (이동 평균)
# 4. Flex 센서 정규화
# 5. 클래스별 가중치 적용
# 6. 데이터 길이 정규화 (300프레임)
# 7. 데이터 증강 (클래스별 차별화)
```

## 📈 **학습 과정 및 결과**

### 🔄 **개발 단계**

1. **초기 과적합 문제** (1.000 정확도)
   - 원인: 데이터 누수 (같은 시나리오 내 중복)
   - 해결: 시나리오 단위 데이터 분할

2. **데이터 품질 분석**
   - yaw 센서 노이즈 문제 발견
   - 클래스별 일관성 점수 분석
   - 데이터 불균형 확인

3. **전처리 파이프라인 개선**
   - 상보 필터 적용으로 IMU 센서 안정화
   - 클래스별 특화 전처리
   - 데이터 증강 최적화

4. **최종 성능 달성**
   - **86.67% 정확도** 달성
   - 13개 클래스가 90% 이상 성능
   - 과적합 위험 확인 및 해결 방안 제시

### 📊 **최종 모델 성능**
- **테스트 정확도**: 86.67%
- **평균 F1-Score**: 0.869
- **평균 Precision**: 0.887
- **평균 Recall**: 0.892
- **모델 크기**: 987KB
- **추론 속도**: 실시간 처리 가능

## 🛠️ **설치 및 실행**

### **환경 설정**
```bash
# 저장소 클론
git clone https://github.com/your-repo/KLP-SignGlove.git
cd KLP-SignGlove

# 의존성 설치
pip install -r requirements.txt
```

### **데이터 품질 분석**
```bash
# 데이터 품질 분석
python training/data_quality_improvement.py
```

### **모델 학습**
```bash
# 개선된 전처리 파이프라인으로 학습
python training/train_with_improved_preprocessing.py
```

### **성능 분석**
```bash
# 클래스별 정확도 분석
python training/class_accuracy_analysis.py

# 과적합 분석
python training/overfitting_analysis.py
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
- **일관성 없는 길이**: 1개 클래스
- **yaw 센서 노이즈**: 4개 클래스에서 높음

## 🔍 **주요 발견사항**

### **데이터 품질 개선 효과**
- **상보 필터**: yaw 센서 노이즈 감소
- **클래스별 최적화**: 일관성 점수 기반 차별화
- **전처리 파이프라인**: 종합적 데이터 품질 향상

### **과적합 문제**
- **위험도**: 높음 (3/4)
- **주요 원인**: 적은 샘플 클래스의 완벽한 성능
- **해결 방안**: 교차 검증, 데이터 증강, 모델 복잡도 감소

## 🎯 **향후 개선 방안**

### **단기 개선 (과적합 해결)**
1. **교차 검증**: 모델 안정성 확인
2. **데이터 증강**: 일반화 성능 향상
3. **모델 복잡도 감소**: 과적합 방지
4. **정규화 강화**: Dropout, Weight Decay 조정

### **중기 개선**
1. **낮은 성능 클래스 특화**: `ㅂ`, `ㅏ`, `ㅗ`, `ㅛ`
2. **앙상블 모델**: 여러 모델 조합
3. **실시간 최적화**: 추론 속도 개선

### **장기 개선**
1. **단어 수준 인식**: 자음/모음 조합
2. **문장 수준 인식**: 문법 규칙 적용
3. **하드웨어 개선**: 센서 보정

## 📝 **주요 파일 설명**

### **핵심 모델 파일**
- `best_improved_preprocessing_model.pth`: 최종 개선 모델
- `training/improved_preprocessing_pipeline.py`: 개선된 전처리 파이프라인
- `training/train_with_improved_preprocessing.py`: 최종 학습 스크립트

### **분석 결과 파일**
- `improved_preprocessing_confusion_matrix.png`: 최종 혼동 행렬
- `improved_preprocessing_training_curves.png`: 학습 곡선
- `class_accuracy_detailed_analysis.png`: 클래스별 정확도 분석
- `overfitting_analysis.png`: 과적합 분석
- `data_quality_analysis.png`: 데이터 품질 분석

### **보고서 파일**
- `improved_preprocessing_classification_report.json`: 상세 분류 보고서
- `class_accuracy_ranking_summary.json`: 클래스별 순위 요약
- `overfitting_analysis_report.json`: 과적합 분석 보고서
- `data_quality_improvement_strategies.json`: 데이터 품질 개선 전략

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
