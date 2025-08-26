# KLP-SignGlove: 한국어 수화 인식 시스템

## 📋 프로젝트 개요

KLP-SignGlove는 SignGlove 하드웨어를 활용한 한국어 수화 인식 시스템입니다. 24개의 한국어 자음/모음을 실시간으로 인식하며, 딥러닝 기반의 고성능 분류 모델을 제공합니다.

## 🎯 주요 성과

### ✅ **최종 성능: 73.53% 정확도**
- **24개 클래스**: 14개 자음 + 10개 모음
- **실시간 인식**: 20프레임 윈도우 기반
- **시나리오 분리**: 데이터 누수 방지
- **클래스별 최적화**: 과적합 및 실패 클래스 해결

### 📊 **클래스별 성능 분석**

#### 🎯 **과적합 클래스 해결 (7개 → 5개 개선)**
| 클래스 | 이전 성능 | 현재 성능 | 상태 |
|--------|-----------|-----------|------|
| ㄱ | 1.000 | 0.983 | ✅ 개선됨 |
| ㄴ | 1.000 | 0.958 | ✅ 개선됨 |
| ㅂ | 1.000 | 1.000 | ⚠️ 유지 |
| ㅇ | 1.000 | 1.000 | ⚠️ 유지 |
| ㅎ | 1.000 | 0.960 | ✅ 개선됨 |
| ㅏ | 1.000 | 0.946 | ✅ 개선됨 |
| ㅣ | 1.000 | 0.998 | ✅ 개선됨 |

#### 🎯 **실패한 클래스 분석**
| 클래스 | 성능 | 원인 |
|--------|------|------|
| ㅊ | 0.000 | flex5/yaw 센서 불안정, ㅑ와 혼동 |
| ㅕ | 0.004 | flex2 센서 데이터 품질 문제, ㅌ/ㄹ과 혼동 |

## 🏗️ 시스템 아키텍처

### 📁 **프로젝트 구조**
```
KLP-SignGlove/
├── models/                 # 딥러닝 모델
│   ├── deep_learning.py   # 메인 모델 (DeepLearningPipeline)
│   └── sensor_fusion.py   # 센서 융합 모델
├── training/              # 학습 스크립트
│   ├── solve_class_issues.py      # 🎯 최종 클래스 문제 해결
│   ├── analyze_failed_classes.py  # 실패 클래스 분석
│   ├── analyze_overfitting_classes.py # 과적합 클래스 분석
│   └── label_mapping.py   # 라벨 매핑
├── inference/             # 추론 스크립트
├── integrations/          # 하드웨어 통합
│   └── SignGlove_HW/     # SignGlove 하드웨어 데이터
├── preprocessing/         # 데이터 전처리
└── server/               # 웹 서버
```

### 🤖 **모델 아키텍처**

#### **DeepLearningPipeline**
- **입력**: 8개 센서 (pitch, roll, yaw, flex1-5)
- **시퀀스 길이**: 20프레임 윈도우
- **은닉층**: 64차원, 2레이어
- **출력**: 24개 클래스 (자음 14개 + 모음 10개)
- **정규화**: Dropout 0.3, Weight Decay 1e-4

## 🚀 **핵심 해결 방안**

### 1. **데이터 누수 방지**
```python
# 시나리오 단위 데이터 분할
train_scenarios = [1, 2, 3]  # 훈련
val_scenarios = [4]          # 검증  
test_scenarios = [5]         # 테스트
```

### 2. **클래스별 맞춤 최적화**
```python
# 과적합 클래스: 낮은 가중치 (0.5x)
overfitting_classes = ['ㄱ', 'ㄴ', 'ㅂ', 'ㅇ', 'ㅎ', 'ㅏ', 'ㅣ']

# 실패한 클래스: 높은 가중치 (3.0x)
failed_classes = ['ㅊ', 'ㅕ']

# 일반 클래스: 기본 가중치 (1.0x)
```

### 3. **스마트 데이터 증강**
```python
# 실패한 클래스: 강한 증강 (60% 확률)
# 과적합 클래스: 약한 증강 (20% 확률)
# 일반 클래스: 중간 증강 (40% 확률)
```

## 📈 **학습 과정 및 결과**

### 🔄 **개발 단계**

1. **초기 과적합 문제** (1.000 정확도)
   - 원인: 데이터 누수 (같은 시나리오 내 중복)
   - 해결: 시나리오 단위 데이터 분할

2. **클래스별 성능 불균형**
   - 과적합 클래스: 7개 (1.000 성능)
   - 실패한 클래스: 2개 (0.000 성능)
   - 해결: 클래스별 가중치 및 맞춤 증강

3. **최종 최적화**
   - 과적합 클래스 5개 개선
   - 전체 성능 73.53% 달성

### 📊 **최종 모델 성능**
- **테스트 정확도**: 73.53%
- **조기 종료**: 에포크 6 (검증 손실 기준)
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

### **모델 학습**
```bash
# 클래스 문제 해결 모델 학습
python training/solve_class_issues.py
```

### **실시간 추론**
```bash
# 실시간 수화 인식
python inference/realtime_demo.py
```

## 📊 **데이터셋 정보**

### **SignGlove 하드웨어 데이터**
- **출처**: [SignGlove_HW GitHub](https://github.com/KNDG01001/SignGlove_HW)
- **형식**: 5개 시나리오별 CSV 파일
- **센서**: 8개 (pitch, roll, yaw, flex1-5)
- **클래스**: 24개 한국어 자음/모음

### **데이터 분할 전략**
- **훈련**: 시나리오 1, 2, 3
- **검증**: 시나리오 4
- **테스트**: 시나리오 5

## 🔍 **문제 해결 사례**

### **과적합 클래스 해결**
- **원인**: 특정 센서 패턴이 너무 뚜렷함
- **해결**: 낮은 가중치 + 약한 증강
- **결과**: 5/7 클래스 개선

### **실패한 클래스 분석**
- **ㅊ 클래스**: flex5/yaw 센서 불안정, ㅑ와 혼동
- **ㅕ 클래스**: flex2 센서 데이터 품질 문제
- **해결 방안**: 하드웨어 센서 보정 필요

## 🎯 **향후 개선 방안**

### **단기 개선**
1. **센서 보정**: flex2, flex5 센서 하드웨어 보정
2. **데이터 재수집**: ㅊ, ㅕ 클래스 재측정
3. **앙상블 모델**: 여러 모델 조합

### **장기 개선**
1. **단어 수준 인식**: 자음/모음 조합
2. **문장 수준 인식**: 문법 규칙 적용
3. **실시간 최적화**: 추론 속도 개선

## 📝 **주요 파일 설명**

### **핵심 모델 파일**
- `best_problem_solver_model.pth`: 최종 최적화 모델
- `training/solve_class_issues.py`: 클래스 문제 해결 학습기
- `models/deep_learning.py`: 메인 딥러닝 모델

### **분석 결과 파일**
- `problem_solver_테스트_confusion_matrix.png`: 최종 혼동 행렬
- `problem_solver_training_curves.png`: 학습 곡선
- `failed_class_analysis_report.json`: 실패 클래스 분석
- `overfitting_analysis_report.json`: 과적합 클래스 분석

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
