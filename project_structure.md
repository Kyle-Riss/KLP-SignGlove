# KLP-SignGlove 프로젝트 구조

## 📁 최종 정리된 프로젝트 구조

```
KLP-SignGlove/
├── 📄 README.md                                    # 프로젝트 메인 문서
├── 📄 project_structure.md                         # 프로젝트 구조 문서
├── 📄 requirements.txt                             # Python 의존성
├── 📄 setup.py                                     # 설치 스크립트
├── 📄 .gitignore                                   # Git 무시 파일
│
├── 🤖 모델 파일
│   ├── 📄 best_problem_solver_model.pth           # 🎯 최종 최적화 모델 (987KB)
│   └── 📄 final_balanced_episode_model.pth        # 이전 최고 성능 모델 (1.8MB)
│
├── 📊 결과 파일
│   ├── 📄 problem_solver_training_curves.png      # 최종 학습 곡선
│   ├── 📄 problem_solver_테스트_confusion_matrix.png # 최종 혼동 행렬
│   ├── 📄 balanced_episode_confusion_matrix.png   # 이전 혼동 행렬
│   ├── 📄 failed_class_analysis_report.json      # 실패 클래스 분석
│   └── 📄 overfitting_analysis_report.json       # 과적합 클래스 분석
│
├── 🧠 학습 스크립트 (training/)
│   ├── 📄 solve_class_issues.py                   # 🎯 최종 클래스 문제 해결
│   ├── 📄 analyze_failed_classes.py              # 실패 클래스 분석
│   ├── 📄 analyze_overfitting_classes.py         # 과적합 클래스 분석
│   ├── 📄 label_mapping.py                       # 라벨 매핑
│   └── 📄 dataset.py                             # 데이터셋 로더
│
├── 🔍 추론 스크립트 (inference/)
│   └── ... (실시간 추론 관련 파일들)
│
├── 🔧 모델 정의 (models/)
│   ├── 📄 deep_learning.py                       # 메인 딥러닝 모델
│   └── 📄 sensor_fusion.py                       # 센서 융합 모델
│
├── 🔌 하드웨어 통합 (integrations/)
│   └── SignGlove_HW/                             # SignGlove 하드웨어 데이터
│
├── 🌐 웹 서버 (server/)
│   └── ... (API 서버 관련 파일들)
│
├── ⚙️ 최적화 (optimization/)
│   └── ... (성능 최적화 관련 파일들)
│
├── 🔄 전처리 (preprocessing/)
│   └── ... (데이터 전처리 관련 파일들)
│
└── 📦 아카이브 (archive/)
    ├── 📁 old_models/                            # 이전 모델 파일들 (11개)
    ├── 📁 old_results/                           # 이전 결과 파일들 (29개)
    └── 📁 old_training_scripts/                  # 이전 학습 스크립트들 (15개)
```

## 🎯 핵심 파일 설명

### **최종 모델**
- `best_problem_solver_model.pth`: **73.53% 정확도** 달성한 최종 최적화 모델
- 클래스별 맞춤 가중치 및 스마트 증강 적용

### **핵심 학습 스크립트**
- `training/solve_class_issues.py`: 클래스 문제 해결 메인 스크립트
- `training/analyze_failed_classes.py`: 실패한 클래스 (ㅊ, ㅕ) 분석
- `training/analyze_overfitting_classes.py`: 과적합 클래스 분석

### **결과 파일**
- `problem_solver_training_curves.png`: 학습 과정 시각화
- `problem_solver_테스트_confusion_matrix.png`: 최종 성능 혼동 행렬
- `failed_class_analysis_report.json`: 실패 클래스 상세 분석
- `overfitting_analysis_report.json`: 과적합 클래스 상세 분석

## 📊 정리 결과

### **파일 정리 전후**
- **정리 전**: 24개 PNG 파일, 11개 PTH 파일, 15개 학습 스크립트
- **정리 후**: 3개 핵심 PNG 파일, 2개 핵심 PTH 파일, 5개 핵심 학습 스크립트
- **정리율**: 약 75% 파일 정리 (불필요한 파일들을 archive/로 이동)

### **보존된 핵심 파일들**
1. **최종 모델**: `best_problem_solver_model.pth`
2. **최종 결과**: `problem_solver_*.png`
3. **분석 보고서**: `*_analysis_report.json`
4. **핵심 스크립트**: `solve_class_issues.py`, `analyze_*.py`

### **아카이브된 파일들**
- **old_models/**: 11개 이전 모델 파일들
- **old_results/**: 29개 이전 결과 파일들  
- **old_training_scripts/**: 15개 이전 학습 스크립트들

## 🚀 사용 방법

### **모델 학습**
```bash
python training/solve_class_issues.py
```

### **클래스 분석**
```bash
python training/analyze_failed_classes.py
python training/analyze_overfitting_classes.py
```

### **실시간 추론**
```bash
python inference/realtime_demo.py
```

## 📈 프로젝트 성과

- **최종 정확도**: 73.53%
- **과적합 클래스 개선**: 5/7 클래스
- **실패한 클래스 분석**: ㅊ, ㅕ 클래스 원인 파악
- **데이터 누수 방지**: 시나리오 단위 분할
- **클래스별 최적화**: 맞춤 가중치 및 증강

---

**이 구조는 프로젝트의 핵심 성과와 해결 방안을 명확하게 보여줍니다!** 🎯
