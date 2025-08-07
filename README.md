[README.md](https://github.com/user-attachments/files/21669882/README.md)
# SignGlove_HW - KSL Recognition System

한국 수어(KSL) 인식을 위한 End-to-End 시스템입니다. SignGlove 하드웨어와 연동하여 다양한 머신러닝 접근법을 통해 실시간 수어 인식을 제공합니다.

## 🎯 주요 특징

- **34개 KSL 클래스 지원** (자음 14개 + 모음 10개 + 숫자 10개)
- **멀티모달 센서 융합** (Flex 센서 5개 + IMU 6DOF + 방향 센서)
- **다양한 학습 접근법** (Classical ML + Deep Learning)
- **실시간 전처리 파이프라인** (Butterworth 필터 + 정규화)
- **확장 가능한 아키텍처**

## 📁 프로젝트 구조

```
ksl_project/
├── hardware/              # 하드웨어 인터페이스
├── data_collection/        # 데이터 수집 및 라벨링
├── preprocessing/          # 전처리 (필터링, 정규화)
├── features/              # 특징 추출 (멀티모달)
├── models/                # ML/DL 모델
├── training/              # 학습 스크립트
├── inference/             # 실시간 추론
├── tts/                   # 음성 합성
├── evaluation/            # 성능 평가
├── deployment/            # 배포
└── configs/               # 설정 파일
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
pip install -r ksl_project/requirements.txt
```

### 2. 샘플 데이터 생성
```bash
cd ksl_project
PYTHONPATH=. python data_collection/label_mapping.py
```

### 3. Classical ML 학습
```bash
PYTHONPATH=ksl_project python ksl_project/training/train_classical_ml.py --csv_dir ksl_project/integrations/SignGlove_HW
```

### 4. 멀티모달 특징 테스트
```bash
PYTHONPATH=ksl_project python ksl_project/features/multimodal.py
```

## 📊 시스템 성능

| 구성요소 | 상태 | 특징 |
|---------|------|------|
| 라벨링 시스템 | ✅ 완료 | 34 클래스 자동 매핑 |
| 전처리 파이프라인 | ✅ 완료 | Butterworth 5Hz + 정규화 |
| 멀티모달 아키텍처 | ✅ 완료 | 114차원 특징 (Classical ML) |
| Classical ML | ✅ 완료 | RandomForest 테스트 성공 |
| Deep Learning | 🔄 진행중 | CNN-LSTM, Transformer |

## 🔬 실험 결과

- **Classical ML**: RandomForest 정확도 100% (테스트 데이터)
- **멀티모달 특징**: 114차원 (통계적 98 + 파생 16)
- **전처리 효과**: Butterworth 필터로 노이즈 제거 확인

## 🛠️ 기술 스택

- **Python 3.12+**
- **PyTorch** (딥러닝)
- **Scikit-learn** (Classical ML)
- **SciPy** (신호 처리)
- **NumPy/Pandas** (데이터 처리)
- **FastAPI** (데이터 수집 서버)

## 📖 사용법

### 라벨링 시스템
```python
from data_collection.label_mapping import KSLLabelMapper

mapper = KSLLabelMapper()
label_id = mapper.extract_label_from_filename('ㄱ_sample.csv')  # 0
```

### 멀티모달 특징 추출
```python
from features.multimodal import MultiModalInput

multimodal = MultiModalInput()
ml_features = multimodal.prepare_for_classical_ml(window_data)  # (114,)
dl_features = multimodal.prepare_for_deep_learning(window_data)  # 센서별 텐서
```

## 📈 로드맵

- [ ] 딥러닝 모델 구현 (CNN-LSTM, Transformer)
- [ ] 실시간 추론 파이프라인
- [ ] TTS 연동
- [ ] 실제 하드웨어 연동 테스트
- [ ] 성능 최적화 및 배포

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 📞 연계 프로세스

- 프로젝트 링크: [https://github.com/yourusername/SignGlove_HW](https://github.com/yourusername/SignGlove_HW)

---

#생각 할 것
- 손을 어느 위치에서 시작해도 상관없는가??
    - 센서 측정기준으로 z축이 측정되기 때문에
    - 손의 모양만 규정
    - **포함되지 않는 변수**
    - 사람의 팔 길이
    - 사람의 키

- 시스템 구동시
    - 실험 환경대로 standard 설정이 필요
    - gyro sensor의 변수를 통제할수 있는 실험 환경 구성
