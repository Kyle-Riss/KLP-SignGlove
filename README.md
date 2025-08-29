# SignGlove - 통합 수어 인식 시스템

## 🎯 **프로젝트 개요**

SignGlove는 한국어 수어 인식을 위한 통합 시스템입니다. 이 저장소는 전체 프로젝트의 **메인 레포지토리**로서, 외부 팀들의 구현체들을 통합하고 관리하는 역할을 합니다.

## 📁 **프로젝트 구조**

```
SignGlove/ (메인 레포지토리)
├── src/                    # 핵심 통합 코드
│   ├── core/              # 핵심 기능
│   │   └── integration_manager.py
│   └── integration/       # 외부 시스템 통합
│       └── api_gateway.py
├── external/              # 외부 의존성 (자동 동기화)
│   ├── KLP-SignGlove/     # 딥러닝 모델 (Kyle-Riss 팀)
│   │   ├── models/        # 모델 구현체
│   │   │   ├── deep_learning.py      # 기본 딥러닝 모델
│   │   │   ├── classical_ml.py       # 전통적 ML 모델
│   │   │   ├── rule_based.py         # 규칙 기반 모델
│   │   │   └── optimized.py          # 최적화된 모델
│   │   ├── training/      # 훈련 코드
│   │   │   ├── cross_validation_training.py    # 교차 검증 훈련
│   │   │   ├── ensemble_model_trainer.py       # 앙상블 모델 훈련
│   │   │   ├── specialized_model_trainer.py    # 특화 모델 훈련
│   │   │   └── optimized_cv_trainer.py         # 최적화된 교차 검증
│   │   ├── preprocessing/ # 데이터 전처리
│   │   ├── inference/     # 추론 코드
│   │   └── server/        # 서버 구현
│   └── SignGlove_HW/      # 하드웨어 구현 (KNDG01001 팀)
│       ├── datasets/      # 데이터 수집
│       ├── integration/   # 통합 코드
│       └── viz/          # 시각화
├── docs/                  # 문서
├── config/                # 설정 파일
├── deploy/                # 배포 스크립트
├── tests/                 # 통합 테스트
└── dependencies/          # 의존성 관리
```

## 🔗 **외부 의존성**

### KLP-SignGlove (딥러닝 모델)
- **저장소**: https://github.com/Kyle-Riss/KLP-SignGlove
- **역할**: 한국어 수어 인식 딥러닝 모델
- **주요 기능**: 
  - 데이터 전처리 및 시각화
  - 다양한 모델 아키텍처 (LSTM, GRU, Transformer)
  - 교차 검증 훈련
  - 앙상블 모델 훈련
  - 모델 성능 분석 및 최적화

### SignGlove_HW (하드웨어)
- **저장소**: https://github.com/KNDG01001/SignGlove_HW
- **역할**: 센서 데이터 수집 및 처리
- **주요 기능**: 
  - IMU 센서 처리
  - 데이터 수집 및 저장
  - 하드웨어 통신
  - 데이터 시각화

## 🚀 **시작하기**

### 1. 저장소 클론
```bash
git clone https://github.com/minuum/SignGlove.git
cd SignGlove
```

### 2. 외부 의존성 동기화
```bash
# 자동 동기화 (GitHub Actions)
# 또는 수동 동기화
cd external/KLP-SignGlove
git clone https://github.com/Kyle-Riss/KLP-SignGlove.git temp
cp -r temp/* .
rm -rf temp

cd ../SignGlove_HW
git clone https://github.com/KNDG01001/SignGlove_HW.git temp
cp -r temp/* .
rm -rf temp
```

### 3. 환경 설정
```bash
# Python 환경 설정
poetry install

# 또는 pip 사용
pip install -r dependencies/requirements.txt
```

## 🔧 **주요 기능**

### 통합 시스템
- 외부 팀 구현체들의 통합 관리
- API 게이트웨이 및 라우팅
- 데이터 파이프라인 관리
- 배포 및 운영 관리

### 딥러닝 모델
- **기본 모델**: LSTM, GRU, Transformer 기반
- **앙상블 모델**: 다중 모델 조합
- **교차 검증**: K-Fold 교차 검증 훈련
- **최적화**: 하이퍼파라미터 튜닝 및 모델 최적화

### 자동화
- GitHub Actions를 통한 자동 동기화
- CI/CD 파이프라인
- 자동 테스트 및 배포

## 📊 **모델 성능**

### 현재 구현된 모델들
- **기본 LSTM**: 시계열 데이터 처리
- **기본 GRU**: 경량화된 순환 신경망
- **Transformer**: 어텐션 메커니즘 기반
- **앙상블 모델**: 다중 모델 조합으로 성능 향상

### 성능 지표
- **정확도**: 모델별 상세 성능 분석
- **교차 검증**: K-Fold 검증 결과
- **최적화**: 하이퍼파라미터 튜닝 결과

## 📚 **문서**

- [API 문서](docs/api/)
- [사용 가이드](docs/guides/)
- [아키텍처 문서](docs/architecture/)
- [외부 의존성 관리](dependencies/external-repos.md)
- [프로젝트 구조 제안](PROJECT_STRUCTURE_PROPOSAL.md)
- [기술적 도전 과제](TECHNICAL_CHALLENGES.md)
- [팀 역할](TEAM_ROLES.md)

## 🤝 **기여하기**

1. 이슈 생성 또는 기존 이슈 확인
2. 브랜치 생성 (`feature/기능명` 또는 `fix/버그명`)
3. 코드 작성 및 테스트
4. Pull Request 생성

## 📋 **개발 상태**

- [x] 프로젝트 구조 설정
- [x] 외부 의존성 통합
- [x] 자동 동기화 설정
- [x] 딥러닝 모델 구현
- [x] 교차 검증 훈련 시스템
- [x] 앙상블 모델 구현
- [x] 모델 성능 분석
- [ ] API 게이트웨이 구현
- [ ] 통합 테스트 작성
- [ ] 배포 파이프라인 구축

## 🔮 **향후 개선 방안**

### SignSpeak 스타일 모델 구현
- **SignSpeakLSTM**: Stacked + Bidirectional + Attention (246,489 파라미터)
- **SignSpeakGRU**: Stacked + Bidirectional + Attention (187,353 파라미터)
- **SignSpeakTransformer**: 6층 Transformer + CLS Token (316,408 파라미터)
- **SignSpeakEnsemble**: 3개 모델 앙상블 (750,253 파라미터)

### 목표 성능
- **정확도**: 92% 이상 달성 (SignSpeak 수준)
- **추론 속도**: 실시간 처리 가능
- **모델 크기**: 경량화 및 최적화

## 📞 **연락처**

- **프로젝트 관리자**: minuum
- **KLP-SignGlove 팀**: Kyle-Riss
- **SignGlove_HW 팀**: KNDG01001

## 📄 **라이선스**

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 외부 의존성들은 각각의 라이선스를 따릅니다. 