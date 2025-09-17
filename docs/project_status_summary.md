# SignGlove 프로젝트 진행사항 요약 (일주일 후 복귀)

## 📅 현재 상황 (2024년 9월 15일 기준)

### 🏗️ 프로젝트 구조
```
SignGlove/
├── README.md                    # SignSpeak 프로젝트 문서 (ASL 번역)
├── src/                         # 메인 소스 코드
│   ├── experiments/             # 실험 및 훈련 스크립트
│   │   ├── LightningTrain.py   # PyTorch Lightning 기반 훈련
│   │   ├── data/data.csv       # SignSpeak 데이터셋
│   │   └── *.sh               # 훈련 스크립트들
│   ├── models/                  # 모델 구현
│   │   ├── LSTM.py            # LSTM 모델
│   │   ├── GRU.py             # GRU 모델
│   │   ├── encoder.py         # Transformer 인코더
│   │   └── LightningModel.py  # Lightning 기본 클래스
│   └── misc/                   # 유틸리티
├── data/                        # 데이터 디렉토리 (비어있음)
├── integrations/                # 하드웨어 통합
│   ├── SignGlove_HW/          # 한국어 수어 샘플 데이터
│   │   ├── ㄱ_sample_data.csv
│   │   ├── ㄴ_sample_data.csv
│   │   ├── ㄷ_sample_data.csv
│   │   ├── ㄹ_sample_data.csv
│   │   └── ㅁ_sample_data.csv
│   └── unified_adapter.py      # 데이터 통합 어댑터
└── requirements.txt            # 의존성
```

## 🎯 프로젝트 목표

### 1. SignSpeak (ASL 번역)
- **목표**: 미국 수화(ASL) A-Z, 1-10 (36 클래스) 번역
- **데이터**: 7200 샘플, 36Hz, 5개 flex 센서
- **성능**: 92% 정확도 달성 (LSTM, GRU, Transformer)
- **논문**: [arXiv:2407.12020](https://arxiv.org/abs/2407.12020)

### 2. SignGlove (한국어 수어)
- **목표**: 한국어 수어 자음/모음 (24 클래스) 인식
- **데이터**: 5개 클래스 샘플 데이터 보유 (ㄱ, ㄴ, ㄷ, ㄹ, ㅁ)
- **상태**: 초기 데이터 수집 단계

## 📊 현재 보유 데이터

### SignSpeak 데이터셋
- **위치**: `src/experiments/data/data.csv`
- **크기**: 7200 샘플
- **클래스**: 36개 (A-Z, 1-10)
- **특징**: 5개 flex 센서, 36Hz

### SignGlove 데이터셋
- **위치**: `integrations/SignGlove_HW/`
- **크기**: 5개 클래스 샘플
- **클래스**: ㄱ, ㄴ, ㄷ, ㄹ, ㅁ (5개)
- **형식**: CSV 파일
- **상태**: 초기 샘플만 보유

## 🤖 구현된 모델들

### 1. LSTM 모델 (`src/models/LSTM.py`)
- Stacked LSTM 구조
- PyTorch Lightning 기반
- 5-Fold 교차 검증 지원

### 2. GRU 모델 (`src/models/GRU.py`)
- Stacked GRU 구조
- PyTorch Lightning 기반
- 5-Fold 교차 검증 지원

### 3. Transformer 인코더 (`src/models/encoder.py`)
- Multi-head attention
- PyTorch Lightning 기반
- 5-Fold 교차 검증 지원

### 4. 공통 기능 (`src/models/LightningModel.py`)
- 훈련/검증 스텝 구현
- 혼동 행렬 계산
- 정확도 메트릭
- Wandb 로깅 지원

## 🚀 훈련 시스템

### PyTorch Lightning 기반
- **메인 스크립트**: `src/experiments/LightningTrain.py`
- **5-Fold 교차 검증**: 자동화된 K-Fold 분할
- **하이퍼파라미터**: 명령행 인자로 설정
- **로깅**: Wandb 통합 지원

### 훈련 명령어 예시
```bash
python LightningTrain.py \
    -layers 2 \
    -model LSTM \
    -hidden_size 64 \
    -lr 0.001 \
    -time_steps 100 \
    -batch_size 32 \
    -epochs 100
```

## 📈 성능 결과

### SignSpeak (ASL)
- **최고 성능**: 92% 정확도
- **모델**: LSTM, GRU, Transformer
- **검증**: 5-Fold 교차 검증
- **논문 발표**: 완료

### SignGlove (한국어 수어)
- **현재 상태**: 데이터 수집 초기 단계
- **보유 데이터**: 5개 클래스 샘플
- **목표 클래스**: 24개 (자음 14개 + 모음 10개)
- **성능**: 아직 측정되지 않음

## 🔧 기술 스택

### 프레임워크
- **PyTorch**: 딥러닝 모델 구현
- **PyTorch Lightning**: 훈련 파이프라인
- **Wandb**: 실험 추적 및 로깅

### 하드웨어
- **센서**: 5개 flex 센서
- **마이크로컨트롤러**: Arduino MEGA 2560
- **샘플링**: 36Hz

## 📋 다음 단계 (우선순위)

### 1. SignGlove 데이터 수집 완료
- [ ] 나머지 19개 클래스 데이터 수집
- [ ] 데이터 품질 검증
- [ ] 통합 데이터셋 구축

### 2. SignGlove 모델 훈련
- [ ] 기존 SignSpeak 모델 적용
- [ ] 한국어 수어 특화 모델 개발
- [ ] 성능 벤치마킹

### 3. 실시간 추론 시스템
- [ ] 하드웨어 통합
- [ ] 실시간 데이터 처리
- [ ] 추론 최적화

### 4. 문서화 및 배포
- [ ] 한국어 수어 모델 문서화
- [ ] API 개발
- [ ] 사용자 가이드 작성

## 🚨 주의사항

### 데이터 부족
- SignGlove 프로젝트는 현재 5개 클래스만 보유
- 24개 클래스 중 19개 클래스 데이터 부족
- 데이터 수집이 최우선 과제

### 모델 검증 필요
- SignSpeak 모델의 한국어 수어 적용성 검증 필요
- 한국어 수어 특성에 맞는 모델 조정 필요

## 📞 연락처

### SignSpeak 팀
- Aditya Makkar: aditya.makkar@uwaterloo.ca
- Divya Makkar: divya.makkar@uwaterloo.ca
- Aarav Patel: aarav.patel@uwaterloo.ca

### SignGlove 팀
- 한국어 수어 프로젝트 담당자 정보 필요

---

**마지막 업데이트**: 2024년 9월 15일
**프로젝트 상태**: SignSpeak 완료, SignGlove 초기 단계
