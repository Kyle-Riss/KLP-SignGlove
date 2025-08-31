# 연구 증거 자료 (Research Evidence)

이 폴더는 SignGlove 프로젝트 연구 과정에서 시도했던 다양한 모델들과 그들이 왜 최종 선택되지 않았는지에 대한 증거 자료들을 포함합니다.

## 📊 연구 과정 요약

### 시도했던 모델들

#### 1. S-GRU (Simplified GRU) 모델들
- **s_gru_model.py**: 기본 S-GRU 모델
- **s_gru_full_training.py**: 전체 데이터 훈련
- **s_gru_complete_training.py**: 완전한 훈련 시스템
- **s_gru_proper_training.py**: 적절한 훈련 설정
- **s_gru_balanced_training.py**: 균형잡힌 데이터 훈련
- **s_gru_filtered_training.py**: 필터링된 데이터 훈련
- **s_gru_angle_independent_training.py**: 각도 독립적 훈련

#### 2. SignSpeak 스타일 모델
- **signspeak_style_gru.py**: SignSpeak 프로젝트 스타일 GRU

#### 3. 기타 모델들
- **trainer_config.py**: 훈련 설정 파일
- **preprocessor_params.pth**: 전처리 파라미터

## 🔍 왜 이 모델들이 최종 선택되지 않았는가?

### 1. 과적합 문제
- **S-GRU 모델들**: 대부분 과적합 발생
- **증거**: 검증 정확도 < 훈련 정확도
- **해결책**: RGRU에서 LayerNorm, Dropout, Weight Decay 적용

### 2. 데이터 불일치 문제
- **원시 데이터 vs 정규화된 데이터**: 전처리 방식 불일치
- **증거**: 3.30% vs 86.3% 정확도 차이
- **해결책**: 일관된 데이터 전처리 적용

### 3. 모델 복잡성 문제
- **단순한 모델**: 표현력 부족
- **복잡한 모델**: 과적합 위험
- **해결책**: RGRU에서 균형잡힌 구조

### 4. 특정 클래스 성능 문제
- **ㄹ/ㅕ 혼동**: 유사한 패턴으로 인한 혼동
- **해결책**: ClassDiscriminator 후처리 필터 적용

## 📈 성능 비교

| 모델 | 정확도 | 과적합 | 최종 선택 |
|------|--------|--------|-----------|
| S-GRU 기본 | 85.2% | 높음 | ❌ |
| S-GRU 완전 | 88.9% | 중간 | ❌ |
| S-GRU 균형 | 91.67% | 낮음 | ❌ |
| SignSpeak 스타일 | 89.3% | 높음 | ❌ |
| **RGRU** | **94.78%** | **없음** | **✅** |

## 🎯 최종 선택 이유

### RGRU의 우수성
1. **과적합 완전 해결**: 과적합 지수 0.1766
2. **높은 정확도**: 94.78% 검증 정확도
3. **안정적인 성능**: 일관된 결과
4. **확장 가능성**: 다양한 정규화 기법 적용

### 기술적 개선점
- **LayerNorm**: 입력 및 중간 레이어 정규화
- **Dropout**: 0.5 비율로 과적합 방지
- **Attention**: 중요 시점 집중
- **Multi-layer Classifier**: 복잡한 패턴 학습

## 📁 파일 구조

```
research_evidence/
├── archive_old_files/           # 이전 버전 파일들
├── s_gru_*.py                   # S-GRU 모델 파일들
├── s_gru_*.pth                  # S-GRU 훈련된 모델들
├── signspeak_style_gru.py       # SignSpeak 스타일 모델
├── trainer_config.py            # 훈련 설정
├── preprocessor_params*.pth     # 전처리 파라미터
└── README.md                    # 이 파일
```

## 💡 연구 교훈

1. **과적합 방지의 중요성**: 정규화 기법이 성능에 결정적
2. **데이터 일관성**: 전처리 방식의 일관성이 필수
3. **모델 구조의 균형**: 복잡성과 단순성의 적절한 조합
4. **후처리 필터의 효과**: 특정 클래스 혼동 해결에 유용

---

**이 자료들은 SignGlove 프로젝트의 연구 과정을 보여주며, 최종 모델 선택의 근거를 제공합니다.** 🔬
