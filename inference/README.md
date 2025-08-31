# 추론 (Inference)

이 폴더는 SignGlove 모델의 추론 관련 파일들을 포함합니다.

## 파일 목록

### `corrected_filtered_inference.py`
- **기능**: 정규화된 데이터를 사용한 추론 시스템
- **사용법**: `python3 corrected_filtered_inference.py`
- **입력**: `real_data_filtered/` 폴더의 정규화된 데이터
- **출력**: `corrected_filtered_results.json`

## 사용 방법

```bash
# 추론 실행
python3 corrected_filtered_inference.py
```

## 주요 특징

- **정규화된 데이터 사용**: 전처리된 데이터로 높은 정확도
- **ㄹ/ㅕ 후처리 필터**: 혼동되는 클래스 구분
- **배치 처리**: 여러 파일 동시 처리
- **진행률 표시**: 실시간 처리 상황 확인

## 결과

- **평균 신뢰도**: 86.3%
- **처리 속도**: 0.7ms
- **인식 문자**: 24개 한글 자음/모음
