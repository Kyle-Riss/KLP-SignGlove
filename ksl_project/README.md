# KSL Recognition Project

이 프로젝트는 SignGlove 하드웨어와 연동하여 KSL(한국 수어) 인식 End-to-End 시스템을 구현하기 위한 구조입니다.

## 주요 디렉토리
- hardware: 센서/하드웨어 연동
- data_collection: 데이터 수집 서버
- preprocessing: 전처리, 필터, 정규화
- features: 특징 추출
- models: Rule-based, ML, DL 모델
- training: 학습/평가 스크립트
- inference: 실시간 추론
- tts: 음성 합성
- evaluation: 평가
- deployment: 엣지 배포
- integrations/SignGlove_HW: 센서 데이터 및 연동
- configs: 설정 파일

## 빠른 시작
1. requirements.txt 설치
2. 데이터 수집 및 학습
3. 추론 및 배포
