# 파일 정리 계획

## 🗑️ 삭제할 파일들

### 1. 오래된 실험 파일들 (training/)
- `safe_training.py` - 안전 훈련 (성능 낮음)
- `stratified_training.py` - 계층적 샘플링 (과적합)
- `file_based_training.py` - 파일 기반 분할 (구식)
- `balanced_training.py` - 균형 훈련 (구식)
- `anti_overfitting_training.py` - 과적합 방지 (구식)
- `optimized_training.py` - 최적화 훈련 (구식)
- `perfect_training.py` - 완벽 훈련 (구식)

### 2. 오래된 모델 파일들
- `safe_training_model.pth` (856KB)
- `stratified_model.pth` (990KB)
- `file_based_model.pth` (988KB)
- `balanced_model.pth` (988KB)
- `anti_overfitting_model.pth` (583KB)
- `optimized_training_model.pth` (664KB)
- `perfect_training_model.pth` (1.8MB)

### 3. 오래된 분석 파일들
- `safe_training_*.json` (3개)
- `stratified_*.json` (3개)
- `file_based_*.json` (3개)
- `balanced_*.json` (3개)
- `anti_overfitting_*.json` (3개)
- `optimized_training_*.json` (3개)
- `perfect_training_*.json` (3개)

### 4. 오래된 시각화 파일들
- `safe_training_curves.png` (187KB)
- `stratified_training_curves.png` (402KB)
- `file_based_training_curves.png` (347KB)
- `balanced_training_curves.png` (392KB)
- `anti_overfitting_training_curves.png` (296KB)
- `optimized_training_curves.png` (219KB)
- `perfect_training_curves.png` (272KB)

### 5. 루트 디렉토리 오래된 파일들
- `overfitting_analysis_report.json`
- `overfitting_analysis.png`
- `class_accuracy_ranking_summary.json`
- `class_accuracy_detailed_analysis.png`
- `improved_preprocessing_*.json/png/pth` (5개)
- `best_improved_preprocessing_model.pth`
- `data_quality_*.txt/json/png` (3개)
- `complementary_filter_analysis_report.txt`
- `dataset_analysis_report.txt`
- `yeo_teul_rieul_sensor_comparison.png`
- `cha_ya_sensor_comparison.png`

### 6. 기타 오래된 파일들
- `complete_pipeline.py`
- `yaw_drift_correction_pipeline.py`
- `train_with_yaw_drift_correction.py`
- `yaw_drift_correction_*.json/png/pth` (5개)
- `class_accuracy_analysis.py`
- `improved_preprocessing_pipeline.py`
- `train_with_improved_preprocessing.py`
- `analyze_improved_preprocessing_results.py`
- `data_quality_improvement.py`
- `train_final_complementary_filter.py`
- `train_enhanced_complementary_filter.py`
- `analyze_low_performance_classes.py`
- `train_with_complementary_filter.py`
- `complementary_filter_analysis.py`
- `sensor_separability_analysis.py`
- `comprehensive_dataset_analysis.py`
- `customized_solution_trainer.py`
- `dataset_analyzer.py`
- `train_ensemble_model_fixed.py`
- `train_ensemble_model.py`
- `train_specialized_model.py`
- `analyze_yeo_teul_rieul_classes.py`
- `analyze_cha_ya_classes.py`
- `solve_class_issues.py`
- `analyze_failed_classes.py`
- `analyze_overfitting_classes.py`

## 💾 보존할 파일들

### 1. 최종 모델들
- `cross_validation_model.pth` - 최고 성능 모델
- `specialized_model.pth` - 특화 모델
- `ensemble_model.pth` - 앙상블 모델
- `optimized_cv_model.pth` - 최적화 모델 (진행 중)

### 2. 최종 분석 파일들
- `final_model_performance_report.json`
- `specialized_model_analysis_report.json`
- `ensemble_model_report.json`
- `cross_validation_results.json`

### 3. 최종 시각화 파일들
- `final_model_analysis.png`
- `specialized_model_analysis.png`
- `ensemble_model_analysis.png`
- `cross_validation_analysis.png`
- `optimized_cv_training_curves.png`

### 4. 최종 훈련 스크립트들
- `cross_validation_training.py`
- `specialized_model_trainer.py`
- `ensemble_model_trainer.py`
- `optimized_cv_trainer.py`

### 5. 분석 스크립트들
- `final_model_analysis.py`
- `specialized_model_analysis.py`

### 6. 핵심 유틸리티
- `label_mapping.py`
- `dataset.py`
- `__init__.py`

### 7. 요약 문서들
- `project_complete_summary.md`
- `optimization_summary.md`

## 📊 정리 후 예상 효과
- **삭제할 파일 수**: 약 80개
- **절약할 용량**: 약 15-20MB
- **정리된 구조**: 핵심 파일들만 남김
- **가독성 향상**: 중요한 파일들만 남아서 찾기 쉬움
