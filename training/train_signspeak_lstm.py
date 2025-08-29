#!/usr/bin/env python3
"""
SignSpeakLSTM 단독 훈련
- SignSpeakLSTM 모델만 K-Fold 교차 검증으로 훈련
- SignSpeak의 92% 정확도에 도전
"""

import sys
sys.path.append('.')
from training.signspeak_cross_validation import SignSpeakTrainer

def main():
    """SignSpeakLSTM 훈련"""
    print("🚀 SignSpeakLSTM 훈련 시작!")
    
    # SignSpeakLSTM 훈련기 생성
    trainer = SignSpeakTrainer(model_type='lstm', n_folds=5)
    
    # 교차 검증 수행
    cv_results, best_model = trainer.cross_validate()
    
    print(f"\n🎉 SignSpeakLSTM 훈련 완료!")
    print(f"📊 평균 검증 정확도: {cv_results['mean_validation_accuracy']:.4f} ± {cv_results['std_validation_accuracy']:.4f}")
    print(f"📊 최고 검증 정확도: {cv_results['max_validation_accuracy']:.4f}")
    
    if cv_results['max_validation_accuracy'] >= 0.92:
        print(f"🎉 SignSpeak 92% 정확도 달성!")
    else:
        gap = 0.92 - cv_results['max_validation_accuracy']
        print(f"📈 SignSpeak 92% 정확도까지 {gap:.4f} 부족")
    
    print(f"\n📁 저장된 파일:")
    print(f"   - signspeak_lstm_cv_model.pth")
    print(f"   - signspeak_lstm_cv_results.json")
    print(f"   - signspeak_lstm_cv_analysis.png")

if __name__ == "__main__":
    main()
