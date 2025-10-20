#!/usr/bin/env python3
"""
Scale-Aware GRU 모델 비교 학습 스크립트
"""
import subprocess
import sys
import time
from datetime import datetime

def run_training(model_name, model_type, description, epochs=50):
    """단일 모델 학습 실행"""
    print(f"\n{'='*80}")
    print(f"🚀 {model_name} 학습 시작")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    cmd = [
        "python3", "src/experiments/LightningTrain.py",
        "-model", model_name,
        "-model_type", model_type,
        "-epochs", str(epochs),
        "-batch_size", "32",
        "-lr", "0.001",
        "-hidden_size", "64",
        "-description", description
    ]
    
    log_file = f"training_output_{model_name}.log"
    
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
            
            if process.returncode != 0:
                print(f"\n❌ {model_name} 학습 실패 (exit code: {process.returncode})")
                return False
        
        elapsed = time.time() - start_time
        print(f"\n✅ {model_name} 학습 완료 (소요 시간: {elapsed:.1f}초)")
        print(f"📄 로그 파일: {log_file}")
        return True
        
    except Exception as e:
        print(f"\n❌ {model_name} 학습 중 오류 발생: {e}")
        return False

def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                  Scale-Aware GRU 모델 비교 학습 시작                         ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"학습 설정:")
    print(f"  - Epochs: 50")
    print(f"  - Batch Size: 32")
    print(f"  - Learning Rate: 0.001")
    print(f"  - Hidden Size: 64")
    print()
    
    models = [
        ("MSCSGRU", "MSCSGRU", "Baseline MSCSGRU for comparison"),
        ("MSCSGRU_ScaleAware", "MSCSGRU_ScaleAware", "Scale-Aware GRU with Sigmoid/Tanh"),
        ("MSCSGRU_ScaleHard", "MSCSGRU_ScaleAware", "Scale-Aware GRU with Hard Functions"),
        ("MSCGRU_ScaleAware", "MSCSGRU_ScaleAware", "Scale-Aware Single GRU"),
    ]
    
    results = {}
    total_start = time.time()
    
    for i, (model_name, model_type, description) in enumerate(models, 1):
        print(f"\n📊 진행: {i}/{len(models)}")
        success = run_training(model_name, model_type, description, epochs=50)
        results[model_name] = "✅ 성공" if success else "❌ 실패"
        
        if not success:
            print(f"\n⚠️  경고: {model_name} 학습 실패, 계속 진행합니다...")
        
        # 다음 모델 학습 전 잠시 대기
        if i < len(models):
            print("\n⏳ 다음 모델 학습 준비 중...")
            time.sleep(2)
    
    total_elapsed = time.time() - total_start
    
    print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                         모든 모델 학습 완료!                                ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 학습 결과 요약:
""")
    
    for model_name, result in results.items():
        print(f"  {model_name}: {result}")
    
    print(f"""
⏱️  총 소요 시간: {total_elapsed/60:.1f}분
📁 로그 파일들:
""")
    
    for model_name, _, _ in models:
        print(f"  - training_output_{model_name}.log")
    
    print(f"""
📈 다음 단계:
  python3 analyze_scale_aware_results.py

종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {e}")
        sys.exit(1)

