#!/usr/bin/env python3
"""
통계 검증 결과 비교표 생성
training_results 폴더와 유사한 형식의 비교표 생성
"""
import json
import re
from pathlib import Path
import numpy as np

def extract_dataset_statistics():
    """로그 파일에서 데이터셋별 통계 추출"""
    log_dir = Path('lightning_logs')
    if not log_dir.exists():
        return None
    
    # 모델 매핑
    model_mapping = {
        'ms3dgru': 'MS3DGRU',
        'gru': 'GRU',
        'stackedgru': 'StackedGRU',
        'ms3dstackedgru': 'MS3DStackedGRU'
    }
    
    # 데이터셋 매핑
    dataset_mapping = {
        'unified': 'Unified',
        'yubeen': 'Yubeen',
        'jaeyeon': 'Jaeyeon'
    }
    
    results = {}
    
    # multi_seed_*.log 파일 찾기
    for log_file in log_dir.glob('multi_seed_*.log'):
        try:
            filename = log_file.stem.replace('multi_seed_', '')
            
            # 모델명 추출
            model_key = None
            for key in model_mapping.keys():
                if filename.startswith(key):
                    model_key = key
                    break
            
            if model_key is None:
                continue
            
            model_name = model_mapping[model_key]
            
            # 데이터셋 추출
            dataset_key = None
            for key in dataset_mapping.keys():
                if key in filename:
                    dataset_key = key
                    break
            
            if dataset_key is None:
                continue
            
            dataset_name = dataset_mapping[dataset_key]
            
            # 시드와 실행 번호 추출
            seed_match = re.search(r'seed(\d+)_run(\d+)', filename)
            if not seed_match:
                continue
            
            # 로그 파일에서 test/accuracy 추출
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                acc_match = None
                # 여러 패턴 시도
                acc_match = re.search(r'test/accuracy\s*[│|\|]\s*([0-9.]+)', content)
                if not acc_match:
                    lines = content.split('\n')
                    for line in reversed(lines):
                        if 'test/accuracy' in line.lower():
                            acc_match = re.search(r'([0-9]+\.[0-9]+)', line)
                            if acc_match:
                                break
                
                if acc_match:
                    acc_value = float(acc_match.group(1)) * 100
                    
                    # 결과 저장
                    key = f"{model_name}_{dataset_name}"
                    if key not in results:
                        results[key] = {
                            'model': model_name,
                            'dataset': dataset_name,
                            'accuracies': []
                        }
                    results[key]['accuracies'].append(acc_value)
        except Exception as e:
            continue
    
    # 통계 계산
    table_data = []
    summary_stats = {}
    
    for key, data in results.items():
        if len(data['accuracies']) >= 3:
            accuracies = data['accuracies']
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            min_acc = np.min(accuracies)
            max_acc = np.max(accuracies)
            
            table_data.append({
                'Model': data['model'],
                'Dataset': data['dataset'],
                'Mean (%)': round(mean_acc, 2),
                'Std (%)': round(std_acc, 2),
                'Min (%)': round(min_acc, 2),
                'Max (%)': round(max_acc, 2),
                'Runs': len(accuracies)
            })
            
            # 모델별 요약 통계
            if data['model'] not in summary_stats:
                summary_stats[data['model']] = {
                    'accuracies': [],
                    'datasets': []
                }
            summary_stats[data['model']]['accuracies'].extend(accuracies)
            summary_stats[data['model']]['datasets'].append(data['dataset'])
    
    # 모델명으로 정렬
    model_order = ['MS3DGRU', 'GRU', 'StackedGRU', 'MS3DStackedGRU']
    dataset_order = ['Unified', 'Yubeen', 'Jaeyeon']
    
    table_data.sort(key=lambda x: (
        model_order.index(x['Model']) if x['Model'] in model_order else 999,
        dataset_order.index(x['Dataset']) if x['Dataset'] in dataset_order else 999
    ))
    
    return table_data, summary_stats

def create_comparison_table(output_dir='visualizations/statistical_validation'):
    """비교표 생성"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("📊 Extracting dataset statistics from log files...")
    table_data, summary_stats = extract_dataset_statistics()
    
    if not table_data:
        print("⚠️  No data found. Please check log files.")
        return
    
    # 1. 텍스트 형식 비교표
    txt_file = output_path / 'statistical_comparison_table.txt'
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Statistical Validation Results Summary (5 Runs per Model-Dataset)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Model':<20} {'Dataset':<15} {'Mean (%)':<12} {'Std (%)':<12} {'Min (%)':<12} {'Max (%)':<12} {'Runs':<8}\n")
        f.write("-" * 80 + "\n")
        
        for row in table_data:
            f.write(f"{row['Model']:<20} {row['Dataset']:<15} {row['Mean (%)']:<12.2f} "
                   f"{row['Std (%)']:<12.2f} {row['Min (%)']:<12.2f} {row['Max (%)']:<12.2f} "
                   f"{row['Runs']:<8}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Model Summary (across all datasets)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Model':<20} {'Mean (%)':<12} {'Std (%)':<12} {'Min (%)':<12} {'Max (%)':<12} {'Total Runs':<12}\n")
        f.write("-" * 80 + "\n")
        
        for model in ['MS3DGRU', 'GRU', 'StackedGRU', 'MS3DStackedGRU']:
            if model in summary_stats:
                accuracies = summary_stats[model]['accuracies']
                f.write(f"{model:<20} {np.mean(accuracies):<12.2f} {np.std(accuracies):<12.2f} "
                       f"{np.min(accuracies):<12.2f} {np.max(accuracies):<12.2f} {len(accuracies):<12}\n")
    
    print(f"✅ Text table saved: {txt_file}")
    
    # 2. JSON 형식 데이터
    json_file = output_path / 'statistical_comparison_table.json'
    json_data = {
        'dataset_results': table_data,
        'model_summary': {}
    }
    
    for model in ['MS3DGRU', 'GRU', 'StackedGRU', 'MS3DStackedGRU']:
        if model in summary_stats:
            accuracies = summary_stats[model]['accuracies']
            json_data['model_summary'][model] = {
                'Mean (%)': round(np.mean(accuracies), 2),
                'Std (%)': round(np.std(accuracies), 2),
                'Min (%)': round(np.min(accuracies), 2),
                'Max (%)': round(np.max(accuracies), 2),
                'Total Runs': len(accuracies),
                'Datasets': list(set(summary_stats[model]['datasets']))
            }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON data saved: {json_file}")
    
    # 3. 간단한 요약 출력
    print("\n" + "=" * 80)
    print("Dataset-wise Results")
    print("=" * 80)
    for row in table_data:
        print(f"{row['Model']:<20} {row['Dataset']:<15} Mean: {row['Mean (%)']:.2f}% "
              f"(Std: {row['Std (%)']:.2f}%, Range: {row['Min (%)']:.2f}%-{row['Max (%)']:.2f}%)")
    
    print("\n" + "=" * 80)
    print("Model Summary (across all datasets)")
    print("=" * 80)
    for model in ['MS3DGRU', 'GRU', 'StackedGRU', 'MS3DStackedGRU']:
        if model in summary_stats:
            accuracies = summary_stats[model]['accuracies']
            print(f"{model:<20} Mean: {np.mean(accuracies):.2f}% "
                  f"(Std: {np.std(accuracies):.2f}%, Range: {np.min(accuracies):.2f}%-{np.max(accuracies):.2f}%, "
                  f"Runs: {len(accuracies)})")

if __name__ == '__main__':
    print("=" * 80)
    print("Statistical Validation Comparison Table Generator")
    print("=" * 80)
    print()
    create_comparison_table()
    print("\n✅ Complete!")







