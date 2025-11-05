#!/usr/bin/env python3
"""
체크포인트 파일 확인 및 분석 스크립트
"""

import sys
sys.path.append('.')

import torch
from pathlib import Path

def analyze_checkpoint(ckpt_path):
    """체크포인트 파일 분석"""
    if not Path(ckpt_path).exists():
        return None
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', {})
        
        # RNN 레이어 확인
        rnn_keys = [k for k in state_dict.keys() if 'RNN' in k]
        l0_count = len([k for k in rnn_keys if 'l0' in k])
        l1_count = len([k for k in rnn_keys if 'l1' in k])
        
        # MS3D 모델 확인
        tower_keys = [k for k in state_dict.keys() if 'tower' in k]
        
        # 모델 타입 결정
        if l1_count > 0:
            model_type = 'StackedGRU'
        elif l0_count > 0:
            model_type = 'GRU'
        elif tower_keys:
            # MS3D 모델 확인
            if any('stacked' in k.lower() for k in state_dict.keys()):
                model_type = 'MS3DStackedGRU'
            else:
                model_type = 'MS3DGRU'
        else:
            model_type = 'Unknown'
        
        return {
            'path': ckpt_path,
            'exists': True,
            'model_type': model_type,
            'rnn_l0_layers': l0_count,
            'rnn_l1_layers': l1_count,
            'tower_keys': len(tower_keys),
            'epoch': ckpt.get('epoch', 'N/A'),
            'total_params': sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
        }
    except Exception as e:
        return {
            'path': ckpt_path,
            'exists': True,
            'error': str(e)
        }

def main():
    print("=" * 80)
    print("체크포인트 파일 확인 및 분석")
    print("=" * 80)
    print()
    
    # 확인할 체크포인트 파일들
    checkpoints = [
        'archive/checkpoints_backup/checkpoints_backup/GRU_best.ckpt',
        'archive/checkpoints_backup/checkpoints_backup/MSCSGRU_best.ckpt',
        'best_model/ms3dgru_best.ckpt',
        'inference/best_models/ms3dgru_best.ckpt',
    ]
    
    results = []
    for ckpt_path in checkpoints:
        result = analyze_checkpoint(ckpt_path)
        if result:
            results.append(result)
    
    print("\n📊 체크포인트 분석 결과:")
    print("-" * 80)
    for result in results:
        if 'error' in result:
            print(f"\n❌ {result['path']}")
            print(f"   오류: {result['error']}")
        else:
            print(f"\n✅ {result['path']}")
            print(f"   모델 타입: {result['model_type']}")
            print(f"   RNN l0 레이어: {result['rnn_l0_layers']}")
            print(f"   RNN l1 레이어: {result['rnn_l1_layers']}")
            if result['tower_keys'] > 0:
                print(f"   Tower 키: {result['tower_keys']}")
            print(f"   Epoch: {result['epoch']}")
            print(f"   총 파라미터 수: {result['total_params']:,}")
    
    print("\n" + "=" * 80)
    print("결론:")
    print("=" * 80)
    
    # GRU와 StackedGRU 구분
    gru_ckpt = None
    stackedgru_ckpt = None
    for result in results:
        if result['model_type'] == 'GRU' and gru_ckpt is None:
            gru_ckpt = result
        elif result['model_type'] == 'StackedGRU' and stackedgru_ckpt is None:
            stackedgru_ckpt = result
    
    if gru_ckpt:
        print(f"\n✅ GRU 체크포인트: {gru_ckpt['path']}")
    else:
        print("\n⚠️  GRU 체크포인트를 찾을 수 없습니다.")
        print("   → archive/checkpoints_backup/checkpoints_backup/GRU_best.ckpt는 실제로 StackedGRU입니다.")
    
    if stackedgru_ckpt:
        print(f"\n✅ StackedGRU 체크포인트: {stackedgru_ckpt['path']}")
    else:
        print("\n⚠️  StackedGRU 체크포인트를 찾을 수 없습니다.")

if __name__ == "__main__":
    main()






