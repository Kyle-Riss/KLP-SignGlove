"""
빠른 성능 평가 (플롯 없이): 정확도/매크로 F1/로스(옵션)만 출력
사용법 예시:
  python scripts/eval_quick.py --model MS3DGRU \
    --ckpt best_model/ms3dgru_best.ckpt \
    --data_dir /home/billy/25-1kp/SignGlove-DataAnalysis/unified/unified_HB
"""

import sys
sys.path.append('.')

import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.misc.DynamicDataModule import DynamicDataModule
from inference import SignGloveInference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['GRU', 'StackedGRU', 'MS3DGRU', 'MS3DStackedGRU'])
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    print('=' * 80)
    print(f'빠른 평가 시작: {args.model} | data_dir={args.data_dir}')
    print('=' * 80)

    engine = SignGloveInference(
        model_path=args.ckpt,
        model_type=args.model,
        device=args.device,
    )

    datamodule = DynamicDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_size=0.2,
        val_size=0.2,
        seed=42,
    )
    # 캐시 분할이 서로 다른 데이터셋 간에 공유되지 않도록, 각 실행 프로세스 내에서만 캐시됨
    datamodule.setup('test')
    loader = datamodule.test_dataloader()

    preds, labels, probs = [], [], []
    engine.model.eval()
    device = torch.device(args.device)
    engine.model.to(device)

    with torch.no_grad():
        for batch in loader:
            x = batch['measurement'].to(device)
            y = batch['label'].to(device)
            logits = engine.model.predict(x)
            p = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(p.cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)
    probs = np.array(probs)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    try:
        ll = log_loss(labels, probs)
    except Exception:
        ll = float('nan')

    print(f'Accuracy: {acc*100:.2f}%')
    print(f'F1 (Macro): {f1_macro:.4f}')
    print(f'LogLoss: {ll:.6f}')


if __name__ == '__main__':
    main()


