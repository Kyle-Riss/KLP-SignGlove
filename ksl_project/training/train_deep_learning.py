import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from training.dataset import KSLCsvDataset
from models.deep_learning import DeepLearningPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = KSLCsvDataset(args.csv_dir)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    pipeline = DeepLearningPipeline().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        total_loss = 0
        for X, y in dl:
            X, y = X.to(device), y.to(device)
            outputs = pipeline(X)
            loss = criterion(outputs['class_logits'], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(dl):.4f}")
