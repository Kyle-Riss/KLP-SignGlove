import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from training.dataset import KSLCsvDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, required=True)
    args = parser.parse_args()

    ds = KSLCsvDataset(args.csv_dir)
    X = np.array([x.flatten() for x,_ in ds])
    y = np.array([y for _,y in ds])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'rf': RandomForestClassifier(n_estimators=200),
        'svm': SVC(kernel='rbf', probability=True),
        'xgb': XGBClassifier()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"{name} accuracy: {acc:.3f}")
