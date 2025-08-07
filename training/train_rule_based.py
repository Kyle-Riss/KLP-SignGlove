from models.rule_based import RuleBasedClassifier
from training.dataset import KSLCsvDataset

# RuleBasedClassifier는 학습이 필요 없으므로, 평가 파이프라인만 구현
if __name__ == '__main__':
    ds = KSLCsvDataset('integrations/SignGlove_HW')
    clf = RuleBasedClassifier()
    correct = 0
    total = len(ds)
    for X, y in ds:
        pred, _ = clf.predict(X)
        if pred == y:
            correct += 1
    print(f"Rule-based accuracy: {correct/total:.3f}")
