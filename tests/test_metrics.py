import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ealz.metrics import evaluate

Y = [0, 0, 1, 1, 2, 2]
P = [
    [0.8, 0.1, 0.1],
    [0.4, 0.5, 0.1],
    [0.1, 0.8, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.2, 0.7],
    [0.5, 0.1, 0.4],
]


def test_counts_based_metrics_match_hand_computation():
    # Predictions are [0, 1, 1, 1, 2, 0], so the confusion matrix is [[1, 1, 0], [0, 2, 0], [1, 0, 1]].
    m = evaluate(Y, P, ("a", "b", "c"), n_bootstrap=200)
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert m["accuracy"] == pytest.approx(4 / 6)
    expected = {
        "a": dict(precision=1 / 2, recall=1 / 2, specificity=3 / 4, npv=3 / 4),
        "b": dict(precision=2 / 3, recall=1.0, specificity=3 / 4, npv=1.0),
        "c": dict(precision=1.0, recall=1 / 2, specificity=1.0, npv=4 / 5),
    }
    for name, values in expected.items():
        for metric, value in values.items():
            assert m["per_class"][name][metric] == pytest.approx(value), (name, metric)
        assert m["per_class"][name]["sensitivity"] == m["per_class"][name]["recall"]
        assert m["per_class"][name]["ppv"] == m["per_class"][name]["precision"]


def test_auc_and_brier():
    m = evaluate(Y, P, ("a", "b", "c"), n_bootstrap=200)
    probs, onehot = np.array(P), np.eye(3)[Y]
    per_class_auc = [roc_auc_score(onehot[:, k], probs[:, k]) for k in range(3)]
    assert m["macro_auc"] == pytest.approx(np.mean(per_class_auc))
    assert m["micro_auc"] == pytest.approx(roc_auc_score(onehot, probs, average="micro"))
    assert m["brier"] == pytest.approx(np.mean([np.mean((probs[:, k] - onehot[:, k]) ** 2) for k in range(3)]))
    lo, hi = m["macro_auc_95ci"]
    assert 0.0 <= lo <= hi <= 1.0
