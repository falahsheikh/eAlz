"""Slice-level evaluation metrics reported in the paper (Tables 8-10)."""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score


def _div(num, den):
    return float(num) / float(den) if den else float("nan")


def macro_auc(onehot, probs):
    """Unweighted mean of the one-vs-rest ROC AUCs."""
    return float(np.mean([roc_auc_score(onehot[:, k], probs[:, k]) for k in range(onehot.shape[1])]))


def bootstrap_ci(onehot, probs, statistic, n_resamples=1000, seed=0, level=0.95):
    """Percentile bootstrap confidence interval, resampling slices with replacement.

    Resamples in which some class is entirely present or entirely absent are skipped,
    since one-vs-rest AUC is undefined for them.
    """
    rng = np.random.default_rng(seed)
    n = len(probs)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        positives = onehot[idx].sum(axis=0)
        if (positives == 0).any() or (positives == n).any():
            continue
        values.append(statistic(onehot[idx], probs[idx]))
    if not values:
        return [float("nan"), float("nan")]
    tail = 100 * (1 - level) / 2
    return [float(v) for v in np.percentile(values, [tail, 100 - tail])]


def evaluate(y_true, probs, class_names, n_bootstrap=1000, seed=0):
    """Compute accuracy, per-class and macro metrics, AUCs and Brier scores.

    y_true: integer labels, shape (n,). probs: predicted probabilities, shape (n, k).
    Sensitivity equals recall and PPV equals precision; both names are reported.
    The Brier score is the mean of the one-vs-rest (class-specific) Brier scores.
    """
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    k = len(class_names)
    onehot = np.eye(k)[y_true]
    y_pred = probs.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn

    per_class = {}
    for i, name in enumerate(class_names):
        precision = _div(tp[i], tp[i] + fp[i])
        recall = _div(tp[i], tp[i] + fn[i])
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": _div(2 * precision * recall, precision + recall),
            "sensitivity": recall,
            "specificity": _div(tn[i], tn[i] + fp[i]),
            "ppv": precision,
            "npv": _div(tn[i], tn[i] + fn[i]),
            "auc": float(roc_auc_score(onehot[:, i], probs[:, i])),
            "brier": float(np.mean((probs[:, i] - onehot[:, i]) ** 2)),
            "support": int(cm[i].sum()),
        }

    macro = {
        m: float(np.mean([per_class[c][m] for c in class_names]))
        for m in ("precision", "recall", "f1", "sensitivity", "specificity", "ppv", "npv")
    }

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_auc": macro_auc(onehot, probs),
        "macro_auc_95ci": bootstrap_ci(onehot, probs, macro_auc, n_resamples=n_bootstrap, seed=seed),
        "micro_auc": float(roc_auc_score(onehot.ravel(), probs.ravel())),
        "brier": float(np.mean((probs - onehot) ** 2)),
        "macro_avg": macro,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
