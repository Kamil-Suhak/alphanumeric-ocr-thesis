from typing import Dict, Any, Tuple
import numpy as np

# 15 par liter nieodróżnialnych bez kontekstu linii bazowej (Cohen et al., 2017)
AMBIGUOUS_CASE_PAIRS = [
    ("C", "c"), ("I", "i"), ("J", "j"), ("K", "k"), ("L", "l"),
    ("M", "m"), ("O", "o"), ("P", "p"), ("S", "s"), ("U", "u"),
    ("V", "v"), ("W", "w"), ("X", "x"), ("Y", "y"), ("Z", "z")
]

def compute_confusion_matrix(targets: np.ndarray, preds: np.ndarray, num_classes: int = 62) -> np.ndarray:
    """Konstrukcja macierzy pomyłek o wymiarach num_classes x num_classes."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm

def compute_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    num_classes: int = 62,
    eps: float = 1e-9,
) -> Dict[str, Any]:
    """Obliczanie podstawowych metryk jakości klasyfikacji: dokładności ogólnej oraz makro-F1."""
    targets = np.asarray(targets)
    preds = np.asarray(preds)

    total_samples = len(targets)
    if total_samples == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}

    accuracy = float((targets == preds).sum() / total_samples)
    cm = compute_confusion_matrix(targets, preds, num_classes=num_classes)

    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = (2 * p * r) / (p + r + eps)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "confusion_matrix": cm,
    }

def compute_case_insensitive_accuracy(
    targets: np.ndarray,
    preds: np.ndarray,
    class_names: list,
) -> float:
    """Dokładność z uwzględnieniem równoważności wielkości liter dla 15 par nierozróżnialnych."""
    from src.data.emnist import CHAR_TO_INDEX

    equivalent_map = {}
    for upper_char, lower_char in AMBIGUOUS_CASE_PAIRS:
        if upper_char in CHAR_TO_INDEX and lower_char in CHAR_TO_INDEX:
            upper_idx = CHAR_TO_INDEX[upper_char]
            lower_idx = CHAR_TO_INDEX[lower_char]
            equivalent_map[lower_idx] = upper_idx

    matched = 0
    for t, p in zip(targets, preds):
        norm_t = equivalent_map.get(t, t)
        norm_p = equivalent_map.get(p, p)
        if norm_t == norm_p:
            matched += 1

    return float(matched / len(targets)) if len(targets) > 0 else 0.0
