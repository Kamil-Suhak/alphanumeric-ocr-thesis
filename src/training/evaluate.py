from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.metrics import compute_metrics, compute_case_insensitive_accuracy
from src.data.emnist import CLASS_NAMES

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Kompleksowa ewaluacja modelu ze zwrotem pełnego wektora predykcji oraz etykiet."""
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets.numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    metrics = compute_metrics(y_true, y_pred, num_classes=len(CLASS_NAMES))
    case_ins_acc = compute_case_insensitive_accuracy(y_true, y_pred, CLASS_NAMES)
    metrics["case_insensitive_accuracy"] = case_ins_acc

    return metrics, y_true, y_pred
