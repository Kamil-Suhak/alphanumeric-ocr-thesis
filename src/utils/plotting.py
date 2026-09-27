from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(history: List[Dict[str, Any]], save_path: Path | str) -> None:
    """Wizualizacja przebiegu funkcji straty oraz dokładności w kolejnych epokach."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, train_loss, label="Uczący", color="#1f77b4", lw=2)
    ax1.plot(epochs, val_loss, label="Walidacyjny", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Funkcja straty (Cross-Entropy)")
    ax1.set_title("Krzywa uczenia - strata")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(epochs, train_acc, label="Uczący", color="#1f77b4", lw=2)
    ax2.plot(epochs, val_acc, label="Walidacyjny", color="#2ca02c", lw=2, linestyle="--")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Dokładność (%)")
    ax2.set_title("Krzywa uczenia - dokładność")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path | str,
    normalize: bool = True,
) -> None:
    """Wizualizacja pełnej macierzy pomyłek dla 62 klas."""
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    else:
        cm_display = cm

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_display, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=7, rotation=90)
    ax.set_yticklabels(class_names, fontsize=7)

    ax.set_xlabel("Klasa predykowana")
    ax.set_ylabel("Klasa rzeczywista")
    ax.set_title("Znormalizowana macierz pomyłek (62 klasy)")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
