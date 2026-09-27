from pathlib import Path
from typing import List, Dict, Any, Callable
import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from src.data.emnist import fix_emnist_orientation, EMNIST_MEAN, EMNIST_STD
from src.data.transforms import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_perspective_distortion,
    BLUR_SEVERITY_LEVELS,
    NOISE_SEVERITY_LEVELS,
    PERSPECTIVE_SEVERITY_LEVELS,
)
from src.training.evaluate import evaluate_model

class ControlledDegradationDataset(Dataset):
    """Zbiór danych nakładający deterministyczną degradację o zadanym poziomie na każdą próbkę."""

    def __init__(
        self,
        base_dataset: Dataset,
        degradation_fn: Callable[[np.ndarray], np.ndarray],
        mean: float = EMNIST_MEAN,
        std: float = EMNIST_STD,
    ):
        self.base_dataset = base_dataset
        self.degradation_fn = degradation_fn
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img_pil, target = self.base_dataset[idx]
        img_pil = fix_emnist_orientation(img_pil).resize((32, 32))
        arr = np.asarray(img_pil, dtype=np.float32) / 255.0

        if self.degradation_fn is not None:
            arr = self.degradation_fn(arr)

        tensor = torch.from_numpy(arr).unsqueeze(0)
        norm_tensor = (tensor - self.mean) / self.std
        return norm_tensor, target

def run_single_sweep(
    model: nn.Module,
    raw_test_dataset: Dataset,
    degradation_type: str,
    levels: List[float],
    device: torch.device,
    batch_size: int = 256,
) -> List[Dict[str, float]]:
    """Ewaluacja modelu na zbiorze testowym wzdłuż zadanych poziomów degradacji."""
    results = []
    rng = np.random.default_rng(123)

    for val in levels:
        if degradation_type == "blur":
            deg_fn = lambda arr, s=val: apply_gaussian_blur(arr, sigma=s)
        elif degradation_type == "noise":
            deg_fn = lambda arr, s=val: apply_gaussian_noise(arr, std=s, rng=rng)
        elif degradation_type == "perspective":
            deg_fn = lambda arr, s=val: apply_perspective_distortion(arr, severity=s, rng=rng)
        else:
            deg_fn = None

        sweep_ds = ControlledDegradationDataset(raw_test_dataset, degradation_fn=deg_fn)
        loader = DataLoader(sweep_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        metrics, _, _ = evaluate_model(model, loader, device=device)
        results.append({
            "severity": val,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "case_insensitive_accuracy": metrics["case_insensitive_accuracy"],
        })
        print(f"  {degradation_type.capitalize()} severity={val:5.2f} | Dokładność={metrics['accuracy']*100:5.2f}%")

    return results

def plot_robustness_curve(
    clean_results: List[Dict[str, float]],
    robust_results: List[Dict[str, float]],
    xlabel: str,
    title: str,
    save_path: Path | str,
) -> None:
    """Wykres porównawczy odporności (Sekcja 34 planu): Model A (Czysty) vs Model B (Odporny)."""
    severities = [r["severity"] for r in clean_results]
    clean_acc = [r["accuracy"] * 100 for r in clean_results]
    robust_acc = [r["accuracy"] * 100 for r in robust_results]

    plt.figure(figsize=(8, 5))
    plt.plot(severities, clean_acc, marker="o", color="#d62728", lw=2, label="Model A (Czysty - Clean CNN)")
    plt.plot(severities, robust_acc, marker="s", color="#2ca02c", lw=2, label="Model B (Odporny - Robust CNN)")

    # Pozioma linia kryterium załamania (50% - sekcja 35 planu)
    plt.axhline(50.0, color="gray", linestyle=":", label="Próg krytyczny (50%)")

    plt.xlabel(xlabel)
    plt.ylabel("Dokładność Top-1 (%)")
    plt.title(title)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

def save_sweep_csv(
    clean_results: List[Dict[str, float]],
    robust_results: List[Dict[str, float]],
    save_path: Path | str,
) -> None:
    """Zapis tabelaryczny wyników testów odporności do pliku CSV."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["severity", "clean_acc", "clean_macro_f1", "robust_acc", "robust_macro_f1"])
        for c, r in zip(clean_results, robust_results):
            writer.writerow([
                f"{c['severity']:.2f}",
                f"{c['accuracy']*100:.2f}",
                f"{c['macro_f1']*100:.2f}",
                f"{r['accuracy']*100:.2f}",
                f"{r['macro_f1']*100:.2f}",
            ])
