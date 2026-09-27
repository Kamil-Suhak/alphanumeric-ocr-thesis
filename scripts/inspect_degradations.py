from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import FIGURES_DIR
from src.data.emnist import fix_emnist_orientation, get_raw_emnist_datasets, index_to_char
from src.data.transforms import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_perspective_distortion,
    BLUR_SEVERITY_LEVELS,
    NOISE_SEVERITY_LEVELS,
    PERSPECTIVE_SEVERITY_LEVELS,
)

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "degradation_samples.png"

    # Wczytanie nieprzetworzonego obrazu demonstracyjnego
    viz_transform = T.Compose([
        T.Lambda(fix_emnist_orientation),
        T.Resize((32, 32)),
        T.ToTensor(),
    ])
    train_set, _, _ = get_raw_emnist_datasets(transform=viz_transform)

    # Wybór próbki 'A' (klasa 10) lub 'B'
    sample_img = None
    sample_char = "A"
    for img, label in train_set:
        if label == 10:
            sample_img = img.squeeze(0).numpy()
            break

    if sample_img is None:
        sample_img = train_set[0][0].squeeze(0).numpy()
        sample_char = index_to_char(train_set[0][1])

    rng = np.random.default_rng(123)

    fig, axes = plt.subplots(3, 7, figsize=(16, 7))

    # Wiersz 1: Rozmycie gaussowskie (sigma: 0.0 - 3.0)
    for col_idx, sigma in enumerate(BLUR_SEVERITY_LEVELS):
        ax = axes[0, col_idx]
        degraded = apply_gaussian_blur(sample_img, sigma=sigma)
        ax.imshow(degraded, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Blur $\\sigma={sigma}$", fontsize=9)
        ax.axis("off")

    # Wiersz 2: Szum gaussowski (std: 0.00 - 0.20)
    for col_idx, std in enumerate(NOISE_SEVERITY_LEVELS):
        ax = axes[1, col_idx]
        degraded = apply_gaussian_noise(sample_img, std=std, rng=rng)
        ax.imshow(degraded, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Noise $\\sigma={std}$", fontsize=9)
        ax.axis("off")
    axes[1, 6].axis("off")

    # Wiersz 3: Zniekształcenie perspektywiczne (severity: 0% - 20%)
    for col_idx, sev in enumerate(PERSPECTIVE_SEVERITY_LEVELS):
        ax = axes[2, col_idx]
        degraded = apply_perspective_distortion(sample_img, severity=sev, rng=rng)
        ax.imshow(degraded, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Persp. {int(sev*100)}%", fontsize=9)
        ax.axis("off")
    axes[2, 5].axis("off")
    axes[2, 6].axis("off")

    plt.suptitle(f"Przegląd syntetycznych degradacji dla znaku '{sample_char}'", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Wykres degradacji zapisano w: {out_path.resolve()}")

if __name__ == "__main__":
    main()
