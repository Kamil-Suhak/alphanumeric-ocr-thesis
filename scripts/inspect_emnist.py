from pathlib import Path
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import FIGURES_DIR
from src.data.emnist import fix_emnist_orientation, get_raw_emnist_datasets, index_to_char

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "emnist_samples.png"

    # Transformacja wizualizacyjna bez normalizacji (wyłącznie korekta orientacji i skalowanie do 32x32)
    viz_transform = T.Compose([
        T.Lambda(fix_emnist_orientation),
        T.Resize((32, 32)),
        T.ToTensor(),
    ])

    train_set, _, _ = get_raw_emnist_datasets(transform=viz_transform)

    # Wybór reprezentatywnego podzbioru: 10 cyfr (0-9), 10 wielkich liter (10-19) oraz 10 małych liter (36-45)
    target_labels = set(range(10)) | set(range(10, 20)) | set(range(36, 46))
    samples = {}

    print("Pobieranie próbek znaków do weryfikacji wizualnej...")
    for img, label in train_set:
        if label in target_labels and label not in samples:
            samples[label] = img
        if len(samples) == len(target_labels):
            break

    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    row_groups = [
        list(range(10)),       # Cyfry: 0-9
        list(range(10, 20)),   # Wielkie litery: A-J
        list(range(36, 46)),   # Małe litery: a-j
    ]

    for row_idx, labels in enumerate(row_groups):
        for col_idx, label in enumerate(labels):
            ax = axes[row_idx, col_idx]
            img_tensor = samples[label].squeeze(0).numpy()
            ax.imshow(img_tensor, cmap="gray")
            char = index_to_char(label)
            ax.set_title(f"'{char}' ({label})", fontsize=10)
            ax.axis("off")

    plt.suptitle("Weryfikacja orientacji oraz etykiet próbek EMNIST ByClass", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Wykres inspekcyjny zapisano w: {out_path.resolve()}")
    print("Należy zweryfikować poprawność orientacji znaków oraz zgodność z etykietami.")

if __name__ == "__main__":
    main()
