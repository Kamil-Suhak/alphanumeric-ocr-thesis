from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import EMNIST
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import RAW_EMNIST_DIR
from src.data.emnist import fix_emnist_orientation

def main() -> None:
    print("Ładowanie zbioru treningowego EMNIST ByClass...")
    transform = T.Compose([
        T.Lambda(fix_emnist_orientation),
        T.Resize((32, 32)),
        T.ToTensor(),
    ])

    full_train = EMNIST(
        root=str(RAW_EMNIST_DIR),
        split="byclass",
        train=True,
        download=False,
        transform=transform,
    )

    # Wydzielenie 10% na zbiór walidacyjny; statystyki liczone wyłącznie na 90% części treningowej
    val_len = int(len(full_train) * 0.1)
    train_len = len(full_train) - val_len
    generator = torch.Generator().manual_seed(123)
    train_set, _ = random_split(full_train, [train_len, val_len], generator=generator)

    loader = DataLoader(train_set, batch_size=4096, shuffle=False, num_workers=0)

    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0

    print(f"Obliczanie średniej i odchylenia standardowego dla {train_len:,} próbek treningowych...")
    for step, (imgs, _) in enumerate(loader):
        imgs_d = imgs.double()
        total_sum += imgs_d.sum().item()
        total_sq_sum += (imgs_d ** 2).sum().item()
        total_pixels += imgs_d.numel()

        if (step + 1) % 25 == 0 or (step + 1) == len(loader):
            print(f"  Przetworzono {min((step + 1) * 4096, train_len):,}/{train_len:,} próbek")

    mean = total_sum / total_pixels
    std = ((total_sq_sum / total_pixels) - (mean ** 2)) ** 0.5

    print("\nObliczenia zakończone:")
    print(f"  EMNIST_MEAN = {mean:.4f}")
    print(f"  EMNIST_STD  = {std:.4f}")
    print("\nNależy zaktualizować plik src/config.py o wyznaczone wartości:")
    print(f"  EMNIST_MEAN = {mean:.4f}")
    print(f"  EMNIST_STD = {std:.4f}")

if __name__ == "__main__":
    main()
