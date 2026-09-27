from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.emnist import load_or_create_split_indices

def main() -> None:
    print("Generowanie plików manifestów podziału zbioru EMNIST ByClass...")
    train_idx, val_idx = load_or_create_split_indices(dataset_length=697932, val_ratio=0.1, seed=123)
    print(f"Zapisano pomyślnie:")
    print(f"  data/manifests/train_indices.csv ({len(train_idx):,} indeksów)")
    print(f"  data/manifests/val_indices.csv   ({len(val_idx):,} indeksów)")

if __name__ == "__main__":
    main()
