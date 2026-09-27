from pathlib import Path
from typing import Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import EMNIST
import torchvision.transforms as T

from src.config import EMNIST_MEAN, EMNIST_STD, RAW_EMNIST_DIR, MANIFESTS_DIR

# 62 klasy wariantu ByClass: 0-9, A-Z, a-z
CLASS_NAMES = (
    list("0123456789")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
)

CHAR_TO_INDEX = {char: idx for idx, char in enumerate(CLASS_NAMES)}

def index_to_char(index: int) -> str:
    return CLASS_NAMES[index]

def char_to_index(char: str) -> int:
    return CHAR_TO_INDEX[char]

def fix_emnist_orientation(img: Image.Image) -> Image.Image:
    # Korekta orientacji: surowe obrazy w formacie EMNIST wymagają transpozycji macierzy pikseli
    return img.transpose(Image.Transpose.TRANSPOSE)

def get_base_transform(mean: float = EMNIST_MEAN, std: float = EMNIST_STD) -> T.Compose:
    return T.Compose([
        T.Lambda(fix_emnist_orientation),
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(mean=(mean,), std=(std,)),
    ])

def get_robust_transform(mean: float = EMNIST_MEAN, std: float = EMNIST_STD, seed: int = 123) -> T.Compose:
    """Potok transformacji uczących z syntetycznymi degradacjami (Model B - Robust)."""
    from src.data.transforms import DegradationPipeline
    return T.Compose([
        T.Lambda(fix_emnist_orientation),
        T.Resize((32, 32)),
        DegradationPipeline(seed=seed),
        T.ToTensor(),
        T.Normalize(mean=(mean,), std=(std,)),
    ])

class TransformedDataset(Dataset):
    """Wrapper nakładający transformację na próbki z wybranego podzbioru danych."""

    def __init__(self, base_dataset: Dataset, transform: T.Compose):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img, target = self.base_dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def load_or_create_split_indices(
    dataset_length: int,
    val_ratio: float = 0.1,
    seed: int = 123,
    manifests_dir: Path | str = MANIFESTS_DIR,
) -> Tuple[List[int], List[int]]:
    """Wczytanie lub utworzenie i utrwalenie indeksów podziału zbioru w plikach CSV."""
    manifest_path = Path(manifests_dir)
    manifest_path.mkdir(parents=True, exist_ok=True)
    train_csv = manifest_path / "train_indices.csv"
    val_csv = manifest_path / "val_indices.csv"

    if train_csv.exists() and val_csv.exists():
        with open(train_csv, "r", encoding="utf-8") as f:
            train_indices = [int(line.strip()) for line in f if line.strip().isdigit()]
        with open(val_csv, "r", encoding="utf-8") as f:
            val_indices = [int(line.strip()) for line in f if line.strip().isdigit()]
        return train_indices, val_indices

    val_len = int(dataset_length * val_ratio)
    train_len = dataset_length - val_len

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(dataset_length, generator=generator).tolist()

    train_indices = shuffled_indices[:train_len]
    val_indices = shuffled_indices[train_len:]

    with open(train_csv, "w", encoding="utf-8") as f:
        f.write("index\n")
        f.write("\n".join(str(idx) for idx in train_indices) + "\n")

    with open(val_csv, "w", encoding="utf-8") as f:
        f.write("index\n")
        f.write("\n".join(str(idx) for idx in val_indices) + "\n")

    return train_indices, val_indices

def get_raw_emnist_datasets(
    data_dir: str = str(RAW_EMNIST_DIR),
    val_ratio: float = 0.1,
    seed: int = 123,
    transform: T.Compose = None,
    train_transform: T.Compose = None,
    eval_transform: T.Compose = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Wczytanie zbioru EMNIST ByClass z jawnym podziałem na podstawie plików manifestów."""
    if transform is not None:
        if train_transform is None:
            train_transform = transform
        if eval_transform is None:
            eval_transform = transform

    if train_transform is None:
        train_transform = get_base_transform()
    if eval_transform is None:
        eval_transform = get_base_transform()

    raw_train = EMNIST(root=data_dir, split="byclass", train=True, download=False, transform=None)
    raw_test = EMNIST(root=data_dir, split="byclass", train=False, download=False, transform=None)

    train_indices, val_indices = load_or_create_split_indices(
        len(raw_train),
        val_ratio=val_ratio,
        seed=seed,
    )

    train_subset = Subset(raw_train, train_indices)
    val_subset = Subset(raw_train, val_indices)

    train_set = TransformedDataset(train_subset, train_transform)
    val_set = TransformedDataset(val_subset, eval_transform)
    test_set = TransformedDataset(raw_test, eval_transform)

    return train_set, val_set, test_set


