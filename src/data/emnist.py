from typing import Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import EMNIST
import torchvision.transforms as T

from src.config import EMNIST_MEAN, EMNIST_STD, RAW_EMNIST_DIR

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

def get_raw_emnist_datasets(
    data_dir: str = str(RAW_EMNIST_DIR),
    val_ratio: float = 0.1,
    seed: int = 123,
    transform: T.Compose = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Wczytanie zbioru EMNIST ByClass z deterministycznym podziałem zbioru uczącego na część treningową i walidacyjną."""
    if transform is None:
        transform = get_base_transform()
    full_train = EMNIST(root=data_dir, split="byclass", train=True, download=False, transform=transform)
    test_set = EMNIST(root=data_dir, split="byclass", train=False, download=False, transform=transform)
    val_len = int(len(full_train) * val_ratio)
    train_len = len(full_train) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_len, val_len], generator=generator)
    return train_set, val_set, test_set
