import gzip
from pathlib import Path
import shutil
import sys
import urllib.request

TARGET_DIR = Path("data/raw/emnist/EMNIST/raw")
TARGET_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "https://huggingface.co/datasets/Royc30ne/emnist-byclass/resolve/main"

FILES = [
    "emnist-byclass-mapping.txt",
    "emnist-byclass-train-images-idx3-ubyte.gz",
    "emnist-byclass-train-labels-idx1-ubyte.gz",
    "emnist-byclass-test-images-idx3-ubyte.gz",
    "emnist-byclass-test-labels-idx1-ubyte.gz",
]

def download_file(filename: str) -> None:
    target_path = TARGET_DIR / filename
    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"[Pominięto] {filename} już istnieje ({target_path.stat().st_size / 1e6:.2f} MB).")
        return
    url = f"{BASE_URL}/{filename}"
    print(f"[Pobieranie] {filename}...")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            sys.stdout.write(f"\r  -> {percent:5.1f}% ({downloaded / 1e6:.1f} MB / {total_size / 1e6:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, target_path, reporthook=reporthook)
    print()

def decompress_files() -> None:
    """Dekompresja archiwów gzip do postaci binarnej wymaganej przez moduł torchvision."""
    for gz_path in TARGET_DIR.glob("*.gz"):
        uncompressed_path = gz_path.with_suffix("")
        if not uncompressed_path.exists():
            print(f"Rozpakowywanie {gz_path.name} -> {uncompressed_path.name}...")
            with gzip.open(gz_path, "rb") as f_in, open(uncompressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            print(f"[Gotowe] {uncompressed_path.name}")

def main() -> None:
    print(f"Katalog docelowy: {TARGET_DIR.resolve()}")
    for f in FILES:
        download_file(f)

    print("Weryfikacja i dekompresja plików binarnych dla torchvision...")
    decompress_files()
    print("Inicjalizacja zbioru EMNIST ByClass zakończona.")

if __name__ == "__main__":
    main()