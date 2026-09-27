import os
from pathlib import Path
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ścieżka wyjściowego archiwum ZIP
OUTPUT_ZIP = PROJECT_ROOT / "alphanumeric_ocr_bundle.zip"

# Katalogi i wzorce wykluczone z archiwum
EXCLUDED_DIRS = {
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".idea",
    ".vscode",
    "raw",
    "runs",
    "manifests",
}


EXCLUDED_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".pt",
    ".pth",
    ".zip",
    ".gz",
}

def should_exclude(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDED_DIRS:
            return True
    if path.suffix in EXCLUDED_EXTENSIONS:
        return True
    return False

def main() -> None:
    print(f"Tworzenie archiwum projektu: {OUTPUT_ZIP.name}...")
    included_count = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            root_path = Path(root)

            # Pomijanie całych wykluczonych katalogów na wczesnym etapie
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

            for file_name in files:
                file_path = root_path / file_name
                rel_path = file_path.relative_to(PROJECT_ROOT)

                if should_exclude(rel_path):
                    continue

                zip_file.write(file_path, arcname=str(rel_path))
                included_count += 1

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Pomyślnie spakowano {included_count} plików do: {OUTPUT_ZIP.resolve()}")
    print(f"Rozmiar archiwum: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
