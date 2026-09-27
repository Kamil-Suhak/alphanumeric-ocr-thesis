from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_EMNIST_DIR = DATA_DIR / "raw" / "emnist"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RUNS_DIR = RESULTS_DIR / "runs"

# Parametry normalizacji wyznaczone eksperymentalnie na 90% zbioru treningowego ByClass (scripts/compute_normalization.py)
EMNIST_MEAN = 0.1739
EMNIST_STD = 0.3170


