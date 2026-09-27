from pathlib import Path
import sys
import torch
from torchvision.datasets import EMNIST

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import RAW_EMNIST_DIR, RESULTS_DIR
from src.data.emnist import CLASS_NAMES
from src.data.transforms import BLUR_SEVERITY_LEVELS, NOISE_SEVERITY_LEVELS, PERSPECTIVE_SEVERITY_LEVELS
from src.models.cnn import AlphanumericCNN
from src.utils.checkpoint import load_checkpoint
from src.experiments.degradation_sweep import (
    run_single_sweep,
    plot_robustness_curve,
    save_sweep_csv,
)

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ewaluacja porównawcza odporności modeli na urządzeniu: {device}")

    runs_dir = RESULTS_DIR / "runs"
    clean_ckpt_path = runs_dir / "EXP-001" / "best.pt"
    robust_ckpt_path = runs_dir / "EXP-002" / "best.pt"

    if not clean_ckpt_path.exists():
        print(f"[BŁĄD] Brak punktu kontrolnego Modelu A (Czystego): {clean_ckpt_path}")
        print("Należy najpierw uruchomić skrypt: python scripts/train_baseline.py")
        return

    if not robust_ckpt_path.exists():
        print(f"[BŁĄD] Brak punktu kontrolnego Modelu B (Odpornego): {robust_ckpt_path}")
        print("Należy najpierw uruchomić skrypt: python scripts/train_robust.py")
        return

    # Inicjalizacja modeli
    clean_model = AlphanumericCNN(num_classes=len(CLASS_NAMES)).to(device)
    load_checkpoint(clean_ckpt_path, clean_model, device=device)
    clean_model.eval()

    robust_model = AlphanumericCNN(num_classes=len(CLASS_NAMES)).to(device)
    load_checkpoint(robust_ckpt_path, robust_model, device=device)
    robust_model.eval()

    # Wczytanie surowego zbioru testowego EMNIST
    raw_test = EMNIST(root=str(RAW_EMNIST_DIR), split="byclass", train=False, download=False, transform=None)
    print(f"Wczytano {len(raw_test):,} surowych próbek zbioru testowego.")

    robustness_out_dir = RESULTS_DIR / "robustness"
    robustness_out_dir.mkdir(parents=True, exist_ok=True)

    sweeps = [
        ("blur", BLUR_SEVERITY_LEVELS, "Rozmycie gaussowskie ($\\sigma$)", "Odporność na rozmycie gaussowskie"),
        ("noise", NOISE_SEVERITY_LEVELS, "Szum gaussowski ($\\sigma$)", "Odporność na szum gaussowski"),
        ("perspective", PERSPECTIVE_SEVERITY_LEVELS, "Zniekształcenie perspektywiczne (%)", "Odporność na zniekształcenia perspektywiczne"),
    ]

    for deg_type, levels, xlabel, title in sweeps:
        print(f"\n--- Rozpoczęcie testu: {deg_type.upper()} ---")
        print("Ewaluacja Modelu A (Czystego)...")
        clean_res = run_single_sweep(clean_model, raw_test, deg_type, levels, device)

        print("Ewaluacja Modelu B (Odpornego)...")
        robust_res = run_single_sweep(robust_model, raw_test, deg_type, levels, device)

        csv_path = robustness_out_dir / f"{deg_type}.csv"
        save_sweep_csv(clean_res, robust_res, csv_path)

        plot_path = robustness_out_dir / f"{deg_type}_curve.png"
        plot_robustness_curve(clean_res, robust_res, xlabel, title, plot_path)
        print(f"Zapisano wykres i tabelę: {plot_path.name}, {csv_path.name}")

    print("\nKompletne testy odporności zakończone pomyślnie.")

if __name__ == "__main__":
    main()
