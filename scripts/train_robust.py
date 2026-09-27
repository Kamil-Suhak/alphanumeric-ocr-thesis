import json
from pathlib import Path
import shutil
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import yaml
    def load_config(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
except ImportError:
    def load_config(path: Path) -> dict:
        return {
            "seed": 123,
            "training": {"batch_size": 128, "epochs": 25, "learning_rate": 0.001, "weight_decay": 0.0001},
            "model": {"dropout": 0.5},
            "experiment": {"name": "EXP-002"},
        }

from src.seed import set_seed
from src.config import RUNS_DIR
from src.data.emnist import CLASS_NAMES, get_raw_emnist_datasets, get_robust_transform, get_base_transform
from src.models.cnn import AlphanumericCNN
from src.training.train import fit
from src.training.evaluate import evaluate_model
from src.utils.checkpoint import load_checkpoint
from src.utils.plotting import plot_training_curves, plot_confusion_matrix

def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "robust.yaml"
    cfg = load_config(config_path)

    seed = cfg.get("seed", 123)
    set_seed(seed)

    exp_name = cfg.get("experiment", {}).get("name", "EXP-002")
    run_dir = RUNS_DIR / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        shutil.copyfile(config_path, run_dir / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rozpoczęcie eksperymentu {exp_name} (Robust CNN) na urządzeniu: {device}")

    # Uczący potok z syntetycznymi degradacjami; walidacyjny i testowy czyste (bez degradacji)
    train_tf = get_robust_transform(seed=seed)
    eval_tf = get_base_transform()

    train_set, val_set, test_set = get_raw_emnist_datasets(
        seed=seed,
        train_transform=train_tf,
        eval_transform=eval_tf,
    )

    batch_size = cfg.get("training", {}).get("batch_size", 512)
    epochs = cfg.get("training", {}).get("epochs", 25)
    lr = cfg.get("training", {}).get("learning_rate", 0.001)
    weight_decay = cfg.get("training", {}).get("weight_decay", 0.0001)
    dropout = cfg.get("model", {}).get("dropout", 0.5)

    num_workers = 2 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = AlphanumericCNN(num_classes=len(CLASS_NAMES), dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(f"Liczba parametrów modelu: {model.count_parameters():,}")
    print(f"Próbki: uczące={len(train_set):,}, walidacyjne={len(val_set):,}, testowe={len(test_set):,}")

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        run_dir=run_dir,
    )

    plot_training_curves(history, run_dir / "training_curve.png")

    best_checkpoint_path = run_dir / "best.pt"
    print(f"\nWczytywanie najlepszego punktu kontrolnego z {best_checkpoint_path}...")
    load_checkpoint(best_checkpoint_path, model, device=device)

    print("Ewaluacja ostateczna na czystym zbiorze testowym...")
    test_metrics, y_true, y_pred = evaluate_model(model, test_loader, device=device)

    cm = test_metrics.pop("confusion_matrix")
    plot_confusion_matrix(cm, CLASS_NAMES, run_dir / "confusion_matrix.png")

    test_results_path = run_dir / "test_results.json"
    with open(test_results_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print("\n" + "=" * 50)
    print(f"Wyniki ostateczne eksperymentu {exp_name} (Robust):")
    print(f"  Dokładność (Top-1 na czystym): {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Makro-F1:                      {test_metrics['macro_f1'] * 100:.2f}%")
    print(f"  Dokładność (case-insensitive): {test_metrics['case_insensitive_accuracy'] * 100:.2f}%")
    print(f"Artefakty zapisano w: {run_dir.resolve()}")
    print("=" * 50)

if __name__ == "__main__":
    main()
