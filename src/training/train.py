import csv
from pathlib import Path
from typing import Tuple, List, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.checkpoint import save_checkpoint

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Pojedyncza epoka uczenia modelu."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == targets).sum().item()
        total_samples += targets.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = (total_correct / total_samples) * 100.0
    return epoch_loss, epoch_acc

def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Ewaluacja modelu na zbiorze walidacyjnym."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            total_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)

    val_loss = total_loss / total_samples
    val_acc = (total_correct / total_samples) * 100.0
    return val_loss, val_acc

def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    run_dir: Path | str,
) -> List[Dict[str, Any]]:
    """Główna pętla uczenia z zapisem najlepszego punktu kontrolnego oraz logowaniem metryk."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = run_path / "metrics.csv"
    best_checkpoint_path = run_path / "best.pt"

    best_val_loss = float("inf")
    history = []

    with open(metrics_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(epoch_record)

        with open(metrics_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.4f}", f"{val_loss:.6f}", f"{val_acc:.4f}"])

        # Sekcja 13: zapis punktu kontrolnego dla minimalnej straty walidacyjnej
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                best_checkpoint_path,
            )
            saved_mark = " [* zapisano checkpoint]"
        else:
            saved_mark = ""

        print(
            f"Epoka {epoch:2d}/{epochs} | "
            f"Uczący: strata={train_loss:.4f}, dokł={train_acc:6.2f}% | "
            f"Walidacyjny: strata={val_loss:.4f}, dokł={val_acc:6.2f}%{saved_mark}"
        )

    return history
