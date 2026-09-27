from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.seed import set_seed
from src.data.emnist import get_raw_emnist_datasets
from src.models.cnn import AlphanumericCNN

def main() -> None:
    set_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie obliczeniowe: {device}")

    # Pobranie małego podzbioru (256 próbek) do testu przeuczenia
    train_set, _, _ = get_raw_emnist_datasets()
    subset_indices = list(range(256))
    tiny_train_set = Subset(train_set, subset_indices)
    loader = DataLoader(tiny_train_set, batch_size=64, shuffle=True)

    model = AlphanumericCNN(num_classes=62).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    epochs = 40
    print(f"Rozpoczęcie testu przeuczenia na {len(tiny_train_set)} próbkach ({epochs} epok)...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0

        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()

        epoch_loss = total_loss / len(tiny_train_set)
        epoch_acc = (total_correct / len(tiny_train_set)) * 100

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoka {epoch:2d}/{epochs} | Strata: {epoch_loss:.4f} | Dokładność: {epoch_acc:6.2f}%")

    if epoch_acc >= 98.0:
        print("\n[SUKCES] Faza 2 zakończona: model z powodzeniem przeucza mały podzbiór danych.")
    else:
        print("\n[OSTRZEŻENIE] Model nie osiągnął pełnego przeuczenia, należy zweryfikować parametry uczenia.")

if __name__ == "__main__":
    main()
