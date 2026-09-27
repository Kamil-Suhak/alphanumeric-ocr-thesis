from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

def save_checkpoint(state: Dict[str, Any], filepath: Path | str) -> None:
    """Zapis punktu kontrolnego modelu z metadanymi."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def load_checkpoint(
    filepath: Path | str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Wczytanie wag modelu oraz stanu optymalizatora z pliku punktu kontrolnego."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
