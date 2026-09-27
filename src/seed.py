import os
import random
import numpy as np
import torch

def set_seed(seed: int = 123) -> None:
    """Inicjalizacja ziarna generatorów liczb losowych dla powtarzalności wyników."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
