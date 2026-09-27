from pathlib import Path
import sys
import time
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.cnn import AlphanumericCNN
from src.config import RUNS_DIR

def benchmark_latency(model: torch.nn.Module, device: torch.device, batch_size: int = 1, iterations: int = 1000) -> float:
    """Pomiar średniego czasu inferencji w milisekundach z rozgrzewką (warm-up)."""
    model.eval()
    dummy_input = torch.randn(batch_size, 1, 32, 32, device=device)

    # Faza rozgrzewki
    with torch.no_grad():
        for _ in range(30):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time

    return (total_time / iterations) * 1000.0

def main() -> None:
    print("--- Benchmark praktyczności modelu (Sekcja 36 planu) ---\n")
    model = AlphanumericCNN(num_classes=62)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Liczba parametrów (ogółem):       {total_params:,}")
    print(f"Liczba parametrów (trenowalnych): {trainable_params:,}")

    # Sprawdzenie rozmiaru pliku checkpointu na dysku
    ckpt_path = RUNS_DIR / "EXP-001" / "best.pt"
    if ckpt_path.exists():
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        print(f"Rozmiar pliku wag (best.pt):      {size_mb:.2f} MB")
    else:
        # Szacunkowy rozmiar wag float32
        est_mb = (total_params * 4) / (1024 * 1024)
        print(f"Szacowany rozmiar wag (FP32):     {est_mb:.2f} MB")

    # Pomiar na CPU
    cpu_device = torch.device("cpu")
    model_cpu = model.to(cpu_device)
    cpu_lat_1 = benchmark_latency(model_cpu, cpu_device, batch_size=1)
    cpu_lat_32 = benchmark_latency(model_cpu, cpu_device, batch_size=32)
    print(f"Czas inferencji CPU (batch=1):    {cpu_lat_1:.3f} ms")
    print(f"Czas inferencji CPU (batch=32):   {cpu_lat_32:.3f} ms (średnio {cpu_lat_32/32:.3f} ms/próbkę)")

    # Pomiar na GPU (jeśli dostępne)
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        model_gpu = model.to(gpu_device)
        gpu_lat_1 = benchmark_latency(model_gpu, gpu_device, batch_size=1)
        gpu_lat_32 = benchmark_latency(model_gpu, gpu_device, batch_size=32)
        print(f"Czas inferencji GPU (batch=1):    {gpu_lat_1:.3f} ms")
        print(f"Czas inferencji GPU (batch=32):   {gpu_lat_32:.3f} ms")

if __name__ == "__main__":
    main()
