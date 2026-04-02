"""
gpu_check.py
Diagnóstico completo del entorno GPU antes de entrenar.
Ejecutar UNA VEZ para verificar que todo está configurado correctamente.

    python IA_BackTests/gpu_check.py
"""
import sys
def check_python():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 9
    print(f"  Python {v.major}.{v.minor}.{v.micro}  {'✓' if ok else '✗ (requiere 3.9+)'}")
    return ok


def check_torch():
    try:
        import torch
        print(f"  PyTorch {torch.__version__}  ✓")

        # CUDA
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            n   = torch.cuda.device_count()
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  CUDA disponible  ✓  ({n} GPU: {gpu}  |  VRAM: {mem:.1f} GB)")
            print(f"  CUDA versión     : {torch.version.cuda}")
            print(f"  cuDNN            : {torch.backends.cudnn.version()}")

            # Benchmark rápido
            import time
            size  = 4096
            a     = torch.randn(size, size, device="cuda")
            b     = torch.randn(size, size, device="cuda")
            torch.cuda.synchronize()
            t0    = time.time()
            _     = torch.matmul(a, b)
            torch.cuda.synchronize()
            ms    = (time.time() - t0) * 1000
            print(f"  Benchmark CUDA   : {size}x{size} matmul en {ms:.1f} ms  ✓")
            return "cuda"

        # Apple Silicon MPS
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if mps_ok:
            print(f"  MPS (Apple Silicon) disponible  ✓")
            return "mps"

        print("  CUDA: no disponible  ✗")
        print("  MPS : no disponible  ✗")
        print("  → Se usará CPU")
        return "cpu"

    except ImportError:
        print("  PyTorch: NO instalado  ✗")
        print("  Instala con: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return None


def check_sb3():
    try:
        import stable_baselines3 as sb3
        print(f"  Stable-Baselines3 {sb3.__version__}  ✓")
        return True
    except ImportError:
        print("  Stable-Baselines3: NO instalado  ✗")
        print("  Instala: pip install stable-baselines3[extra]")
        return False


def check_multiprocessing():
    try:
        import multiprocessing
        n = multiprocessing.cpu_count()
        print(f"  CPU cores disponibles: {n}  ✓")
        return n
    except Exception as e:
        print(f"  CPU cores: error  ({e})")
        return 1


def recommend_config(device: str, vram_gb: float = 0, cpu_cores: int = 4):
    """Imprime la configuración óptima para el hardware detectado."""
    print(f"\n{'─'*55}")
    print("  Configuración recomendada para tu hardware:")
    print(f"{'─'*55}")

    if device == "cuda":
        if vram_gb >= 8:
            n_envs, batch, n_steps = 8, 2048, 4096
        elif vram_gb >= 4:
            n_envs, batch, n_steps = 4, 1024, 2048
        else:
            n_envs, batch, n_steps = 2, 512, 2048

        print(f"""
  # Pega esto en train_model.py → sección CONFIGURACIÓN GPU:

  GPU_DEVICE   = "cuda"
  N_ENVS       = {n_envs}      # Entornos paralelos
  CUSTOM_CONFIG = PPOConfig(
      batch_size    = {batch},
      n_steps       = {n_steps},
      n_epochs      = 10,
      learning_rate = 3e-4,
  )
  # Timesteps equivalentes en GPU vs CPU:
  # CPU  500_000 ~40 min  →  GPU 2_000_000 ~15 min (4x más rápido)
        """)

    elif device == "mps":
        print(f"""
  # Apple Silicon (M1/M2/M3):

  GPU_DEVICE   = "mps"
  N_ENVS       = 4
  CUSTOM_CONFIG = PPOConfig(
      batch_size    = 512,
      n_steps       = 2048,
  )
        """)

    else:
        n_envs = min(cpu_cores, 8)
        print(f"""
  # Sin GPU — optimización para CPU:

  GPU_DEVICE   = "cpu"
  N_ENVS       = {n_envs}      # Paralelismo en CPU
  CUSTOM_CONFIG = PPOConfig(
      batch_size    = 256,
      n_steps       = 2048,
  )
        """)


def main():
    print("╔═══════════════════════════════════════════════╗")
    print("║         GPU Check — AutoTrader  IA_BackTests            ║")
    print("╚═══════════════════════════════════════════════╝\n")

    print("── Sistema ──────────────────────────────────────")
    check_python()
    cpu_cores = check_multiprocessing()

    print("\n── PyTorch / GPU ────────────────────────────────")
    device = check_torch()

    print("\n── Dependencias RL ──────────────────────────────")
    check_sb3()

    # VRAM para la recomendación
    vram = 0
    if device == "cuda":
        try:
            import torch
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            pass

    if device:
        recommend_config(device, vram, cpu_cores)

    print("─" * 55)
    if device == "cuda":
        print("  ✓ GPU lista. Ejecuta: python train_model.py")
    elif device == "mps":
        print("  ✓ Apple GPU lista. Ejecuta: python train_model.py")
    elif device == "cpu":
        print("  ⚠ Sin GPU. El entrenamiento será más lento.")
        print("  Consejo: usa N_ENVS > 1 para paralelismo en CPU.")
    else:
        print("  ✗ Instala PyTorch antes de continuar.")
    print()


if __name__ == "__main__":
    main()