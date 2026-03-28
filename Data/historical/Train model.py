"""
train_model.py
Script de entrenamiento con soporte GPU/CPU completo.

Ejecución:
    python train_model.py
"""

import sys
from pathlib import Path
import pandas as pd
from Data.historical.Datadownloader import DataManager
from Data.historical.Datapipeline   import DataPipeline
from IA.ModelTrainer import ModelTrainer, PPOConfig


# ╔═══════════════════════════════════════════════════════════╗
# ║              CONFIGURACIÓN — edita solo aquí              ║
# ╚═══════════════════════════════════════════════════════════╝

SYMBOL      = "AAL"       # Símbolo a entrenar
INTERVAL    = "1m"         # Timeframe: "1m" "5m" "15m" "1h" "1d"
START       = "2013-01-01" # Inicio del histórico
SOURCE      = "yfinance"   # "yfinance" | "ibkr" | "av"

# ── GPU / paralelismo ──────────────────────────────────────────────────────
GPU_DEVICE  = "cuda"   # "auto" detecta CUDA→MPS→CPU automáticamente
                       # "cuda" fuerza NVIDIA GPU
                       # "mps"  fuerza Apple Silicon
                       # "cpu"  fuerza CPU (útil para debug)

N_ENVS      = 16        # 0 = auto según hardware
                       # GPU  RTX 30/40: usa 8
                       # Apple M-series: usa 4
                       # Solo CPU      : usa cantidad de núcleos

# ── Timesteps ─────────────────────────────────────────────────────────────
# CPU 500_000  ≈ 40 min   →   GPU 2_000_000  ≈ 15 min  (mismo resultado)
TIMESTEPS   = 10_000_000
# TIMESTEPS = 500_000    # Si usas solo CPU

EVAL_FREQ       = 20_000  # Evaluar cada N pasos
N_EVAL_EPISODES = 5
EVAL_SPLIT      = 0.20    # 20% reservado para evaluación

MODE = "B"   # "A" | "B" | "C" | "D"
"""
    Modos:
    A → Pipeline completo (descarga + features + entrena)  ← primer uso
    B → Datos ya en disco
    C → Reanudar desde checkpoint
    D → Optimización de hiperparámetros (Optuna)
"""
# ── Config manual (opcional — None = auto según GPU/CPU) ──────────────────
# CUSTOM_CONFIG = None
# Ejemplo GPU RTX 3060 8GB:
CUSTOM_CONFIG = PPOConfig(
    policy="MlpPolicy",
    n_steps       = 8192,
    batch_size    = 4096,
    n_epochs      = 10,
    learning_rate = 3e-4,
)

# Optuna (solo Modo D)
OPTUNA_TRIALS      = 20
OPTUNA_STEPS_TRIAL = 500_000


# ══════════════════════════════════════════════════════════════════════════════


def make_trainer(features_ready: bool = False) -> ModelTrainer:
    return ModelTrainer(
        symbol         = SYMBOL,
        config         = CUSTOM_CONFIG,
        device         = GPU_DEVICE,
        n_envs         = N_ENVS,
        features_ready = features_ready,
    )


def modo_a():
    """Pipeline completo en un solo flujo."""
    print("\n▶  Modo A — Pipeline completo")
    pipeline         = DataPipeline(source=SOURCE)
    train_df, test_df = pipeline.run(SYMBOL, INTERVAL, START)

    trainer = make_trainer(features_ready=True)
    trainer.train(train_df, TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES, eval_split=0.0)
    return trainer, trainer.evaluate(test_df, episodes=10)


def modo_b():
    """Datos ya descargados en disco."""
    print("\n▶  Modo B — Datos desde disco")
    train_path = Path(f"IA/Data/historical/{SYMBOL}/{SYMBOL}_{INTERVAL}_train.csv")
    test_path  = Path(f"IA/Data/historical/{SYMBOL}/{SYMBOL}_{INTERVAL}_test.csv")

    if train_path.exists():
        train_df       = pd.read_csv(train_path, index_col=0, parse_dates=True)
        test_df        = pd.read_csv(test_path,  index_col=0, parse_dates=True)
        features_ready = "close_norm" in train_df.columns
    else:
        df             = DataManager().load(SYMBOL, INTERVAL)
        split          = int(len(df) * 0.8)
        train_df, test_df = df.iloc[:split], df.iloc[split:]
        features_ready = False

    trainer = make_trainer(features_ready=features_ready)
    trainer.train(train_df, TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES,
                  eval_split=EVAL_SPLIT if not features_ready else 0.0)
    return trainer, trainer.evaluate(test_df, episodes=10)


def modo_c():
    """Reanudar desde checkpoint."""
    print("\n▶  Modo C — Reanudar desde checkpoint")
    dm = DataManager()
    try:
        df = dm.load(SYMBOL, INTERVAL)
    except FileNotFoundError:
        df = dm.download(SYMBOL, INTERVAL, START, source=SOURCE)

    split    = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    trainer = make_trainer()
    trainer.train(train_df, TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES, resume=True)
    return trainer, trainer.evaluate(test_df, episodes=10)


def modo_d():
    """Optimización de hiperparámetros con Optuna."""
    print(f"\n▶  Modo D — Optuna  ({OPTUNA_TRIALS} trials × {OPTUNA_STEPS_TRIAL:,} steps)")
    pipeline          = DataPipeline(source=SOURCE)
    train_df, test_df = pipeline.run(SYMBOL, INTERVAL, START)

    trainer     = make_trainer(features_ready=True)
    best_config = trainer.optimize_hyperparams(
        train_df, n_trials=OPTUNA_TRIALS, n_steps_per_trial=OPTUNA_STEPS_TRIAL
    )

    trainer2 = ModelTrainer(
        symbol=SYMBOL, config=best_config,
        device=GPU_DEVICE, n_envs=N_ENVS, features_ready=True
    )
    trainer2.train(train_df, TIMESTEPS, EVAL_FREQ, eval_split=0.0)
    return trainer2, trainer2.evaluate(test_df, episodes=10)


def print_header():
    print("\n╔═══════════════════════════════════════════════╗")
    print("║      Model Trainer — AutoTrader IA_BackTests  🚀        ║")
    print("╚═══════════════════════════════════════════════╝")
    print(f"  Símbolo  : {SYMBOL}  |  {INTERVAL}")
    print(f"  Steps    : {TIMESTEPS:,}")
    print(f"  Device   : {GPU_DEVICE}  |  Envs: {N_ENVS if N_ENVS else 'auto'}")
    print(f"  Modo     : {MODE}")


def print_summary(m: dict):
    print("\n╔═══════════════════════════════════════════════╗")
    print("║              Resumen final                    ║")
    print("╚═══════════════════════════════════════════════╝")
    print(f"  Dispositivo  : {m.get('device_used','?').upper()}")
    print(f"  Win rate     : {m['win_rate']:.1%}")
    print(f"  Retorno      : {m['mean_return_pct']:+.2f}%")
    print(f"  Sharpe       : {m['sharpe_ratio']:.3f}")
    print(f"  Max DD       : {m['max_drawdown_pct']:.2f}%")

    ready = (m["win_rate"] >= 0.55 and m["sharpe_ratio"] >= 0.5
             and m["max_drawdown_pct"] <= 20.0)
    print()
    if ready:
        print("  ✓ Modelo aprobado. Próximos pasos:")
        print("    1. Abre TWS en modo Paper Trading (puerto 7497)")
        print("    2. python 'IBKR Bot.py'")
    else:
        print("  ✗ Modelo necesita ajustes:")
        if m["win_rate"]         < 0.55: print("    → Aumenta TIMESTEPS")
        if m["sharpe_ratio"]     < 0.5:  print("    → Prueba Modo D (Optuna)")
        if m["max_drawdown_pct"] > 20.0: print("    → Revisa RiskManager config")
    print()


if __name__ == "__main__":
    print_header()

    modos = {"A": modo_a, "B": modo_b, "C": modo_c, "D": modo_d}
    if MODE not in modos:
        print(f"ERROR: MODE='{MODE}' no válido. Usa A/B/C/D.")
        sys.exit(1)

    trainer, metrics = modos[MODE]()
    print_summary(metrics)