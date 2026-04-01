"""
Script de entrenamiento con soporte GPU/CPU completo.
"""
import sys
from pathlib import Path
import pandas as pd
from Data.historical.Datadownloader import DataManager
from Data.historical.Datapipeline import DataPipeline
from src.brain.ModelTrainer import ModelTrainer, PPOConfig
from config import settings as IBKR_SETTINGS

CUSTOM_CONFIG = PPOConfig(
    policy=IBKR_SETTINGS.policy,
    n_steps       = IBKR_SETTINGS.n_steps,
    batch_size    = IBKR_SETTINGS.batch_size,
    n_epochs      = IBKR_SETTINGS.n_epochs,
    learning_rate = IBKR_SETTINGS.learning_rate,
)

def make_trainer(features_ready: bool = False) -> ModelTrainer:
    return ModelTrainer(
        symbol         = IBKR_SETTINGS.SYMBOL,
        config         = CUSTOM_CONFIG,
        device         = IBKR_SETTINGS.GPU_DEVICE,
        n_envs         = IBKR_SETTINGS.N_ENVS,
        features_ready = features_ready,
    )


def modo_a():
    """Pipeline completo en un solo flujo."""
    print("\n▶  Modo A — Pipeline completo")
    pipeline         = DataPipeline(source=IBKR_SETTINGS.SOURCE)
    train_df, test_df = pipeline.run(IBKR_SETTINGS.SYMBOL, IBKR_SETTINGS.INTERVAL, IBKR_SETTINGS.START)

    trainer = make_trainer(features_ready=True)
    trainer.train(train_df, IBKR_SETTINGS.TIMESTEPS, IBKR_SETTINGS.EVAL_FREQ, IBKR_SETTINGS.N_EVAL_EPISODES, eval_split=0.0)
    return trainer, trainer.evaluate(test_df, episodes=10)


def modo_b():
    """Datos ya descargados en disco."""
    print("\n▶  Modo B — Datos desde disco")
    train_path = Path(f"raw/{IBKR_SETTINGS.SYMBOL}/{IBKR_SETTINGS.SYMBOL}_{IBKR_SETTINGS.INTERVAL}_train.csv")
    test_path  = Path(f"raw/{IBKR_SETTINGS.SYMBOL}/{IBKR_SETTINGS.SYMBOL}_{IBKR_SETTINGS.INTERVAL}_test.csv")

    if train_path.exists():
        train_df       = pd.read_csv(train_path, index_col=0, parse_dates=True)
        test_df        = pd.read_csv(test_path,  index_col=0, parse_dates=True)
        features_ready = "close_norm" in train_df.columns
    else:
        df             = DataManager().load(IBKR_SETTINGS.SYMBOL, IBKR_SETTINGS.INTERVAL)
        split          = int(len(df) * 0.8)
        train_df, test_df = df.iloc[:split], df.iloc[split:]
        features_ready = False

    trainer = make_trainer(features_ready=features_ready)
    trainer.train(train_df, IBKR_SETTINGS.TIMESTEPS, IBKR_SETTINGS.EVAL_FREQ, IBKR_SETTINGS.N_EVAL_EPISODES,
                  eval_split=IBKR_SETTINGS.EVAL_SPLIT if not features_ready else 0.0)
    return trainer, trainer.evaluate(test_df, episodes=10)


def modo_c():
    """Reanudar desde checkpoint."""
    print("\n▶  Modo C — Reanudar desde checkpoint")
    dm = DataManager()
    try:
        df = dm.load(IBKR_SETTINGS.SYMBOL, IBKR_SETTINGS.INTERVAL)
    except FileNotFoundError:
        df = dm.download(IBKR_SETTINGS.SYMBOL, IBKR_SETTINGS.INTERVAL, IBKR_SETTINGS.START, source=IBKR_SETTINGS.SOURCE)

    split    = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    trainer = make_trainer()
    trainer.train(train_df, IBKR_SETTINGS.TIMESTEPS, IBKR_SETTINGS.EVAL_FREQ, IBKR_SETTINGS.N_EVAL_EPISODES, resume=True)
    return trainer, trainer.evaluate(test_df, episodes=10)


def modo_d():
    """Optimización de hiperparámetros con Optuna."""
    print(f"\n▶  Modo D — Optuna  ({IBKR_SETTINGS.OPTUNA_TRIALS} trials × {IBKR_SETTINGS.OPTUNA_STEPS_TRIAL:,} steps)")
    pipeline          = DataPipeline(source=IBKR_SETTINGS.SOURCE)
    train_df, test_df = pipeline.run(IBKR_SETTINGS.SYMBOL, IBKR_SETTINGS.INTERVAL, IBKR_SETTINGS.START)

    trainer     = make_trainer(features_ready=True)
    best_config = trainer.optimize_hyperparams(
        train_df, n_trials=IBKR_SETTINGS.OPTUNA_TRIALS, n_steps_per_trial=IBKR_SETTINGS.OPTUNA_STEPS_TRIAL
    )

    trainer2 = ModelTrainer(
        symbol=IBKR_SETTINGS.SYMBOL, config=best_config,
        device=IBKR_SETTINGS.GPU_DEVICE, n_envs=IBKR_SETTINGS.N_ENVS, features_ready=True
    )
    trainer2.train(train_df, IBKR_SETTINGS.TIMESTEPS, IBKR_SETTINGS.EVAL_FREQ, eval_split=0.0)
    return trainer2, trainer2.evaluate(test_df, episodes=10)


def print_header():
    print("\n╔═══════════════════════════════════════════════╗")
    print("║      Model Trainer — AutoTrader IA_BackTests  🚀        ║")
    print("╚═══════════════════════════════════════════════╝")
    print(f"  Símbolo  : {IBKR_SETTINGS.SYMBOL}  |  {IBKR_SETTINGS.INTERVAL}")
    print(f"  Steps    : {IBKR_SETTINGS.TIMESTEPS:,}")
    print(f"  Device   : {IBKR_SETTINGS.GPU_DEVICE}  |  Envs: {IBKR_SETTINGS.N_ENVS if IBKR_SETTINGS.N_ENVS else 'auto'}")
    print(f"  Modo     : {IBKR_SETTINGS.MODE}")


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
    if IBKR_SETTINGS.MODE not in modos:
        print(f"ERROR: MODE='{IBKR_SETTINGS.MODE}' no válido. Usa A/B/C/D.")
        sys.exit(1)

    trainer, metrics = modos[IBKR_SETTINGS.MODE]()
    print_summary(metrics)