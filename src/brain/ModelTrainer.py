from __future__ import annotations
"""
ModelTrainer.py
Entrena, evalúa, optimiza hiperparámetros y persiste el agente PPO.
Soporte completo para GPU (CUDA), Apple Silicon (MPS) y CPU.

VERSIÓN EXTREMA: Optimizada con DummyVecEnv y torch.set_num_threads(1)
para maximizar steps/s eliminando la latencia IPC.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch

# ── OPTIMIZACIÓN CRÍTICA PARA CPU/GPU ─────────────────────────────────────────
# Evita que la CPU pelee consigo misma intentando usar todos los hilos
# en cada entorno. Mantiene la CPU libre para alimentar a la GPU.
torch.set_num_threads(3)
# ──────────────────────────────────────────────────────────────────────────────

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
)
from src.brain.TradingEnvironment import TradingEnvironment
from src.brain.FeatureEngineering import FeatureEngineer

MODELS_DIR = Path(f"C:\\Users\\artur\\Programming\\PycharmProjects\\python_autotrader\\IA\\models")
LOGS_DIR   = Path("C:\\Users\\artur\\Programming\\PycharmProjects\\python_autotrader\\IA\\logs")


# ══════════════════════════════════════════════════════════════════════════════
# Detección de hardware
# ══════════════════════════════════════════════════════════════════════════════

def detect_device() -> str:
    """
    Detecta el mejor dispositivo disponible en orden de preferencia:
    CUDA → MPS (Apple Silicon) → CPU
    """
    try:
        if torch.cuda.is_available():
            gpu   = torch.cuda.get_device_name(0)
            vram  = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[GPU] CUDA detectado: {gpu}  ({vram:.1f} GB VRAM)")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[GPU] Apple MPS detectado (Silicon)")
            return "mps"
    except ImportError:
        pass
    print("[GPU] Sin GPU disponible — usando CPU")
    return "cpu"

def recommended_n_envs(device: str) -> int:
    """
    Número de entornos paralelos recomendado según el hardware.
    Con DummyVecEnv puro, podemos usar más entornos sin penalización IPC.
    """
    import multiprocessing
    cores = multiprocessing.cpu_count()

    if device == "cuda":
        return 16  # Subimos a 16 por defecto para saturar la GPU
    if device == "mps":
        return 8
    return min(cores, 8)

# ══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════════

class TrainingProgressCallback(BaseCallback):
    """Imprime métricas de entrenamiento cada N pasos con velocidad y ETA."""

    def __init__(self, log_every: int = 10_000, total_steps: int = 0, verbose: int = 1):
        super().__init__(verbose)
        self.log_every   = log_every
        self.total_steps = total_steps
        self.start_time  = None
        self._ep_rewards: list[float] = []

    def _on_training_start(self):
        self.start_time = time.time()
        print(f"\n{'─'*60}")
        print(f"  Entrenamiento iniciado  |  {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'─'*60}")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])

        if self.num_timesteps % self.log_every == 0 and self.num_timesteps > 0:
            elapsed  = time.time() - self.start_time
            speed    = self.num_timesteps / max(elapsed, 1)
            mean_r   = np.mean(self._ep_rewards[-50:]) if self._ep_rewards else 0.0

            eta_s    = (self.total_steps - self.num_timesteps) / max(speed, 1)
            eta_min  = eta_s / 60
            progress = self.num_timesteps / max(self.total_steps, 1) * 100

            print(
                f"  [{progress:5.1f}%  {self.num_timesteps:>9,} steps]  "
                f"reward: {mean_r:+7.2f}  |  "
                f"{speed:,.0f} steps/s  |  "
                f"ETA: {eta_min:.0f} min"
            )
            self._ep_rewards = self._ep_rewards[-200:]
        return True

    def _on_training_end(self):
        elapsed = time.time() - self.start_time
        print(f"\n{'─'*60}")
        print(f"  Entrenamiento finalizado  |  Duración: {elapsed/60:.1f} min")
        print(f"{'─'*60}\n")

class EarlyStoppingCallback(BaseCallback):
    """Para el entrenamiento si no mejora en `patience` evaluaciones."""

    # patience = 12 y min_dela = 0.05 para modo OPTUNA
    #patience = 30 y min_delta = 0.01 para modo A
    def __init__(self, patience: int = 12, min_delta: float = 0.05, verbose: int = 1):
        super().__init__(verbose)
        self.patience     = patience
        self.min_delta    = min_delta
        self._best        = -np.inf
        self._no_improve  = 0

    def _on_step(self) -> bool:
        if hasattr(self.parent, "best_mean_reward"):
            cur = self.parent.best_mean_reward
            if cur > self._best + self.min_delta:
                self._best       = cur
                self._no_improve = 0
            else:
                self._no_improve += 1
                if self.verbose:
                    print(
                        f"  [EarlyStopping] Sin mejora "
                        f"{self._no_improve}/{self.patience}  (mejor={self._best:.2f})"
                    )
                if self._no_improve >= self.patience:
                    print(f"  [EarlyStopping] Deteniendo entrenamiento.")
                    return False
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Configuración
# ══════════════════════════════════════════════════════════════════════════════

class PPOConfig:
    # Arquitectura
    policy:           str   = "MlpPolicy"
    net_arch:         list  = None

    # PPO core
    learning_rate:    float = 3e-4
    n_steps:          int   = 8192
    batch_size:       int   = 4096
    n_epochs:         int   = 10
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_range:       float = 0.2
    clip_range_vf:    float = None

    # Regularización
    ent_coef:         float = 0.01
    vf_coef:          float = 0.5
    max_grad_norm:    float = 0.5

    # Normalización
    normalize_obs:    bool  = True
    normalize_reward: bool  = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"[PPOConfig] Parámetro desconocido: '{k}'")
            setattr(self, k, v)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__class__.__dict__.items()
                if not k.startswith("_") and not callable(v)}

    @classmethod
    def for_gpu(cls, vram_gb: float = 8.0, n_envs: int = 16) -> "PPOConfig":
        """Preset optimizado para GPU masiva."""
        if vram_gb >= 8:
            n_steps, batch = 8192, 4096
        elif vram_gb >= 4:
            n_steps, batch = 4096, 2048
        else:
            n_steps, batch = 2048, 512

        total_buffer = n_steps * n_envs
        while total_buffer % batch != 0:
            batch //= 2

        return cls(
            n_steps       = n_steps,
            batch_size    = batch,
            n_epochs      = 10,
            learning_rate = 3e-4,
            ent_coef      = 0.01,
            gamma         = 0.99,
        )

    @classmethod
    def for_cpu(cls, n_envs: int = 4) -> "PPOConfig":
        total_buffer = 2048 * n_envs
        batch        = 256
        while total_buffer % batch != 0:
            batch //= 2
        return cls(n_steps=2048, batch_size=batch)

# ══════════════════════════════════════════════════════════════════════════════
# ModelTrainer
# ══════════════════════════════════════════════════════════════════════════════

class ModelTrainer:
    def __init__(
        self,
        symbol:         str                = "MODEL",
        config:         Optional[PPOConfig] = None,
        device:         str                = "auto",
        n_envs:         int                = 0,
        features_ready: bool               = False,
    ):
        self.symbol         = symbol.upper()
        self.features_ready = features_ready
        self.fe             = FeatureEngineer()

        self.device = detect_device() if device == "auto" else device
        self.n_envs = n_envs if n_envs > 0 else recommended_n_envs(self.device)

        if config is not None:
            self.config = config
        else:
            if self.device == "cuda":
                try:
                    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                except Exception:
                    vram = 4.0
                self.config = PPOConfig.for_gpu(vram_gb=vram, n_envs=self.n_envs)
                print(f"[Trainer] Config GPU automática (n_steps={self.config.n_steps}, batch={self.config.batch_size})")
            else:
                self.config = PPOConfig.for_cpu(n_envs=self.n_envs)

        self.model:   Optional[PPO]          = None
        self.vec_env: Optional[VecNormalize] = None

        (MODELS_DIR / self.symbol / "checkpoints").mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Entrenamiento
    # ══════════════════════════════════════════════════════════════════════════

    def train(
        self,
        df:              pd.DataFrame,
        total_timesteps: int   = 20_000_000,
        eval_freq:       int   = 50_000,
        n_eval_episodes: int   = 15,
        eval_split:      float = 0.20,
        resume:          bool  = False,
    ) -> "ModelTrainer":
        print(f"\n{'═'*60}")
        print(
            f"  ModelTrainer  |  {self.symbol}  |  {total_timesteps:,} steps\n"
            f"  Device: {self.device.upper()}  |  "
            f"Envs paralelos: {self.n_envs}  |  "
            f"Batch: {self.config.batch_size}"
        )
        print(f"{'═'*60}")

        if self.features_ready:
            df_feat = df.copy()
        else:
            print("[Trainer] Calculando features...")
            df_feat = self.fe.transform(df)

        if len(df_feat) < 500:
            raise ValueError(f"Solo {len(df_feat)} filas. Descarga más datos históricos.")

        if eval_split > 0:
            split    = int(len(df_feat) * (1 - eval_split))
            df_train = df_feat.iloc[:split].copy()
            df_eval  = df_feat.iloc[split:].copy()
        else:
            df_train = df_feat.copy()
            df_eval  = df_feat.iloc[-max(200, len(df_feat)//5):].copy()

        # ── 3. Entornos paralelos (MÁXIMA VELOCIDAD) ──────────────────────────
        def make_env(df_local: pd.DataFrame, rank: int):
            def _init():
                env = TradingEnvironment(df_local)
                return Monitor(env)
            return _init

        print(f"[Trainer] Creando {self.n_envs} entornos (DummyVecEnv para latencia cero)...")
        train_fns = [make_env(df_train, i) for i in range(self.n_envs)]
        raw_train = DummyVecEnv(train_fns)

        self.vec_env = VecNormalize(
            raw_train,
            norm_obs    = self.config.normalize_obs,
            norm_reward = self.config.normalize_reward,
            clip_obs    = 10.0,
        )

        eval_env = VecNormalize(
            DummyVecEnv([make_env(df_eval, 0)]),
            norm_obs    = self.config.normalize_obs,
            norm_reward = False,
            training    = False,
            clip_obs    = 10.0,
        )

        # ── 4. Callbacks ──────────────────────────────────────────────────────
        progress_cb   = TrainingProgressCallback(
            log_every   = max(eval_freq // 2, 5_000),
            total_steps = total_timesteps,
        )
        # patience = 12 para modo OPTUNA
        # patience = 30 para modo A
        early_stop_cb = EarlyStoppingCallback(patience=12)
        eval_cb       = EvalCallback(
            eval_env,
            callback_after_eval  = early_stop_cb,
            best_model_save_path = str(MODELS_DIR / self.symbol),
            log_path             = str(LOGS_DIR / self.symbol),
            eval_freq            = max(eval_freq // self.n_envs, 1),
            n_eval_episodes      = n_eval_episodes,
            deterministic        = True,
            verbose              = 1,
        )
        ckpt_cb = CheckpointCallback(
            save_freq   = max(total_timesteps // 10, 50_000) // self.n_envs,
            save_path   = str(MODELS_DIR / self.symbol / "checkpoints"),
            name_prefix = f"ppo_{self.symbol.lower()}",
            save_vecnormalize = True,
            verbose           = 0,
        )
        callbacks = CallbackList([progress_cb, eval_cb, ckpt_cb])

        # ── 5. Modelo PPO con device ──────────────────────────────────────────
        policy_kwargs = {}
        if self.config.net_arch:
            policy_kwargs["net_arch"] = self.config.net_arch

        if self.device == "cuda" and not self.config.net_arch:
            policy_kwargs["net_arch"] = [256, 256]

        if resume:
            ckpt = self._find_latest_checkpoint()
            if ckpt:
                print(f"[Trainer] Reanudando desde: {ckpt}")
                self.model = PPO.load(
                    ckpt, env=self.vec_env,
                    device=self.device, verbose=1,
                    tensorboard_log=str(LOGS_DIR),
                )
            else:
                resume = False

        if not resume:
            self.model = PPO(
                policy          = self.config.policy,
                env             = self.vec_env,
                learning_rate   = self.config.learning_rate,
                n_steps         = self.config.n_steps,
                batch_size      = self.config.batch_size,
                n_epochs        = self.config.n_epochs,
                gamma           = self.config.gamma,
                gae_lambda      = self.config.gae_lambda,
                clip_range      = self.config.clip_range,
                ent_coef        = self.config.ent_coef,
                vf_coef         = self.config.vf_coef,
                max_grad_norm   = self.config.max_grad_norm,
                tensorboard_log = str(LOGS_DIR),
                policy_kwargs   = policy_kwargs or None,
                device          = self.device,
                verbose         = 1,
            )

        self._log_model_summary(total_timesteps)

        # ── 6. Entrenar ───────────────────────────────────────────────────────
        t0 = time.time()
        self.model.learn(
            total_timesteps     = total_timesteps,
            callback            = callbacks,
            reset_num_timesteps = not resume,
            progress_bar        = True,
            tb_log_name         = f"{self.symbol}_{self.device}",
        )
        elapsed = time.time() - t0

        # ── 7. Guardar ────────────────────────────────────────────────────────
        self._save_model()
        self._save_training_meta(total_timesteps, elapsed)
        print(f"[Trainer] ✓ Completado en {elapsed/60:.1f} min  "
              f"({total_timesteps / max(elapsed, 1):,.0f} steps/s)")
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # Evaluación, Optuna y Persistencia permanecen igual
    # ══════════════════════════════════════════════════════════════════════════

    def evaluate(self, df: pd.DataFrame, episodes: int = 10, verbose: bool = True) -> dict:
        if self.model is None:
            self.load_model()
        df_feat = df.copy() if self.features_ready else self.fe.transform(df)
        env     = TradingEnvironment(df_feat, render_mode="human")
        rewards, balances, trades = [], [], []
        all_pnls: list[float] = []
        for _ in range(episodes):
            obs, _ = env.reset()
            done   = False
            ep_r   = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_r += reward
                done  = terminated or truncated
            rewards.append(ep_r)
            balances.append(info["balance"])
            trades.append(info["total_trades"])
            all_pnls.append(info["balance"] - TradingEnvironment.INITIAL_BALANCE)

        init   = TradingEnvironment.INITIAL_BALANCE
        rets   = [(b - init) / init for b in balances]
        gains  = [p for p in all_pnls if p > 0]
        losses = [abs(p) for p in all_pnls if p < 0]

        metrics = {
            "mean_reward": float(round(np.mean(rewards), 4)),
            "win_rate": float(round(sum(b > init for b in balances) / episodes, 3)),
            "mean_final_balance": float(round(np.mean(balances), 2)),
            "mean_return_pct": float(round(np.mean(rets) * 100, 2)),
            "sharpe_ratio": float(round(self._sharpe(rets), 3)),
            "sortino_ratio": float(round(self._sortino(rets), 3)),
            "max_drawdown_pct": float(round(self._max_drawdown(balances) * 100, 2)),
            "mean_trades": float(round(np.mean(trades), 1)),
            "profit_factor": float(round(sum(gains) / max(sum(losses), 1e-9), 3)),
            "device_used": str(self.device),
            "n_envs": int(self.n_envs),
        }
        if verbose:
            self._print_metrics(metrics)
        with open(MODELS_DIR / self.symbol / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    def optimize_hyperparams(self, df: pd.DataFrame, n_trials: int = 20, n_steps_per_trial: int = 100_000) -> PPOConfig:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("pip install optuna")

        if not self.features_ready:
            df = self.fe.transform(df)
            self.features_ready = True

        split    = int(len(df) * 0.8)
        df_train = df.iloc[:split]
        df_eval  = df.iloc[split:]

        def objective(trial: optuna.Trial) -> float:
            cfg = PPOConfig(
                learning_rate = trial.suggest_float("lr",1e-5, 1e-3, log=True),
                n_steps       = trial.suggest_categorical("n_steps", [4096, 8192, 16384]),
                batch_size    = trial.suggest_categorical("batch",   [2048, 4096, 8192]),
                n_epochs      = trial.suggest_int("epochs",   5, 15),
                gamma         = trial.suggest_float("gamma",  0.90, 0.9999),
                ent_coef      = trial.suggest_float("ent",    0.00001, 0.01, log=True),
                clip_range    = trial.suggest_float("clip",   0.1, 0.4),
            )
            total_buf = cfg.n_steps * self.n_envs
            while total_buf % cfg.batch_size != 0:
                cfg.batch_size //= 2

            try:
                tmp = ModelTrainer(
                    symbol         = f"{self.symbol}_t{trial.number}",
                    config         = cfg,
                    device         = self.device,
                    n_envs         = self.n_envs,
                    features_ready = True,
                )
                tmp.train(df_train, total_timesteps=n_steps_per_trial, eval_split=0.0, n_eval_episodes=3)
                m = tmp.evaluate(df_eval, episodes=5, verbose=False)
                return m["sharpe_ratio"]
            except Exception as e:
                print(f"  Trial {trial.number} falló: {e}")
                return -999.0

        print(f"\n[Optuna] {n_trials} trials en {self.device.upper()}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best = study.best_params
        print(f"\n[Optuna] Mejor config → Sharpe: {study.best_value:.4f}")
        for k, v in best.items():
            print(f"  {k}: {v}")

        best_cfg = PPOConfig(
            learning_rate = best["lr"],
            n_steps       = best["n_steps"],
            batch_size    = best["batch"],
            n_epochs      = best["epochs"],
            gamma         = best["gamma"],
            ent_coef      = best["ent"],
            clip_range    = best["clip"],
        )
        with open(MODELS_DIR / self.symbol / "optuna_best.json", "w") as f:
            json.dump({"params": best, "sharpe": study.best_value}, f, indent=2)

        self.config         = best_cfg
        self.features_ready = True
        return best_cfg

    def save(self):
        self._save_model()

    def load_model(self, path: Optional[str] = None) -> "ModelTrainer":
        model_path = path or str(MODELS_DIR / self.symbol / "best_model")
        norm_path  = str(MODELS_DIR / self.symbol / "vec_normalize.pkl")
        if not Path(model_path + ".zip").exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}.zip")
        self.model = PPO.load(model_path, device=self.device, verbose=1)
        if os.path.exists(norm_path):
            dummy = DummyVecEnv([lambda: TradingEnvironment(pd.DataFrame(columns=["open","high","low","close","volume"]))])
            self.vec_env          = VecNormalize.load(norm_path, dummy)
            self.vec_env.training = False
        print(f"[Trainer] Modelo cargado  ({self.device.upper()})")
        return self

    def _save_model(self):
        path = MODELS_DIR / self.symbol / "best_model"
        self.model.save(str(path))
        if self.vec_env is not None:
            self.vec_env.save(str(MODELS_DIR / self.symbol / "vec_normalize.pkl"))
        print(f"[Trainer] Guardado → {path}.zip")

    def _save_training_meta(self, total_timesteps: int, elapsed: float):
        meta = {
            "symbol":          self.symbol,
            "trained_at":      datetime.now().isoformat(),
            "total_timesteps": total_timesteps,
            "elapsed_seconds": round(elapsed, 1),
            "steps_per_sec":   round(total_timesteps / max(elapsed, 1), 0),
            "device":          self.device,
            "n_envs":          self.n_envs,
            "config":          self.config.to_dict(),
        }
        with open(MODELS_DIR / self.symbol / "training_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def _find_latest_checkpoint(self) -> Optional[str]:
        ckpt_dir = MODELS_DIR / self.symbol / "checkpoints"
        zips     = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
        return str(zips[-1]).replace(".zip", "") if zips else None

    @staticmethod
    def _sharpe(returns: list[float], rf: float = 0.0, periods: int = 252) -> float:
        arr = np.array(returns)
        return 0.0 if arr.std() < 1e-9 else float((arr.mean() - rf) / arr.std() * np.sqrt(periods))

    @staticmethod
    def _sortino(returns: list[float], rf: float = 0.0, periods: int = 252) -> float:
        arr     = np.array(returns)
        excess  = arr - rf
        neg     = excess[excess < 0]
        down_sd = neg.std() if len(neg) > 1 else 1e-9
        return 0.0 if down_sd < 1e-9 else float(excess.mean() / down_sd * np.sqrt(periods))

    @staticmethod
    def _max_drawdown(balances: list[float]) -> float:
        arr  = np.array(balances)
        peak = np.maximum.accumulate(arr)
        return float(((peak - arr) / np.maximum(peak, 1e-9)).max())

    def _log_model_summary(self, total_timesteps: int):
        cfg = self.config
        total_buf = cfg.n_steps * self.n_envs
        print(
            f"\n  Configuración:\n"
            f"    device         : {self.device.upper()}\n"
            f"    n_envs         : {self.n_envs}  (buffer total: {total_buf:,})\n"
            f"    n_steps        : {cfg.n_steps}\n"
            f"    batch_size     : {cfg.batch_size}  "
            f"({total_buf // cfg.batch_size} mini-batches/update)\n"
            f"    n_epochs       : {cfg.n_epochs}\n"
            f"    learning_rate  : {cfg.learning_rate}\n"
            f"    total_timesteps: {total_timesteps:,}\n"
        )

    @staticmethod
    def _print_metrics(m: dict):
        init = TradingEnvironment.INITIAL_BALANCE
        print(
            f"\n{'─'*50}\n"
            f"  Resultados (device: {m.get('device_used','?').upper()})\n"
            f"{'─'*50}\n"
            f"  Win rate         : {m['win_rate']:.1%}\n"
            f"  Retorno medio    : {m['mean_return_pct']:+.2f}%\n"
            f"  Balance final    : ${m['mean_final_balance']:,.2f}  (inicial: ${init:,.2f})\n"
            f"  Sharpe ratio     : {m['sharpe_ratio']:.3f}\n"
            f"  Sortino ratio    : {m['sortino_ratio']:.3f}\n"
            f"  Max drawdown     : {m['max_drawdown_pct']:.2f}%\n"
            f"  Profit factor    : {m['profit_factor']:.2f}\n"
            f"  Trades/episodio  : {m['mean_trades']:.1f}\n"
            f"{'─'*50}\n"
        )
        ready = (
            m["win_rate"]         >= 0.55 and
            m["sharpe_ratio"]     >= 0.5  and
            m["max_drawdown_pct"] <= 20.0
        )
        if ready:
            print("  ✓ MODELO APROBADO — listo para paper trading\n")
        else:
            issues = []
            if m["win_rate"]         < 0.55: issues.append(f"win_rate {m['win_rate']:.1%}")
            if m["sharpe_ratio"]     < 0.5:  issues.append(f"Sharpe {m['sharpe_ratio']:.2f}")
            if m["max_drawdown_pct"] > 20.0: issues.append(f"DD {m['max_drawdown_pct']:.1f}%")
            print(f"  ✗ Ajustes necesarios: {', '.join(issues)}\n")