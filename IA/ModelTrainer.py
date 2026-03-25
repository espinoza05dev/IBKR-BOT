"""
ModelTrainer.py
Entrena, evalua y persiste el agente de Reinforcement Learning.
Usa Stable-Baselines3 (PPO) sobre el TradingEnvironment personalizado.
"""

import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from IA.TradingEnvironment import TradingEnvironment
from IA.FeatureEngineering import FeatureEngineer


MODELS_DIR = Path("IA/models")
LOGS_DIR   = Path("IA/logs")


class ModelTrainer:
    """
    Entrena un agente PPO para trading autonomo.

    Ciclo completo:
        1. Preparar datos con FeatureEngineer
        2. Crear entornos train / eval
        3. Entrenar con PPO
        4. Evaluar y guardar metricas
        5. Persistir modelo

    Uso:
        trainer = ModelTrainer()
        trainer.train(df_ohlcv, total_timesteps=500_000)
        trainer.evaluate(df_test)
    """

    def __init__(self, symbol: str = "MODEL", model_path: str | None = None):
        self.symbol      = symbol
        self.fe          = FeatureEngineer()
        self.model       = None
        self.vec_env     = None
        self.model_path  = model_path

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        total_timesteps: int = 500_000,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
    ) -> "ModelTrainer":
        print(f"[Trainer] Preparando features para {self.symbol}...")
        df_feat = self.fe.transform(df)

        # Split 80/20
        split    = int(len(df_feat) * 0.8)
        df_train = df_feat.iloc[:split]
        df_eval  = df_feat.iloc[split:]

        print(f"[Trainer] Train: {len(df_train)} velas | Eval: {len(df_eval)} velas")

        # Entornos
        train_env = DummyVecEnv([lambda: Monitor(TradingEnvironment(df_train))])
        self.vec_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

        eval_env  = DummyVecEnv([lambda: Monitor(TradingEnvironment(df_eval))])
        eval_env  = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

        # Callbacks
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(MODELS_DIR / self.symbol),
            log_path=str(LOGS_DIR / self.symbol),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=1,
        )
        checkpoint_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=str(MODELS_DIR / self.symbol / "checkpoints"),
            name_prefix="ppo_trading",
        )
        callbacks = CallbackList([eval_cb, checkpoint_cb])

        # Modelo PPO
        self.model = PPO(
            policy             = "MlpPolicy",
            env                = self.vec_env,
            learning_rate      = 3e-4,
            n_steps            = 2048,
            batch_size         = 64,
            n_epochs           = 10,
            gamma              = 0.99,
            gae_lambda         = 0.95,
            clip_range         = 0.2,
            ent_coef           = 0.01,      # Exploracion
            vf_coef            = 0.5,
            max_grad_norm      = 0.5,
            tensorboard_log    = str(LOGS_DIR),
            verbose            = 1,
        )

        print(f"[Trainer] Iniciando entrenamiento — {total_timesteps:,} steps...")
        start = datetime.now()
        self.model.learn(
            total_timesteps = total_timesteps,
            callback        = callbacks,
            progress_bar    = True,
        )
        elapsed = (datetime.now() - start).seconds
        print(f"[Trainer] Entrenamiento completado en {elapsed}s")

        self._save_model()
        return self

    # ── Evaluacion ────────────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame, episodes: int = 10) -> dict:
        """Evalua el modelo en datos no vistos y retorna metricas."""
        if self.model is None:
            self.load_model()

        df_feat  = self.fe.transform(df)
        env      = TradingEnvironment(df_feat, render_mode="human")

        all_rewards, all_trades, final_balances = [], [], []

        for ep in range(episodes):
            obs, _   = env.reset()
            done     = False
            ep_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_reward += reward
                done       = terminated or truncated

            all_rewards.append(ep_reward)
            all_trades.append(info["total_trades"])
            final_balances.append(info["balance"])

        metrics = {
            "mean_reward":       round(float(np.mean(all_rewards)), 4),
            "std_reward":        round(float(np.std(all_rewards)), 4),
            "mean_trades":       round(float(np.mean(all_trades)), 1),
            "mean_final_balance": round(float(np.mean(final_balances)), 2),
            "win_rate":          round(
                sum(b > TradingEnvironment.INITIAL_BALANCE for b in final_balances) / episodes, 3
            ),
        }
        print("[Trainer] Metricas de evaluacion:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        # Persistir metricas
        metrics_path = MODELS_DIR / self.symbol / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    # ── Persistencia ─────────────────────────────────────────────────────────

    def _save_model(self):
        path = MODELS_DIR / self.symbol / "best_model"
        self.model.save(str(path))
        self.vec_env.save(str(MODELS_DIR / self.symbol / "vec_normalize.pkl"))
        print(f"[Trainer] Modelo guardado en {path}")

    def load_model(self, path: str | None = None):
        model_path = path or str(MODELS_DIR / self.symbol / "best_model")
        norm_path  = str(MODELS_DIR / self.symbol / "vec_normalize.pkl")

        self.model = PPO.load(model_path)
        if os.path.exists(norm_path):
            self.vec_env = VecNormalize.load(
                norm_path,
                DummyVecEnv([lambda: TradingEnvironment(pd.DataFrame())]),
            )
        print(f"[Trainer] Modelo cargado desde {model_path}")
        return self