"""
TradingAI.py
Cerebro autonomo del bot: carga el modelo entrenado, recibe barras en vivo,
consulta la KnowledgeBase para contexto de mercado y emite ordenes
validadas por el RiskManager.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from IA.FeatureEngineering import FeatureEngineer
from IA.RiskManager import RiskManager, RiskConfig
from IA.TradingEnvironment import TradingEnvironment

MODELS_DIR = Path("IA/models")


class TradingAI:
    """
    Agente de trading autonomo basado en PPO + RiskManager.

    Flujo en vivo:
        1. Recibe nueva barra via on_new_bar()
        2. Calcula features con FeatureEngineer
        3. Pide accion al modelo (PPO.predict)
        4. Valida con RiskManager
        5. Emite senal via callback → Portfolio

    Uso:
        ai = TradingAI(symbol="AAPL")
        ai.load()
        ai.set_order_callback(portfolio.place_bracket_order)
        # Conectar al MarketDataHandler:
        market_data.set_signal_callback(ai.on_new_bar)
    """

    WARMUP_BARS = 60      # Barras minimas antes de operar

    def __init__(
        self,
        symbol: str,
        risk_config: Optional[RiskConfig] = None,
        confidence_threshold: float = 0.60,
    ):
        self.symbol               = symbol
        self.fe                   = FeatureEngineer()
        self.risk                 = RiskManager(config=risk_config)
        self.confidence_threshold = confidence_threshold
        self.model: Optional[PPO]              = None
        self.vec_normalize: Optional[VecNormalize] = None

        self._bar_buffer: list[dict] = []    # Buffer OHLCV en bruto
        self._lock       = threading.Lock()
        self._order_cb   = None              # Callback hacia Portfolio

        # Mapa accion → texto
        self._action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}

    # ── Setup ─────────────────────────────────────────────────────────────────

    def load(self) -> "TradingAI":
        """Carga el modelo PPO y el normalizador desde disco."""
        model_path = MODELS_DIR / self.symbol / "best_model"
        norm_path  = MODELS_DIR / self.symbol / "vec_normalize.pkl"

        if not model_path.with_suffix(".zip").exists():
            raise FileNotFoundError(
                f"[TradingAI] Modelo no encontrado: {model_path}\n"
                f"Ejecuta ModelTrainer().train(df) primero."
            )

        self.model = PPO.load(str(model_path))

        if norm_path.exists():
            dummy_env       = DummyVecEnv([lambda: TradingEnvironment(pd.DataFrame())])
            self.vec_normalize = VecNormalize.load(str(norm_path), dummy_env)
            self.vec_normalize.training = False

        print(f"[TradingAI] Modelo cargado para {self.symbol}")
        return self

    def set_order_callback(self, callback):
        """Registra el callback que se llamara al emitir una orden."""
        self._order_cb = callback

    # ── Ingesta de datos en vivo ──────────────────────────────────────────────

    def on_new_bar(self, bar) -> None:
        """
        Punto de entrada de datos en tiempo real.
        Compatible con MarketDataHandler.set_signal_callback().
        """
        with self._lock:
            self._bar_buffer.append({
                "open":   bar.open,
                "high":   bar.high,
                "low":    bar.low,
                "close":  bar.close,
                "volume": bar.volume,
            })

            if len(self._bar_buffer) < self.WARMUP_BARS:
                print(f"[TradingAI] Calentando ({len(self._bar_buffer)}/{self.WARMUP_BARS})...")
                return

            self._decide()

    # ── Decision ──────────────────────────────────────────────────────────────

    def _decide(self) -> None:
        """Genera y ejecuta (si aplica) la siguiente accion."""
        df_raw  = pd.DataFrame(self._bar_buffer[-200:])   # Ventana de 200 barras
        df_feat = self.fe.transform(df_raw)

        if len(df_feat) < TradingEnvironment(pd.DataFrame()).window:
            return

        obs = self._build_observation(df_feat)

        # Prediccion del modelo
        raw_action, _ = self.model.predict(obs, deterministic=True)
        action        = int(raw_action)

        # Confianza via distribucion de probabilidades
        confidence = self._estimate_confidence(obs)
        action_name = self._action_names.get(action, "UNKNOWN")

        print(
            f"[TradingAI] Accion: {action_name} | "
            f"Confianza: {confidence:.2f} | "
            f"Precio: {df_feat['close'].iloc[-1]:.2f}"
        )

        # Consultar RiskManager
        current_price = float(df_feat["close"].iloc[-1])
        atr_norm      = float(df_feat.get("atr_norm", pd.Series([0])).iloc[-1])
        atr_abs       = atr_norm * current_price

        allowed, reason = self.risk.check(action, confidence, current_price, atr_abs)
        if not allowed:
            print(f"[TradingAI] Bloqueado por RiskManager: {reason}")
            return

        # Emitir orden
        if action in (1, 2) and self._order_cb:
            size          = self.risk.position_size(current_price)
            profit_target = self.risk.dynamic_take_profit(current_price, atr_abs)
            stop_loss     = self.risk.dynamic_stop_loss(current_price, atr_abs)

            self._order_cb(
                action        = "BUY" if action == 1 else "SELL",
                quantity      = size,
                profit_target = profit_target,
                stop_loss     = stop_loss,
            )

    # ── Utilidades ────────────────────────────────────────────────────────────

    def _build_observation(self, df_feat: pd.DataFrame) -> np.ndarray:
        """Construye la observacion en el formato esperado por el modelo."""
        env = TradingEnvironment(df_feat)
        env.current_step = len(df_feat) - 1
        obs = env._get_observation()

        if self.vec_normalize:
            obs = self.vec_normalize.normalize_obs(obs[np.newaxis])[0]

        return obs

    def _estimate_confidence(self, obs: np.ndarray) -> float:
        """
        Estima confianza via entropia de la distribucion de politica.
        Mayor concentracion en una accion = mayor confianza.
        """
        try:
            import torch
            obs_tensor = torch.FloatTensor(obs[np.newaxis])
            with torch.no_grad():
                dist    = self.model.policy.get_distribution(obs_tensor)
                probs   = dist.distribution.probs.numpy()[0]
            entropy = -np.sum(probs * np.log(probs + 1e-9))
            max_entropy = np.log(len(probs))
            confidence  = 1.0 - (entropy / max_entropy)
            return float(confidence)
        except Exception:
            return 0.75    # Valor por defecto si no se puede calcular

    def get_status(self) -> dict:
        return {
            "symbol":      self.symbol,
            "bars_loaded": len(self._bar_buffer),
            "risk":        self.risk.get_status(),
            "model_ready": self.model is not None,
        }