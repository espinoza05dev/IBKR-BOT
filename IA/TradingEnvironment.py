# """
# TradingEnvironment.py
# Entorno Gymnasium personalizado para entrenamiento del agente de RL.
# El agente observa indicadores de mercado y decide: BUY / HOLD / SELL.
# """
#
# import numpy as np
# import pandas as pd
# import gymnasium as gym
# from gymnasium import spaces
#
#
# class TradingEnvironment(gym.Env):
#     """
#     Entorno de trading compatible con Stable-Baselines3.
#
#     Espacio de observacion (18 features):
#         close_norm, returns_1, returns_5, returns_10,
#         sma20_dist, sma50_dist, ema12_dist, ema26_dist,
#         rsi, macd, macd_signal, bb_upper_dist, bb_lower_dist,
#         atr_norm, volume_norm, stoch_k, stoch_d, adx
#
#     Espacio de acciones:
#         0 = HOLD  |  1 = BUY  |  2 = SELL
#
#     Recompensa:
#         PnL realizado + penalizacion por drawdown excesivo.
#     """
#
#     metadata = {"render_modes": ["human"]}
#
#     # Parametros de riesgo
#     INITIAL_BALANCE     = 10_000.0
#     TRANSACTION_COST    = 0.001      # 0.1% por operacion
#     MAX_DRAWDOWN_LIMIT  = 0.15       # -15% antes de terminar el episodio
#     REWARD_SCALE        = 100.0
#
#     def __init__(self, df: pd.DataFrame, window: int = 20, render_mode=None):
#         """
#         Args:
#             df: DataFrame con columnas OHLCV ya procesadas + features tecnicas.
#             window: ventana de observacion (numero de velas hacia atras).
#         """
#         super().__init__()
#         self.df            = df.reset_index(drop=True)
#         self.window        = window
#         self.render_mode   = render_mode
#         self.n_features    = 18
#
#         # Espacios
#         self.action_space = spaces.Discrete(3)   # 0=HOLD 1=BUY 2=SELL
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf,
#             shape=(window, self.n_features),
#             dtype=np.float32,
#         )
#
#         # Estado interno
#         self._reset_state()
#
#     # ── Gym API ───────────────────────────────────────────────────────────────
#
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self._reset_state()
#         obs = self._get_observation()
#         return obs, {}
#
#     def step(self, action: int):
#         current_price = self._current_price()
#         reward = 0.0
#
#         # Ejecutar accion
#         if action == 1 and not self.in_position:          # BUY
#             self.in_position      = True
#             self.entry_price      = current_price
#             self.balance         -= current_price * self.TRANSACTION_COST
#             self.entry_step       = self.current_step
#
#         elif action == 2 and self.in_position:             # SELL
#             pnl = (current_price - self.entry_price) / self.entry_price
#             reward = pnl * self.REWARD_SCALE
#             self.balance         += current_price - self.entry_price
#             self.balance         -= current_price * self.TRANSACTION_COST
#             self.total_trades    += 1
#             self.in_position      = False
#             self.entry_price      = 0.0
#
#         # Penalizar por mantener posicion perdedora demasiado tiempo
#         if self.in_position:
#             unrealized = (current_price - self.entry_price) / max(self.entry_price, 1e-9)
#             if unrealized < -0.02:          # mas de -2% sin cerrar
#                 reward -= 0.1
#
#         # Actualizar drawdown
#         self.peak_balance = max(self.peak_balance, self.balance)
#         drawdown = (self.peak_balance - self.balance) / max(self.peak_balance, 1e-9)
#         if drawdown > self.MAX_DRAWDOWN_LIMIT:
#             reward -= 10.0
#             terminated = True
#         else:
#             terminated = False
#
#         self.current_step += 1
#         truncated = self.current_step >= len(self.df) - 1
#
#         obs = self._get_observation()
#         info = {
#             "balance":      self.balance,
#             "drawdown":     drawdown,
#             "total_trades": self.total_trades,
#             "in_position":  self.in_position,
#         }
#         return obs, reward, terminated, truncated, info
#
#     def render(self):
#         if self.render_mode == "human":
#             pnl = self.balance - self.INITIAL_BALANCE
#             print(
#                 f"Step {self.current_step:4d} | "
#                 f"Balance: ${self.balance:9.2f} | "
#                 f"PnL: ${pnl:+.2f} | "
#                 f"Trades: {self.total_trades}"
#             )
#
#     # ── Internos ──────────────────────────────────────────────────────────────
#
#     def _reset_state(self):
#         self.current_step  = self.window
#         self.balance       = self.INITIAL_BALANCE
#         self.peak_balance  = self.INITIAL_BALANCE
#         self.in_position   = False
#         self.entry_price   = 0.0
#         self.entry_step    = 0
#         self.total_trades  = 0
#
#     def _current_price(self) -> float:
#         return float(self.df["close"].iloc[self.current_step])
#
#     def _get_observation(self) -> np.ndarray:
#         feature_cols = [
#             "close_norm", "returns_1", "returns_5", "returns_10",
#             "sma20_dist", "sma50_dist", "ema12_dist", "ema26_dist",
#             "rsi", "macd", "macd_signal", "bb_upper_dist", "bb_lower_dist",
#             "atr_norm", "volume_norm", "stoch_k", "stoch_d", "adx",
#         ]
#         # Usar solo columnas disponibles
#         available = [c for c in feature_cols if c in self.df.columns]
#         window_df = self.df.iloc[
#             self.current_step - self.window : self.current_step
#         ][available].values.astype(np.float32)
#
#         # Rellenar NaN con 0
#         window_df = np.nan_to_num(window_df, nan=0.0)
#
#         # Padding si faltan columnas
#         if window_df.shape[1] < self.n_features:
#             pad = np.zeros((self.window, self.n_features - window_df.shape[1]), dtype=np.float32)
#             window_df = np.hstack([window_df, pad])
#
#         return window_df

"""
TradingEnvironment.py
Entorno Gymnasium personalizado para entrenamiento del agente de RL.
Versión Optimizada con NumPy puro para máximo rendimiento (steps/s).
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    # Parámetros de riesgo
    INITIAL_BALANCE = 10_000.0
    TRANSACTION_COST = 0.001  # 0.1% por operación
    MAX_DRAWDOWN_LIMIT = 0.15  # -15% antes de terminar el episodio
    REWARD_SCALE = 100.0

    def __init__(self, df: pd.DataFrame, window: int = 20, render_mode=None):
        super().__init__()
        self.window = window
        self.render_mode = render_mode
        self.n_features = 18

        # ── OPTIMIZACIÓN CRÍTICA: Pre-procesamiento NumPy ─────────────
        # 1. Extraer los precios de cierre a un arreglo 1D super rápido
        self.close_prices = df["close"].values.astype(np.float32)
        self.max_steps = len(self.close_prices) - 1

        # 2. Extraer y formatear la matriz de observaciones UNA SOLA VEZ
        feature_cols = [
            "close_norm", "returns_1", "returns_5", "returns_10",
            "sma20_dist", "sma50_dist", "ema12_dist", "ema26_dist",
            "rsi", "macd", "macd_signal", "bb_upper_dist", "bb_lower_dist",
            "atr_norm", "volume_norm", "stoch_k", "stoch_d", "adx",
        ]

        available_cols = [c for c in feature_cols if c in df.columns]
        raw_features = df[available_cols].values.astype(np.float32)
        raw_features = np.nan_to_num(raw_features, nan=0.0)

        # Hacer padding si faltan columnas
        if raw_features.shape[1] < self.n_features:
            pad = np.zeros((len(df), self.n_features - raw_features.shape[1]), dtype=np.float32)
            self.features_array = np.hstack([raw_features, pad])
        else:
            self.features_array = raw_features
        # ──────────────────────────────────────────────────────────────

        # Espacios
        self.action_space = spaces.Discrete(3)  # 0=HOLD 1=BUY 2=SELL
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window, self.n_features),
            dtype=np.float32,
        )

        self._reset_state()

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}

    def step(self, action: int):
        # Lectura O(1) super rápida
        current_price = self.close_prices[self.current_step]
        reward = 0.0

        if action == 1 and not self.in_position:  # BUY
            self.in_position = True
            self.entry_price = current_price
            self.balance -= current_price * self.TRANSACTION_COST
            self.entry_step = self.current_step

        elif action == 2 and self.in_position:  # SELL
            pnl = (current_price - self.entry_price) / self.entry_price
            reward = pnl * self.REWARD_SCALE
            self.balance += current_price - self.entry_price
            self.balance -= current_price * self.TRANSACTION_COST
            self.total_trades += 1
            self.in_position = False
            self.entry_price = 0.0

        if self.in_position:
            unrealized = (current_price - self.entry_price) / max(self.entry_price, 1e-9)
            if unrealized < -0.02:
                reward -= 0.1

        self.peak_balance = max(self.peak_balance, self.balance)
        drawdown = (self.peak_balance - self.balance) / max(self.peak_balance, 1e-9)

        terminated = False
        if drawdown > self.MAX_DRAWDOWN_LIMIT:
            reward -= 10.0
            terminated = True

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        info = {
            "balance": self.balance,
            "drawdown": drawdown,
            "total_trades": self.total_trades,
            "in_position": self.in_position,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            pnl = self.balance - self.INITIAL_BALANCE
            print(
                f"Step {self.current_step:4d} | "
                f"Balance: ${self.balance:9.2f} | "
                f"PnL: ${pnl:+.2f} | "
                f"Trades: {self.total_trades}"
            )

    # ── Internos ──────────────────────────────────────────────────────────────

    def _reset_state(self):
        self.current_step = self.window
        self.balance = self.INITIAL_BALANCE
        self.peak_balance = self.INITIAL_BALANCE
        self.in_position = False
        self.entry_price = 0.0
        self.entry_step = 0
        self.total_trades = 0

    def _get_observation(self) -> np.ndarray:
        # Slicing de NumPy: Toma microsegundos en comparación a Pandas
        return self.features_array[self.current_step - self.window: self.current_step]