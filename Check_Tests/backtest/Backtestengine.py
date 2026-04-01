from __future__ import annotations
"""
BacktestEngine.py
Motor de backtesting que ejecuta el modelo entrenado sobre datos históricos
de forma cronológica (una sola pasada, sin reset), registrando cada trade.

Diferencia clave vs ModelTrainer.evaluate():
    evaluate()      → N episodios aleatorios, solo mide reward de RL
    BacktestEngine  → 1 pasada cronológica, registra cada trade con
                      precio de entrada/salida, PnL, duración, comisiones
                      y compara contra Buy & Hold como benchmark

Incluye:
    - BacktestEngine   : motor principal (1 run)
    - WalkForwardEngine: validación walk-forward (múltiples ventanas)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from src.brain.TradingEnvironment import TradingEnvironment
from src.brain.FeatureEngineering import FeatureEngineer

MODELS_DIR = Path("C:\\Users\\artur\\Programming\\PycharmProjects\\python_autotrader\\IA\\models")


# ══════════════════════════════════════════════════════════════════════════════
# Estructuras de datos
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """Registro completo de un trade individual."""
    trade_id:        int
    entry_date:      pd.Timestamp
    exit_date:       Optional[pd.Timestamp]
    entry_price:     float
    exit_price:      float      = 0.0
    quantity:        int        = 1
    commission:      float      = 0.0
    pnl_gross:       float      = 0.0    # Sin comisiones
    pnl_net:         float      = 0.0    # Con comisiones
    pnl_pct:         float      = 0.0    # % sobre entry_price
    duration_bars:   int        = 0      # Barras que duró abierto
    is_winner:       bool       = False
    exit_reason:     str        = ""     # "signal" | "stop_loss" | "end_of_data"


@dataclass
class BacktestResult:
    """Resultado completo de un backtest."""
    symbol:           str
    interval:         str
    start_date:       pd.Timestamp
    end_date:         pd.Timestamp
    initial_balance:  float
    final_balance:    float
    trades:           list[Trade]             = field(default_factory=list)
    equity_curve:     pd.Series               = field(default_factory=pd.Series)
    drawdown_series:  pd.Series               = field(default_factory=pd.Series)
    actions_series:   pd.Series               = field(default_factory=pd.Series)
    benchmark_equity: pd.Series               = field(default_factory=pd.Series)
    metrics:          dict                    = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return len([t for t in self.trades if t.exit_date is not None])

    @property
    def total_return(self) -> float:
        return (self.final_balance - self.initial_balance) / self.initial_balance


# ══════════════════════════════════════════════════════════════════════════════
# Motor principal
# ══════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Ejecuta el modelo PPO sobre datos históricos paso a paso.

    Simula exactamente lo que haría el bot en vivo:
        - Una observación por vela
        - El modelo predice BUY / HOLD / SELL
        - Se registra cada entrada y salida con todos sus detalles
        - Se construye la curva de equity barra a barra

    Uso:
        engine = BacktestEngine(symbol="AAPL")
        result = engine.run(df_test)
        print(result.metrics)
    """

    COMMISSION = 0.001    # 0.1% por operación (entrada + salida)

    def __init__(
        self,
        symbol:           str,
        initial_balance:  float = 10_000.0,
        commission:       float = 0.001,
        features_ready:   bool  = False,
    ):
        self.symbol          = symbol.upper()
        self.initial_balance = initial_balance
        self.commission      = commission
        self.features_ready  = features_ready
        self.fe              = FeatureEngineer()
        self._model: Optional[PPO]          = None
        self._norm:  Optional[VecNormalize] = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def load_model(self, model_path: Optional[str] = None) -> "BacktestEngine":
        """Carga el modelo y el normalizador desde disco."""
        mp = model_path or str(MODELS_DIR / self.symbol / "best_model")
        np_ = str(MODELS_DIR / self.symbol / "vec_normalize.pkl")

        if not Path(mp + ".zip").exists():
            raise FileNotFoundError(
                f"Modelo no encontrado: {mp}.zip\n"
                f"Entrena primero con: python train_model.py"
            )

        self._model = PPO.load(mp, device="cpu")   # CPU para backtest (más estable)

        if Path(np_).exists():
            # Cargar primero el modelo para saber el observation_space
            tmp_env = TradingEnvironment(
                pd.DataFrame({c: [0.0] * 50
                              for c in ["close_norm", "returns_1", "returns_5", "returns_10",
                                        "sma20_dist", "sma50_dist", "ema12_dist", "ema26_dist",
                                        "rsi", "macd", "macd_signal", "bb_upper_dist",
                                        "bb_lower_dist", "atr_norm", "volume_norm",
                                        "stoch_k", "stoch_d", "adx",
                                        "open", "high", "low", "close", "volume"]})
            )
            dummy = DummyVecEnv([lambda: tmp_env])
            self._norm = VecNormalize.load(np_, dummy)
            self._norm.training = False
            self._norm.norm_reward = False

        print(f"[Backtest] Modelo cargado: {self.symbol}")
        return self

    # ── Run principal ─────────────────────────────────────────────────────────

    def run(
        self,
        df:       pd.DataFrame,
        interval: str = "1h",
        verbose:  bool = True,
    ) -> BacktestResult:
        """
        Ejecuta el backtest completo sobre el DataFrame.

        Args:
            df:       DataFrame con columnas OHLCV (+ features si features_ready=True).
            interval: Timeframe del df. Usado para calcular duración de trades.
            verbose:  Si True, imprime progreso y resumen.
        """
        if self._model is None:
            self.load_model()

        # ── 1. Features ───────────────────────────────────────────────────────
        if self.features_ready:
            df_feat = df.copy()
        else:
            if verbose:
                print("[Backtest] Calculando features...")
            df_feat = self.fe.transform(df)

        df_feat = df_feat.reset_index(drop=False)   # Conservar fecha como columna

        n_bars = len(df_feat)
        window = 20   # Coincide con TradingEnvironment.window

        if verbose:
            print(
                f"[Backtest] Ejecutando  {self.symbol}  |  "
                f"{n_bars:,} barras  |  "
                f"{df_feat['datetime'].iloc[window] if 'datetime' in df_feat.columns else 'N/A'}"
                f" → {df_feat['datetime'].iloc[-1] if 'datetime' in df_feat.columns else 'N/A'}"
            )

        # ── 2. Simulación barra a barra ───────────────────────────────────────
        env     = TradingEnvironment(df_feat.drop(columns=["datetime"], errors="ignore"))
        obs, _  = env.reset()

        balance         = self.initial_balance
        peak_balance    = self.initial_balance
        in_position     = False
        entry_price     = 0.0
        entry_step      = 0
        trade_id        = 0

        equity_values   = [balance] * window
        drawdown_values = [0.0] * window
        action_values   = [0] * window
        trades: list[Trade] = []

        for step in range(window, n_bars - 1):
            current_price = float(df_feat["close"].iloc[step])

            # Normalizar observación si hay VecNormalize
            if self._norm is not None:
                obs_norm = self._norm.normalize_obs(obs[np.newaxis])[0]
            else:
                obs_norm = obs

            raw_action, _ = self._model.predict(obs_norm, deterministic=True)
            action        = int(raw_action)
            action_values.append(action)

            # ── Ejecutar acción ───────────────────────────────────────────────
            if action == 1 and not in_position:          # BUY
                in_position  = True
                entry_price  = current_price
                entry_step   = step
                commission   = current_price * self.commission
                balance     -= commission
                trade_id    += 1

            elif action == 2 and in_position:            # SELL
                exit_price  = current_price
                commission  = exit_price * self.commission
                pnl_gross   = exit_price - entry_price
                pnl_net     = pnl_gross - commission - (entry_price * self.commission)
                balance    += pnl_gross - commission

                date_col  = "datetime" if "datetime" in df_feat.columns else None
                entry_date = pd.Timestamp(df_feat[date_col].iloc[entry_step]) if date_col else None
                exit_date  = pd.Timestamp(df_feat[date_col].iloc[step])       if date_col else None

                trades.append(Trade(
                    trade_id      = trade_id,
                    entry_date    = entry_date,
                    exit_date     = exit_date,
                    entry_price   = entry_price,
                    exit_price    = exit_price,
                    commission    = commission + entry_price * self.commission,
                    pnl_gross     = pnl_gross,
                    pnl_net       = pnl_net,
                    pnl_pct       = pnl_gross / entry_price * 100,
                    duration_bars = step - entry_step,
                    is_winner     = pnl_net > 0,
                    exit_reason   = "signal",
                ))
                in_position = False
                entry_price = 0.0

            # ── Actualizar equity ─────────────────────────────────────────────
            # Si hay posición abierta, el balance incluye el PnL no realizado
            unrealized = (current_price - entry_price) if in_position else 0.0
            total_equity = balance + unrealized
            equity_values.append(total_equity)

            peak_balance    = max(peak_balance, total_equity)
            dd              = (peak_balance - total_equity) / max(peak_balance, 1e-9)
            drawdown_values.append(dd)

            # Avanzar entorno
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        # Cerrar posición abierta al final del período
        if in_position:
            exit_price = float(df_feat["close"].iloc[-1])
            pnl_gross  = exit_price - entry_price
            commission = (exit_price + entry_price) * self.commission
            pnl_net    = pnl_gross - commission
            balance   += pnl_gross - exit_price * self.commission

            date_col   = "datetime" if "datetime" in df_feat.columns else None
            trades.append(Trade(
                trade_id      = trade_id,
                entry_date    = pd.Timestamp(df_feat[date_col].iloc[entry_step]) if date_col else None,
                exit_date     = pd.Timestamp(df_feat[date_col].iloc[-1])         if date_col else None,
                entry_price   = entry_price,
                exit_price    = exit_price,
                commission    = commission,
                pnl_gross     = pnl_gross,
                pnl_net       = pnl_net,
                pnl_pct       = pnl_gross / entry_price * 100,
                duration_bars = len(df_feat) - 1 - entry_step,
                is_winner     = pnl_net > 0,
                exit_reason   = "end_of_data",
            ))

        # ── 3. Series temporales ──────────────────────────────────────────────
        idx = df_feat["datetime"] if "datetime" in df_feat.columns else pd.RangeIndex(n_bars)
        idx_trimmed = idx[:len(equity_values)]

        equity_series   = pd.Series(equity_values,   index=idx_trimmed, name="equity")
        drawdown_series = pd.Series(drawdown_values,  index=idx_trimmed, name="drawdown")
        actions_series  = pd.Series(action_values,    index=idx_trimmed, name="action")

        # Buy & Hold benchmark
        first_price = float(df_feat["close"].iloc[window])
        bh_equity = (df_feat["close"].iloc[window:].values / first_price) * self.initial_balance

        # 1. CREAMOS SOLO LA LISTA (Sin pd.Series todavía)
        benchmark_vals = list(np.full(window, self.initial_balance)) + list(bh_equity)

        # 2. CALCULAMOS EL TAMAÑO CORRECTO
        min_len = min(len(benchmark_vals), len(idx_trimmed))

        # 3. AHORA SÍ CREAMOS LA SERIE YA RECORTADA
        benchmark = pd.Series(
            benchmark_vals[:min_len],
            index=idx_trimmed[:min_len],
            name="buy_hold"
        )


        # ── 4. Resultado ──────────────────────────────────────────────────────
        result = BacktestResult(
            symbol           = self.symbol,
            interval         = interval,
            start_date       = idx_trimmed.iloc[window] if hasattr(idx_trimmed, "iloc") else idx_trimmed[window],
            end_date         = idx_trimmed.iloc[-1]     if hasattr(idx_trimmed, "iloc") else idx_trimmed[-1],
            initial_balance  = self.initial_balance,
            final_balance    = balance,
            trades           = trades,
            equity_curve     = equity_series,
            drawdown_series  = drawdown_series,
            actions_series   = actions_series,
            benchmark_equity = benchmark,
        )

        if verbose:
            self._print_summary(result)

        return result

    # ── Utilidades ────────────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(r: BacktestResult):
        ret   = r.total_return * 100
        bh_r  = (r.benchmark_equity.iloc[-1] / r.initial_balance - 1) * 100
        alpha = ret - bh_r
        print(
            f"\n[Backtest] ─── Resumen rápido ──────────────────\n"
            f"  Período       : {r.start_date} → {r.end_date}\n"
            f"  Balance final : ${r.final_balance:,.2f}  "
            f"(inicio: ${r.initial_balance:,.2f})\n"
            f"  Retorno       : {ret:+.2f}%  |  "
            f"Buy&Hold: {bh_r:+.2f}%  |  "
            f"Alpha: {alpha:+.2f}%\n"
            f"  Trades        : {r.n_trades}\n"
            f"────────────────────────────────────────────────"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Validation
# ══════════════════════════════════════════════════════════════════════════════

class WalkForwardEngine:
    """
    Validación Walk-Forward: entrena en ventana deslizante y prueba en la siguiente.

    Elimina el sesgo de look-ahead: el modelo NUNCA ve datos futuros durante
    el entrenamiento de cada ventana.

    Esquema:
        Ventana 1:  TRAIN [0 : 70%]  →  TEST [70% : 85%]
        Ventana 2:  TRAIN [15%: 85%] →  TEST [85% :100%]
        ...

    Uso:
        wf = WalkForwardEngine("AAPL", n_windows=4, train_pct=0.70, test_pct=0.15)
        results = wf.run(df_full, timesteps_per_window=300_000)
        print(wf.aggregate_metrics(results))
    """

    def __init__(
        self,
        symbol:      str,
        n_windows:   int   = 4,
        train_pct:   float = 0.70,
        test_pct:    float = 0.15,
        step_pct:    float = 0.15,
        device:      str   = "auto",
        n_envs:      int   = 0,
    ):
        self.symbol    = symbol.upper()
        self.n_windows = n_windows
        self.train_pct = train_pct
        self.test_pct  = test_pct
        self.step_pct  = step_pct
        self.device    = device
        self.n_envs    = n_envs

    def run(
        self,
        df:                    pd.DataFrame,
        timesteps_per_window:  int  = 300_000,
        features_ready:        bool = False,
        verbose:               bool = True,
    ) -> list[BacktestResult]:
        """Entrena y prueba en N ventanas deslizantes. Retorna lista de BacktestResult."""
        from src.brain.ModelTrainer import ModelTrainer
        from src.brain.FeatureEngineering import FeatureEngineer

        fe = FeatureEngineer()

        if not features_ready:
            if verbose:
                print("[WalkForward] Calculando features para el dataset completo...")
            df = fe.transform(df)
            features_ready = True

        n      = len(df)
        train_n = int(n * self.train_pct)
        test_n  = int(n * self.test_pct)
        step_n  = int(n * self.step_pct)

        results: list[BacktestResult] = []

        for w in range(self.n_windows):
            train_start = w * step_n
            train_end   = train_start + train_n
            test_end    = train_end   + test_n

            if test_end > n:
                print(f"[WalkForward] Ventana {w+1}: fuera de rango. Terminando.")
                break

            df_train = df.iloc[train_start : train_end]
            df_test  = df.iloc[train_end   : test_end]

            date_col = "datetime" if "datetime" in df.columns else None
            if verbose:
                d0 = df_train[date_col].iloc[0]  if date_col else train_start
                d1 = df_test[date_col].iloc[-1]  if date_col else test_end
                print(
                    f"\n{'═'*55}\n"
                    f"  Walk-Forward  Ventana {w+1}/{self.n_windows}\n"
                    f"  Train: {len(df_train):,} barras  |  Test: {len(df_test):,} barras\n"
                    f"  {d0}  →  {d1}\n"
                    f"{'═'*55}"
                )

            # Entrenar en esta ventana
            symbol_w = f"{self.symbol}_wf{w+1}"
            trainer  = ModelTrainer(
                symbol         = symbol_w,
                device         = self.device,
                n_envs         = self.n_envs,
                features_ready = True,
            )
            trainer.train(
                df              = df_train,
                total_timesteps = timesteps_per_window,
                eval_split      = 0.0,
                n_eval_episodes = 3,
            )

            # Backtest en ventana de test
            engine = BacktestEngine(symbol=symbol_w, features_ready=True)
            engine.load_model()
            result = engine.run(df_test, verbose=verbose)
            results.append(result)

        if verbose:
            self._print_aggregate(results)

        return results

    def aggregate_metrics(self, results: list[BacktestResult]) -> dict:
        """Agrega métricas de todas las ventanas en un resumen estadístico."""
        from tests.backtest.Backtestmetrics import BacktestMetrics

        all_metrics = [BacktestMetrics(r).compute() for r in results]

        def agg(key):
            vals = [m[key] for m in all_metrics if key in m]
            return {
                "mean":   round(float(np.mean(vals)),   4),
                "std":    round(float(np.std(vals)),    4),
                "min":    round(float(np.min(vals)),    4),
                "max":    round(float(np.max(vals)),    4),
            }

        keys = ["total_return_pct", "sharpe_ratio", "sortino_ratio",
                "max_drawdown_pct", "win_rate", "profit_factor"]

        return {k: agg(k) for k in keys}

    def _print_aggregate(self, results: list[BacktestResult]):
        from tests.backtest.Backtestmetrics import BacktestMetrics

        print(f"\n{'═'*55}")
        print(f"  Walk-Forward — {len(results)} ventanas completadas")
        print(f"{'═'*55}")
        for i, r in enumerate(results):
            m   = BacktestMetrics(r).compute()
            ret = m.get("total_return_pct", 0)
            bh  = m.get("benchmark_return_pct", 0)
            print(
                f"  Ventana {i+1}: retorno {ret:+.2f}%  "
                f"B&H {bh:+.2f}%  "
                f"Sharpe {m.get('sharpe_ratio', 0):.2f}  "
                f"Trades {m.get('n_trades', 0)}"
            )
        print()