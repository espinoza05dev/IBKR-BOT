"""
BacktestMetrics.py
Calcula todas las métricas financieras profesionales sobre un BacktestResult.

Métricas implementadas:
    Retorno         total, anualizado, vs benchmark (alpha)
    Riesgo          volatilidad, max drawdown, drawdown promedio
    Ajustado riesgo Sharpe, Sortino, Calmar, Omega ratio
    Trades          win rate, profit factor, payoff ratio, avg duración
    Estadísticas    VaR 95%, CVaR 95%, skewness, kurtosis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from IA.backtest.BacktestEngine import BacktestResult


class BacktestMetrics:
    """
    Calcula el conjunto completo de métricas financieras.

    Uso:
        metrics = BacktestMetrics(result).compute()
        print(metrics["sharpe_ratio"])
    """

    TRADING_DAYS_YEAR = 252
    RISK_FREE_RATE    = 0.05   # 5% anual

    def __init__(self, result: "BacktestResult"):
        self.r = result

    def compute(self) -> dict:
        """Calcula y retorna todas las métricas en un dict."""
        equity  = self.r.equity_curve
        bh      = self.r.benchmark_equity
        trades  = [t for t in self.r.trades if t.exit_date is not None]

        daily_returns  = equity.pct_change().dropna()
        bh_returns     = bh.pct_change().dropna()

        m: dict = {}

        # ── Retorno ───────────────────────────────────────────────────────────
        m["initial_balance"]      = round(self.r.initial_balance, 2)
        m["final_balance"]        = round(self.r.final_balance,   2)
        m["total_return_pct"]     = round(self.r.total_return * 100, 2)
        m["benchmark_return_pct"] = round(
            (bh.iloc[-1] / self.r.initial_balance - 1) * 100, 2
        )
        m["alpha_pct"]            = round(
            m["total_return_pct"] - m["benchmark_return_pct"], 2
        )

        # Retorno anualizado (CAGR)
        n_bars   = len(equity)
        years    = n_bars / (self.TRADING_DAYS_YEAR * 6.5)   # 6.5h por día de trading
        if years > 0 and self.r.final_balance > 0:
            m["cagr_pct"] = round(
                ((self.r.final_balance / self.r.initial_balance) ** (1 / years) - 1) * 100, 2
            )
        else:
            m["cagr_pct"] = 0.0

        # ── Riesgo ────────────────────────────────────────────────────────────
        if len(daily_returns) > 1:
            m["volatility_annual_pct"] = round(
                float(daily_returns.std()) * np.sqrt(self.TRADING_DAYS_YEAR) * 100, 2
            )
        else:
            m["volatility_annual_pct"] = 0.0

        dd                         = self.r.drawdown_series
        m["max_drawdown_pct"]      = round(float(dd.max()) * 100, 2)
        m["avg_drawdown_pct"]      = round(float(dd[dd > 0].mean()) * 100, 2) if (dd > 0).any() else 0.0
        m["drawdown_duration_bars"] = int(self._max_drawdown_duration(dd))

        # ── Ratios ajustados al riesgo ────────────────────────────────────────
        rf_per_bar = self.RISK_FREE_RATE / (self.TRADING_DAYS_YEAR * 6.5)

        m["sharpe_ratio"]  = round(self._sharpe(daily_returns, rf_per_bar), 3)
        m["sortino_ratio"] = round(self._sortino(daily_returns, rf_per_bar), 3)
        m["calmar_ratio"]  = round(self._calmar(m["cagr_pct"], m["max_drawdown_pct"]), 3)
        m["omega_ratio"]   = round(self._omega(daily_returns, rf_per_bar), 3)

        # ── Estadísticas de distribución ──────────────────────────────────────
        if len(daily_returns) > 4:
            from scipy import stats as sp_stats
            m["skewness"] = round(float(sp_stats.skew(daily_returns)), 4)
            m["kurtosis"] = round(float(sp_stats.kurtosis(daily_returns)), 4)
            # VaR y CVaR al 95%
            m["var_95_pct"]  = round(float(np.percentile(daily_returns, 5)) * 100, 3)
            m["cvar_95_pct"] = round(
                float(daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean()) * 100, 3
            )
        else:
            m["skewness"] = m["kurtosis"] = m["var_95_pct"] = m["cvar_95_pct"] = 0.0

        # ── Métricas de trades ────────────────────────────────────────────────
        m["n_trades"]          = len(trades)
        m["n_winners"]         = sum(1 for t in trades if t.is_winner)
        m["n_losers"]          = sum(1 for t in trades if not t.is_winner)
        m["win_rate"]          = round(m["n_winners"] / max(m["n_trades"], 1), 3)

        wins   = [t.pnl_net for t in trades if t.is_winner]
        losses = [abs(t.pnl_net) for t in trades if not t.is_winner]

        m["avg_win"]           = round(float(np.mean(wins))   if wins   else 0.0, 2)
        m["avg_loss"]          = round(float(np.mean(losses)) if losses else 0.0, 2)
        m["largest_win"]       = round(float(max(wins))       if wins   else 0.0, 2)
        m["largest_loss"]      = round(float(max(losses))     if losses else 0.0, 2)
        m["profit_factor"]     = round(sum(wins) / max(sum(losses), 1e-9), 3)
        m["payoff_ratio"]      = round(m["avg_win"] / max(m["avg_loss"], 1e-9), 3)
        m["total_commissions"] = round(sum(t.commission for t in trades), 2)
        m["expectancy"]        = round(
            m["win_rate"] * m["avg_win"] - (1 - m["win_rate"]) * m["avg_loss"], 2
        )
        m["avg_duration_bars"] = round(
            float(np.mean([t.duration_bars for t in trades])) if trades else 0.0, 1
        )

        # ── Correlación con benchmark ─────────────────────────────────────────
        if len(daily_returns) > 4 and len(bh_returns) > 4:
            min_len = min(len(daily_returns), len(bh_returns))
            m["beta"] = round(
                float(np.cov(
                    daily_returns.values[:min_len],
                    bh_returns.values[:min_len]
                )[0, 1] / max(float(bh_returns.var()), 1e-12)), 3
            )
            m["correlation_with_bh"] = round(
                float(np.corrcoef(
                    daily_returns.values[:min_len],
                    bh_returns.values[:min_len]
                )[0, 1]), 3
            )
        else:
            m["beta"] = m["correlation_with_bh"] = 0.0

        # ── Veredicto final ───────────────────────────────────────────────────
        m["approved_for_live"] = self._verdict(m)

        self.r.metrics = m
        return m

    # ── Fórmulas ──────────────────────────────────────────────────────────────

    def _sharpe(self, returns: pd.Series, rf: float) -> float:
        excess = returns - rf
        return 0.0 if excess.std() < 1e-9 else float(
            excess.mean() / excess.std() * np.sqrt(self.TRADING_DAYS_YEAR * 6.5)
        )

    def _sortino(self, returns: pd.Series, rf: float) -> float:
        excess  = returns - rf
        neg     = excess[excess < 0]
        down_sd = neg.std() if len(neg) > 1 else 1e-9
        return 0.0 if down_sd < 1e-9 else float(
            excess.mean() / down_sd * np.sqrt(self.TRADING_DAYS_YEAR * 6.5)
        )

    @staticmethod
    def _calmar(cagr_pct: float, max_dd_pct: float) -> float:
        return 0.0 if max_dd_pct < 1e-9 else cagr_pct / max_dd_pct

    @staticmethod
    def _omega(returns: pd.Series, threshold: float) -> float:
        excess   = returns - threshold
        gains    = excess[excess > 0].sum()
        losses   = abs(excess[excess < 0].sum())
        return gains / max(losses, 1e-9)

    @staticmethod
    def _max_drawdown_duration(dd: pd.Series) -> int:
        """Duración máxima del drawdown en barras."""
        max_dur = 0
        cur_dur = 0
        for v in dd:
            if v > 0:
                cur_dur += 1
                max_dur  = max(max_dur, cur_dur)
            else:
                cur_dur  = 0
        return max_dur

    @staticmethod
    def _verdict(m: dict) -> bool:
        """
        Criterios mínimos para operar con dinero real.
        Todos deben cumplirse simultáneamente.
        """
        return (
            m["win_rate"]          >= 0.50 and
            m["sharpe_ratio"]      >= 0.50 and
            m["sortino_ratio"]     >= 0.75 and
            m["max_drawdown_pct"]  <= 20.0 and
            m["profit_factor"]     >= 1.20 and
            m["n_trades"]          >= 10   and
            m["alpha_pct"]         >= 0.0
        )