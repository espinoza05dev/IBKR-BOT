"""
RealTime_Market_Data.py
Clase Bar y MarketDataHandler.

MarketDataHandler recibe barras crudas de IBApi y las ensambla en velas
del timeframe configurado. Cuando cierra una vela notifica via callbacks.

Cambios vs versión anterior:
    - strategy ahora es opcional (None = solo IA decide)
    - Nuevo callback set_bar_close_callback() → notifica al Bot/IA con la Bar cerrada
    - BUG FIX: condición del low era ">" en lugar de "<"
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional, Callable

import numpy as np
import pandas as pd
import pytz
import ta

from core.Strategy import SMAStrategy


class Bar:
    """Representa una vela OHLCV."""

    def __init__(self):
        self.open:   float = 0.0
        self.high:   float = 0.0
        self.low:    float = 0.0
        self.close:  float = 0.0
        self.volume: float = 0.0
        self.date:   str   = ""

    def __repr__(self):
        return (
            f"Bar(date={self.date}, O={self.open:.2f}, H={self.high:.2f}, "
            f"L={self.low:.2f}, C={self.close:.2f}, V={self.volume})"
        )

    def to_dict(self) -> dict:
        return {
            "open": self.open, "high": self.high,
            "low":  self.low,  "close": self.close,
            "volume": self.volume,
        }


class MarketDataHandler:
    """
    Ensambla barras de IB en velas del timeframe configurado.

    Callbacks disponibles:
        set_signal_callback(fn)     → fn(close_price) cuando SMAStrategy señala entrada
        set_bar_close_callback(fn)  → fn(Bar) cada vez que cierra una vela (para la IA)
    """

    TZ = pytz.timezone("America/New_York")

    def __init__(
        self,
        barsize:  int,
        strategy: Optional[SMAStrategy] = None,
    ):
        self.barsize   = barsize
        self.strategy  = strategy      # Puede ser None si solo usa IA
        self.bars:     list[Bar] = []
        self.current_bar = Bar()
        self._init_time  = datetime.now().astimezone(self.TZ)

        self._signal_cb:    Optional[Callable] = None   # SMAStrategy → Bot viejo
        self._bar_close_cb: Optional[Callable] = None   # Bar cerrada → TradingAI

    # ── Configuración ─────────────────────────────────────────────────────────

    def set_signal_callback(self, callback: Callable):
        """Para compatibilidad con el bot original (SMAStrategy)."""
        self._signal_cb = callback

    def set_bar_close_callback(self, callback: Callable):
        """
        Se llama con la Bar completa cada vez que cierra una vela.
        Usar este para conectar con TradingAI.
        """
        self._bar_close_cb = callback

    # ── Punto de entrada de datos ─────────────────────────────────────────────

    def on_bar_update(self, reqId, bar, realtime: bool):
        if not realtime:
            self._store_historical(bar)
        else:
            self._process_realtime(bar)

    # ── Datos históricos ──────────────────────────────────────────────────────

    def _store_historical(self, bar):
        b = Bar()
        b.open   = bar.open
        b.high   = bar.high
        b.low    = bar.low
        b.close  = bar.close
        b.volume = bar.volume
        b.date   = bar.date
        self.bars.append(b)

    # ── Datos en tiempo real ──────────────────────────────────────────────────

    def _process_realtime(self, bar):
        try:
            bartime = datetime.strptime(bar.date, "%Y%m%d %H:%M:%S").astimezone(self.TZ)
        except Exception:
            bartime = datetime.now().astimezone(self.TZ)

        minutes_diff = (bartime - self._init_time).total_seconds() / 60
        self.current_bar.date = bartime

        if minutes_diff > 0 and math.floor(minutes_diff) % self.barsize == 0:
            self._on_bar_close(bar)

        self._update_current_bar(bar)

    def _on_bar_close(self, bar):
        """Lógica al cierre de cada vela del timeframe."""
        if not self.bars:
            return

        # Completar la barra actual
        closed          = Bar()
        closed.open     = self.current_bar.open  or bar.open
        closed.high     = self.current_bar.high  or bar.high
        closed.low      = self.current_bar.low   or bar.low
        closed.close    = bar.close
        closed.volume   = bar.volume
        closed.date     = self.current_bar.date

        # ── Callback para TradingAI ───────────────────────────────────────────
        if self._bar_close_cb:
            try:
                self._bar_close_cb(closed)
            except Exception as e:
                print(f"[MarketData] Error en bar_close_callback: {e}")

        # ── SMAStrategy (modo legado) ─────────────────────────────────────────
        if self.strategy and self._signal_cb and len(self.bars) >= self.strategy.sma_period:
            sma_series = self._compute_sma()
            last_bar   = self.bars[-1]
            if self.strategy.check_entry(
                current_close = bar.close,
                current_low   = self.current_bar.low,
                last_bar      = last_bar,
                sma_series    = sma_series,
            ):
                self._signal_cb(bar.close)

        # Archivar barra cerrada
        last_bar = self.bars[-1]
        if closed.date != last_bar.date:
            self.bars.append(closed)

        # Reiniciar barra en curso
        self.current_bar       = Bar()
        self.current_bar.open  = bar.open

    def _update_current_bar(self, bar):
        if self.current_bar.open == 0:
            self.current_bar.open = bar.open
        if self.current_bar.high == 0 or bar.high > self.current_bar.high:
            self.current_bar.high = bar.high
        # BUG FIX original: era ">" — correcto es "<"
        if self.current_bar.low == 0 or bar.low < self.current_bar.low:
            self.current_bar.low = bar.low

    def _compute_sma(self) -> pd.Series:
        closes = pd.Series([b.close for b in self.bars], dtype=float)
        return ta.trend.sma_indicator(closes, window=self.strategy.sma_period)