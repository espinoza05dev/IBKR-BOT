import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import ta

from core.Strategy import SMAStrategy


class Bar:
    """Representa una vela OHLCV."""

    def __init__(self):
        self.open: float = 0.0
        self.high: float = 0.0
        self.low: float = 0.0
        self.close: float = 0.0
        self.volume: float = 0.0
        self.date: str = ""

    def __repr__(self):
        return (
            f"Bar(date={self.date}, O={self.open}, H={self.high}, "
            f"L={self.low}, C={self.close}, V={self.volume})"
        )


class MarketDataHandler:
    """
    Recibe barras historicas y en tiempo real de IB,
    construye las velas del timeframe solicitado y
    notifica al Bot cuando se dispara una senal de la estrategia.
    """

    TZ = pytz.timezone("America/New_York")

    def __init__(self, barsize: int, strategy: SMAStrategy):
        self.barsize = barsize
        self.strategy = strategy
        self.bars = []
        self.current_bar = Bar()
        self._init_time = datetime.now().astimezone(self.TZ)
        self._on_signal = None

    def set_signal_callback(self, callback):
        """El callback recibe (close_price: float) cuando hay senal de entrada."""
        self._on_signal = callback

    def on_bar_update(self, reqId, bar, realtime: bool):
        if not realtime:
            self.bars.append(bar)
            return
        self._process_realtime_bar(bar)

    def _process_realtime_bar(self, bar):
        bartime = datetime.strptime(bar.date, "%Y%m%d %H:%M:%S").astimezone(self.TZ)
        minutes_diff = (bartime - self._init_time).total_seconds() / 60
        self.current_bar.date = bartime

        if minutes_diff > 0 and math.floor(minutes_diff) % self.barsize == 0:
            self._on_bar_close(bar)

        self._update_current_bar(bar)

    def _on_bar_close(self, bar):
        if len(self.bars) < self.strategy.sma_period:
            print(f"[MarketData] Acumulando datos ({len(self.bars)}/{self.strategy.sma_period})")
            return

        sma_series = self._compute_sma()
        print(f"[MarketData] SMA({self.strategy.sma_period}): {sma_series.iloc[-1]:.4f}")

        last_bar = self.bars[-1]
        signal = self.strategy.check_entry(
            current_close=bar.close,
            current_low=self.current_bar.low,
            last_bar=last_bar,
            sma_series=sma_series,
        )

        if signal:
            print("[MarketData] Senal de entrada detectada")
            if self._on_signal:
                self._on_signal(bar.close)

        self.current_bar.close = bar.close
        if self.current_bar.date != last_bar.date:
            print("[MarketData] Nueva vela anyadida al historial")
            self.bars.append(self.current_bar)

        self.current_bar = Bar()
        self.current_bar.open = bar.open

    def _update_current_bar(self, bar):
        if self.current_bar.open == 0:
            self.current_bar.open = bar.open
        if self.current_bar.high == 0 or bar.high > self.current_bar.high:
            self.current_bar.high = bar.high
        # BUG FIX original: usaba ">" en lugar de "<" para el low
        if self.current_bar.low == 0 or bar.low < self.current_bar.low:
            self.current_bar.low = bar.low

    def _compute_sma(self) -> pd.Series:
        closes = pd.Series([b.close for b in self.bars], dtype=float)
        return ta.trend.sma_indicator(closes, window=self.strategy.sma_period)