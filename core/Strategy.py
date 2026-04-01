import pandas as pd

class SMAStrategy:
    """
    Estrategia de entrada basada en cruce de SMA con Higher Highs / Higher Lows.

    Condiciones de entrada (LONG):
        1. El cierre actual supera el máximo de la última vela (Higher High).
        2. El mínimo actual supera el mínimo de la última vela (Higher Low).
        3. El cierre actual está por encima de la SMA (cruce al alza).
        4. El cierre anterior estaba por debajo de la SMA (confirmación del cruce).
    """

    def __init__(self, sma_period: int = 50, profit_pct: float = 0.02, stop_pct: float = 0.01):
        self.sma_period = sma_period
        self.profit_pct = profit_pct   # 2 % profit target
        self.stop_pct = stop_pct       # 1 % stop loss

    def check_entry(
        self,
        current_close: float,
        current_low: float,
        last_bar,
        sma_series: pd.Series,
    ) -> bool:
        """Devuelve True si se cumplen todas las condiciones de entrada."""
        if len(sma_series.dropna()) < 2:
            return False

        last_sma = sma_series.iloc[-1]
        prev_sma = sma_series.iloc[-2]

        return (
            current_close > last_bar.high       # Higher High
            and current_low > last_bar.low      # Higher Low
            and current_close > last_sma        # Por encima de la SMA
            and last_bar.close < prev_sma       # Cruce desde abajo
        )

    def calculate_targets(self, entry_price: float) -> tuple[float, float]:
        """Calcula profit target y stop loss a partir del precio de entrada."""
        profit_target = entry_price * (1 + self.profit_pct)
        stop_loss = entry_price * (1 - self.stop_pct)
        return profit_target, stop_loss