"""
RiskManager.py
Capa de gestion de riesgo que envuelve las decisiones del agente IA_BackTests.
Nunca permite que el agente opere fuera de los limites de riesgo definidos.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class RiskConfig:
    max_position_pct:   float = 0.10   # Max 10% del capital por trade
    max_daily_loss_pct: float = 0.03   # Detener si pierde 3% en el dia
    max_drawdown_pct:   float = 0.15   # Detener si drawdown supera 15%
    max_trades_per_day: int   = 10     # Maximo de operaciones diarias
    min_confidence:     float = 0.60   # Confianza minima del modelo (0-1)
    atr_multiplier:     float = 2.0    # Stop loss = entrada - ATR * multiplier


@dataclass
class PortfolioState:
    balance:        float = 10_000.0
    peak_balance:   float = 10_000.0
    daily_start:    float = 10_000.0
    trades_today:   int   = 0
    open_positions: dict  = field(default_factory=dict)
    last_reset_day: Optional[str] = None


class RiskManager:
    """
    Verifica cada decision del agente IA_BackTests contra las reglas de riesgo.

    Uso:
        rm = RiskManager(config=RiskConfig())
        allowed, reason = rm.check(action, confidence, current_price, atr)
        if allowed:
            size = rm.position_size(current_price)
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.state  = PortfolioState()

    # ── API publica ───────────────────────────────────────────────────────────

    def check(
        self,
        action: int,
        confidence: float,
        current_price: float,
        atr: float = 0.0,
    ) -> tuple[bool, str]:
        """
        Verifica si el agente puede ejecutar la accion.

        Returns:
            (permitido: bool, razon: str)
        """
        self._maybe_reset_daily()

        if action == 0:                          # HOLD siempre permitido
            return True, "HOLD"

        # Confianza minima
        if confidence < self.config.min_confidence:
            return False, f"Confianza insuficiente: {confidence:.2f} < {self.config.min_confidence}"

        # Limite de operaciones diarias
        if self.state.trades_today >= self.config.max_trades_per_day:
            return False, f"Limite diario alcanzado: {self.state.trades_today} trades"

        # Perdida diaria maxima
        daily_pnl = (self.state.balance - self.state.daily_start) / self.state.daily_start
        if daily_pnl < -self.config.max_daily_loss_pct:
            return False, f"Perdida diaria maxima alcanzada: {daily_pnl:.2%}"

        # Drawdown maximo
        drawdown = (self.state.peak_balance - self.state.balance) / self.state.peak_balance
        if drawdown > self.config.max_drawdown_pct:
            return False, f"Drawdown maximo alcanzado: {drawdown:.2%}"

        return True, "OK"

    def position_size(self, price: float) -> int:
        """Calcula el numero de acciones a comprar segun el riesgo permitido."""
        max_capital = self.state.balance * self.config.max_position_pct
        shares = int(max_capital / price)
        return max(shares, 1)

    def dynamic_stop_loss(self, entry_price: float, atr: float) -> float:
        """Stop loss dinamico basado en ATR."""
        return entry_price - (atr * self.config.atr_multiplier)

    def dynamic_take_profit(self, entry_price: float, atr: float) -> float:
        """Take profit dinamico: riesgo/beneficio 1:2."""
        risk   = atr * self.config.atr_multiplier
        return entry_price + (risk * 2)

    def update_after_trade(self, pnl: float):
        """Actualiza el estado del portfolio despues de cerrar una posicion."""
        self.state.balance       += pnl
        self.state.peak_balance   = max(self.state.peak_balance, self.state.balance)
        self.state.trades_today  += 1
        print(
            f"[RiskManager] Trade cerrado | PnL: ${pnl:+.2f} | "
            f"Balance: ${self.state.balance:.2f} | "
            f"Trades hoy: {self.state.trades_today}"
        )

    def get_status(self) -> dict:
        drawdown   = (self.state.peak_balance - self.state.balance) / self.state.peak_balance
        daily_pnl  = (self.state.balance - self.state.daily_start) / self.state.daily_start
        return {
            "balance":      round(self.state.balance, 2),
            "drawdown":     round(drawdown, 4),
            "daily_pnl":    round(daily_pnl, 4),
            "trades_today": self.state.trades_today,
        }

    # ── Internos ──────────────────────────────────────────────────────────────

    def _maybe_reset_daily(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if self.state.last_reset_day != today:
            self.state.daily_start    = self.state.balance
            self.state.trades_today   = 0
            self.state.last_reset_day = today
            print(f"[RiskManager] Reset diario — Balance inicio: ${self.state.daily_start:.2f}")