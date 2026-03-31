from __future__ import annotations
"""
PaperTradingMonitor.py
Dashboard de trading en tiempo real en la terminal.
Muestra P&L, posición actual, historial de trades y estado del RiskManager.

Actualiza cada 5 segundos con la pantalla limpia para simular un dashboard live.
Compatible con paper trading y live trading.
"""

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from Sessionlogger import SessionLogger


@dataclass
class Position:
    """Posición abierta actual."""
    symbol:        str
    quantity:      int
    entry_price:   float
    entry_time:    datetime
    current_price: float   = 0.0
    profit_target: float   = 0.0
    stop_loss:     float   = 0.0

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pct(self) -> float:
        return (self.current_price - self.entry_price) / max(self.entry_price, 1e-9) * 100


@dataclass
class ClosedTrade:
    """Trade cerrado con resultado final."""
    symbol:      str
    quantity:    int
    entry_price: float
    exit_price:  float
    entry_time:  datetime
    exit_time:   datetime
    pnl:         float
    pnl_pct:     float
    is_winner:   bool


class PaperTradingMonitor:
    """
    Monitor de sesión de trading en tiempo real.

    Funciones:
        - Lleva cuenta del balance y P&L acumulado
        - Registra cada posición abierta y cerrada
        - Muestra dashboard actualizado en terminal
        - Detecta fills de IBKR para confirmar ejecuciones
        - Calcula métricas de sesión en tiempo real

    Uso:
        monitor = PaperTradingMonitor("AAPL", initial_cash=10_000)
        monitor.on_order_placed("BUY", 10, 185.32, 189.00, 183.00)
        monitor.update_price(186.50)
        monitor.run()   # Bloquea mostrando el dashboard
    """

    REFRESH_SECS = 5   # Refrescar pantalla cada N segundos

    def __init__(
        self,
        symbol:       str,
        initial_cash: float = 10_000.0,
        paper_mode:   bool  = True,
        logger:       Optional["SessionLogger"] = None,
    ):
        self.symbol       = symbol.upper()
        self.initial_cash = initial_cash
        self.cash         = initial_cash
        self.paper_mode   = paper_mode
        self.logger       = logger

        self.current_price: float         = 0.0
        self.position:      Optional[Position] = None
        self.closed_trades: list[ClosedTrade]  = []
        self._start_time   = datetime.now()
        self._running      = False
        self._lock         = threading.Lock()

    # ══════════════════════════════════════════════════════════════════════════
    # Eventos de trading
    # ══════════════════════════════════════════════════════════════════════════

    def update_price(self, price: float):
        """Actualizar precio actual (llamado en cada barra)."""
        with self._lock:
            self.current_price = price
            if self.position:
                self.position.current_price = price

    def on_order_placed(
        self,
        action:       str,
        quantity:     int,
        price:        float,
        profit_target: float = 0.0,
        stop_loss:    float  = 0.0,
    ):
        """Registrar que se envió una orden (puede no haberse ejecutado aún)."""
        with self._lock:
            if action == "BUY" and self.position is None:
                self.position = Position(
                    symbol        = self.symbol,
                    quantity      = quantity,
                    entry_price   = price,
                    entry_time    = datetime.now(),
                    current_price = price,
                    profit_target = profit_target,
                    stop_loss     = stop_loss,
                )
                self.cash -= price * quantity
                print(f"\n[Monitor] 📈 POSICIÓN ABIERTA  {self.symbol} × {quantity}  @ ${price:.2f}")

            elif action == "SELL" and self.position:
                pos    = self.position
                pnl    = (price - pos.entry_price) * pos.quantity
                pnl_p  = (price - pos.entry_price) / pos.entry_price * 100

                self.closed_trades.append(ClosedTrade(
                    symbol      = self.symbol,
                    quantity    = pos.quantity,
                    entry_price = pos.entry_price,
                    exit_price  = price,
                    entry_time  = pos.entry_time,
                    exit_time   = datetime.now(),
                    pnl         = pnl,
                    pnl_pct     = pnl_p,
                    is_winner   = pnl > 0,
                ))
                self.cash    += price * pos.quantity
                self.position = None

                icon = "✓" if pnl > 0 else "✗"
                print(
                    f"\n[Monitor] {icon} POSICIÓN CERRADA  "
                    f"PnL: ${pnl:+.2f} ({pnl_p:+.2f}%)"
                )

    def on_fill(self, fill_price: float, action: str, quantity: int):
        """
        Llamado cuando IBKR confirma la ejecución de una orden (fill real).
        En paper trading confirma que la orden simulada se procesó.
        """
        print(f"[Monitor] ✅ FILL confirmado  {action} {quantity} × ${fill_price:.2f}")
        if self.logger:
            self.logger.log_trade("FILL", {
                "action": action, "quantity": quantity, "price": fill_price
            })

    # ══════════════════════════════════════════════════════════════════════════
    # Dashboard
    # ══════════════════════════════════════════════════════════════════════════

    def run(self):
        """
        Muestra el dashboard en un loop.
        Bloquea el hilo principal — llamar al final del __init__ del bot.
        """
        self._running = True
        try:
            while self._running:
                self._render()
                time.sleep(self.REFRESH_SECS)
        except KeyboardInterrupt:
            pass

    def stop(self):
        self._running = False

    def _render(self):
        """Limpia la pantalla y dibuja el dashboard completo."""
        os.system("cls" if os.name == "nt" else "clear")

        mode_str  = "📄 PAPER TRADING" if self.paper_mode else "💰 LIVE TRADING"
        elapsed   = datetime.now() - self._start_time
        h, rem    = divmod(int(elapsed.total_seconds()), 3600)
        m, s      = divmod(rem, 60)

        with self._lock:
            equity     = self._equity()
            total_pnl  = equity - self.initial_cash
            total_pct  = total_pnl / self.initial_cash * 100
            n_trades   = len(self.closed_trades)
            n_wins     = sum(1 for t in self.closed_trades if t.is_winner)
            win_rate   = n_wins / max(n_trades, 1) * 100

        # ── Header ────────────────────────────────────────────────────────────
        print(f"{'═'*60}")
        print(f"  {mode_str}  |  {self.symbol}  |  "
              f"{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
        print(f"  Tiempo en sesión: {h:02d}:{m:02d}:{s:02d}")
        print(f"{'═'*60}")

        # ── Balance ───────────────────────────────────────────────────────────
        pnl_sym  = "+" if total_pnl >= 0 else ""
        pnl_clr  = ""   # La terminal no siempre soporta ANSI, evitamos problemas
        print(f"\n  BALANCE")
        print(f"  {'Capital inicial':<20}: ${self.initial_cash:>10,.2f}")
        print(f"  {'Equity actual':<20}: ${equity:>10,.2f}")
        print(f"  {'P&L sesión':<20}: ${pnl_sym}{total_pnl:>9,.2f}  ({pnl_sym}{total_pct:.2f}%)")
        print(f"  {'Precio actual':<20}: ${self.current_price:>10.2f}")

        # ── Posición abierta ──────────────────────────────────────────────────
        print(f"\n  POSICIÓN ACTUAL")
        with self._lock:
            pos = self.position
        if pos:
            u_pnl = pos.unrealized_pnl
            u_sym = "+" if u_pnl >= 0 else ""
            dur   = int((datetime.now() - pos.entry_time).total_seconds() / 60)
            print(f"  {'Símbolo':<20}: {pos.symbol}")
            print(f"  {'Cantidad':<20}: {pos.quantity}")
            print(f"  {'Precio entrada':<20}: ${pos.entry_price:.2f}")
            print(f"  {'Precio actual':<20}: ${pos.current_price:.2f}")
            print(f"  {'P&L no realizado':<20}: {u_sym}${u_pnl:.2f} ({u_sym}{pos.unrealized_pct:.2f}%)")
            print(f"  {'Take Profit':<20}: ${pos.profit_target:.2f}")
            print(f"  {'Stop Loss':<20}: ${pos.stop_loss:.2f}")
            print(f"  {'Duración':<20}: {dur} min")
        else:
            print(f"  Sin posición abierta")

        # ── Métricas de sesión ────────────────────────────────────────────────
        print(f"\n  MÉTRICAS DE SESIÓN")
        print(f"  {'Trades cerrados':<20}: {n_trades}")
        print(f"  {'Ganados / Perdidos':<20}: {n_wins} / {n_trades - n_wins}")
        print(f"  {'Win rate':<20}: {win_rate:.1f}%")

        with self._lock:
            ct = self.closed_trades

        if ct:
            best  = max(ct, key=lambda t: t.pnl)
            worst = min(ct, key=lambda t: t.pnl)
            print(f"  {'Mejor trade':<20}: ${best.pnl:+.2f}  ({best.pnl_pct:+.2f}%)")
            print(f"  {'Peor trade':<20}: ${worst.pnl:+.2f}  ({worst.pnl_pct:+.2f}%)")

        # ── Últimos 5 trades ──────────────────────────────────────────────────
        if ct:
            print(f"\n  ÚLTIMOS TRADES")
            print(f"  {'Entrada':>8}  {'Salida':>8}  {'PnL':>8}  {'%':>7}  Resultado")
            print(f"  {'─'*52}")
            for t in reversed(ct[-5:]):
                res  = "WIN " if t.is_winner else "LOSS"
                print(
                    f"  ${t.entry_price:>7.2f}  ${t.exit_price:>7.2f}  "
                    f"${t.pnl:>+7.2f}  {t.pnl_pct:>+6.2f}%  {res}"
                )

        print(f"\n{'─'*60}")
        print(f"  Refresca cada {self.REFRESH_SECS}s  |  Ctrl+C para cerrar")
        print(f"{'═'*60}\n")

    def session_summary(self) -> dict:
        """Resumen final de la sesión para el log."""
        with self._lock:
            n      = len(self.closed_trades)
            wins   = [t.pnl for t in self.closed_trades if t.is_winner]
            losses = [t.pnl for t in self.closed_trades if not t.is_winner]
            return {
                "initial_cash":  self.initial_cash,
                "final_equity":  round(self._equity(), 2),
                "total_pnl":     round(self._equity() - self.initial_cash, 2),
                "total_pnl_pct": round((self._equity() - self.initial_cash) / self.initial_cash * 100, 2),
                "n_trades":      n,
                "n_winners":     len(wins),
                "n_losers":      len(losses),
                "win_rate":      round(len(wins) / max(n, 1) * 100, 1),
                "avg_win":       round(sum(wins) / max(len(wins), 1), 2),
                "avg_loss":      round(sum(losses) / max(len(losses), 1), 2),
                "profit_factor": round(sum(wins) / max(abs(sum(losses)), 1e-9), 3),
                "duration_min":  round((datetime.now() - self._start_time).seconds / 60, 1),
            }

    def _equity(self) -> float:
        """Equity = cash + valor de posición abierta al precio actual."""
        if self.position:
            return self.cash + self.current_price * self.position.quantity
        return self.cash