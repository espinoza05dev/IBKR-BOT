"""
SessionLogger.py
Registra todo lo que ocurre durante una sesión de trading:
    - Eventos del sistema (conexión, desconexión, errores)
    - Barras recibidas
    - Decisiones de la IA_BackTests (acción, confianza)
    - Órdenes enviadas y fills recibidos
    - Resumen final de la sesión

Guarda en: IA_BackTests/logs/sessions/AAPL_paper_20240315_143022.jsonl
(Un JSON por línea para fácil análisis posterior)
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

SESSIONS_DIR = Path("LogsSession/sessions")


class SessionLogger:
    """
    Logger de sesión thread-safe.
    Escribe cada evento como una línea JSON (JSONL format).

    Uso:
        logger = SessionLogger(symbol="AAPL", paper=True)
        logger.log_event("BAR_CLOSE", {"close": 185.32, "volume": 12000})
        logger.log_trade("ORDER_BUY",  {"price": 185.32, "qty": 5})
        logger.save()
    """

    def __init__(self, symbol: str, paper: bool = True):
        self.symbol    = symbol.upper()
        self.paper     = paper
        self._events   : list[dict] = []
        self._trades   : list[dict] = []
        self._lock     = threading.Lock()
        self._start_ts = datetime.now()

        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

        mode   = "paper" if paper else "live"
        ts     = self._start_ts.strftime("%Y%m%d_%H%M%S")
        self._path = SESSIONS_DIR / f"{self.symbol}_{mode}_{ts}.jsonl"

        self.log_event("SESSION_START", {
            "symbol":    self.symbol,
            "mode":      mode,
            "started_at": self._start_ts.isoformat(),
        })

    # ── API pública ───────────────────────────────────────────────────────────

    def log_event(self, event_type: str, data: dict):
        """Registra un evento del sistema."""
        with self._lock:
            entry = {
                "ts":    datetime.now().isoformat(),
                "type":  event_type,
                "data":  data,
            }
            self._events.append(entry)
            self._write_line(entry)

    def log_trade(self, trade_type: str, data: dict):
        """Registra una operación de trading (orden, fill, cancelación)."""
        with self._lock:
            entry = {
                "ts":    datetime.now().isoformat(),
                "type":  trade_type,
                "data":  data,
            }
            self._trades.append(entry)
            self._write_line(entry)

    def log_ai_decision(
        self,
        action:     str,
        confidence: float,
        price:      float,
        blocked:    bool   = False,
        reason:     str    = "",
    ):
        """Registra la decisión del modelo de IA_BackTests."""
        self.log_event("AI_DECISION", {
            "action":     action,
            "confidence": round(confidence, 4),
            "price":      round(price, 4),
            "blocked":    blocked,
            "reason":     reason,
        })

    def save(self):
        """Guarda resumen de sesión al final."""
        summary = {
            "symbol":        self.symbol,
            "mode":          "paper" if self.paper else "live",
            "started_at":    self._start_ts.isoformat(),
            "ended_at":      datetime.now().isoformat(),
            "duration_min":  round((datetime.now() - self._start_ts).seconds / 60, 1),
            "total_events":  len(self._events),
            "total_trades":  len(self._trades),
            "log_file":      str(self._path),
        }
        self.log_event("SESSION_END", summary)
        print(f"[Logger] Sesión guardada → {self._path}")
        return self._path

    def get_trades(self) -> list[dict]:
        with self._lock:
            return list(self._trades)

    # ── Interno ───────────────────────────────────────────────────────────────

    def _write_line(self, entry: dict):
        """Escribe una línea JSON al archivo (append)."""
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            print(f"[Logger] Error escribiendo log: {e}")