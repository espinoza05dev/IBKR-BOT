"""
BrokerConecction.py
Capa de conexión con Interactive Brokers TWS / IB Gateway.
Usa callbacks para desacoplar completamente la lógica del bot.
"""

from __future__ import annotations

from typing import Optional, Callable

from ibapi.client  import EClient
from ibapi.wrapper import EWrapper
from ibapi.common  import OrderId


class IBApi(EWrapper, EClient):
    """
    Maneja la conexión con TWS/Gateway.
    Todos los eventos se delegan via callbacks registrados externamente.
    """

    def __init__(self):
        EClient.__init__(self, self)
        self._on_bar_update:     Optional[Callable] = None
        self._on_next_order_id:  Optional[Callable] = None
        self._on_order_status:   Optional[Callable] = None
        self._on_exec_details:   Optional[Callable] = None

    def set_callbacks(
        self,
        on_bar_update:    Callable,
        on_next_order_id: Callable,
        on_order_status:  Optional[Callable] = None,
        on_exec_details:  Optional[Callable] = None,
    ):
        self._on_bar_update    = on_bar_update
        self._on_next_order_id = on_next_order_id
        self._on_order_status  = on_order_status
        self._on_exec_details  = on_exec_details

    # ── Datos históricos ──────────────────────────────────────────────────────

    def historicalData(self, reqId: int, bar):
        try:
            if self._on_bar_update:
                self._on_bar_update(reqId, bar, realtime=False)
        except Exception as e:
            print(f"[IB] historicalData error: {e}")

    def historicalDataUpdate(self, reqId: int, bar):
        try:
            if self._on_bar_update:
                self._on_bar_update(reqId, bar, realtime=True)
        except Exception as e:
            print(f"[IB] historicalDataUpdate error: {e}")

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        print(f"[IB] Datos históricos completos  ReqId={reqId}  {start} → {end}")

    # ── Datos en tiempo real ──────────────────────────────────────────────────

    def realtimeBar(self, reqId, time, open_, high, low, close, volume, wap, count):
        super().realtimeBar(reqId, time, open_, high, low, close, volume, wap, count)
        try:
            if self._on_bar_update:
                self._on_bar_update(reqId, time, open_, high, low, close, volume, wap, count)
        except Exception as e:
            print(f"[IB] realtimeBar error: {e}")

    # ── Órdenes ───────────────────────────────────────────────────────────────

    def nextValidId(self, orderId: int):
        if self._on_next_order_id:
            self._on_next_order_id(orderId)

    def orderStatus(
        self, orderId, status, filled, remaining,
        avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice
    ):
        print(
            f"[IB] Orden {orderId}  status={status}  "
            f"filled={filled}  avgPrice={avgFillPrice:.2f}"
        )
        if self._on_order_status:
            self._on_order_status(orderId, status, filled, avgFillPrice)

    def execDetails(self, reqId, contract, execution):
        print(
            f"[IB] Fill  {execution.side}  {execution.shares}  "
            f"@ ${execution.price:.2f}  OrderId={execution.orderId}"
        )
        if self._on_exec_details:
            self._on_exec_details(execution)

    # ── Errores ───────────────────────────────────────────────────────────────

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        # Códigos informativos que no son errores reales
        info_codes = {2104, 2106, 2158, 2119, 2108, 2107, 10197}
        if errorCode in info_codes:
            print(f"[IB Info {errorCode}] {errorString}")
        else:
            print(f"[IB ERROR {errorCode}] ReqId={reqId}  {errorString}")