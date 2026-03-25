from ibapi.client import EClient
from ibapi.wrapper import EWrapper


class IBApi(EWrapper, EClient):
    """
    Maneja la conexión con Interactive Brokers TWS/Gateway.
    Usa callbacks para desacoplar la lógica del bot de la capa de red.
    """

    def __init__(self):
        EClient.__init__(self, self)
        self._on_bar_update = None
        self._on_next_order_id = None

    def set_callbacks(self, on_bar_update: callable, on_next_order_id: callable):
        """Registra los callbacks que el Bot usará para recibir eventos."""
        self._on_bar_update = on_bar_update
        self._on_next_order_id = on_next_order_id

    # ── Datos históricos ──────────────────────────────────────────────────────

    def historicalData(self, reqId: int, bar):
        try:
            if self._on_bar_update:
                self._on_bar_update(reqId, bar, realtime=False)
        except Exception as e:
            print(f"[historicalData] Error: {e}")

    def historicalDataUpdate(self, reqId: int, bar):
        try:
            if self._on_bar_update:
                self._on_bar_update(reqId, bar, realtime=True)
        except Exception as e:
            print(f"[historicalDataUpdate] Error: {e}")

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        print(f"[IB] Datos históricos completos — ReqId: {reqId} | {start} → {end}")

    # ── Datos en tiempo real ──────────────────────────────────────────────────

    def realtimeBar(self, reqId, time, open_, high, low, close, volume, wap, count):
        super().realtimeBar(reqId, time, open_, high, low, close, volume, wap, count)
        try:
            if self._on_bar_update:
                self._on_bar_update(reqId, time, open_, high, low, close, volume, wap, count)
        except Exception as e:
            print(f"[realtimeBar] Error: {e}")

    # ── Órdenes ───────────────────────────────────────────────────────────────

    def nextValidId(self, orderId: int):
        if self._on_next_order_id:
            self._on_next_order_id(orderId)

    # ── Errores ───────────────────────────────────────────────────────────────

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        print(f"[IB Error {errorCode}] ReqId={reqId} — {errorString}")