from __future__ import annotations
"""
Portfolio.py
Gestiona la creación y envío de órdenes a Interactive Brokers.
Soporte completo para bracket orders, market orders y fills.
"""
from typing import Optional, Callable
from ibapi.contract import Contract
from ibapi.order    import Order

class Portfolio:
    """
    Gestiona órdenes hacia IBKR.

    Tipos de orden soportados:
        place_bracket_order()  → BUY + Take Profit + Stop Loss (3 órdenes OCA)
        place_market_order()   → Market order simple (SELL para cerrar)
    """

    def __init__(self, ib, symbol: str):
        self.ib     = ib
        self.symbol = symbol.upper()
        self._fill_cb: Optional[Callable] = None

    def set_fill_callback(self, callback: Callable):
        """
        Registra un callback para cuando IBKR confirme un fill.
        El callback recibe (fill_price, action, quantity).
        """
        self._fill_cb = callback

    # ── Contratos ─────────────────────────────────────────────────────────────

    def _contract(self) -> Contract:
        c          = Contract()
        c.symbol   = self.symbol
        c.secType  = "STK"
        c.exchange = "SMART"
        c.currency = "USD"
        return c

    # ── Bracket Order ─────────────────────────────────────────────────────────

    def build_bracket_order(
        self,
        parent_order_id: int,
        action:          str,
        quantity:        int,
        profit_target:   float,
        stop_loss:       float,
    ) -> list[Order]:
        """
        Construye las 3 órdenes de un bracket:
            1. Entrada (MKT)
            2. Take Profit (LMT)
            3. Stop Loss (STP)
        Las 3 van en el mismo OCA group.
        """
        # Entrada
        parent                = Order()
        parent.orderId        = parent_order_id
        parent.orderType      = "MKT"
        parent.action         = action
        parent.totalQuantity  = quantity
        parent.transmit       = False

        # Take Profit
        tp                    = Order()
        tp.orderId            = parent_order_id + 1
        tp.orderType          = "LMT"
        tp.action             = "SELL" if action == "BUY" else "BUY"
        tp.totalQuantity      = quantity
        tp.lmtPrice           = round(profit_target, 2)
        tp.parentId           = parent_order_id
        tp.transmit           = False

        # Stop Loss
        sl                    = Order()
        sl.orderId            = parent_order_id + 2
        sl.orderType          = "STP"
        sl.action             = "SELL" if action == "BUY" else "BUY"
        sl.totalQuantity      = quantity
        sl.auxPrice           = round(stop_loss, 2)
        sl.parentId           = parent_order_id
        sl.transmit           = True   # Transmite las 3 juntas

        return [parent, tp, sl]

    def place_bracket_order(
        self,
        order_id:      int,
        action:        str,
        quantity:      int,
        profit_target: float,
        stop_loss:     float,
    ) -> int:
        """
        Construye y envía el bracket order a IBKR.
        Retorna el próximo order_id disponible (order_id + 3).
        """
        orders   = self.build_bracket_order(order_id, action, quantity, profit_target, stop_loss)
        contract = self._contract()
        oca_grp  = f"OCA_{order_id}"

        for o in orders:
            o.ocaGroup = oca_grp
            o.ocaType  = 2
            self.ib.placeOrder(o.orderId, contract, o)
            print(
                f"[Portfolio] Orden enviada  "
                f"ID={o.orderId}  {o.orderType}  {o.action}  qty={o.totalQuantity}"
            )

        return order_id + 3

    # ── Market Order simple ───────────────────────────────────────────────────

    def place_market_order(
        self,
        order_id: int,
        action:   str,
        quantity: int,
    ) -> int:
        """Orden de mercado simple. Usada para cerrar posiciones."""
        o                 = Order()
        o.orderId         = order_id
        o.orderType       = "MKT"
        o.action          = action
        o.totalQuantity   = quantity
        o.transmit        = True

        self.ib.placeOrder(order_id, self._contract(), o)
        print(f"[Portfolio] MKT {action} × {quantity}  ID={order_id}")
        return order_id + 1