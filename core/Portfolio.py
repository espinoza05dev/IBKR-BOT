from ibapi.contract import Contract
from ibapi.order import Order


class Portfolio:
    """
    Gestiona la creación y el envío de órdenes a Interactive Brokers.
    """

    def __init__(self, ib, symbol: str):
        self.ib = ib
        self.symbol = symbol.upper()

    # ── Contrato ──────────────────────────────────────────────────────────────

    def _create_contract(self) -> Contract:
        contract = Contract()
        contract.symbol = self.symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    # ── Construcción de bracket order ────────────────────────────────────────

    def build_bracket_order(
        self,
        parent_order_id: int,
        action: str,
        quantity: int,
        profit_target: float,
        stop_loss: float,
    ) -> list[Order]:
        """
        Construye las tres órdenes que forman un bracket:
            1. Orden de entrada (MKT)
            2. Take Profit (LMT)
            3. Stop Loss (STP)
        """
        # Entrada
        parent = Order()
        parent.orderId = parent_order_id
        parent.orderType = "MKT"
        parent.action = action              # "BUY" / "SELL"
        parent.totalQuantity = quantity
        parent.transmit = False

        # Take Profit
        take_profit = Order()
        take_profit.orderId = parent_order_id + 1
        take_profit.orderType = "LMT"
        take_profit.action = "SELL"
        take_profit.totalQuantity = quantity
        take_profit.lmtPrice = round(profit_target, 2)
        take_profit.parentId = parent_order_id
        take_profit.transmit = False

        # Stop Loss
        stop = Order()
        stop.orderId = parent_order_id + 2
        stop.orderType = "STP"
        stop.action = "SELL"
        stop.totalQuantity = quantity
        stop.auxPrice = round(stop_loss, 2)
        stop.parentId = parent_order_id
        stop.transmit = True               # Transmite las 3 juntas

        return [parent, take_profit, stop]

    # ── Envío ─────────────────────────────────────────────────────────────────

    def place_bracket_order(
        self,
        order_id: int,
        action: str,
        quantity: int,
        profit_target: float,
        stop_loss: float,
    ) -> int:
        """
        Construye y envía el bracket order.
        Retorna el próximo order_id disponible (order_id + 3).
        """
        orders = self.build_bracket_order(order_id, action, quantity, profit_target, stop_loss)
        contract = self._create_contract()
        oca_group = f"OCA_{order_id}"

        for order in orders:
            order.ocaGroup = oca_group
            order.ocaType = 2
            self.ib.placeOrder(order.orderId, contract, order)
            print(f"[Portfolio] Orden enviada — ID={order.orderId} | Tipo={order.orderType} | Acción={order.action}")

        return order_id + 3