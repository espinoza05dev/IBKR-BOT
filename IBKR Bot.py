from __future__ import annotations
"""
IBKR Bot.py - Punto de entrada del autotrader.
Orquesta la conexion, datos de mercado, estrategia y portfolio.
"""
"""
IBKR Bot.py
Punto de entrada del autotrader con IA_BackTests integrada.
Soporta Paper Trading (simulado) y Live Trading (dinero real).

Flujo completo:
    1. Conectar a TWS/Gateway (paper o live)
    2. Cargar modelo PPO entrenado (TradingAI)
    3. Descargar barras históricas para calentamiento (warmup)
    4. Suscribir a barras en tiempo real
    5. Cada barra → TradingAI decide → RiskManager valida → Portfolio ejecuta
    6. SessionLogger registra todo → PaperTradingMonitor muestra dashboard

Antes de ejecutar:
    - Abre Trader Workstation (TWS) o IB Gateway
    - Activa API: TWS → Edit → Global Configuration → API → Settings
      ✓ Enable ActiveX and Socket Clients
      ✓ Allow connections from localhost only
    - Modo Paper: puerto 7497 (TWS) o 4002 (Gateway)
    - Modo Live:  puerto 7496 (TWS) o 4001 (Gateway)  ← CUIDADO: dinero real
"""
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import pytz
from ibapi.contract import Contract
from core.BrockerConnection  import IBApi
from core.Portfolio         import Portfolio
from Data.RealTime_Market_Data import MarketDataHandler, Bar
from IA.TradingAI           import TradingAI
from IA.RiskManager         import RiskConfig
from IA.Sessionlogger       import SessionLogger
from IA.Papertradingmonitor import PaperTradingMonitor


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — edita solo esta sección
# ══════════════════════════════════════════════════════════════════════════════

SYMBOL       = "PG"    # Símbolo a operar, la IA_BackTests debe de haber sido entrenada con el simbolo antes de ejecutal el BOT
INTERVAL_MIN = 60        # Tamaño de vela en minutos (ej: 60 = velas de 1h)
INITIAL_CASH = 10_000.0  # Capital inicial (informativo para el monitor)

# ── Conexión IB ───────────────────────────────────────────────────────────────
PAPER_MODE = True        # True = Paper Trading | False = LIVE (dinero real)

TWS_HOST   = "127.0.0.1"
TWS_PORT   = 7497        # Paper: 7497 (TWS) / 4002 (Gateway)
                         # Live:  7496 (TWS) / 4001 (Gateway)
CLIENT_ID  = 1           # ID único de cliente (cambia si hay varias conexiones)

# ── IA_BackTests / Riesgo ───────────────────────────────────────────────────────────────
WARMUP_BARS          = 80    # Barras históricas antes de operar
CONFIDENCE_THRESHOLD = 0.60  # Confianza mínima del modelo para actuar

RISK = RiskConfig(
    max_position_pct   = 0.10,   # Max 10% del capital por trade
    max_daily_loss_pct = 0.03,   # Detener si pierde 3% en el día
    max_drawdown_pct   = 0.15,   # Detener si drawdown supera 15%
    max_trades_per_day = 10,     # Máximo de operaciones diarias
    min_confidence     = CONFIDENCE_THRESHOLD,
    atr_multiplier     = 2.0,    # Stop loss = entrada - ATR × 2
)

# ── Datos históricos (warmup) ─────────────────────────────────────────────────
HISTORY_DURATION = "5 D"  # Cuánto historial pedir para calentar la IA_BackTests
                           # "1 D" | "5 D" | "1 M" | "6 M" | "1 Y"
# ══════════════════════════════════════════════════════════════════════════════
class AITradingBot:
    """
    Bot de trading con IA_BackTests integrada.

    Componentes:
        IBApi              → Conexión con TWS/Gateway
        TradingAI          → Cerebro: modelo PPO + RiskManager
        Portfolio          → Envío y gestión de órdenes
        MarketDataHandler  → Construcción de velas desde barras IB
        SessionLogger      → Log detallado de toda la sesión
        PaperTradingMonitor→ Dashboard de P&L en tiempo real
    """

    def __init__(self):
        self._running   = False
        self._order_id  = 1
        self._lock      = threading.Lock()

        mode_str = "📄 PAPER TRADING" if PAPER_MODE else "💰 LIVE TRADING"
        print(f"\n{'═'*55}")
        print(f"  AutoTrader IA_BackTests  |  {mode_str}")
        print(f"  Símbolo: {SYMBOL}  |  Velas: {INTERVAL_MIN} min")
        print(f"  Host: {TWS_HOST}:{TWS_PORT}  |  ClientID: {CLIENT_ID}")
        print(f"{'═'*55}\n")

        if not PAPER_MODE:
            self._confirm_live_mode()

        # ── 1. Logger (primero, para registrar todo) ──────────────────────────
        self.logger  = SessionLogger(symbol=SYMBOL, paper=PAPER_MODE)

        # ── 2. Monitor de P&L ────────────────────────────────────────────────
        self.monitor = PaperTradingMonitor(
            symbol          = SYMBOL,
            initial_cash    = INITIAL_CASH,
            paper_mode      = PAPER_MODE,
            logger          = self.logger,
        )

        # ── 3. Conexión IB ────────────────────────────────────────────────────
        self.ib = IBApi()
        self.ib.set_callbacks(
            on_bar_update    = self._on_bar_update,
            on_next_order_id = self._set_order_id,
        )
        print(f"[Bot] Conectando a TWS en {TWS_HOST}:{TWS_PORT}...")
        self.ib.connect(TWS_HOST, TWS_PORT, CLIENT_ID)

        ib_thread = threading.Thread(target=self.ib.run, daemon=True)
        ib_thread.start()
        time.sleep(2)   # Dar tiempo a que llegue nextValidId

        if not self.ib.isConnected():
            raise ConnectionError(
                "No se pudo conectar a TWS/IB Gateway.\n"
                "Verifica que TWS esté abierto y la API esté habilitada.\n"
                f"Puerto configurado: {TWS_PORT}"
            )
        print(f"[Bot] ✓ Conectado a IB  (order_id={self._order_id})")

        # ── 4. Portfolio ──────────────────────────────────────────────────────
        self.portfolio = Portfolio(ib=self.ib, symbol=SYMBOL)
        self.portfolio.set_fill_callback(self.monitor.on_fill)   # Notificar fills

        # ── 5. TradingAI ──────────────────────────────────────────────────────
        print(f"[Bot] Cargando modelo IA_BackTests para {SYMBOL}...")
        self.ai = TradingAI(
            symbol               = SYMBOL,
            risk_config          = RISK,
            confidence_threshold = CONFIDENCE_THRESHOLD,
        )
        self.ai.load()
        self.ai.set_order_callback(self._execute_ai_order)
        print(f"[Bot] ✓ Modelo IA_BackTests cargado")

        # ── 6. MarketDataHandler (envuelve barras IB y alimenta a la IA_BackTests) ──────
        self.market_data = MarketDataHandler(
            barsize  = INTERVAL_MIN,
            strategy = None,      # Sin SMAStrategy — la IA_BackTests toma las decisiones
        )
        self.market_data.set_bar_close_callback(self._on_bar_closed)

        # ── 7. Solicitar datos ────────────────────────────────────────────────
        self.ib.reqIds(-1)
        self._request_data()

        self._running = True
        self.logger.log_event("BOT_START", {
            "symbol": SYMBOL, "interval": INTERVAL_MIN,
            "paper": PAPER_MODE, "port": TWS_PORT,
        })

        # ── 8. Señal de apagado limpio ────────────────────────────────────────
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        print(f"\n[Bot] ✓ Sistema iniciado — esperando barras ({WARMUP_BARS} para calentamiento)")
        print(f"[Bot] Presiona Ctrl+C para cerrar limpiamente\n")

        # Mantener el proceso vivo
        self.monitor.run()   # Bloquea aquí mostrando el dashboard

    # ══════════════════════════════════════════════════════════════════════════
    # Flujo de datos
    # ══════════════════════════════════════════════════════════════════════════

    def _request_data(self):
        """Pide historial (warmup) + suscripción a barras en tiempo real."""
        contract = self._make_contract()
        min_text = " mins" if INTERVAL_MIN > 1 else " min"
        self.ib.reqHistoricalData(
            reqId          = 0,
            contract       = contract,
            endDateTime    = "",
            durationStr    = HISTORY_DURATION,
            barSizeSetting = f"{INTERVAL_MIN}{min_text}",
            whatToShow     = "TRADES",
            useRTH         = 1,
            formatDate     = 1,
            keepUpToDate   = True,   # ← Continúa con barras en tiempo real
            chartOptions   = [],
        )
        print(f"[Bot] Solicitando {HISTORY_DURATION} de historial + streaming en tiempo real...")

    def _on_bar_update(self, reqId, bar, realtime: bool):
        """Callback de IBApi. Redirige al MarketDataHandler."""
        self.market_data.on_bar_update(reqId, bar, realtime)

    def _on_bar_closed(self, closed_bar: Bar):
        """
        Callback llamado por MarketDataHandler al cierre de cada vela.
        Aquí la IA_BackTests toma la decisión.
        """
        self.ai.on_new_bar(closed_bar)

        # Actualizar monitor con el precio actual
        self.monitor.update_price(closed_bar.close)

        self.logger.log_event("BAR_CLOSE", {
            "date":   str(closed_bar.date),
            "open":   closed_bar.open,
            "high":   closed_bar.high,
            "low":    closed_bar.low,
            "close":  closed_bar.close,
            "volume": closed_bar.volume,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # Ejecución de órdenes
    # ══════════════════════════════════════════════════════════════════════════

    def _execute_ai_order(
        self,
        action:       str,
        quantity:     int,
        profit_target: float,
        stop_loss:    float,
    ):
        """
        Recibe la orden de la IA_BackTests y la envía a IBKR via Portfolio.
        Este callback es el puente entre TradingAI y Portfolio.
        """
        with self._lock:
            price = self.monitor.current_price

            print(
                f"\n[Bot] ▶ ORDEN IA_BackTests\n"
                f"  Acción       : {action}\n"
                f"  Cantidad     : {quantity}\n"
                f"  Precio actual: ${price:.2f}\n"
                f"  Take Profit  : ${profit_target:.2f}\n"
                f"  Stop Loss    : ${stop_loss:.2f}\n"
                f"  Modo         : {'PAPER' if PAPER_MODE else 'LIVE'}"
            )

            if action == "BUY":
                new_order_id = self.portfolio.place_bracket_order(
                    order_id      = self._order_id,
                    action        = "BUY",
                    quantity      = quantity,
                    profit_target = profit_target,
                    stop_loss     = stop_loss,
                )
                self._order_id = new_order_id

                self.monitor.on_order_placed(action, quantity, price, profit_target, stop_loss)
                self.logger.log_trade("ORDER_BUY", {
                    "quantity": quantity, "price": price,
                    "tp": profit_target, "sl": stop_loss,
                    "order_id": self._order_id,
                })

            elif action == "SELL":
                # Cierre de posición existente
                new_order_id = self.portfolio.place_market_order(
                    order_id = self._order_id,
                    action   = "SELL",
                    quantity = quantity,
                )
                self._order_id = new_order_id
                self.monitor.on_order_placed(action, quantity, price)
                self.logger.log_trade("ORDER_SELL", {
                    "quantity": quantity, "price": price,
                    "order_id": self._order_id,
                })

    # ══════════════════════════════════════════════════════════════════════════
    # Utilidades
    # ══════════════════════════════════════════════════════════════════════════

    def _set_order_id(self, order_id: int):
        with self._lock:
            self._order_id = order_id
        print(f"[Bot] Order ID válido: {order_id}")

    def _make_contract(self) -> Contract:
        c          = Contract()
        c.symbol   = SYMBOL
        c.secType  = "STK"
        c.exchange = "SMART"
        c.currency = "USD"
        return c

    def _shutdown(self, sig=None, frame=None):
        """Cierre limpio: cancela suscripciones, guarda log, desconecta."""
        if not self._running:
            return
        self._running = False
        print(f"\n[Bot] Cerrando sesión...")

        self.monitor.stop()
        summary = self.monitor.session_summary()
        self.logger.log_event("BOT_STOP", summary)
        self.logger.save()

        self.ib.cancelHistoricalData(0)
        time.sleep(0.5)
        self.ib.disconnect()
        print(f"[Bot] ✓ Desconectado de IB")
        print(f"\n{'═'*55}")
        print(f"  Sesión finalizada  |  {datetime.now().strftime('%H:%M:%S')}")
        for k, v in summary.items():
            print(f"  {k:<25}: {v}")
        print(f"{'═'*55}\n")
        sys.exit(0)

    @staticmethod
    def _confirm_live_mode():
        """Doble confirmación antes de operar con dinero real."""
        print("\n" + "!"*55)
        print("  ⚠  ATENCIÓN: MODO LIVE TRADING ACTIVADO")
        print("  ⚠  Las órdenes se ejecutarán con DINERO REAL")
        print("!"*55)
        resp = input("\n  Escribe 'CONFIRMO' para continuar: ").strip()
        if resp != "CONFIRMO":
            print("[Bot] Cancelado. Cambia PAPER_MODE=True para usar paper trading.")
            sys.exit(0)
        print()
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bot = AITradingBot()

#--------------------------------------------------------------------BOT FUNCIONAL----------------------------------------------------------------
    # from datetime import datetime, timedelta
    #
    # import ibapi
    # from ibapi.client import EClient
    # from ibapi.common import BarData
    # from ibapi.wrapper import EWrapper
    # import ta
    # import numpy as np
    # import pandas as pd
    # import pytz
    # import math
    # from ibapi.contract import Contract
    # from ibapi.order import *
    # import threading
    # import time
    #
    # orderId = 1
    #
    # #Connection
    # class IBApi(EWrapper,EClient):
    #     def __init__(self):
    #         EClient.__init__(self,self)
    #     def historicalData(self, reqId, bar):
    #         try:
    #             bot.on_bar_update(reqId, bar)
    #         except Exception as e:
    #             print(e)
    #     #On realtime Bar after historical data finishes
    #     def historicalDataUpate(self,reqId, bar):
    #         try:
    #             bot.on_bar_update(reqId, bar)
    #         except Exception as e:
    #             print(e)
    #     # On historical Data End
    #     def historicalDataEnd(self,reqId, start, end):
    #         print(reqId)
    #
    #     #GEt next order id we can use
    #     def nextValidId(self, nextorderId):
    #         global orderId
    #         orderId = nextorderId
    #     #list for realtime bars
    #     def realtimeBar(self, reqId, time, open_, high, low, close, volume, wap, count):
    #         super().realtimeBar(reqId, time, open_, high, low, close, volume, wap, count)
    #         try:
    #             bot.on_bar_update(reqId,time, open_, high,low,close,volume, wap, count)
    #         except Exception as e:
    #             print(e)
    #     def error(self, id, errorCode, errorString):
    #         print(errorCode)
    #         print(errorString)
    #
    # #Bar object
    # class Bar:
    #     open = 0
    #     low = 0
    #     high = 0
    #     close = 0
    #     volume = 0
    #     date = ''
    #     def __init__(self, open, low, high, close, volume, date):
    #         self.open = 0
    #         self.low = 0
    #         self.high =0
    #         self.close = 0
    #         self.volume = 0
    #         self.date = ''
    #
    #
    # #Bot Logic
    # class Bot():
    #     ib = None
    #     barsize = Bar()
    #     currentBar = Bar
    #     bars = []
    #     reqId = 0
    #     global orderId
    #     smaPeriod = 50
    #     symbol = ""
    #     initializebartime = datetime.now().astimezone(pytz.timezone("America/New_York"))
    #     def __init__(self):
    #         #Connect to IB on init
    #         self.ib = IBApi()
    #         self.ib.connect("127.0.0.1", 7496,1)
    #         ib_thread = threading.Thread(target=self.run_loop, daemon=True)
    #         ib_thread.start()
    #         time.sleep(1)
    #         currentBar = Bar()
    #         #Get the symbol
    #         self.symbol = input("Ingrese el simbolo en el que quieres invertir: ")
    #         #get the bar size
    #         self.barsize = input("enter the barsize you watn to trade in minutes: ")
    #         mintext = " min"
    #         if (int(self.barsize) > 1):
    #             mintext = " mins"
    #         queryTime = (datetime.now().astimezone(pytz.timezone("America/New_York"))-timedelta(days=1)).replace(hour=16, minute=0, second=0, microsecond=0).strftime("%Y%m%d %H:%M:%S")
    #         #Create our IB contrart Object
    #         contract = Contract()
    #         contract.symbol = self.symbol.upper()
    #         contract.secType = "STK"
    #         contract.exchange = "SMART"
    #         contract.currency = "USD"
    #         self.ib.reqIds(-1)
    #         #request market data
    #         #self.ib.reqRealTimeBars(1,contract,5,"TRADES",1,[])
    #         self.ib.reqHistoricalData(self.reqId,contract,"","2 D",str(self.barsize)+mintext,"TRADES",1,1,True,[])
    #     #listen to socket i separate thread
    #     def run_loop(self):
    #         self.ib.run()
    #
    #     #bracet ORder Setup
    #     def bracketORder(self,parentOrderId,action,quantity,profitTarget,stopLoss):
    #         contract = Contract()
    #         contract.symbol = self.symbol.upper()
    #         contract.secType = "STK"
    #         contract.exchange = "SMART"
    #         contract.currency = "USD"
    #         #Create Parent Order / Initial entry
    #         parent = Order()
    #         parent.orderId = parentOrderId
    #         parent.orderType = "MKT"
    #         parent.action = action
    #         parent.quantity = quantity
    #         parent.transmit = False
    #         #Profit Target
    #         profitTargetOrder = Order()
    #         profitTargetOrder.orderId = parent.orderId + 1
    #         profitTargetOrder.orderType = "LMT"
    #         profitTargetOrder.action = "SELL"
    #         profitTargetOrder.totalQuantity = quantity
    #         profitTargetOrder.lmtPrice = round(profitTarget,2)
    #         profitTargetOrder.transmit = False
    #         #Stop Loss
    #         stopLossOrder = Order()
    #         stopLossOrder.orderId = parent.orderId + 2
    #         stopLossOrder.orderType = "STP"
    #         stopLossOrder.action = "SELL"
    #         stopLossOrder.totalQuantity = quantity
    #         stopLossOrder.auxPrice = round(stopLoss, 2)
    #         stopLossOrder.transmit = True
    #
    #         bracketOrders = [parent,profitTargetOrder,stopLossOrder]
    #         return bracketOrders
    #
    #     #pass reatime bar data back to our bot object
    #     def on_bar_update(self, reqId, bar,realtime):
    #         #historical data to catch up
    #         if(realtime == False):
    #             self.bars.append(bar)
    #         else:
    #             bartime = datetime.strptime(bar.date,"%Y%m%d %H:%M:%S").astimezone(pytz.timezone("America/New_York"))
    #             minutes_diff = (bartime - self.initializebartime).total_seconds() / 60
    #             self.currentBar.date = bartime
    #             #ON bar close
    #             if(minutes_diff > 0 and math.floor(minutes_diff) % self.barsize == 0):
    #                 #Entry - If we have a higher high, a higher low and we cross the 50 SMA Buy
    #                 #SMA
    #                 closes = []
    #                 for bar in self.bars:
    #                     closes.append(bar.close)
    #                 self.close_array = pd.Series(np.asarray(closes))
    #                 self.sma = ta.trend.sma(self.close_array, sma_period=self.smaPeriod)
    #                 print("SMA : " + str(self.sma[len(self.sma)-1]))
    #                 #Calculate higher highs and lows
    #                 lastlow = self.bars[len(self.bars)-1].low
    #                 lasthigh = self.bars[len(self.bars)-1].high
    #                 lastclose = self.bars[len(self.bars)-1].close
    #                 lastbar = self.bars[len(self.bars)-1]
    #                 #Check Criteria
    #                 if(bar.close > lasthigh
    #                     and self.currentBar.low > lastlow
    #                     and bar.close > str(self.sma[len(self.sma)-1])
    #                     and lastclose < str(self.sma[len(self.sma)-2])):
    #                     #Bracket order @% profit target 1% stop loss
    #                     profitTarget = bar.close * 1.02
    #                     stopLoss = bar.close * 0.99
    #                     quantity = 1
    #                     bracket = self.bracketOrder(self.orderId,"EUY",quantity,profitTarget,stopLoss)
    #                     contract = Contract()
    #                     contract = Contract()
    #                     contract.symbol = self.symbol.upper()
    #                     contract.secType = "STK"
    #                     contract.exchange = "SMART"
    #                     contract.currency = "USD"
    #                     #Place Bracket Order
    #                     for o in bracket:
    #                         o.ocaGroup = "OCA " + str(orderId)
    #                         o.ocaType = 2
    #                         self.ib.placeOrder(o.orderId,o.contract,o)
    #                     orderId += 3
    #                  #bar closed append
    #                 self.currentBar.close = bar.close
    #                 if(self.currentBar.date != lastbar.date):
    #                     print("New bar!")
    #                     self.bars.append(self.currentBar)
    #                 self.currentBar.open = bar.open
    #         #Build reatime bar
    #         if(self.currentBar.open == 0):
    #             self.currentBar.open = bar.open
    #         if(self.currentBar.high == 0 or bar.high > self.currentBar.high):
    #             self.currentBar.high = bar.high
    #         if (self.currentBar.low == 0 or bar.low > self.currentBar.low):
    #             self.currentBar.low = bar.low
    #
    # #Start bot
    # if __name__ == "__main__":
    #     bot = Bot()
    # ------------------------------------------BOT FUNCIONAL----------------------------------------------------------------