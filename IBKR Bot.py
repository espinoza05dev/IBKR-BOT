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
#------------------------------------------BOT FUNCIONAL----------------------------------------------------------------

"""
IBKR Bot.py - Punto de entrada del autotrader.
Orquesta la conexion, datos de mercado, estrategia y portfolio.
"""

import threading
import time

from ibapi.contract import Contract

from core.BrockerConnection import IBApi
from core.Portfolio import Portfolio
from core.Strategy import SMAStrategy
from Data.RealTime_Market_Data import MarketDataHandler


class Bot:
    """
    Orquestador principal.
    Inicializa y conecta todos los componentes del sistema.
    """

    HOST      = "127.0.0.1"
    PAPER_PORT = 7496
    PORT      = 7497
    CLIENT_ID = 1

    def __init__(self):
        self.order_id: int = 1
        self.symbol: str = ""
        self.barsize: int = 1

        # Conexion IB
        self.ib = IBApi()
        self.ib.set_callbacks(
            on_bar_update=self._on_bar_update,
            on_next_order_id=self._set_order_id,
        )
        self.ib.connect(self.HOST, self.PAPER_PORT, self.CLIENT_ID)

        ib_thread = threading.Thread(target=self.ib.run, daemon=True)
        ib_thread.start()
        time.sleep(1)

        # Input del usuario
        self.symbol  = input("Ingrese el simbolo en el que quieres invertir: ").strip().upper()
        self.barsize = int(input("Ingrese el tamano de vela en minutos: ").strip())

        # Componentes
        self.strategy    = SMAStrategy(sma_period=50, profit_pct=0.02, stop_pct=0.01)
        self.market_data = MarketDataHandler(barsize=self.barsize, strategy=self.strategy)
        self.market_data.set_signal_callback(self._on_trade_signal)
        self.portfolio   = Portfolio(ib=self.ib, symbol=self.symbol)

        # Solicitar datos historicos + streaming
        self.ib.reqIds(-1)
        min_text = " mins" if self.barsize > 1 else " min"
        self.ib.reqHistoricalData(
            reqId=0,
            contract=self._create_contract(),
            endDateTime="",
            durationStr="2 D",
            barSizeSetting=f"{self.barsize}{min_text}",
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=True,
            chartOptions=[],
        )
        print(f"[Bot] Autotrader iniciado | {self.symbol} | Velas de {self.barsize} min")

    # Helpers

    def _create_contract(self) -> Contract:
        c = Contract()
        c.symbol   = self.symbol
        c.secType  = "STK"
        c.exchange = "SMART"
        c.currency = "USD"
        return c

    def _set_order_id(self, order_id: int):
        self.order_id = order_id
        print(f"[Bot] Proximo order ID: {order_id}")

    # Callbacks de datos

    def _on_bar_update(self, reqId, bar, realtime: bool):
        self.market_data.on_bar_update(reqId, bar, realtime)

    # Logica de trading

    def _on_trade_signal(self, close_price: float):
        """Llamado por MarketDataHandler cuando la estrategia detecta una entrada."""
        profit_target, stop_loss = self.strategy.calculate_targets(close_price)

        print(
            f"[Bot] Orden bracket | Entry={close_price:.2f} "
            f"TP={profit_target:.2f} SL={stop_loss:.2f}"
        )
        self.order_id = self.portfolio.place_bracket_order(
            order_id=self.order_id,
            action="BUY",        # BUG FIX: el original tenia "EUY"
            quantity=1,
            profit_target=profit_target,
            stop_loss=stop_loss,
        )


# Entry point

if __name__ == "__main__":
    bot = Bot()
