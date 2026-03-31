from __future__ import annotations
"""
DataDownloader.py
Descarga y normaliza datos OHLCV históricos desde múltiples fuentes:
    - yfinance     (Yahoo Finance  — gratis, sin cuenta)
    - IBKR         (Interactive Brokers — requiere TWS/Gateway activo)
    - Alpha Vantage (requiere API key gratuita)
    - CSV local     (importar datos propios)

Salida unificada: DataFrame con columnas → open, high, low, close, volume
guardado en Data/historical/<SYMBOL>/<SYMBOL>_<interval>.csv
"""

import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd

DATA_DIR = Path(f"C:\\Users\\artur\\Programming\\PycharmProjects\\IBKR_AI_BOT\\Data\\raw")


# ══════════════════════════════════════════════════════════════════════════════
# Clase base
# ══════════════════════════════════════════════════════════════════════════════

class OHLCVDownloader:
    """
    Interfaz unificada.  Todas las fuentes heredan de aquí y
    devuelven un DataFrame con el mismo esquema.

    Columnas garantizadas:
        datetime (index, tz-aware UTC)
        open, high, low, close  → float64
        volume                  → float64
    """

    REQUIRED_COLS = {"open", "high", "low", "close", "volume"}

    def download(
        self,
        symbol:   str,
        interval: str  = "1h",
        start:    str  = "2020-01-01",
        end:      str  | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    # ── Utilidades compartidas ────────────────────────────────────────────────

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estandariza columnas, elimina nulos/duplicados y
        garantiza que el index sea DatetimeIndex UTC.
        """
        # Minúsculas
        df.columns = [c.lower().strip() for c in df.columns]

        # Renombrar alias comunes
        rename = {
            "adj close": "close",
            "adj_close": "close",
            "vol":       "volume",
            "date":      "datetime",
            "timestamp": "datetime",
        }
        df = df.rename(columns=rename)

        # Si el index es la fecha, conviértelo en columna
        if df.index.name and df.index.name.lower() in {"date", "datetime", "timestamp"}:
            df = df.reset_index().rename(columns={df.index.name: "datetime"})

        # Columna datetime → index
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df = df.set_index("datetime")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        else:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

        df.index.name = "datetime"

        # Solo las columnas necesarias
        available = [c for c in self.REQUIRED_COLS if c in df.columns]
        df = df[available].copy()

        # Tipos
        for col in available:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Limpiar
        df = df.dropna(subset=["close"])
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        # Eliminar velas con valores imposibles
        df = df[(df["close"] > 0) & (df["high"] >= df["low"])]

        return df

    def _save(self, df: pd.DataFrame, symbol: str, interval: str) -> Path:
        folder = DATA_DIR / symbol.upper()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{symbol.upper()}_{interval}.csv"
        df.to_csv(path)
        print(f"[DataDownloader] Guardado → {path}  ({len(df):,} velas)")
        return path

    def _print_summary(self, df: pd.DataFrame, source: str, symbol: str):
        if df.empty:
            print(f"[{source}] Sin datos para {symbol}")
            return
        print(
            f"\n[{source}] {symbol} — {len(df):,} velas\n"
            f"  Desde : {df.index[0]}\n"
            f"  Hasta : {df.index[-1]}\n"
            f"  Close : min={df['close'].min():.2f}  "
            f"max={df['close'].max():.2f}  "
            f"último={df['close'].iloc[-1]:.2f}\n"
            f"  Nulos : {df.isnull().sum().to_dict()}\n"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fuente 1 — Yahoo Finance  (yfinance)
# ══════════════════════════════════════════════════════════════════════════════

class YFinanceDownloader(OHLCVDownloader):
    """
    Descarga gratis sin cuenta.

    Intervalos válidos:
        Intraday : 1m  2m  5m  15m  30m  60m  90m  1h
        Diario+  : 1d  5d  1wk  1mo  3mo

    Límites de historia:
        1m  → 7 días  |  5m/15m/30m → 60 días  |  1h → 730 días  |  1d → ilimitado
    """

    INTERVAL_LIMITS = {
        "1m": 7, "2m": 60, "5m": 60, "15m": 60,
        "30m": 60, "60m": 730, "1h": 730,
        "90m": 60, "1d": 9999, "5d": 9999,
        "1wk": 9999, "1mo": 9999, "3mo": 9999,
    }

    def download(
        self,
        symbol:   str,
        interval: str       = "1h",
        start:    str       = "2020-01-01",
        end:      str | None = None,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("Instala yfinance:  pip install yfinance")

        end = end or datetime.utcnow().strftime("%Y-%m-%d")
        limit_days = self.INTERVAL_LIMITS.get(interval, 9999)

        # Respetar el límite de historia para datos intraday
        start_dt  = datetime.strptime(start, "%Y-%m-%d")
        max_start = datetime.utcnow() - timedelta(days=limit_days - 1)
        if start_dt < max_start and limit_days < 9999:
            print(
                f"[YFinance] Intervalo '{interval}' tiene límite de {limit_days} días. "
                f"Ajustando start → {max_start.strftime('%Y-%m-%d')}"
            )
            start = max_start.strftime("%Y-%m-%d")

        print(f"[YFinance] Descargando {symbol} {interval}  {start} → {end} ...")
        ticker = yf.Ticker(symbol)
        raw    = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)

        if raw.empty:
            print(f"[YFinance] Sin datos para {symbol}. ¿Símbolo incorrecto?")
            return pd.DataFrame()

        df = self._normalize(raw)
        self._print_summary(df, "YFinance", symbol)
        self._save(df, symbol, interval)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# Fuente 2 — Interactive Brokers  (ibapi)
# ══════════════════════════════════════════════════════════════════════════════

class IBKRDownloader(OHLCVDownloader):
    """
    Descarga desde tu cuenta IBKR via TWS o IB Gateway (debe estar abierto).

    Requiere:
        - TWS o IB Gateway corriendo en localhost
        - ibapi instalado

    Intervalos IB → barSizeSetting:
        "1m"   → "1 min"      "5m"  → "5 mins"
        "15m"  → "15 mins"    "1h"  → "1 hour"
        "1d"   → "1 day"
    """

    IB_BAR_SIZES = {
        "1m":  "1 min",   "2m":  "2 mins",  "3m":  "3 mins",
        "5m":  "5 mins",  "10m": "10 mins", "15m": "15 mins",
        "30m": "30 mins", "1h":  "1 hour",  "2h":  "2 hours",
        "4h":  "4 hours", "1d":  "1 day",   "1w":  "1 week",
    }

    IB_DURATION = {
        "1m": "3 D",  "5m": "10 D",  "15m": "20 D",
        "30m": "1 M", "1h": "6 M",   "1d":  "5 Y",
    }

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 99):
        self.host      = host
        self.port      = port
        self.client_id = client_id
        self._bars: list = []
        self._done  = threading.Event()

    def download(
        self,
        symbol:   str,
        interval: str       = "1h",
        start:    str       = "2020-01-01",
        end:      str | None = None,
        duration: str | None = None,
    ) -> pd.DataFrame:
        try:
            from ibapi.client  import EClient
            from ibapi.wrapper import EWrapper
            from ibapi.contract import Contract
        except ImportError:
            raise ImportError("ibapi no encontrado. Instala el TWS API desde IBKR.")

        bar_size = self.IB_BAR_SIZES.get(interval)
        if not bar_size:
            raise ValueError(f"Intervalo '{interval}' no válido. Usa: {list(self.IB_BAR_SIZES)}")

        duration_str = duration or self.IB_DURATION.get(interval, "1 Y")
        self._bars   = []
        self._done.clear()

        class _IBApp(EWrapper, EClient):
            def __init__(self_ib):
                EClient.__init__(self_ib, self_ib)

            def historicalData(self_ib, reqId, bar):
                self._bars.append({
                    "datetime": bar.date,
                    "open":     bar.open,
                    "high":     bar.high,
                    "low":      bar.low,
                    "close":    bar.close,
                    "volume":   bar.volume,
                })

            def historicalDataEnd(self_ib, reqId, start_str, end_str):
                print(f"[IBKR] Descarga completada — {len(self._bars)} barras")
                self._done.set()

            def error(self_ib, reqId, code, msg, *args):
                if code not in (2104, 2106, 2158, 2119):   # Info messages
                    print(f"[IBKR Error {code}] {msg}")

        contract          = Contract()
        contract.symbol   = symbol.upper()
        contract.secType  = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        app = _IBApp()
        app.connect(self.host, self.port, self.client_id)

        t = threading.Thread(target=app.run, daemon=True)
        t.start()
        time.sleep(1)

        end_dt = end or ""
        print(f"[IBKR] Solicitando {symbol} {bar_size} duración={duration_str} ...")
        app.reqHistoricalData(
            reqId          = 1,
            contract       = contract,
            endDateTime    = end_dt,
            durationStr    = duration_str,
            barSizeSetting = bar_size,
            whatToShow     = "TRADES",
            useRTH         = 1,
            formatDate     = 1,
            keepUpToDate   = False,
            chartOptions   = [],
        )

        self._done.wait(timeout=60)
        app.disconnect()

        if not self._bars:
            print("[IBKR] Sin datos recibidos. ¿TWS está abierto y conectado?")
            return pd.DataFrame()

        df = pd.DataFrame(self._bars)
        df = self._normalize(df)
        self._print_summary(df, "IBKR", symbol)
        self._save(df, symbol, interval)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# Fuente 3 — Alpha Vantage  (API key gratuita)
# ══════════════════════════════════════════════════════════════════════════════

class AlphaVantageDownloader(OHLCVDownloader):
    """
    API key gratuita en https://www.alphavantage.co/support/#api-key
    Plan free: 25 requests/día, 5 req/min.

    Intervalos: "1m" "5m" "15m" "30m" "60m" "1d" "1w" "1mo"
    """

    BASE_URL = "https://www.alphavantage.co/query"

    AV_FUNCTION = {
        "1m": "TIME_SERIES_INTRADAY",  "5m":  "TIME_SERIES_INTRADAY",
        "15m": "TIME_SERIES_INTRADAY", "30m": "TIME_SERIES_INTRADAY",
        "60m": "TIME_SERIES_INTRADAY", "1h":  "TIME_SERIES_INTRADAY",
        "1d": "TIME_SERIES_DAILY_ADJUSTED",
        "1w": "TIME_SERIES_WEEKLY_ADJUSTED",
        "1mo":"TIME_SERIES_MONTHLY_ADJUSTED",
    }

    def __init__(self, api_key: str):
        self.api_key = api_key

    def download(
        self,
        symbol:   str,
        interval: str       = "1h",
        start:    str       = "2020-01-01",
        end:      str | None = None,
    ) -> pd.DataFrame:
        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests")

        function = self.AV_FUNCTION.get(interval)
        if not function:
            raise ValueError(f"Intervalo '{interval}' no soportado por Alpha Vantage.")

        av_interval = interval.replace("m", "min").replace("1h", "60min")
        params: dict = {
            "function":   function,
            "symbol":     symbol.upper(),
            "apikey":     self.api_key,
            "outputsize": "full",
            "datatype":   "json",
        }
        if "INTRADAY" in function:
            params["interval"] = av_interval

        print(f"[AlphaVantage] Descargando {symbol} {interval} ...")
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Detectar la clave con los datos
        ts_key = next((k for k in data if "Time Series" in k), None)
        if not ts_key:
            msg = data.get("Information") or data.get("Note") or str(data)
            raise RuntimeError(f"[AlphaVantage] Error de API: {msg}")

        records = []
        for dt_str, vals in data[ts_key].items():
            row = {"datetime": dt_str}
            for k, v in vals.items():
                # "1. open" → "open"
                clean = k.split(". ")[-1].strip().lower().replace(" ", "_")
                row[clean] = float(v)
            records.append(row)

        df = pd.DataFrame(records)
        df = self._normalize(df)

        # Filtrar por rango de fechas
        start_ts = pd.Timestamp(start, tz="UTC")
        df = df[df.index >= start_ts]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]

        self._print_summary(df, "AlphaVantage", symbol)
        self._save(df, symbol, interval)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# Fuente 4 — CSV local
# ══════════════════════════════════════════════════════════════════════════════

class CSVDownloader(OHLCVDownloader):
    """
    Importa datos propios desde CSV.
    El CSV puede tener cualquier nombre de columnas — se auto-detectan.
    """

    def download(
        self,
        symbol:   str,
        interval: str        = "1d",
        start:    str        = "2000-01-01",
        end:      str | None = None,
        filepath: str        = "",
    ) -> pd.DataFrame:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"[CSV] Archivo no encontrado: {path}")

        print(f"[CSV] Leyendo {path} ...")
        df = pd.read_csv(path)
        df = self._normalize(df)

        start_ts = pd.Timestamp(start, tz="UTC")
        df = df[df.index >= start_ts]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]

        self._print_summary(df, "CSV", symbol)
        self._save(df, symbol, interval)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# Fachada principal
# ══════════════════════════════════════════════════════════════════════════════

class DataManager:
    """
    Punto de entrada único.  Elige la fuente, descarga, valida
    y devuelve un DataFrame listo para FeatureEngineer + ModelTrainer.

    Uso rápido:
        dm = DataManager()

        # Desde Yahoo Finance (recomendado para empezar)
        df = dm.get("AAPL", interval="1h", start="2022-01-01")

        # Desde IBKR (requiere TWS abierto)
        df = dm.get("AAPL", source="ibkr", interval="1h")

        # Cargar CSV propio
        df = dm.get("AAPL", source="csv", filepath="mi_data.csv")

        # Cargar lo que ya está en disco sin descargar
        df = dm.load("AAPL", interval="1h")

        # Multi-símbolo para portafolio
        dfs = dm.download_many(["AAPL","MSFT","NVDA"], interval="1d")
    """

    SOURCES = {
        "yfinance": YFinanceDownloader,
        "ibkr":     IBKRDownloader,
        "av":       AlphaVantageDownloader,
        "csv":      CSVDownloader,
    }

    def __init__(self, av_api_key: str = "", ibkr_port: int = 7497):
        self._av_key   = av_api_key
        self._ibkr_port = ibkr_port

    # ── API pública ───────────────────────────────────────────────────────────

    def get(
        self,
        symbol:   str,
        interval: str       = "1h",
        start:    str       = "2020-01-01",
        end:      str | None = None,
        source:   str       = "yfinance",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Descarga o carga desde caché.
        Si los datos ya existen en disco y tienen menos de 24h,
        los carga directamente sin descargar.
        """
        cached = self._load_cache(symbol, interval)
        if cached is not None:
            print(f"[DataManager] Usando caché  →  {symbol} {interval}  ({len(cached):,} velas)")
            return cached

        return self.download(symbol, interval, start, end, source, **kwargs)

    def download(
        self,
        symbol:   str,
        interval: str       = "1h",
        start:    str       = "2000-01-01",
        end:      str | None = None,
        source:   str       = "yfinance",
        **kwargs,
    ) -> pd.DataFrame:
        """Descarga siempre (ignora caché)."""
        src = source.lower()
        if src not in self.SOURCES:
            raise ValueError(f"Fuente '{source}' no válida. Usa: {list(self.SOURCES)}")

        # Instanciar el downloader correcto
        if src == "av":
            if not self._av_key:
                raise ValueError("Necesitas una API key de Alpha Vantage. "
                                 "Obtén una gratis en https://www.alphavantage.co")
            dl = AlphaVantageDownloader(self._av_key)
        elif src == "ibkr":
            dl = IBKRDownloader(port=self._ibkr_port)
        else:
            dl = self.SOURCES[src]()

        return dl.download(symbol, interval, start, end, **kwargs)

    def load(self, symbol: str, interval: str = "1h") -> pd.DataFrame:
        """Carga datos desde disco sin descargar."""
        path = DATA_DIR / symbol.upper() / f"{symbol.upper()}_{interval}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No hay datos en disco para {symbol} {interval}.\n"
                f"Descarga primero con: DataManager().download('{symbol}', interval='{interval}')"
            )
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        print(f"[DataManager] Cargado desde disco → {len(df):,} velas")
        return df

    def download_many(
        self,
        symbols:  list[str],
        interval: str       = "1d",
        start:    str       = "2020-01-01",
        source:   str       = "yfinance",
    ) -> dict[str, pd.DataFrame]:
        """
        Descarga múltiples símbolos.
        Retorna dict: {"AAPL": df, "MSFT": df, ...}
        """
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.download(sym, interval, start, source=source)
                time.sleep(0.5)   # Evitar rate-limit
            except Exception as e:
                print(f"[DataManager] Error en {sym}: {e}")
        return result

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Valida la calidad del DataFrame antes de entrenar.
        Retorna un reporte con advertencias.
        """
        report = {
            "total_rows":   len(df),
            "null_pct":     df.isnull().mean().to_dict(),
            "date_range":   f"{df.index[0]} → {df.index[-1]}",
            "warnings":     [],
            "ready":        True,
        }

        if len(df) < 500:
            report["warnings"].append(
                f"Solo {len(df)} filas — se recomiendan al menos 500 para entrenar"
            )
            report["ready"] = False

        null_close = df["close"].isnull().mean()
        if null_close > 0.02:
            report["warnings"].append(f"Close tiene {null_close:.1%} nulos")
            report["ready"] = False

        # Gaps (para intraday)
        if len(df) > 1:
            diffs  = df.index.to_series().diff().dropna()
            median = diffs.median()
            gaps   = (diffs > median * 5).sum()
            if gaps > 0:
                report["warnings"].append(f"{gaps} gaps detectados en el índice temporal")

        # Precios negativos o cero
        bad_prices = ((df[["open","high","low","close"]] <= 0).any(axis=1)).sum()
        if bad_prices > 0:
            report["warnings"].append(f"{bad_prices} filas con precios ≤ 0")

        print("\n[Validación]")
        for k, v in report.items():
            print(f"  {k}: {v}")
        return report

    # ── Caché ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_cache(symbol: str, interval: str) -> Optional[pd.DataFrame]:
        path = DATA_DIR / symbol.upper() / f"{symbol.upper()}_{interval}.csv"
        if not path.exists():
            return None
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > 24:
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df