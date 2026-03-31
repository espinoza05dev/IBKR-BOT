"""
FeatureEngineering.py
Transforma datos OHLCV crudos en features tecnicas normalizadas
listas para ser consumidas por el agente de RL y la estrategia IA_BackTests.
"""
import pandas as pd
import ta


class FeatureEngineer:
    """
    Genera 18 features tecnicas a partir de un DataFrame OHLCV.

    Uso:
        fe = FeatureEngineer()
        df_features = fe.transform(df_ohlcv)
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recibe DataFrame con columnas: open, high, low, close, volume.
        Devuelve el mismo DataFrame enriquecido con features normalizadas.
        """
        df = df.copy()
        self._validate(df)

        df = self._add_returns(df)
        df = self._add_trend(df)
        df = self._add_momentum(df)
        df = self._add_volatility(df)
        df = self._add_volume(df)
        df = self._normalize(df)
        df = df.dropna().reset_index(drop=True)

        return df

    # ── Features ──────────────────────────────────────────────────────────────

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["returns_1"]  = df["close"].pct_change(1)
        df["returns_5"]  = df["close"].pct_change(5)
        df["returns_10"] = df["close"].pct_change(10)
        return df

    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        # SMAs
        df["sma20"] = ta.trend.sma_indicator(df["close"], window=20)
        df["sma50"] = ta.trend.sma_indicator(df["close"], window=50)

        # EMAs para MACD
        df["ema12"] = ta.trend.ema_indicator(df["close"], window=12)
        df["ema26"] = ta.trend.ema_indicator(df["close"], window=26)

        # MACD
        macd_obj         = ta.trend.MACD(df["close"])
        df["macd"]       = macd_obj.macd()
        df["macd_signal"] = macd_obj.macd_signal()

        # ADX
        adx_obj  = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
        df["adx"] = adx_obj.adx()

        # Distancias relativas al precio (para normalizar)
        df["sma20_dist"] = (df["close"] - df["sma20"]) / df["sma20"]
        df["sma50_dist"] = (df["close"] - df["sma50"]) / df["sma50"]
        df["ema12_dist"] = (df["close"] - df["ema12"]) / df["ema12"]
        df["ema26_dist"] = (df["close"] - df["ema26"]) / df["ema26"]
        return df

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()

        stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        bb = ta.volatility.BollingerBands(df["close"])
        df["bb_upper_dist"] = (bb.bollinger_hband() - df["close"]) / df["close"]
        df["bb_lower_dist"] = (df["close"] - bb.bollinger_lband()) / df["close"]

        atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"])
        df["atr_norm"] = atr.average_true_range() / df["close"]
        return df

    def _add_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        vol_ma = df["volume"].rolling(20).mean().replace(0, 1)
        df["volume_norm"] = df["volume"] / vol_ma
        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza close entre 0 y 1 dentro de ventanas rodantes de 50 velas."""
        roll_min = df["close"].rolling(50).min()
        roll_max = df["close"].rolling(50).max()
        rng = (roll_max - roll_min).replace(0, 1)
        df["close_norm"] = (df["close"] - roll_min) / rng

        # RSI y Stoch → 0-1
        df["rsi"]     = df["rsi"] / 100.0
        df["stoch_k"] = df["stoch_k"] / 100.0
        df["stoch_d"] = df["stoch_d"] / 100.0
        df["adx"]     = df["adx"] / 100.0
        return df

    # ── Validacion ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(df: pd.DataFrame):
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"[FeatureEngineer] Faltan columnas: {missing}")