from __future__ import annotations
"""
DataPipeline.py
Orquesta el flujo completo:
    Descargar → Validar → Limpiar → Enriquecer → Partir → Guardar

Uso:
    pipeline = DataPipeline()
    train_df, test_df = pipeline.run("AAPL", interval="1h", start="2021-01-01")
"""
from pathlib import Path
import pandas as pd

from Data.historical.Datadownloader import DataManager
from IA.FeatureEngineering import FeatureEngineer


class DataPipeline:
    """
    Pipeline completo desde símbolo hasta DataFrames listos para ModelTrainer.

    run() devuelve:
        (train_df, test_df)  →  split 80/20 cronológico
    """

    def __init__(
        self,
        source:     str   = "yfinance",
        test_split: float = 0.20,
        av_api_key: str   = "",
    ):
        self.dm         = DataManager(av_api_key=av_api_key)
        self.fe         = FeatureEngineer()
        self.source     = source
        self.test_split = test_split

    def run(
        self,
        symbol:   str,
        interval: str = "1h",
        start:    str = "2005-01-01",
        end:      str | None = None,
        force_download: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        print(f"\n{'='*55}")
        print(f"  DataPipeline  |  {symbol}  |  {interval}")
        print(f"{'='*55}")

        # 1. Descargar / cargar
        if force_download:
            raw = self.dm.download(symbol, interval, start, end, self.source)
        else:
            raw = self.dm.get(symbol, interval, start, end, self.source)

        if raw.empty:
            raise RuntimeError(f"No se obtuvieron datos para {symbol}")

        # 2. Validar
        report = self.dm.validate(raw)
        if not report["ready"]:
            print("[Pipeline] ADVERTENCIA: datos con problemas de calidad")

        # 3. Feature engineering
        print("\n[Pipeline] Calculando features técnicas...")
        enriched = self.fe.transform(raw)
        print(f"[Pipeline] Features calculadas: {enriched.shape[1]} columnas, {len(enriched):,} filas")

        # 4. Partir train/test (cronológico, sin shuffle)
        split_idx = int(len(enriched) * (1 - self.test_split))
        train_df  = enriched.iloc[:split_idx].copy()
        test_df   = enriched.iloc[split_idx:].copy()

        print(
            f"\n[Pipeline] Split completado:\n"
            f"  Train : {len(train_df):,} velas  "
            f"({train_df.index[0]} → {train_df.index[-1]})\n"
            f"  Test  : {len(test_df):,} velas  "
            f"({test_df.index[0]} → {test_df.index[-1]})"
        )

        # 5. Guardar splits
        self._save_splits(train_df, test_df, symbol, interval)

        return train_df, test_df

    def run_many(
        self,
        symbols:  list[str],
        interval: str = "1d",
        start:    str = "2005-01-01",
    ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
        """Corre el pipeline para múltiples símbolos."""
        results = {}
        for sym in symbols:
            try:
                results[sym] = self.run(sym, interval, start)
            except Exception as e:
                print(f"[Pipeline] Error en {sym}: {e}")
        return results

    @staticmethod
    def _save_splits(train: pd.DataFrame, test: pd.DataFrame, symbol: str, interval: str):
        folder = Path(f"IA/Data/historical") / symbol.upper()
        folder.mkdir(parents=True, exist_ok=True)
        train.to_csv(folder / f"{symbol.upper()}_{interval}_train.csv")
        test.to_csv(folder  / f"{symbol.upper()}_{interval}_test.csv")
        print(f"[Pipeline] Splits guardados en {folder}/")
#
# if __name__ == "__main__":
#     pipeline = DataPipeline()
#     train_df, test_df = pipeline.run("MSFT", interval="1h", start="2014-01-01")