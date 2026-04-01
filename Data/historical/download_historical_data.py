"""
download_data.py
Script listo para ejecutar desde la terminal.
"""
from Data.historical.Datapipeline import DataPipeline
from config import settings as IBKR_SETTINGS

def main():
    print("\n╔══════════════════════════════════════╗")
    print("║   OHLCV Data Downloader — AutoTrader  ║")
    print("╚══════════════════════════════════════╝\n")

    # ── Opción A: Pipeline completo (recomendado para entrenar) ───────────────
    pipeline = DataPipeline(source=IBKR_SETTINGS.SOURCE)

    for symbol in IBKR_SETTINGS.SYMBOLS:
        try:
            train_df, test_df = pipeline.run(
                symbol   = symbol,
                interval = IBKR_SETTINGS.INTERVAL,
                start    = IBKR_SETTINGS.START,
            )
            print(f"\n✓ {symbol} listo — Train: {len(train_df):,}  Test: {len(test_df):,}\n")
        except Exception as e:
            print(f"\n✗ Error en {symbol}: {e}\n")

    # ── Opción B: Descarga simple sin features ────────────────────────────────
    # dm = DataManager()
    # df = dm.get("AAPL", interval="1d", start="2018-01-01")
    # print(df.tail())

    # ── Opción C: Multi-símbolo a la vez ──────────────────────────────────────
    # dm = DataManager()
    # dfs = dm.download_many(["AAPL", "MSFT", "TSLA"], interval="1d")

    print("\n✓ Descarga completada. Datos en Data/raw")
    print("  Siguiente paso: python train_model.py\n")


if __name__ == "__main__":
    main()