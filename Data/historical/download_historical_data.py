"""
download_data.py
Script listo para ejecutar desde la terminal.
"""
from Data.historical.Datapipeline   import DataPipeline

def main():
    print("\n╔══════════════════════════════════════╗")
    print("║   OHLCV Data Downloader — AutoTrader  ║")
    print("╚══════════════════════════════════════╝\n")

    # ── Configuración ─────────────────────────────────────────────────────────
    SYMBOLS  = ["AAL"]
    INTERVAL = "1h"                        # "1m" "5m" "15m" "1h" "1d"
    START    = "2010-01-01"                # Fecha de inicio
    SOURCE   = "ibkr"                  # "yfinance" | "ibkr" | "av" | "csv"

    # ── Opción A: Pipeline completo (recomendado para entrenar) ───────────────
    pipeline = DataPipeline(source=SOURCE)

    for symbol in SYMBOLS:
        try:
            train_df, test_df = pipeline.run(
                symbol   = symbol,
                interval = INTERVAL,
                start    = START,
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

    print("\n✓ Descarga completada. Datos en IA/Data/historical/")
    print("  Siguiente paso: python train_model.py\n")


if __name__ == "__main__":
    main()