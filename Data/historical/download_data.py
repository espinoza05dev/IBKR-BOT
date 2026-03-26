"""
download_data.py
Script listo para ejecutar desde la terminal.

Uso:
    python Data/historical/download_data.py
"""

from Data.historical.Datadownloader  import DataManager
from Data.historical.Datapipeline   import DataPipeline


def main():
    print("\n╔══════════════════════════════════════╗")
    print("║   OHLCV Data Downloader — AutoTrader  ║")
    print("╚══════════════════════════════════════╝\n")

    # ── Configuración ─────────────────────────────────────────────────────────
    SYMBOLS  = ["AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","BRK.B","AVGO","LLY","JPM","V","UNH","MA","XOM","WMT","COST","PG","JNJ","HD","ORCL","NFLX","ABBV","BAC","KO","CRM","ADBE","CVX","PEP","ACN","TMO","AMD","LIN","MCD","CSCO","ABT","TMUS","QCOM","INTU","INTC","DHR","GE","AMGN","CAT","PGR","TXN","IBM","VZ","AMAT","PM","UNP","NOW","ISRG","GS","HON","BKNG","SPGI","AXP","MS","SYK","RTX","LOW","C","NEE","PFE","TJX","VRTX","BLK","BSX","LRCX","ETN","GEV","MMC","ADI","MU","PANW","COP","REGN","MDT","PLTR","SCHW","CB","BMY","LMT","DE","CI","PLD","ADP","GILD","MDLZ","UBER","FI","APH","SNE","T","BA","SYY","ZTS","MO","EL","SNPS","CDNS","KLAC","VRSK","ROP","CPRT","MAR","KMB","AON","BKR","CEG","A","NXPI","ADSK","IDXX","PAYX","ORLY","CTAS","ANSS","CHTR","MCHP","CDW","TEAM","WDAY","EXPE","DDOG","LULU","DLTR","MNST","KDP","ROST","EBAY","FAST","ODFL","PCAR","AZN","BIIB","CSX","CP","EPAM","FSLR","FTNT","GFS","IDXX","ILMN","MDB","ON","TTWO","WBD","ZS","ABNB","AKAM","ALGN","ARE","ALB","ALL","ALLE","LNT","AMCR","AEE","AAL","AEP","AMT","AWK","AMP","AME","APA","APTV","AJG","AIZ","ATO","AVB","AVY","BBY","BIO","BXP","CHRW","COF","KCC","CF","CRL","CHD","CHUB","CLX","CMA","CAG","ED","GLW","CTVA","CSGP","CTRA","CMI","DHI","DRI","DVA","XRAY","DVN","DXCM","FANG","DLR","DFS","DIS","DG","DLTR","D","DPZ","DOV","DOW","DTE","DUK","DD","EMN","EMR","EOG","EFX","EQIX","EQT","ETR","ECL","EW","EA","EXC","EXPD","EXR","F","FDS","FMC","FE","FIS","FLT","FITB","FSLR","BEN","FCX","GRMN","IT","GD","GIS","GPC","GPN","GWW","HAL","HBI","HAS","HCA","HSIC","HSY","HES","HPE","HILTON","HMC","HRL","HST","HPQ","HUM","HBAN","IEX","IDXX","ITW","ILMN","INCY","IR","INTC","ICE","IP","IPG","IFF","IVZ","IRM","JBHT","JKHY","J","JCI","K","KEYS","KIM","KMI","KHC","LHX","LH","LRCX","LEG","LEN","LNC","LYB","MTB","MRO","MPC","MKTX","MLM","MAS","MTCH","MKC","MCK","MDT","MRK","MET","MTD","MGM","KUA","NDAQ","NTAP","NWL","NEM","NWSA","NWS","NEE","NLSN","NKE","NI","NSC","NTRS","NOC","NLY","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","OMC","OKE","OTIS","PCG","PKG","PH","PAYX","PAYC","PYPL","PNR","PEP","PKI","PRU","PEG","PSA","PHM","PVH","PWR","DGX","RJF","O","REG","RHI","ROK","ROL","RSG","SPG","SLB","STX","SEE","SRE","SHW","SPG","SWKS","SNA","SO","LUV","SWK","SBUX","STT","STE","STK","TEL","FTV","TGT","TDY","TFX","TER","TSLA","TXN","TXT","TMO","TRV","TFC","TYL","TSN","USB","UDR","ULTA","URI","VFC","VLO","VTR","VRSN","VRTX","VFC","VMC","VNO","VRSK","VTRS","VRTX","VZ","WBA","WBD","WM","WAT","WEC","WFC","WELL","WDC","WRK","WY","WHR","WMB","WLTW","WYNN","XEL","XYL","YUM","ZBH","ZION","ZTS"]   # Cambia según tus necesidades
    INTERVAL = "1h"                        # "1m" "5m" "15m" "1h" "1d"
    START    = "2018-01-01"                # Fecha de inicio
    SOURCE   = "yfinance"                  # "yfinance" | "ibkr" | "av" | "csv"

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