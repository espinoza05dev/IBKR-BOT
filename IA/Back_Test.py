"""
backtest.py
Script de backtesting y evaluación completa del modelo.

Modos:
    single      → Un backtest sobre el período de test
    walkforward → Validación walk-forward en N ventanas
    compare     → Backtestea múltiples símbolos y compara

Ejecución:
    python backtest.py
"""

import sys
from pathlib import Path

import pandas as pd

from Data.historical.DataDownloader import DataManager
from Data.historical.DataPipeline   import DataPipeline
from IA.backtest.BacktestEngine     import BacktestEngine, WalkForwardEngine
from IA.backtest.BacktestMetrics    import BacktestMetrics
from IA.backtest.BacktestReport     import BacktestReport


# ╔════════════════════════════════════════════════════════╗
# ║           CONFIGURACIÓN — edita solo aquí              ║
# ╚════════════════════════════════════════════════════════╝

SYMBOL          = "AAPL"       # Símbolo del modelo a evaluar
INTERVAL        = "1h"
SOURCE          = "yfinance"
START_TRAIN     = "2021-01-01"  # Inicio del dataset completo
START_TEST      = "2023-01-01"  # Los datos de TEST deben ser POSTERIORES al entrenamiento
END_TEST        = None          # None = hasta hoy

INITIAL_BALANCE = 10_000.0
COMMISSION      = 0.001        # 0.1% por operación

MODE = "single"   # "single" | "walkforward" | "compare"

# Walk-forward (solo Modo walkforward)
WF_N_WINDOWS          = 4
WF_TIMESTEPS_WINDOW   = 300_000   # Steps por ventana (más rápido que entrenamiento completo)

# Multi-símbolo (solo Modo compare)
COMPARE_SYMBOLS = ["AAPL", "MSFT", "NVDA"]

OPEN_REPORT_IN_BROWSER = True   # Abrir HTML al terminar


# ═════════════════════════════════════════════════════════════════════════════


def modo_single():
    """
    Backtest sobre datos de test (posteriores al entrenamiento).
    Es la validación más importante: el modelo nunca vio estos datos.
    """
    print("\n▶  Modo: Backtest único")

    # Cargar datos de test (posteriores al entrenamiento)
    dm = DataManager()
    try:
        df_raw = dm.load(SYMBOL, INTERVAL)
    except FileNotFoundError:
        print(f"[Main] Descargando {SYMBOL}...")
        df_raw = dm.download(SYMBOL, INTERVAL, START_TRAIN, source=SOURCE)

    # Filtrar solo el período de test
    if hasattr(df_raw.index, "tz") and df_raw.index.tz is None:
        df_raw.index = df_raw.index.tz_localize("UTC")

    start_ts = pd.Timestamp(START_TEST, tz="UTC")
    df_test  = df_raw[df_raw.index >= start_ts]
    if END_TEST:
        df_test = df_test[df_test.index <= pd.Timestamp(END_TEST, tz="UTC")]

    if len(df_test) < 100:
        print(f"[Main] Solo {len(df_test)} barras en el período de test. "
              f"Ajusta START_TEST a una fecha más temprana.")
        sys.exit(1)

    print(f"[Main] Período de test: {df_test.index[0].date()} → {df_test.index[-1].date()}  "
          f"({len(df_test):,} barras)")

    # Ejecutar backtest
    engine = BacktestEngine(
        symbol          = SYMBOL,
        initial_balance = INITIAL_BALANCE,
        commission      = COMMISSION,
    )
    result = engine.run(df_test, interval=INTERVAL)

    # Métricas completas
    metrics = BacktestMetrics(result).compute()
    _print_full_metrics(metrics)

    # Reporte HTML
    report = BacktestReport(result, metrics)
    if OPEN_REPORT_IN_BROWSER:
        report.show()
    else:
        path = report.save()
        print(f"\n[Main] Reporte HTML → {path}")

    return result, metrics


def modo_walkforward():
    """
    Validación walk-forward: la forma más rigurosa de validar un modelo.
    Entrena en ventanas deslizantes y prueba siempre en datos no vistos.
    Elimina completamente el sesgo de look-ahead.
    """
    print("\n▶  Modo: Walk-Forward Validation")
    print(f"   Ventanas: {WF_N_WINDOWS}  |  Steps/ventana: {WF_TIMESTEPS_WINDOW:,}")

    # Dataset completo
    pipeline          = DataPipeline(source=SOURCE)
    train_df, test_df = pipeline.run(SYMBOL, INTERVAL, START_TRAIN)
    df_full = pd.concat([train_df, test_df])

    # Walk-forward
    wf = WalkForwardEngine(
        symbol    = SYMBOL,
        n_windows = WF_N_WINDOWS,
        train_pct = 0.70,
        test_pct  = 0.15,
    )
    results = wf.run(df_full, timesteps_per_window=WF_TIMESTEPS_WINDOW, features_ready=True)

    # Métricas agregadas
    agg = wf.aggregate_metrics(results)

    print(f"\n{'═'*55}")
    print("  Walk-Forward — Métricas agregadas (mean ± std)")
    print(f"{'═'*55}")
    for metric, stats in agg.items():
        print(
            f"  {metric:30s}: "
            f"{stats['mean']:+7.3f} ± {stats['std']:.3f}  "
            f"[{stats['min']:+.3f}, {stats['max']:+.3f}]"
        )

    # Reporte de la última ventana
    if results:
        last_metrics = BacktestMetrics(results[-1]).compute()
        report = BacktestReport(results[-1], last_metrics)
        if OPEN_REPORT_IN_BROWSER:
            report.show()
        else:
            report.save()

    return results, agg


def modo_compare():
    """
    Backtest en múltiples símbolos y compara los resultados.
    Útil para elegir en qué activos el modelo funciona mejor.
    """
    print(f"\n▶  Modo: Comparación multi-símbolo  {COMPARE_SYMBOLS}")

    dm       = DataManager()
    summary  = []

    for sym in COMPARE_SYMBOLS:
        print(f"\n[Main] ── {sym} ──────────────────────────────")
        try:
            df_raw = dm.load(sym, INTERVAL)
        except FileNotFoundError:
            df_raw = dm.download(sym, INTERVAL, START_TRAIN, source=SOURCE)

        # Período de test
        if df_raw.index.tz is None:
            df_raw.index = df_raw.index.tz_localize("UTC")
        df_test = df_raw[df_raw.index >= pd.Timestamp(START_TEST, tz="UTC")]

        try:
            engine  = BacktestEngine(sym, INITIAL_BALANCE, COMMISSION)
            result  = engine.run(df_test, interval=INTERVAL, verbose=False)
            metrics = BacktestMetrics(result).compute()
            summary.append({
                "symbol":      sym,
                "return_pct":  metrics["total_return_pct"],
                "bh_pct":      metrics["benchmark_return_pct"],
                "alpha_pct":   metrics["alpha_pct"],
                "sharpe":      metrics["sharpe_ratio"],
                "max_dd":      metrics["max_drawdown_pct"],
                "win_rate":    metrics["win_rate"],
                "n_trades":    metrics["n_trades"],
                "approved":    metrics["approved_for_live"],
            })
        except Exception as e:
            print(f"  Error en {sym}: {e}")

    # Tabla comparativa
    if summary:
        print(f"\n{'═'*80}")
        print(f"  {'SÍMBOLO':<8} {'RETORNO':>8} {'B&H':>7} {'ALPHA':>7} "
              f"{'SHARPE':>8} {'MAX DD':>8} {'WIN%':>7} {'TRADES':>7} {'OK'}")
        print(f"{'─'*80}")
        for s in sorted(summary, key=lambda x: x["return_pct"], reverse=True):
            ok = "✓" if s["approved"] else "✗"
            print(
                f"  {s['symbol']:<8} "
                f"{s['return_pct']:>+7.2f}% "
                f"{s['bh_pct']:>+6.2f}% "
                f"{s['alpha_pct']:>+6.2f}% "
                f"{s['sharpe']:>8.3f} "
                f"{s['max_dd']:>7.2f}% "
                f"{s['win_rate']:>6.1%} "
                f"{s['n_trades']:>7}   {ok}"
            )
        print(f"{'═'*80}\n")

    return summary


def _print_full_metrics(m: dict):
    approved = m.get("approved_for_live", False)
    status   = "✓ APROBADO" if approved else "✗ NO APROBADO"
    color    = ""

    print(f"\n{'═'*55}")
    print(f"  RESULTADO FINAL: {status}")
    print(f"{'═'*55}")

    sections = [
        ("Retorno", [
            ("Retorno total",         f"{m.get('total_return_pct',0):+.2f}%"),
            ("Buy & Hold",            f"{m.get('benchmark_return_pct',0):+.2f}%"),
            ("Alpha",                 f"{m.get('alpha_pct',0):+.2f}%"),
            ("CAGR",                  f"{m.get('cagr_pct',0):+.2f}%"),
        ]),
        ("Riesgo", [
            ("Volatilidad anual",     f"{m.get('volatility_annual_pct',0):.2f}%"),
            ("Max drawdown",          f"{m.get('max_drawdown_pct',0):.2f}%"),
            ("DD promedio",           f"{m.get('avg_drawdown_pct',0):.2f}%"),
            ("VaR 95% diario",        f"{m.get('var_95_pct',0):.3f}%"),
            ("CVaR 95% diario",       f"{m.get('cvar_95_pct',0):.3f}%"),
        ]),
        ("Ratios", [
            ("Sharpe ratio",          f"{m.get('sharpe_ratio',0):.3f}"),
            ("Sortino ratio",         f"{m.get('sortino_ratio',0):.3f}"),
            ("Calmar ratio",          f"{m.get('calmar_ratio',0):.3f}"),
            ("Omega ratio",           f"{m.get('omega_ratio',0):.3f}"),
            ("Beta vs B&H",           f"{m.get('beta',0):.3f}"),
        ]),
        ("Trades", [
            ("Total trades",          str(m.get("n_trades",0))),
            ("Ganadores / Perdedores",f"{m.get('n_winners',0)} / {m.get('n_losers',0)}"),
            ("Win rate",              f"{m.get('win_rate',0):.1%}"),
            ("Profit factor",         f"{m.get('profit_factor',0):.2f}"),
            ("Payoff ratio",          f"{m.get('payoff_ratio',0):.2f}"),
            ("Ganancia media",        f"${m.get('avg_win',0):.2f}"),
            ("Pérdida media",         f"${m.get('avg_loss',0):.2f}"),
            ("Expectancy",            f"${m.get('expectancy',0):+.2f}"),
            ("Comisiones totales",    f"${m.get('total_commissions',0):.2f}"),
            ("Duración media",        f"{m.get('avg_duration_bars',0):.1f} barras"),
        ]),
    ]

    for title, rows in sections:
        print(f"\n  {title}:")
        for label, value in rows:
            print(f"    {label:<28}: {value}")

    print()
    if approved:
        print("  ✓ Próximos pasos:")
        print("    1. Conecta TWS en modo Paper Trading (puerto 7497)")
        print("    2. python 'IBKR Bot.py'")
        print("    3. Monitorea mínimo 2 semanas antes de dinero real")
    else:
        issues = []
        if m.get("win_rate",0)        < 0.50: issues.append("Win rate < 50%")
        if m.get("sharpe_ratio",0)    < 0.50: issues.append("Sharpe < 0.5")
        if m.get("sortino_ratio",0)   < 0.75: issues.append("Sortino < 0.75")
        if m.get("max_drawdown_pct",0)> 20.0: issues.append("Max DD > 20%")
        if m.get("profit_factor",0)   < 1.20: issues.append("PF < 1.2")
        if m.get("alpha_pct",0)       < 0.0:  issues.append("Alpha negativo")
        print(f"  ✗ Razones: {' | '.join(issues)}")
        print("  → Reentrena con más timesteps o más datos históricos")
    print()


if __name__ == "__main__":
    modos = {
        "single":      modo_single,
        "walkforward": modo_walkforward,
        "compare":     modo_compare,
    }
    if MODE not in modos:
        print(f"ERROR: MODE='{MODE}' no válido. Usa: {list(modos)}")
        sys.exit(1)

    print("\n╔═══════════════════════════════════════════════╗")
    print("║       Backtester — AutoTrader IA              ║")
    print("╚═══════════════════════════════════════════════╝")
    print(f"  Símbolo : {SYMBOL}  |  Intervalo: {INTERVAL}")
    print(f"  Test    : {START_TEST} → {END_TEST or 'hoy'}")
    print(f"  Modo    : {MODE}\n")

    modos[MODE]()