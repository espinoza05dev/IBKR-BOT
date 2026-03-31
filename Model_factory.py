from __future__ import annotations
"""
model_factory.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pipeline autónomo de producción de modelos aptos para trading.

OBJETIVO: dado un símbolo, entrenar N modelos con configuraciones
variadas hasta conseguir TARGET_APPROVED modelos que pasen el backtest.

Flujo por intento:
    1. Samplear configuración PPO única (seed + hiperparámetros)
    2. Entrenar con esa config
    3. Backtest sobre período de test
    4. Si pasa criterios → guardar en IA/models/<SYMBOL>/approved/model_N/
    5. Repetir hasta tener TARGET_APPROVED modelos o agotar MAX_ATTEMPTS

Al final genera:
    - IA/models/<SYMBOL>/approved/      ← modelos aptos (zip + vec_normalize)
    - IA/models/<SYMBOL>/factory_report.json ← métricas de todos los intentos
    - IA/models/<SYMBOL>/factory_report.html ← reporte visual interactivo

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIGURACIÓN — edita solo esta sección
"""
# ╔════════════════════════════════════════════════════════╗
# ║  CONFIGURA AQUÍ ANTES DE EJECUTAR                      ║
# ╚════════════════════════════════════════════════════════╝

SYMBOL          = "AAPL"         # Símbolo objetivo
INTERVAL        = "1h"          # "1h" | "1d" | "30m"
SOURCE          = "yfinance"

START_TRAIN     = "2013-01-01"  # Inicio del historial de entrenamiento
START_TEST      = "2023-06-01"  # Los datos de test NUNCA se ven durante entrenamiento
END_TEST        = None          # None = hasta hoy

INITIAL_BALANCE = 10_000.0
COMMISSION      = 0.001         # 0.1% por operación (IBKR)

# ── Metas ────────────────────────────────────────────────
TARGET_APPROVED = 20            # Cuántos modelos aptos quiero
MAX_ATTEMPTS    = 200           # Intentos máximos antes de rendirse
STOP_ON_TARGET  = True          # True = parar al llegar al objetivo
                                # False = seguir hasta MAX_ATTEMPTS

# ── Entrenamiento ────────────────────────────────────────
TIMESTEPS_PER_MODEL = 2_000_000 # Steps por intento (recomendado ≥ 1.5M)
                                 # GPU ~15min | CPU ~60min por modelo

# ── Criterios de aprobación ──────────────────────────────
# Ajusta según la volatilidad del símbolo:
#   Símbolos volátiles (AAL, AMC, GME): sé más permisivo
#   Símbolos estables (AAPL, MSFT, SPY): puedes ser más exigente
APPROVAL = {
    "win_rate":        0.48,   # ≥ 48%  (baja de 50% para acciones volátiles)
    "sharpe_ratio":    0.35,   # ≥ 0.35 (Sharpe de mercado ~ 0.4-0.6)
    "sortino_ratio":   0.50,   # ≥ 0.50
    "max_drawdown_pct": 25.0,  # ≤ 25%  (AAL puede caer fuerte)
    "profit_factor":   1.10,   # ≥ 1.10 (cada $1 perdido se ganan $1.10)
    "alpha_pct":       -5.0,   # ≥ -5%  (permisivo: que no sea catastrófico)
    "n_trades":        8,      # ≥ 8 trades para tener muestra estadística
}

# ── Espacio de búsqueda de hiperparámetros ───────────────
# Cada intento samplea aleatoriamente de estos rangos.
# Cuanta más variedad, más probabilidad de encontrar buenas configs.
SEARCH_SPACE = {
    "learning_rate":  [1e-4, 3e-4, 5e-4, 7e-4, 1e-3],
    "n_steps":        [2048, 4096, 8192],
    "batch_size":     [2048, 4096, 8192],
    "gamma":          [0.95, 0.97, 0.99, 0.995],
    "gae_lambda":     [0.90, 0.92, 0.95, 0.98],
    "ent_coef":       [0.001, 0.005, 0.01, 0.02, 0.05],
    "clip_range":     [0.1, 0.15, 0.2, 0.25, 0.3],
    "net_arch_key":   ["small", "medium", "large", "deep"],
    "n_envs":         [2, 4, 6, 8],
}

NET_ARCHS = {
    "small":  [64, 64],
    "medium": [128, 128],
    "large":  [256, 256],
    "deep":   [128, 128, 64],
}

# ══════════════════════════════════════════════════════════════════════════════
# Código del factory (no modificar salvo que sepas lo que haces)
# ══════════════════════════════════════════════════════════════════════════════

import json
import random
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
import pandas as pd
from Data.historical.Datadownloader import DataManager
from IA.ModelTrainer                import ModelTrainer, PPOConfig, detect_device
from IA.backtest.Backtestengine     import BacktestEngine
from IA.backtest.Backtestmetrics    import BacktestMetrics


# Directorios de salida
APPROVED_DIR  = Path(f"IA/models/{SYMBOL}/approved")
FACTORY_LOG   = Path(f"IA/models/{SYMBOL}/factory_report.json")
APPROVED_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades
# ══════════════════════════════════════════════════════════════════════════════

def sample_config(attempt: int) -> tuple[PPOConfig, int, dict]:
    """
    Samplea una configuración PPO única para este intento.
    Combina:
        - Seed determinístico derivado del intento (reproducible)
        - Hiperparámetros aleatorios del espacio de búsqueda
        - Variación de n_envs para diversidad en la exploración
    """
    seed       = attempt * 137 + 42          # Pseudo-único y reproducible
    rng        = random.Random(seed)

    lr         = rng.choice(SEARCH_SPACE["learning_rate"])
    n_steps    = rng.choice(SEARCH_SPACE["n_steps"])
    batch_raw  = rng.choice(SEARCH_SPACE["batch_size"])
    gamma      = rng.choice(SEARCH_SPACE["gamma"])
    gae_lambda = rng.choice(SEARCH_SPACE["gae_lambda"])
    ent_coef   = rng.choice(SEARCH_SPACE["ent_coef"])
    clip_range = rng.choice(SEARCH_SPACE["clip_range"])
    arch_key   = rng.choice(SEARCH_SPACE["net_arch_key"])
    n_envs     = rng.choice(SEARCH_SPACE["n_envs"])

    # batch_size debe dividir exactamente (n_steps × n_envs)
    total_buf  = n_steps * n_envs
    batch_size = batch_raw
    while total_buf % batch_size != 0:
        batch_size //= 2
    batch_size = max(batch_size, 32)

    cfg = PPOConfig(
        learning_rate = lr,
        n_steps       = n_steps,
        batch_size    = batch_size,
        gamma         = gamma,
        gae_lambda    = gae_lambda,
        ent_coef      = ent_coef,
        clip_range    = clip_range,
        net_arch      = NET_ARCHS[arch_key],
        n_epochs      = 10,
    )

    meta = {
        "seed":       seed,
        "lr":         lr,
        "n_steps":    n_steps,
        "batch_size": batch_size,
        "gamma":      gamma,
        "gae_lambda": gae_lambda,
        "ent_coef":   ent_coef,
        "clip_range": clip_range,
        "net_arch":   arch_key,
        "n_envs":     n_envs,
    }
    return cfg, n_envs, meta

def passes_approval(metrics: dict) -> tuple[bool, list[str]]:
    """
    Evalúa si un modelo pasa los criterios de aprobación.
    Retorna (aprobado, lista_de_fallos).
    """
    fails = []

    if metrics.get("win_rate", 0)           < APPROVAL["win_rate"]:
        fails.append(f"win_rate={metrics.get('win_rate',0):.1%} < {APPROVAL['win_rate']:.0%}")

    if metrics.get("sharpe_ratio", 0)        < APPROVAL["sharpe_ratio"]:
        fails.append(f"sharpe={metrics.get('sharpe_ratio',0):.3f} < {APPROVAL['sharpe_ratio']}")

    if metrics.get("sortino_ratio", 0)       < APPROVAL["sortino_ratio"]:
        fails.append(f"sortino={metrics.get('sortino_ratio',0):.3f} < {APPROVAL['sortino_ratio']}")

    if metrics.get("max_drawdown_pct", 100)  > APPROVAL["max_drawdown_pct"]:
        fails.append(f"maxDD={metrics.get('max_drawdown_pct',0):.1f}% > {APPROVAL['max_drawdown_pct']}%")

    if metrics.get("profit_factor", 0)       < APPROVAL["profit_factor"]:
        fails.append(f"PF={metrics.get('profit_factor',0):.2f} < {APPROVAL['profit_factor']}")

    if metrics.get("alpha_pct", -999)        < APPROVAL["alpha_pct"]:
        fails.append(f"alpha={metrics.get('alpha_pct',0):.1f}% < {APPROVAL['alpha_pct']}%")

    if metrics.get("n_trades", 0)            < APPROVAL["n_trades"]:
        fails.append(f"n_trades={metrics.get('n_trades',0)} < {APPROVAL['n_trades']}")

    return (len(fails) == 0), fails

def save_approved_model(attempt: int, approved_count: int, metrics: dict, config_meta: dict):
    """
    Copia el modelo aprobado a la carpeta de aprobados con un nombre descriptivo.
    Guarda el modelo, el normalizador y las métricas juntos.
    """
    score  = metrics.get("sharpe_ratio", 0) + metrics.get("sortino_ratio", 0)
    label  = f"model_{approved_count:03d}_sharpe{metrics.get('sharpe_ratio',0):.2f}"
    dst    = APPROVED_DIR / label
    dst.mkdir(parents=True, exist_ok=True)

    src_model = Path(f"IA/models/{SYMBOL}/best_model.zip")
    src_norm  = Path(f"IA/models/{SYMBOL}/vec_normalize.pkl")

    if src_model.exists():
        shutil.copy2(src_model, dst / "best_model.zip")
    if src_norm.exists():
        shutil.copy2(src_norm, dst / "vec_normalize.pkl")

    # Guardar métricas + config del modelo aprobado
    model_info = {
        "approved_id":  approved_count,
        "attempt":      attempt,
        "saved_at":     datetime.now().isoformat(),
        "symbol":       SYMBOL,
        "interval":     INTERVAL,
        "timesteps":    TIMESTEPS_PER_MODEL,
        "config":       config_meta,
        "metrics": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in metrics.items()
            if k in [
                "total_return_pct", "benchmark_return_pct", "alpha_pct",
                "sharpe_ratio", "sortino_ratio", "calmar_ratio",
                "max_drawdown_pct", "win_rate", "profit_factor",
                "payoff_ratio", "n_trades", "expectancy", "cagr_pct",
            ]
        },
    }
    with open(dst / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2, default=str)

    print(f"\n  ✓ GUARDADO → {dst}")
    return str(dst)

def print_attempt_result(attempt: int, approved: int, elapsed: float,
                         metrics: dict, passed: bool, fails: list[str],
                         config_meta: dict):
    """Imprime el resultado de un intento de forma clara."""
    status = "✓ APROBADO" if passed else "✗ NO APROBADO"
    print(f"\n{'─'*60}")
    print(
        f"  Intento {attempt:>3}  |  {status}  |  "
        f"Aprobados: {approved}/{TARGET_APPROVED}  |  "
        f"{elapsed/60:.1f} min"
    )
    print(
        f"  WR={metrics.get('win_rate',0):.1%}  "
        f"Sharpe={metrics.get('sharpe_ratio',0):.2f}  "
        f"Sortino={metrics.get('sortino_ratio',0):.2f}  "
        f"MaxDD={metrics.get('max_drawdown_pct',0):.1f}%  "
        f"PF={metrics.get('profit_factor',0):.2f}  "
        f"Trades={metrics.get('n_trades',0)}"
    )
    print(
        f"  Config: lr={config_meta['lr']:.0e}  "
        f"n_steps={config_meta['n_steps']}  "
        f"arch={config_meta['net_arch']}  "
        f"ent={config_meta['ent_coef']:.3f}  "
        f"γ={config_meta['gamma']}"
    )
    if not passed:
        print(f"  Fallos: {' | '.join(fails)}")
    print(f"{'─'*60}")

def generate_html_report(all_results: list[dict], approved_count: int):
    """Genera un reporte HTML interactivo con el historial completo del factory."""

    approved  = [r for r in all_results if r["passed"]]
    failed    = [r for r in all_results if not r["passed"]]

    # Datos para las gráficas
    attempts   = [r["attempt"] for r in all_results]
    sharpes    = [r["metrics"].get("sharpe_ratio", 0) for r in all_results]
    win_rates  = [r["metrics"].get("win_rate", 0) * 100 for r in all_results]
    passed_idx = [r["attempt"] for r in approved]
    passed_sh  = [r["metrics"].get("sharpe_ratio", 0) for r in approved]

    approved_rows = ""
    for r in sorted(approved, key=lambda x: x["metrics"].get("sharpe_ratio", 0), reverse=True):
        m = r["metrics"]
        approved_rows += f"""
        <tr>
          <td>#{r['approved_id']}</td>
          <td>{r['attempt']}</td>
          <td>{m.get('sharpe_ratio',0):.3f}</td>
          <td>{m.get('sortino_ratio',0):.3f}</td>
          <td>{m.get('win_rate',0):.1%}</td>
          <td>{m.get('profit_factor',0):.2f}</td>
          <td>{m.get('max_drawdown_pct',0):.1f}%</td>
          <td>{m.get('total_return_pct',0):+.1f}%</td>
          <td>{m.get('alpha_pct',0):+.1f}%</td>
          <td>{m.get('n_trades',0)}</td>
          <td><code>{r['config']['lr']:.0e} / {r['config']['net_arch']} / ent={r['config']['ent_coef']:.3f}</code></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Model Factory — {SYMBOL}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; padding: 24px; }}
  h1   {{ font-size: 1.8rem; color: #00d4aa; margin-bottom: 4px; }}
  h2   {{ font-size: 1.1rem; color: #8b9dc3; font-weight: 400; margin-bottom: 24px; }}
  h3   {{ font-size: 1.1rem; color: #b0c4de; margin: 24px 0 12px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 32px; }}
  .card  {{ background: #1a1d27; border: 1px solid #2d3142; border-radius: 10px; padding: 16px; text-align: center; }}
  .card .val {{ font-size: 2rem; font-weight: 700; color: #00d4aa; }}
  .card .lbl {{ font-size: 0.78rem; color: #6b7280; margin-top: 4px; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 32px; }}
  .chart-box {{ background: #1a1d27; border: 1px solid #2d3142; border-radius: 10px; padding: 16px; }}
  table {{ width: 100%; border-collapse: collapse; background: #1a1d27; border-radius: 10px; overflow: hidden; }}
  th {{ background: #2d3142; color: #8b9dc3; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .05em; padding: 10px 12px; text-align: left; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #2d3142; font-size: 0.85rem; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #252839; }}
  code {{ background: #252839; padding: 2px 6px; border-radius: 4px; font-size: 0.78rem; color: #a78bfa; }}
  .green {{ color: #34d399; }}
  .red   {{ color: #f87171; }}
  .approval-box {{ background: #1a1d27; border: 1px solid #2d3142; border-radius: 10px; padding: 16px; margin-bottom: 24px; }}
  .approval-box h3 {{ margin: 0 0 10px; }}
  .criterion {{ display: inline-block; background: #252839; border-radius: 6px; padding: 4px 10px; margin: 3px; font-size: 0.82rem; }}
</style>
</head>
<body>
<h1>Model Factory — {SYMBOL}</h1>
<h2>Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  
    {len(all_results)} intentos  |  {approved_count} modelos aptos</h2>

<div class="stats">
  <div class="card"><div class="val">{approved_count}</div><div class="lbl">Modelos aptos</div></div>
  <div class="card"><div class="val">{len(all_results)}</div><div class="lbl">Total intentos</div></div>
  <div class="card"><div class="val">{approved_count/max(len(all_results),1)*100:.0f}%</div><div class="lbl">Tasa de éxito</div></div>
  <div class="card"><div class="val">{max((r["metrics"].get("sharpe_ratio",0) for r in approved), default=0):.2f}</div><div class="lbl">Mejor Sharpe</div></div>
  <div class="card"><div class="val">{max((r["metrics"].get("win_rate",0) for r in approved), default=0):.0%}</div><div class="lbl">Mejor Win Rate</div></div>
  <div class="card"><div class="val">{sum(r["duration_sec"] for r in all_results)/3600:.1f}h</div><div class="lbl">Tiempo total</div></div>
</div>

<div class="approval-box">
  <h3>Criterios de aprobación usados</h3>
  {"".join(f'<span class="criterion">{k}: {v}</span>' for k,v in APPROVAL.items())}
</div>

<div class="charts">
  <div class="chart-box"><canvas id="sharpeChart"></canvas></div>
  <div class="chart-box"><canvas id="wrChart"></canvas></div>
</div>

<h3>Modelos Aprobados (ordenados por Sharpe)</h3>
<table>
  <thead>
    <tr>
      <th>#</th><th>Intento</th><th>Sharpe</th><th>Sortino</th>
      <th>Win Rate</th><th>PF</th><th>Max DD</th>
      <th>Retorno</th><th>Alpha</th><th>Trades</th><th>Config</th>
    </tr>
  </thead>
  <tbody>{approved_rows}</tbody>
</table>

<script>
const attempts  = {json.dumps(attempts)};
const sharpes   = {json.dumps(sharpes)};
const winRates  = {json.dumps(win_rates)};
const passedIdx = {json.dumps(passed_idx)};
const passedSh  = {json.dumps(passed_sh)};
const threshold = {APPROVAL['sharpe_ratio']};

const colors = attempts.map((a,i) => passedIdx.includes(a) ? '#00d4aa' : '#374151');

new Chart(document.getElementById('sharpeChart'), {{
  type: 'bar',
  data: {{
    labels: attempts,
    datasets: [
      {{ label: 'Sharpe Ratio', data: sharpes, backgroundColor: colors, borderRadius: 3 }},
      {{ label: 'Umbral mínimo', data: attempts.map(() => threshold),
         type: 'line', borderColor: '#f87171', borderDash: [5,5],
         pointRadius: 0, borderWidth: 1.5, fill: false }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: 'Sharpe Ratio por intento', color: '#b0c4de' }},
               legend: {{ labels: {{ color: '#b0c4de' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#6b7280', maxTicksLimit: 20 }}, grid: {{ color: '#2d3142' }} }},
      y: {{ ticks: {{ color: '#6b7280' }}, grid: {{ color: '#2d3142' }} }}
    }}
  }}
}});

new Chart(document.getElementById('wrChart'), {{
  type: 'scatter',
  data: {{
    datasets: [{{
      label: 'Todos los intentos',
      data: attempts.map((a,i) => ({{x: sharpes[i], y: winRates[i]}})),
      backgroundColor: colors,
      pointRadius: 5,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: 'Sharpe vs Win Rate', color: '#b0c4de' }},
               legend: {{ labels: {{ color: '#b0c4de' }} }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Sharpe Ratio', color: '#6b7280' }},
             ticks: {{ color: '#6b7280' }}, grid: {{ color: '#2d3142' }} }},
      y: {{ title: {{ display: true, text: 'Win Rate (%)', color: '#6b7280' }},
             ticks: {{ color: '#6b7280' }}, grid: {{ color: '#2d3142' }} }}
    }}
  }}
}});
</script>
</body></html>"""

    report_path = Path(f"IA/models/{SYMBOL}/factory_report.html")
    report_path.write_text(html, encoding="utf-8")
    print(f"\n[Factory] Reporte HTML → {report_path}")
    return report_path

# ══════════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def run_factory():
    print(f"\n{'═'*65}")
    print(f"  MODEL FACTORY  |  {SYMBOL}  |  Objetivo: {TARGET_APPROVED} modelos aptos")
    print(f"  Max intentos: {MAX_ATTEMPTS}  |  Timesteps/modelo: {TIMESTEPS_PER_MODEL:,}")
    print(f"{'═'*65}\n")

    device = detect_device()

    # ── 1. Cargar datos ───────────────────────────────────────────────────────
    dm = DataManager()
    try:
        df_raw = dm.load(SYMBOL, INTERVAL)
        print(f"[Factory] Datos cargados desde disco: {len(df_raw):,} filas")
    except FileNotFoundError:
        print(f"[Factory] Descargando {SYMBOL} desde {SOURCE}...")
        df_raw = dm.download(SYMBOL, INTERVAL, START_TRAIN, source=SOURCE)

    # Normalizar timezone
    if df_raw.index.tz is None:
        df_raw.index = df_raw.index.tz_localize("UTC")

    # Split entrenamiento / test (sin overlap — el test nunca toca el entrenamiento)
    start_test_ts = pd.Timestamp(START_TEST, tz="UTC")
    df_train_raw  = df_raw[df_raw.index < start_test_ts].copy()
    df_test_raw   = df_raw[df_raw.index >= start_test_ts].copy()

    if END_TEST:
        df_test_raw = df_test_raw[df_test_raw.index <= pd.Timestamp(END_TEST, tz="UTC")]

    print(f"[Factory] Train: {len(df_train_raw):,} filas  |  Test: {len(df_test_raw):,} filas")

    if len(df_train_raw) < 500:
        print(f"[Factory] ERROR: Solo {len(df_train_raw)} barras de entrenamiento.")
        print(f"          Ajusta START_TRAIN a una fecha más antigua o descarga más datos.")
        sys.exit(1)

    if len(df_test_raw) < 100:
        print(f"[Factory] ERROR: Solo {len(df_test_raw)} barras de test.")
        print(f"          Ajusta START_TEST a una fecha más antigua.")
        sys.exit(1)

    # ── 2. Loop de intentos ───────────────────────────────────────────────────
    approved_count = 0
    all_results    = []
    factory_start  = time.time()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if STOP_ON_TARGET and approved_count >= TARGET_APPROVED:
            print(f"\n[Factory] 🎯 Objetivo alcanzado: {approved_count} modelos aptos")
            break

        print(f"\n{'═'*65}")
        print(
            f"  INTENTO {attempt}/{MAX_ATTEMPTS}  |  "
            f"Aprobados: {approved_count}/{TARGET_APPROVED}  |  "
            f"Tiempo total: {(time.time()-factory_start)/60:.0f} min"
        )
        print(f"{'═'*65}")

        attempt_start = time.time()
        cfg, n_envs, meta = sample_config(attempt)

        print(
            f"[Factory] Config sampledada:\n"
            f"  lr={meta['lr']:.0e}  n_steps={meta['n_steps']}  "
            f"batch={meta['batch_size']}  γ={meta['gamma']}\n"
            f"  arch={meta['net_arch']}  ent={meta['ent_coef']:.3f}  "
            f"clip={meta['clip_range']}  n_envs={n_envs}"
        )

        metrics   = {}
        passed    = False
        fails     = []
        error_msg = None

        try:
            # ── Entrenamiento ─────────────────────────────────────────────────
            trainer = ModelTrainer(
                symbol  = SYMBOL,
                config  = cfg,
                device  = device,
                n_envs  = n_envs,
            )
            trainer.train(
                df              = df_train_raw,
                total_timesteps = TIMESTEPS_PER_MODEL,
                eval_split      = 0.15,   # 15% del train para eval interna
            )

            # ── Backtest sobre datos de test (nunca vistos) ───────────────────
            engine  = BacktestEngine(
                symbol          = SYMBOL,
                initial_balance = INITIAL_BALANCE,
                commission      = COMMISSION,
            )
            result  = engine.run(df_test_raw, interval=INTERVAL)
            metrics = BacktestMetrics(result).compute()

            # ── Evaluar aprobación ────────────────────────────────────────────
            passed, fails = passes_approval(metrics)

            if passed:
                approved_count += 1
                save_approved_model(attempt, approved_count, metrics, meta)

        except KeyboardInterrupt:
            print("\n[Factory] Interrumpido por el usuario")
            break
        except Exception as e:
            error_msg = str(e)
            print(f"[Factory] ERROR en intento {attempt}: {e}")
            traceback.print_exc()

        duration = time.time() - attempt_start
        print_attempt_result(attempt, approved_count, duration, metrics, passed, fails, meta)

        # Guardar historial del intento
        all_results.append({
            "attempt":      attempt,
            "passed":       passed,
            "approved_id":  approved_count if passed else None,
            "duration_sec": round(duration, 1),
            "config":       meta,
            "metrics":      {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in metrics.items()},
            "fails":        fails,
            "error":        error_msg,
        })

        # Guardar JSON incremental (para no perder datos si se interrumpe)
        FACTORY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(FACTORY_LOG, "w") as f:
            json.dump({
                "symbol":          SYMBOL,
                "target":          TARGET_APPROVED,
                "approved":        approved_count,
                "attempts":        len(all_results),
                "approval_config": APPROVAL,
                "results":         all_results,
            }, f, indent=2, default=str)

    # ── 3. Resumen final ──────────────────────────────────────────────────────
    total_time = time.time() - factory_start
    approved_list = [r for r in all_results if r["passed"]]

    print(f"\n{'═'*65}")
    print(f"  FACTORY COMPLETADO  —  {SYMBOL}")
    print(f"{'═'*65}")
    print(f"  Intentos realizados : {len(all_results)}")
    print(f"  Modelos aprobados   : {approved_count}")
    print(f"  Tasa de aprobación  : {approved_count/max(len(all_results),1)*100:.1f}%")
    print(f"  Tiempo total        : {total_time/3600:.2f} h  ({total_time/60:.0f} min)")

    if approved_list:
        best = max(approved_list, key=lambda r: r["metrics"].get("sharpe_ratio", 0))
        print(f"\n  Mejor modelo:")
        print(f"    Intento    : #{best['attempt']}")
        print(f"    Sharpe     : {best['metrics'].get('sharpe_ratio',0):.3f}")
        print(f"    Sortino    : {best['metrics'].get('sortino_ratio',0):.3f}")
        print(f"    Win Rate   : {best['metrics'].get('win_rate',0):.1%}")
        print(f"    Max DD     : {best['metrics'].get('max_drawdown_pct',0):.1f}%")
        print(f"    Retorno    : {best['metrics'].get('total_return_pct',0):+.1f}%")
        print(f"    Config     : {best['config']}")
        print(f"\n  Modelos guardados en → {APPROVED_DIR}")

    print(f"  Reporte JSON  → {FACTORY_LOG}")

    # ── 4. Reporte HTML ───────────────────────────────────────────────────────
    if all_results:
        report_path = generate_html_report(all_results, approved_count)
        try:
            import webbrowser
            webbrowser.open(report_path.resolve().as_uri())
        except Exception:
            pass

    print(f"\n{'═'*65}\n")
    return approved_count, all_results

if __name__ == "__main__":
    run_factory()