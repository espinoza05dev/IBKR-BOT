"""
BacktestReport.py
Genera un reporte HTML interactivo con:
    - Curva de equity vs Buy & Hold
    - Curva de drawdown
    - Distribución de trades (winners vs losers)
    - Tabla completa de métricas
    - Log de todos los trades
    - Veredicto final con semáforo de aprobación
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.backtest.Backtestengine import BacktestResult

# REPORTS_DIR = Path("IA_BackTests/backtest/reports")
REPORTS_DIR = Path("C:\\Users\\artur\\Programming\\PycharmProjects\\python_autotrader\\IA\\IA_BackTests\\backtest\\reports")

class BacktestReport:
    """
    Genera un reporte HTML autocontenido (sin dependencias externas).
    Usa Chart.js via CDN para los gráficos interactivos.

    Uso:
        report = BacktestReport(result, metrics)
        path   = report.save()          # Guarda HTML en IA_BackTests/backtest/reports/
        report.show()                   # Abre en el navegador por defecto
    """

    def __init__(self, result: "BacktestResult", metrics: dict):
        self.r = result
        self.m = metrics
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str | None = None) -> Path:
        """Guarda el reporte HTML y retorna la ruta."""
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname    = filename or f"backtest_{self.r.symbol}_{ts}.html"
        path     = REPORTS_DIR / fname
        html     = self._build_html()
        path.write_text(html, encoding="utf-8")
        print(f"[Report] Reporte guardado → {path}")
        return path

    def show(self):
        """Abre el reporte en el navegador del sistema."""
        import webbrowser
        path = self.save()
        webbrowser.open(f"file://{path.resolve()}")

    # ── Construcción HTML ─────────────────────────────────────────────────────

    def _build_html(self) -> str:
        equity_data  = self._series_to_json(self.r.equity_curve)
        bh_data      = self._series_to_json(self.r.benchmark_equity)
        dd_data      = self._series_to_json(self.r.drawdown_series * 100)
        labels       = self._labels_json(self.r.equity_curve)
        trades_rows  = self._trades_table_rows()
        metrics_rows = self._metrics_table_rows()
        verdict      = self.m.get("approved_for_live", False)
        verdict_html = self._verdict_html(verdict)
        pnl_hist     = self._pnl_histogram_data()

        return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest — {self.r.symbol}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f1117; color: #e2e8f0; font-size: 14px; }}
  .header {{ background: linear-gradient(135deg, #1a1f2e, #16213e);
             padding: 28px 32px; border-bottom: 1px solid #2d3748; }}
  .header h1 {{ font-size: 22px; font-weight: 600; color: #fff; }}
  .header .sub {{ color: #718096; font-size: 13px; margin-top: 4px; }}
  .content {{ max-width: 1200px; margin: 0 auto; padding: 24px 20px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
  .grid3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 16px; }}
  .grid4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }}
  .card {{ background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px;
           padding: 20px; }}
  .kpi  {{ text-align: center; }}
  .kpi .val {{ font-size: 26px; font-weight: 700; margin: 6px 0; }}
  .kpi .lbl {{ font-size: 12px; color: #718096; text-transform: uppercase; letter-spacing: .05em; }}
  .pos {{ color: #68d391; }} .neg {{ color: #fc8181; }} .neu {{ color: #63b3ed; }}
  .section-title {{ font-size: 15px; font-weight: 600; color: #a0aec0;
                    margin: 24px 0 10px; text-transform: uppercase; letter-spacing: .06em; }}
  canvas {{ width: 100% !important; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #2d3748; color: #a0aec0; padding: 8px 12px;
       text-align: left; font-weight: 500; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #2d3748; color: #e2e8f0; }}
  tr:hover td {{ background: #2d3748; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 11px; font-weight: 600; }}
  .badge-win {{ background: #276749; color: #9ae6b4; }}
  .badge-loss{{ background: #742a2a; color: #feb2b2; }}
  @media (max-width: 700px) {{ .grid2,.grid3,.grid4 {{ grid-template-columns: 1fr 1fr; }} }}
</style>
</head>
<body>
<div class="header">
  <h1>Backtest Report — {self.r.symbol}</h1>
  <div class="sub">
    {self.r.start_date} → {self.r.end_date} &nbsp;|&nbsp;
    Intervalo: {self.r.interval} &nbsp;|&nbsp;
    Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")}
  </div>
</div>

<div class="content">

{verdict_html}

<!-- KPIs principales -->
<div class="grid4">
  {self._kpi("Retorno Total", f"{self.m.get('total_return_pct',0):+.2f}%", "pos" if self.m.get('total_return_pct',0)>=0 else "neg")}
  {self._kpi("Sharpe Ratio",  f"{self.m.get('sharpe_ratio',0):.3f}", "pos" if self.m.get('sharpe_ratio',0)>=0.5 else "neg")}
  {self._kpi("Max Drawdown",  f"-{self.m.get('max_drawdown_pct',0):.2f}%", "neg")}
  {self._kpi("Win Rate",      f"{self.m.get('win_rate',0):.1%}", "pos" if self.m.get('win_rate',0)>=0.5 else "neg")}
</div>
<div class="grid4">
  {self._kpi("Balance Final",  f"${self.m.get('final_balance',0):,.2f}", "neu")}
  {self._kpi("Alpha vs B&H",   f"{self.m.get('alpha_pct',0):+.2f}%", "pos" if self.m.get('alpha_pct',0)>=0 else "neg")}
  {self._kpi("Profit Factor",  f"{self.m.get('profit_factor',0):.2f}", "pos" if self.m.get('profit_factor',0)>=1.2 else "neg")}
  {self._kpi("Trades",         str(self.m.get('n_trades',0)), "neu")}
</div>

<!-- Curva de Equity -->
<div class="section-title">Curva de Equity</div>
<div class="card" style="margin-bottom:16px">
  <canvas id="equityChart" height="90"></canvas>
</div>

<!-- Drawdown -->
<div class="section-title">Drawdown</div>
<div class="card" style="margin-bottom:16px">
  <canvas id="ddChart" height="55"></canvas>
</div>

<!-- Histograma PnL + tabla métricas -->
<div class="section-title">Análisis de Trades</div>
<div class="grid2">
  <div class="card"><canvas id="pnlChart" height="180"></canvas></div>
  <div class="card" style="overflow:auto">
    <table>{metrics_rows}</table>
  </div>
</div>

<!-- Log de trades -->
<div class="section-title">Log de Trades ({self.m.get('n_trades',0)} operaciones)</div>
<div class="card" style="overflow:auto">
  <table>
    <thead><tr>
      <th>#</th><th>Entrada</th><th>Salida</th>
      <th>Precio Entrada</th><th>Precio Salida</th>
      <th>PnL Neto</th><th>PnL %</th><th>Duración</th><th>Resultado</th>
    </tr></thead>
    <tbody>{trades_rows}</tbody>
  </table>
</div>

</div><!-- /content -->

<script>
const labels = {labels};
const equity = {equity_data};
const bh     = {bh_data};
const dd     = {dd_data};
const pnlBins   = {pnl_hist["bins"]};
const pnlCounts = {pnl_hist["counts"]};
const pnlColors = {pnl_hist["colors"]};

// Equity Chart
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{ label: 'Modelo IA_BackTests', data: equity, borderColor: '#68d391',
         backgroundColor: 'rgba(104,211,145,.08)', borderWidth: 2,
         pointRadius: 0, tension: .3, fill: true }},
      {{ label: 'Buy & Hold', data: bh, borderColor: '#63b3ed',
         backgroundColor: 'transparent', borderWidth: 1.5,
         pointRadius: 0, tension: .3, borderDash: [5,3] }}
    ]
  }},
  options: {{
    responsive: true, animation: false,
    plugins: {{ legend: {{ labels: {{ color: '#a0aec0' }} }} }},
    scales: {{
      x: {{ ticks: {{ color:'#718096', maxTicksLimit:10 }}, grid: {{ color:'#2d3748' }} }},
      y: {{ ticks: {{ color:'#718096', callback: v=>'$'+v.toLocaleString() }},
            grid: {{ color:'#2d3748' }} }}
    }}
  }}
}});

// Drawdown Chart
new Chart(document.getElementById('ddChart'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [{{
      label: 'Drawdown %', data: dd,
      borderColor: '#fc8181', backgroundColor: 'rgba(252,129,129,.15)',
      borderWidth: 1.5, pointRadius: 0, tension: .3, fill: true
    }}]
  }},
  options: {{
    responsive: true, animation: false,
    plugins: {{ legend: {{ labels: {{ color:'#a0aec0' }} }} }},
    scales: {{
      x: {{ ticks: {{ color:'#718096', maxTicksLimit:10 }}, grid: {{ color:'#2d3748' }} }},
      y: {{ ticks: {{ color:'#718096', callback: v=>v.toFixed(1)+'%' }},
            grid: {{ color:'#2d3748' }}, reverse: false }}
    }}
  }}
}});

// PnL Histogram
new Chart(document.getElementById('pnlChart'), {{
  type: 'bar',
  data: {{
    labels: pnlBins,
    datasets: [{{ label: 'PnL por trade ($)', data: pnlCounts,
                  backgroundColor: pnlColors, borderRadius: 3 }}]
  }},
  options: {{
    responsive: true, animation: false,
    plugins: {{ legend: {{ labels: {{ color:'#a0aec0' }} }},
               title: {{ display:true, text:'Distribución de PnL por trade',
                         color:'#a0aec0' }} }},
    scales: {{
      x: {{ ticks: {{ color:'#718096' }}, grid: {{ color:'#2d3748' }} }},
      y: {{ ticks: {{ color:'#718096' }}, grid: {{ color:'#2d3748' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

    # ── Helpers HTML ──────────────────────────────────────────────────────────

    def _verdict_html(self, approved: bool) -> str:
        if approved:
            color = "#276749"; icon = "✓"; text = "MODELO APROBADO — Listo para paper trading"
            bg    = "rgba(39,103,73,.25)"
        else:
            issues = []
            m = self.m
            if m.get("win_rate",0)        < 0.50: issues.append(f"Win rate {m['win_rate']:.1%} < 50%")
            if m.get("sharpe_ratio",0)    < 0.50: issues.append(f"Sharpe {m['sharpe_ratio']:.2f} < 0.5")
            if m.get("max_drawdown_pct",0)> 20.0: issues.append(f"Max DD {m['max_drawdown_pct']:.1f}% > 20%")
            if m.get("profit_factor",0)   < 1.20: issues.append(f"PF {m['profit_factor']:.2f} < 1.2")
            color = "#742a2a"; icon = "✗"; bg = "rgba(116,42,42,.25)"
            text  = "MODELO NO APROBADO — " + " | ".join(issues)

        return f"""<div style="background:{bg};border:1px solid {color};border-radius:8px;
padding:14px 20px;margin-bottom:20px;display:flex;align-items:center;gap:12px">
  <span style="color:{color};font-size:22px;font-weight:700">{icon}</span>
  <span style="color:{color};font-weight:600;font-size:15px">{text}</span>
</div>"""

    def _kpi(self, label: str, value: str, color_cls: str) -> str:
        return f"""<div class="card kpi">
  <div class="lbl">{label}</div>
  <div class="val {color_cls}">{value}</div>
</div>"""

    def _metrics_table_rows(self) -> str:
        display = {
            "total_return_pct":     "Retorno total %",
            "benchmark_return_pct": "B&H retorno %",
            "alpha_pct":            "Alpha %",
            "cagr_pct":             "CAGR %",
            "volatility_annual_pct":"Volatilidad anual %",
            "sharpe_ratio":         "Sharpe ratio",
            "sortino_ratio":        "Sortino ratio",
            "calmar_ratio":         "Calmar ratio",
            "omega_ratio":          "Omega ratio",
            "max_drawdown_pct":     "Max drawdown %",
            "avg_drawdown_pct":     "DD promedio %",
            "win_rate":             "Win rate",
            "profit_factor":        "Profit factor",
            "payoff_ratio":         "Payoff ratio",
            "expectancy":           "Expectancy $",
            "var_95_pct":           "VaR 95% diario %",
            "cvar_95_pct":          "CVaR 95% diario %",
            "beta":                 "Beta vs B&H",
            "total_commissions":    "Comisiones totales $",
        }
        rows = ""
        for key, label in display.items():
            val = self.m.get(key, "—")
            rows += f"<tr><td style='color:#718096'>{label}</td><td>{val}</td></tr>"
        return rows

    def _trades_table_rows(self) -> str:
        rows  = ""
        trades = [t for t in self.r.trades if t.exit_date is not None]
        for t in sorted(trades, key=lambda x: x.entry_date or 0):
            color  = "#9ae6b4" if t.is_winner else "#feb2b2"
            badge  = f'<span class="badge badge-{"win" if t.is_winner else "loss"}">'
            badge += ("WIN" if t.is_winner else "LOSS") + "</span>"
            rows += f"""<tr>
              <td>{t.trade_id}</td>
              <td>{str(t.entry_date)[:16] if t.entry_date else '—'}</td>
              <td>{str(t.exit_date)[:16]  if t.exit_date  else '—'}</td>
              <td>${t.entry_price:.2f}</td>
              <td>${t.exit_price:.2f}</td>
              <td style="color:{color}">${t.pnl_net:+.2f}</td>
              <td style="color:{color}">{t.pnl_pct:+.2f}%</td>
              <td>{t.duration_bars} barras</td>
              <td>{badge}</td>
            </tr>"""
        return rows or "<tr><td colspan='9' style='text-align:center;color:#718096'>Sin trades</td></tr>"

    # ── Datos para Chart.js ───────────────────────────────────────────────────

    @staticmethod
    def _series_to_json(s: "pd.Series") -> str:
        vals = s.round(2).tolist()
        # Submuestrear si hay demasiados puntos (Chart.js se ralentiza con >5k)
        if len(vals) > 3000:
            step = len(vals) // 3000
            vals = vals[::step]
        return json.dumps(vals)

    @staticmethod
    def _labels_json(s: "pd.Series") -> str:
        if hasattr(s.index, "strftime"):
            labels = [str(x)[:16] for x in s.index]
        else:
            labels = list(range(len(s)))
        if len(labels) > 3000:
            step   = len(labels) // 3000
            labels = labels[::step]
        return json.dumps(labels)

    def _pnl_histogram_data(self) -> dict:
        import numpy as np
        trades = [t for t in self.r.trades if t.exit_date is not None]
        if not trades:
            return {"bins": [], "counts": [], "colors": []}

        pnls   = [t.pnl_net for t in trades]
        counts, edges = np.histogram(pnls, bins=min(15, len(pnls)))
        bins   = [f"${e:.0f}" for e in edges[:-1]]
        colors = ["rgba(104,211,145,.75)" if (e + edges[i+1])/2 >= 0
                  else "rgba(252,129,129,.75)"
                  for i, e in enumerate(edges[:-1])]
        return {
            "bins":   json.dumps(bins),
            "counts": json.dumps(counts.tolist()),
            "colors": json.dumps(colors),
        }