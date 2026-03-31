# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN - IBKR Bot — edita solo esta sección
# ══════════════════════════════════════════════════════════════════════════════

SYMBOL       = "AAL"    # Símbolo a operar, la IA_BackTests debe de haber sido entrenada con el simbolo antes de ejecutal el BOT
INTERVAL_MIN = 60        # Tamaño de vela en minutos (ej: 60 = velas de 1h)
INITIAL_CASH = 10_000.0  # Capital inicial (informativo para el monitor)

# ── Conexión IB ───────────────────────────────────────────────────────────────
PAPER_MODE = True        # True = Paper Trading | False = LIVE (dinero real)

TWS_HOST   = "127.0.0.1"
TWS_PORT   = 7497        # Paper: 7497 (TWS) / 4002 (Gateway)
                         # Live:  7496 (TWS) / 4001 (Gateway)
CLIENT_ID  = 1           # ID único de cliente (cambia si hay varias conexiones)

# ── IA_BackTests / Riesgo ───────────────────────────────────────────────────────────────
WARMUP_BARS          = 80    # Barras históricas antes de operar
CONFIDENCE_THRESHOLD = 0.60  # Confianza mínima del modelo para actuar

max_position_pct   = 0.10   # Max 10% del capital por trade
max_daily_loss_pct = 0.03   # Detener si pierde 3% en el día
max_drawdown_pct   = 0.15   # Detener si drawdown supera 15%
Max_trades_per_day = 10     # Máximo de operaciones diarias
atr_multiplier     = 2.0    # Stop loss = entrada - ATR × 2


# ── Datos históricos (warmup) ─────────────────────────────────────────────────
HISTORY_DURATION = "5 D"  # Cuánto historial pedir para calentar la IA_BackTests
                           # "1 D" | "5 D" | "1 M" | "6 M" | "1 Y"
# ══════════════════════════════════════════════════════════════════════════════


#rutas de alojamiento de archivos tipo: logs,csv,html etc
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas de Datos
DATA_DIR = BASE_DIR / "Data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Rutas de Modelos y Logs
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Crear carpetas si no existen automáticamente
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)