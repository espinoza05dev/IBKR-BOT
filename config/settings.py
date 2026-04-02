# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN - IBKR Bot & AI Training
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Importaciones y Rutas (Paths) ──────────────────────────────────────────
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas
IA_DIR = BASE_DIR / 'IA'
DATA_DIR = BASE_DIR / "Data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = IA_DIR / "models"
LOGS_DIR = IA_DIR / "logs"
KNOWLEDGE_DIR = IA_DIR / "knowledgeBase"
INGESTION_DIR = KNOWLEDGE_DIR / "IngestionLog"
DB_DIR = KNOWLEDGE_DIR / "db"
SESSIONS_DIR = IA_DIR / "IA_BackTests" / "LogsSession" / "sessions"

# Crear carpetas si no existen automáticamente
DB_DIR.mkdir(parents=True,exist_ok=True)
INGESTION_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── 2. Parámetros Core del Bot ────────────────────────────────────────────────
SYMBOL       = "A"     # Símbolo a operar, la IA_BackTests debe de haber sido entrenada con el simbolo antes de ejecutal el BOT
INITIAL_CASH = 10_000.0  # Capital inicial (informativo para el monitor)
INTERVAL_MIN = 60        # Tamaño de vela en minutos (ej: 60 = velas de 1h)
INTERVAL     = "1h"      # "1m" "5m" "15m" "1h" "1d"

# ── 3. Conexión IBKR ──────────────────────────────────────────────────────────
PAPER_MODE = True        # True = Paper Trading | False = LIVE (dinero real)

TWS_HOST   = "127.0.0.1"
TWS_PORT   = 7497        # Paper: 7497 (TWS) / 4002 (Gateway)
                         # Live:  7496 (TWS) / 4001 (Gateway)
CLIENT_ID  = 1           # ID único de cliente (cambia si hay varias conexiones)

# ── 4. Reglas de Riesgo e IA_BackTests ────────────────────────────────────────
START_TEST_BACKTEST      = "2025-01-01"  # Los datos de TEST deben ser POSTERIORES al entrenamiento
START_TRAIN  = "2010-01-01"     # Fecha de inicio
WARMUP_BARS          = 80    # Barras históricas antes de operar
CONFIDENCE_THRESHOLD = 0.60  # Confianza mínima del modelo para actuar

"""
Modos:
    single      → Un backtest sobre el período de test
    walkforward → Validación walk-forward en N ventanas
    compare     → Backtestea múltiples símbolos y compara
"""

BACKTEST_MODE = "single"   # "single" | "walkforward" | "compare"

# Walk-forward (solo Modo walkforward)
WF_N_WINDOWS          = 4
WF_TIMESTEPS_WINDOW   = 10_000_000   # Steps por ventana (más rápido que entrenamiento completo)

# Multi-símbolo (solo Modo compare)
COMPARE_SYMBOLS = ["AAPL", "MSFT", "NVDA",""]

OPEN_REPORT_IN_BROWSER = True   # Abrir HTML al terminar

max_position_pct   = 0.10   # Max 10% del capital por trade
max_daily_loss_pct = 0.03   # Detener si pierde 3% en el día
max_drawdown_pct   = 0.15   # Detener si drawdown supera 15%
Max_trades_per_day = 10     # Máximo de operaciones diarias
atr_multiplier     = 2.0    # Stop loss = entrada - ATR × 2

# ── 5. Datos Históricos (Descarga y Warmup) ───────────────────────────────────
HISTORY_DURATION = "5 D"  # Cuánto historial pedir para calentar la IA_BackTests
                          # "1 D" | "5 D" | "1 M" | "6 M" | "1 Y"
START  = "2010-01-01"     # Fecha de inicio
SOURCE = "yfinance"       # "yfinance" | "ibkr" | "av" | "csv"

"""
───────────────────── Download Historical Data & Train Model & Model factory ────────────────────────────
─────────────────────────────────────── Configuración ───────────────────────────────────
 en symbols puede ser multiples, este es solamente para Download Historical Data
"""
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK.B", "TSLA", "V", "UNH", "JNJ", "XOM", "WMT", "JPM", "MA", "PG", "AVGO", "HD", "ORCL", "CVX", "LLY", "ABBV", "KO", "PEP", "MRK", "COST", "ADBE", "TMO", "CSCO", "PFE", "BAC", "CRM", "ACN", "ABT", "LIN", "NFLX", "AMD", "DIS", "TXN", "DHR", "INTC", "PM", "VZ", "RTX", "NEE", "AMGN", "LOW", "WFC", "IBM", "UNP", "HON", "COP", "CAT", "MS", "GE", "INTU", "SBUX", "GS", "DE", "PLD", "BLK", "BMY", "AMT", "LMT", "BA", "AXP", "AMAT", "ELV", "T", "GILD", "MDT", "SYK", "TJX", "ADI", "BKNG", "ISRG", "VRTX", "MMC", "REGN", "ADP", "NOW", "VLO", "CI", "CB", "LRCX", "SLB", "MDLZ", "CVS", "PGR", "ZTS", "MO", "SCHW", "BSX", "BDX", "ETN", "MU", "EQIX", "DUK", "C", "SO", "EOG", "AON", "HUM", "WM", "ITW", "KLAC", "ICE", "PANW", "MPC", "APD", "MCD", "EW", "MCK", "ATVI", "ORLY", "SNPS", "SHW", "MAR", "FDX", "HCA", "EMR", "APH", "NXPI", "MCO", "USB", "CDNS", "MSI", "ECL", "ROP", "F", "ADM", "CL", "ADSK", "GD", "PSA", "D", "PXD", "AIG", "DOW", "MET", "AJG", "EXC", "NSC", "PCAR", "TEL", "AEP", "TGT", "TRV", "O", "CTAS", "AZO", "PAYX", "COF", "PH", "KMB", "MGM", "CMI", "BKR", "WELL", "ED", "NOC", "WMB", "STZ", "AEL", "IDXX", "IQV", "OTIS", "MSCI", "CARR", "DXCM", "JCI", "BK", "VICI", "CHTR", "CSX", "NEM", "HUM", "DLR", "ALL", "SPG", "GIS", "SYY", "CNC", "EIX", "VRSK", "SRE", "A", "KDP", "LVS", "GPN", "KR", "ROK", "MCHP", "MTD", "PRU", "PR", "HLT", "EBAY", "PEG", "COR", "FAST", "TRGP", "CTSH", "FTNT", "AME", "DD", "BBY", "WBD", "WES", "DAL", "UAL", "AAL", "CCL", "RCL", "NCLH", "PARA", "HPE", "HPQ", "MU", "NKE", "TROW", "DLTR", "ROST", "EXPE", "DFS", "FITB", "K", "STT", "KEYS", "VMC", "GWW", "CBRE", "WY", "TSCO", "HWM", "EFX", "VTR", "DHI", "LEN", "MAA", "AVB", "EQR", "INVH", "ARE", "BXP", "CPT", "UDR", "ESS", "EXR", "KIM", "REG", "FRT", "GL", "BEN", "IVZ", "NWL", "WHR", "HBI", "PVH", "RL", "UAA", "HAS", "MAT", "GEN", "CTRA", "MRO", "DVN", "APA", "HAL", "HES", "FANG", "CTVA", "FMC", "MOS", "CF", "CE", "EMN", "SHW", "VMC", "MLM", "EXP", "NUE", "STLD", "FCX", "NEM", "ALB", "LTHM", "WRK", "PKG", "IP", "SEE", "AMCR", "BALL", "LYB", "IFF", "RPM", "DD", "DOW", "CTVA", "XYL", "PH", "DOV", "SNA", "SWK", "IEX", "AME", "IR", "TT", "PNR", "ZURN", "EMR", "CHRW", "EXPD", "JBHT", "ODFL", "NSC", "CSX", "UNP", "KSU", "CP", "CNI", "LUV", "ALK", "HA", "FDX", "UPS", "LDO", "GD", "HII", "TDY", "TXT", "LHX", "LDOS", "SAIC", "MANT", "ACM", "KBR", "VRSN", "AKAM", "FTNT", "PANW", "CRWD", "NET", "OKTA", "ZS", "DDOG", "MDB", "SPLK", "TEAM", "WDAY", "SNOW", "DOCU", "TWLO", "RNG", "ZEN", "AYX", "BOX", "DBX", "PATH", "UI", "U", "UNITY", "STNE", "PAGS", "MELI", "SE", "SHOP", "ETSY", "EBAY", "GRUB", "DASH", "ABNB", "BKNG", "EXPE", "TRIP", "TCOM", "HLT", "MAR", "H", "WH", "RHP", "CHH", "PLYA", "MTN", "ERI", "CZR", "WYNN", "LVS", "MGM", "PENN", "DKNG", "BYD", "BALY", "IGT", "SGMS", "AAL", "DAL", "UAL", "LUV", "ALK", "SAVE", "JBLU", "HA", "SKYW", "MESA", "ALGT", "F", "GM", "TSLA", "RIVN", "LCID", "FSR", "NKLA", "HMC", "TM", "STLA", "VWAGY", "BMWYY", "DDAIF", "HYMTF", "NSANY", "MZDAY", "FUJHY", "TTM", "NIO", "XPEV", "LI", "BYDDY", "GE", "MMM", "HON", "RTX", "BA", "LMT", "NOC", "GD", "LHX", "TXT", "HWM", "TDG", "HEI", "BWXT", "TGI", "SPR", "DUK", "SO", "AEP", "EXC", "D", "NEE", "SRE", "EIX", "PEG", "ED", "WEC", "ES", "ETR", "FE", "AEE", "CMS", "LNT", "ATO", "NI", "PNW", "OGE", "CNP", "XEL", "NRG", "VST", "AES", "AWK", "WTRG", "MCD", "SBUX", "YUM", "CMG", "DRI", "DPZ", "WEN", "QSR", "JACK", "EAT", "BLMN", "TXRH", "CAKE", "PZZA", "CBRL", "DIN", "RRGB", "DENN", "BJRI", "CHUY"]

"""
── 6. Model Factory (Aprobación y Búsqueda de Hiperparámetros) ───────────────
─────────────────────────────Exclusivo para Model Factory──────────────────────────────────────────
"""
START_TEST      = "2023-06-01"  # Los datos de test NUNCA se ven durante entrenamiento
END_TEST        = None          # None = hasta hoy
COMMISSION      = 0.001         # 0.1% por operación (IBKR)
"""────────────────────────────────────────────────────────────────────────────────────"""
# ──────────────────────────────────────────── Metas ────────────────────────────────────────────────
TARGET_APPROVED = 20            # Cuántos modelos aptos quiero
MAX_ATTEMPTS    = 200           # Intentos máximos antes de rendirse
STOP_ON_TARGET  = True          # True = parar al llegar al objetivo
                                # False = seguir hasta MAX_ATTEMPTS

# approval
win_rate:        0.52   # ≥ 52%  (Al ser más estable, la IA debería acertar más de la mitad de las veces)
sharpe_ratio:    0.60   # ≥ 0.60 (Menor volatilidad general permite exigir un mejor retorno ajustado al riesgo)
sortino_ratio:   0.80   # ≥ 0.80 (Agilent tiene menos riesgo a la baja que AAL, el Sortino debe ser mayor)
max_drawdown_pct: 15.0  # ≤ 15%  (Un 25% es demasiado permisivo para esta acción; ajustamos para proteger el capital)
profit_factor:   1.25   # ≥ 1.25 (Por cada $1 perdido, debería ganar $1.25, aprovechando tendencias limpias)
alpha_pct:       0.0    # ≥ 0%   (No deberías permitir un alpha negativo fuerte; el bot debe al menos igualar al mercado)
n_trades:        10     # ≥ 10   (Exigimos un par de trades más para asegurar que el modelo no dependa de 1 o 2 "golpes de suerte")
#cambiar valores dependiendo de empresa

#search space
learning_rate:  [1e-4, 3e-4, 5e-4, 7e-4, 1e-3]
n_steps:        [2048, 4096, 8192]
batch_size:     [2048, 4096, 8192]
gamma:          [0.95, 0.97, 0.99, 0.995]
gae_lambda:     [0.90, 0.92, 0.95, 0.98]
ent_coef:       [0.001, 0.005, 0.01, 0.02, 0.05]
clip_range:     [0.1, 0.15, 0.2, 0.25, 0.3]
net_arch_key:   ["small", "medium", "large", "deep"]
n_envs:         [2, 4, 6, 8]

#net archs
small:  [64, 64]
medium: [128, 128]
large:  [256, 256]
deep:   [128, 128, 64]

""" 
── 7. Entrenamiento del Modelo (Train Model) ─────────────────────────────────
────────────────────────Config especifica de Train Model────────────────────────
"""

MODE = "A"   # "A" | "B" | "C" | "D"
"""
    Modos:
    A → Pipeline completo (descarga + features + entrena)  ← primer uso
    B → Datos ya en disco
    C → Reanudar desde checkpoint
    D → Optimización de hiperparámetros (Optuna)
"""

# ── GPU / paralelismo ──────────────────────────────────────────────────────
GPU_DEVICE  = "cuda"   # "auto" detecta CUDA→MPS→CPU automáticamente
                       # "cuda" fuerza NVIDIA GPU
                       # "mps"  fuerza Apple Silicon
                       # "cpu"  fuerza CPU (útil para debug)

N_ENVS      = 16       # 0 = auto según hardware
                       # GPU  RTX 30/40: usa 8
                       # Apple M-series: usa 4
                       # Solo CPU      : usa cantidad de núcleos

""" 
── OPTIMIZACIÓN CRÍTICA PARA CPU/GPU ─────────────────────────────────────────
 Evita que la CPU pelee consigo misma intentando usar todos los hilos
 en cada entorno. Mantiene la CPU libre para alimentar a la GPU.
    el numero de hilos debe estar entre 2 - 4
"""
SET_THREAD_NUMBER = 3
#-----------------

# ── Timesteps y Evaluación ─────────────────────────────────────────────────
# CPU 500_000  ≈ 40 min   →   GPU 2_000_000  ≈ 15 min  (mismo resultado)
# TIMESTEPS = 500_000    # Si usas solo CPU
TIMESTEPS   = 20_000_000 #pasos totales que hara la IA

EVAL_FREQ       = 50_000  # Evaluar cada N pasos
N_EVAL_EPISODES = 15
EVAL_SPLIT      = 0.20    # 20% reservado para evaluación

# ── Optuna (solo Modo D) ───────────────────────────────────────────────────
OPTUNA_TRIALS      = 50 #cantidad de entrenamiento, cada uno es diferente
OPTUNA_STEPS_TRIAL = 2_000_000 #cantidad de pasos que dara en cada entrenamiento

# ── Config manual (opcional — None = auto según GPU/CPU) ──────────────────
# CUSTOM_CONFIG = None

policy        = "MlpPolicy"
n_steps       = 8192
batch_size    = 8192
n_epochs      = 10
learning_rate = 2.5e-4