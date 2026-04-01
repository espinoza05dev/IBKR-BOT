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

"""
───────────────────── Download Historical Data & Train Model & Model factory ────────────────────────────
─────────────────────────────────────── Configuración ───────────────────────────────────
 en symbols puede ser multiples, este es solamente para Download Historical Data
"""
SYMBOLS = ["AAL"]
#─────────────────────

INTERVAL = "1h"  # "1m" "5m" "15m" "1h" "1d"
START = "2010-01-01"  # Fecha de inicio
SOURCE = "ibkr"  # "yfinance" | "ibkr" | "av" | "csv"

#─────────────────────────────Exclusivo para Model Factory──────────────────────────────────────────
START_TEST      = "2023-06-01"  # Los datos de test NUNCA se ven durante entrenamiento
END_TEST        = None          # None = hasta hoy
COMMISSION      = 0.001         # 0.1% por operación (IBKR)
# ── Metas ────────────────────────────────────────────────
TARGET_APPROVED = 20            # Cuántos modelos aptos quiero
MAX_ATTEMPTS    = 200           # Intentos máximos antes de rendirse
STOP_ON_TARGET  = True          # True = parar al llegar al objetivo
                                # False = seguir hasta MAX_ATTEMPTS

# approval
win_rate:        0.48   # ≥ 48%  (baja de 50% para acciones volátiles)
sharpe_ratio:    0.35   # ≥ 0.35 (Sharpe de mercado ~ 0.4-0.6)
sortino_ratio:   0.50   # ≥ 0.50
max_drawdown_pct: 25.0  # ≤ 25%  (AAL puede caer fuerte)
profit_factor:   1.10   # ≥ 1.10 (cada $1 perdido se ganan $1.10)
alpha_pct:       -5.0   # ≥ -5%  (permisivo: que no sea catastrófico)
n_trades:        8     # ≥ 8 trades para tener muestra estadística

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

#
#───────────────────────────────────────────────────────────────────────────────────────────────────

#────────────────────────Config especifica de Train Model────────────────────────

# ── GPU / paralelismo ──────────────────────────────────────────────────────
GPU_DEVICE  = "cuda"   # "auto" detecta CUDA→MPS→CPU automáticamente
                       # "cuda" fuerza NVIDIA GPU
                       # "mps"  fuerza Apple Silicon
                       # "cpu"  fuerza CPU (útil para debug)

N_ENVS      = 16        # 0 = auto según hardware
                       # GPU  RTX 30/40: usa 8
                       # Apple M-series: usa 4
                       # Solo CPU      : usa cantidad de núcleos

# ── Timesteps ─────────────────────────────────────────────────────────────
# CPU 500_000  ≈ 40 min   →   GPU 2_000_000  ≈ 15 min  (mismo resultado)
# TIMESTEPS = 500_000    # Si usas solo CPU

EVAL_FREQ       = 50_000  # Evaluar cada N pasos
N_EVAL_EPISODES = 15
EVAL_SPLIT      = 0.20    # 20% reservado para evaluación

MODE = "B"   # "A" | "B" | "C" | "D"
"""
    Modos:
    A → Pipeline completo (descarga + features + entrena)  ← primer uso
    B → Datos ya en disco
    C → Reanudar desde checkpoint
    D → Optimización de hiperparámetros (Optuna)
"""
# Optuna (solo Modo D)
OPTUNA_TRIALS      = 50 #cantidad de entrenamiento, cada uno es diferente
OPTUNA_STEPS_TRIAL = 2_000_000 #cantidad de pasos que dara en cada entrenamiento

# ── Config manual (opcional — None = auto según GPU/CPU) ──────────────────
# CUSTOM_CONFIG = None

policy="MlpPolicy"
n_steps       = 8192
batch_size    = 8192
n_epochs      = 10
learning_rate = 2.5e-4
#────────────────────────────────────────────────────────────────────────────────

""" 
── OPTIMIZACIÓN CRÍTICA PARA CPU/GPU ─────────────────────────────────────────
 Evita que la CPU pelee consigo misma intentando usar todos los hilos
 en cada entorno. Mantiene la CPU libre para alimentar a la GPU.
    el numero de hilos debe estar entre 2 - 4
"""

SET_THREAD_NUMBER = 3
#-----------------

TIMESTEPS   = 1_000_000 #pasos totales que hara la IA


#rutas de alojamiento de archivos tipo: logs,csv,html etc
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