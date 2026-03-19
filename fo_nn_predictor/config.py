"""
config.py
=========
Consolidated configuration for the F&O Neural Network project.
All constants, paths, and hyperparameters are defined here.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
DATA_DIR         = BASE_DIR / "data"
BHAVCOPY_DIR     = DATA_DIR / "bhavcopy"
VIX_FILE         = DATA_DIR / "vix" / "india_vix.csv"
FII_FILE         = DATA_DIR / "fii_dii" / "fii_dii_data.csv"
MODEL_DIR        = BASE_DIR / "models"
OUTPUT_DIR       = BASE_DIR / "output"
LOG_DIR          = BASE_DIR / "logs"

# ── Data & Training ───────────────────────────────────────────────────────────
SYMBOLS           = ["NIFTY", "BANKNIFTY"]
CAPITAL           = 1_000_000
RISK_FREE_RATE    = 0.065
LOOKBACK_WINDOW   = 20
N_CV_FOLDS        = 5
BATCH_SIZE        = 64
MAX_EPOCHS        = 100

# ── Feature Engineering ───────────────────────────────────────────────────────
EMA_FAST          = 12
EMA_SLOW          = 26
RSI_WINDOW        = 14
ATR_WINDOW        = 14

# ── TFT Model ─────────────────────────────────────────────────────────────────
TFT_HIDDEN_SIZE           = 128
TFT_LSTM_LAYERS           = 2
TFT_ATTENTION_HEADS       = 4
TFT_HIDDEN_CONTINUOUS     = 16
TFT_MAX_ENCODER_LENGTH    = 20
TFT_MAX_PREDICTION_LENGTH = 1
TFT_LEARNING_RATE         = 1e-3

# ── Ensemble Model ────────────────────────────────────────────────────────────
ENSEMBLE_MODELS           = ["tft", "lgbm", "xgb", "logreg"]
LGBM_LEARNING_RATE        = 0.05
XGB_LEARNING_RATE         = 0.05
TEMPERATURE_INIT          = 1.5

# ── Loss Functions ────────────────────────────────────────────────────────────
FOCAL_ALPHA               = 0.25
FOCAL_GAMMA               = 2.0

# ── Optimization ──────────────────────────────────────────────────────────────
OPTUNA_TRIALS             = 100
N_SNAPSHOTS               = 5

# ── Backtest & Risk ───────────────────────────────────────────────────────────
BROKERAGE_PER_LEG         = 20
SLIPPAGE_PCT              = 0.0005
STT_PCT                   = 0.0001
MIN_OI_LOTS               = 500
MIN_DAILY_VOLUME          = 200
MAX_SPREAD_PCT            = 0.03
MIN_DTE                   = 2
ATR_STOP_MULT             = 1.5
MAX_POSITION_PCT          = 0.20
KELLY_FRACTION            = 0.25
TARGET_VOL                = 0.15
CIRCUIT_BREAKER_DD        = 0.05

# ── Drift Monitoring ──────────────────────────────────────────────────────────
PSI_THRESHOLD             = 0.2
ADWIN_DELTA               = 0.002
RETRAIN_INTERVAL_DAYS     = 7

# ── Scheduling ────────────────────────────────────────────────────────────────
PREDICTOR_TIME            = "15:00"
TIMEZONE                  = "Asia/Kolkata"

# ── Signal Output ─────────────────────────────────────────────────────────────
TOP_N_SIGNALS             = 5
MIN_CONFIDENCE            = 0.55
