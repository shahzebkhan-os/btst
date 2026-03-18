"""
config.py — All project-wide constants and hyperparameters.
Modify this file to tune the system without touching model code.
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

# ── Data ──────────────────────────────────────────────────────────────────────
SYMBOLS           = ["NIFTY", "BANKNIFTY"]
TRAIN_START_DATE  = "2020-01-01"
TRAIN_END_DATE    = "2023-12-31"
VAL_START_DATE    = "2024-01-01"
VAL_END_DATE      = "2024-12-31"

# ── Feature Engineering ───────────────────────────────────────────────────────
LOOKBACK_WINDOW   = 20     # trading days of history per sample
ATM_STRIKE_RANGE  = 10     # number of strikes above/below ATM for PCR
MIN_HISTORY_ROWS  = 252    # minimum rows per symbol before feature computation

# ── Model Architecture ────────────────────────────────────────────────────────
CNN_FILTERS_1     = 64
CNN_FILTERS_2     = 128
CNN_KERNEL        = 3
LSTM_UNITS_1      = 128
LSTM_UNITS_2      = 64
ATTENTION_HEADS   = 4
ATTENTION_KEY_DIM = 16
DENSE_UNITS       = 64
DROPOUT_CNN       = 0.20
DROPOUT_LSTM      = 0.30
DROPOUT_DENSE     = 0.20
NUM_CLASSES       = 3      # DOWN=0, FLAT=1, UP=2

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE        = 64
MAX_EPOCHS        = 200
LEARNING_RATE     = 1e-3
EARLY_STOP_PATIENCE = 20
LR_REDUCE_PATIENCE  = 8
LR_REDUCE_FACTOR    = 0.5
TRAIN_WINDOW_DAYS   = 504   # 2 years of trading days per fold
VAL_WINDOW_DAYS     = 63    # 1 quarter per fold
N_CV_FOLDS          = 5

# ── Risk & Execution ──────────────────────────────────────────────────────────
CAPITAL            = 1_000_000   # ₹ 10 lakh
MAX_POSITION_PCT   = 0.20        # max 20% capital per trade
ATR_STOP_MULT      = 1.5         # stop-loss = 1.5 × ATR from entry
BROKERAGE_PER_LEG  = 20          # ₹ 20 per order
SLIPPAGE_PCT       = 0.0005      # 0.05% slippage assumption
STT_PCT            = 0.0001      # 0.01% STT on sell side
MIN_OI_LOTS        = 500         # minimum OI (lots) to consider instrument
MIN_DAILY_VOLUME   = 200         # minimum contracts per day
MAX_SPREAD_PCT     = 0.03        # skip if (high-low)/close > 3%
MIN_DTE            = 2           # skip contracts expiring in < 2 days
RISK_FREE_RATE     = 0.065       # 6.5% for Sharpe calculation

# ── Signal Output ─────────────────────────────────────────────────────────────
TOP_N_SIGNALS      = 5           # top N instruments in daily output
MIN_CONFIDENCE     = 0.55        # minimum softmax probability to trade
DIRECTION_THRESHOLD = 0.005      # 0.5% return threshold for UP/DOWN label

# ── Scheduling ────────────────────────────────────────────────────────────────
PREDICTOR_RUN_TIME = "15:00"     # 3:00 PM IST
MARKET_CLOSE_TIME  = "15:30"     # 3:30 PM IST
TIMEZONE           = "Asia/Kolkata"
