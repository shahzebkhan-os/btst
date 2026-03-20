"""
config_test.py — Test configuration for 1-5 day trial period
This config overrides the main config.py for quick testing purposes.
"""

from pathlib import Path
import datetime as dt

# ── Paths (same as main config) ───────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
DATA_DIR         = BASE_DIR / "data"
BHAVCOPY_DIR     = DATA_DIR / "bhavcopy"
VIX_FILE         = DATA_DIR / "vix" / "india_vix.csv"
FII_FILE         = DATA_DIR / "fii_dii" / "fii_dii_data.csv"
MODEL_DIR        = BASE_DIR / "models_test"
OUTPUT_DIR       = BASE_DIR / "output_test"
LOG_DIR          = BASE_DIR / "logs"

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

# ── Data (TEST MODE: Last 5 trading days only) ────────────────────────────────
SYMBOLS           = ["NIFTY", "BANKNIFTY"]

# For testing, use last 10 days of data (to ensure we have enough for features)
today = dt.datetime.now()
TEST_END_DATE     = today.strftime("%Y-%m-%d")
TEST_START_DATE   = (today - dt.timedelta(days=10)).strftime("%Y-%m-%d")

# For training test, use last 30 days as training and last 5 days as validation
TRAIN_START_DATE  = (today - dt.timedelta(days=30)).strftime("%Y-%m-%d")
TRAIN_END_DATE    = (today - dt.timedelta(days=5)).strftime("%Y-%m-%d")
VAL_START_DATE    = (today - dt.timedelta(days=5)).strftime("%Y-%m-%d")
VAL_END_DATE      = today.strftime("%Y-%m-%d")

# ── Feature Engineering ───────────────────────────────────────────────────────
LOOKBACK_WINDOW   = 5      # Reduced from 20 for testing
ATM_STRIKE_RANGE  = 5      # Reduced from 10 for testing
MIN_HISTORY_ROWS  = 10     # Reduced from 252 for testing

# ── Model Architecture (Reduced for quick testing) ────────────────────────────
CNN_FILTERS_1     = 32     # Reduced from 64
CNN_FILTERS_2     = 64     # Reduced from 128
CNN_KERNEL        = 3
LSTM_UNITS_1      = 64     # Reduced from 128
LSTM_UNITS_2      = 32     # Reduced from 64
ATTENTION_HEADS   = 2      # Reduced from 4
ATTENTION_KEY_DIM = 8      # Reduced from 16
DENSE_UNITS       = 32     # Reduced from 64
DROPOUT_CNN       = 0.20
DROPOUT_LSTM      = 0.30
DROPOUT_DENSE     = 0.20
NUM_CLASSES       = 3      # DOWN=0, FLAT=1, UP=2

# ── Training (Reduced for quick testing) ──────────────────────────────────────
BATCH_SIZE        = 32     # Reduced from 64
MAX_EPOCHS        = 5      # Reduced from 200 for quick test
LEARNING_RATE     = 1e-3
EARLY_STOP_PATIENCE = 3    # Reduced from 20
LR_REDUCE_PATIENCE  = 2    # Reduced from 8
LR_REDUCE_FACTOR    = 0.5
TRAIN_WINDOW_DAYS   = 20   # Reduced from 504 for testing
VAL_WINDOW_DAYS     = 5    # Reduced from 63 for testing
N_CV_FOLDS          = 2    # Reduced from 5 for testing

# ── M1/M2 Mac Optimization ────────────────────────────────────────────────────
M1_BATCH_SIZE           = 16    # Further reduced for testing
M1_GRADIENT_ACCUM_STEPS = 2
M1_NUM_WORKERS          = 2     # Reduced from 4

# ── Risk & Execution ──────────────────────────────────────────────────────────
CAPITAL            = 100_000    # ₹ 1 lakh for testing (reduced from 10 lakh)
MAX_POSITION_PCT   = 0.20
ATR_STOP_MULT      = 1.5
BROKERAGE_PER_LEG  = 20
SLIPPAGE_PCT       = 0.0005
STT_PCT            = 0.0001
MIN_OI_LOTS        = 100        # Reduced from 500 for testing
MIN_DAILY_VOLUME   = 50         # Reduced from 200 for testing
MAX_SPREAD_PCT     = 0.05       # Relaxed from 0.03 for testing
MIN_DTE            = 1          # Reduced from 2 for testing
RISK_FREE_RATE     = 0.065

# ── Signal Output ─────────────────────────────────────────────────────────────
TOP_N_SIGNALS      = 3          # Reduced from 5 for testing
MIN_CONFIDENCE     = 0.50       # Relaxed from 0.55 for testing
DIRECTION_THRESHOLD = 0.005

# ── Scheduling ────────────────────────────────────────────────────────────────
PREDICTOR_RUN_TIME = "15:00"
MARKET_CLOSE_TIME  = "15:30"
TIMEZONE           = "Asia/Kolkata"

# ── TFT (Temporal Fusion Transformer) - Reduced for testing ───────────────────
TFT_HIDDEN_SIZE           = 32        # Reduced from 64
TFT_LSTM_LAYERS           = 1         # Reduced from 2
TFT_ATTENTION_HEADS       = 2         # Reduced from 4
TFT_DROPOUT               = 0.15
TFT_HIDDEN_CONTINUOUS     = 8         # Reduced from 16
TFT_MAX_ENCODER_LENGTH    = 5         # Reduced from 20
TFT_MAX_PREDICTION_LENGTH = 1
TFT_LEARNING_RATE         = 1e-3
TFT_GRADIENT_CLIP_VAL     = 0.1
TFT_REDUCE_ON_PLATEAU_PATIENCE = 2    # Reduced from 4

# ── Ensemble Model (Reduced for testing) ──────────────────────────────────────
ENSEMBLE_MODELS = ["lgbm", "xgb"]    # Exclude TFT and logreg for quick test
LGBM_N_ESTIMATORS         = 50        # Reduced from 500
LGBM_MAX_DEPTH            = 5         # Reduced from 10
LGBM_LEARNING_RATE        = 0.1       # Increased for faster learning
LGBM_NUM_LEAVES           = 31        # Reduced from 63
XGB_N_ESTIMATORS          = 50        # Reduced from 500
XGB_MAX_DEPTH             = 5         # Reduced from 8
XGB_LEARNING_RATE         = 0.1       # Increased for faster learning
XGB_SUBSAMPLE             = 0.8
LOGREG_C                  = 1.0
LOGREG_MAX_ITER           = 500       # Reduced from 1000
META_LEARNER_TYPE         = "logreg"

# ── Focal Loss ────────────────────────────────────────────────────────────────
FOCAL_ALPHA               = [0.25, 0.5, 0.25]
FOCAL_GAMMA               = 2.0

# ── Optuna Hyperparameter Optimization (Reduced for testing) ──────────────────
OPTUNA_N_TRIALS           = 5         # Reduced from 100
OPTUNA_TIMEOUT            = 300       # 5 minutes timeout for testing
OPTUNA_N_JOBS             = 1         # Single job for testing
OPTUNA_PRUNER             = "median"
OPTUNA_SAMPLER            = "tpe"

# ── Calibration ───────────────────────────────────────────────────────────────
TEMP_SCALING_ENABLED      = True
CONFORMAL_ALPHA           = 0.1
CONFORMAL_METHOD          = "lac"

# ── Kelly Criterion & Position Sizing ─────────────────────────────────────────
KELLY_FRACTION            = 0.25
VOLATILITY_TARGET         = 0.15
DRAWDOWN_CIRCUIT_BREAKER  = 0.10
MAX_LEVERAGE              = 1.0

# ── Drift Detection & Retraining (Disabled for testing) ───────────────────────
ADWIN_DELTA               = 0.002
DRIFT_RETRAIN_DAYS        = 30        # Increased to avoid retraining during test
FEATURE_DRIFT_THRESHOLD   = 0.05

# ── Explainability (Reduced for testing) ──────────────────────────────────────
SHAP_BACKGROUND_SIZE      = 20        # Reduced from 100
SHAP_N_SAMPLES            = 50        # Reduced from 500
EXTRACT_ATTENTION_WEIGHTS = False     # Disabled for testing
USE_INTEGRATED_GRADIENTS  = False     # Disabled for testing

# ── Testing specific flags ────────────────────────────────────────────────────
IS_TEST_MODE              = True
VERBOSE                   = True
