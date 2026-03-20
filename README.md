# F&O Neural Network Predictor

Production-grade Neural Network system for predicting optimal F&O instruments to trade near market close (2:30–3:25 PM IST) for maximum next-day returns in Indian derivatives markets (NSE F&O).

## 🎯 System Overview

This system uses state-of-the-art machine learning to predict F&O trading opportunities by combining:

- **Primary Model**: Temporal Fusion Transformer (TFT) with multi-head attention
- **Ensemble**: TFT + LightGBM + XGBoost + Logistic Regression with meta-learner stacking
- **175+ Features**: 75+ technical indicators + 100+ global market signals
- **Production-Ready**: Walk-forward validation, drift detection, automatic retraining

---

## 📊 Neural Network Data Flow & Training Pipeline

### Complete Data Pipeline (From Raw Data to Predictions)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    1. DATA COLLECTION LAYER                          │
│                     (data_collector.py)                              │
├─────────────────────────────────────────────────────────────────────┤
│ INPUT:                                                               │
│   • NSE F&O Bhavcopy (OHLCV + Open Interest)                        │
│     Source: data/bhavcopy/*.csv                                     │
│     Format: Daily NSE derivatives data                              │
│   • India VIX (Volatility Index)                                     │
│     Source: data/vix/india_vix.csv                                  │
│   • FII/DII Flows (Foreign/Domestic Institutional Investment)       │
│     Source: data/fii_dii/fii_dii_data.csv                           │
│                                                                       │
│ OUTPUT:                                                              │
│   • Unified DataFrame with columns:                                  │
│     [DATE, SYMBOL, INSTRUMENT, EXPIRY_DT, STRIKE_PR, OPTION_TYP,    │
│      OPEN, HIGH, LOW, CLOSE, SETTLE_PR, CONTRACTS, OPEN_INT,        │
│      CHG_IN_OI, VIX, FII_NET, DII_NET]                              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  2. GLOBAL MARKET DATA LAYER                         │
│                  (market_data_extended.py)                           │
├─────────────────────────────────────────────────────────────────────┤
│ DOWNLOADS 100+ GLOBAL SIGNALS via yfinance:                         │
│                                                                       │
│ • Commodities (5):                                                   │
│   Gold (GC=F), Silver (SI=F), Copper (HG=F),                        │
│   Crude Oil (CL=F), Natural Gas (NG=F)                              │
│                                                                       │
│ • Global Indices (7):                                                │
│   S&P 500 (^GSPC), NASDAQ (^IXIC), Nikkei (^N225),                 │
│   Hang Seng (^HSI), Shanghai (000001.SS), DAX (^GDAXI),            │
│   FTSE 100 (^FTSE)                                                   │
│                                                                       │
│ • Currencies (6):                                                    │
│   USD-INR (INR=X), EUR-USD (EURUSD=X), DXY (DX-Y.NYB),             │
│   JPY (JPY=X), CNY (CNY=X), AUD (AUD=X)                            │
│                                                                       │
│ • Bonds (4):                                                         │
│   US 10Y (^TNX), US 2Y (^IRX), US 30Y (^TYX),                      │
│   India 10Y (^NSEI)                                                  │
│                                                                       │
│ • Volatility Indices (5):                                            │
│   VIX (^VIX), VXN (^VXN), OVX (^OVX), GVZ (^GVZ), VVIX            │
│                                                                       │
│ • Cryptocurrency (2):                                                │
│   Bitcoin (BTC-USD), Ethereum (ETH-USD)                             │
│                                                                       │
│ • India Sector Indices (10):                                         │
│   IT, Bank, Auto, Metal, Pharma, FMCG, Energy, Realty, PSU, Media  │
│                                                                       │
│ OUTPUT:                                                              │
│   • DataFrame with global features aligned to trading dates          │
│   • Columns: [DATE, gold_close, spy_close, usd_inr, vix, ...]      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   3. FEATURE ENGINEERING LAYER                       │
│                   (feature_engineering.py)                           │
├─────────────────────────────────────────────────────────────────────┤
│ COMPUTES 75+ TECHNICAL FEATURES:                                    │
│                                                                       │
│ A. TREND INDICATORS (12 features):                                   │
│    • EMA (9, 21, 50, 200 periods)                                   │
│    • MACD (line, signal, histogram)                                 │
│    • ADX (Average Directional Index)                                │
│    • Supertrend                                                      │
│    • VWAP (Volume Weighted Average Price)                           │
│    • Ichimoku Cloud (Tenkan, Kijun, Senkou A/B)                    │
│    • Parabolic SAR                                                   │
│                                                                       │
│ B. MOMENTUM INDICATORS (14 features):                                │
│    • RSI (7, 14, 21 periods)                                        │
│    • Stochastic Oscillator (%K, %D)                                 │
│    • MFI (Money Flow Index)                                         │
│    • Williams %R                                                     │
│    • ROC (Rate of Change)                                           │
│    • CMO (Chande Momentum Oscillator)                               │
│    • DPO (Detrended Price Oscillator)                               │
│    • CCI (Commodity Channel Index)                                  │
│                                                                       │
│ C. VOLATILITY INDICATORS (12 features):                              │
│    • Bollinger Bands (upper, middle, lower, %B, bandwidth)         │
│    • Bollinger Band Squeeze                                         │
│    • Keltner Channels                                               │
│    • ATR (Average True Range)                                       │
│    • Donchian Channels                                              │
│    • NATR (Normalized ATR)                                          │
│    • Historical Volatility (10, 20, 60 day)                         │
│                                                                       │
│ D. VOLUME INDICATORS (8 features):                                   │
│    • OBV (On-Balance Volume)                                        │
│    • CMF (Chaikin Money Flow)                                       │
│    • AD Line (Accumulation/Distribution)                            │
│    • RVOL (Relative Volume)                                         │
│    • Volume Moving Averages (5, 20 day)                             │
│    • Volume Trend                                                    │
│                                                                       │
│ E. F&O SPECIFIC INDICATORS (20 features):                            │
│    • PCR (Put-Call Ratio) - 4 variants:                            │
│      - PCR OI (Open Interest based)                                 │
│      - PCR Volume                                                    │
│      - PCR Premium                                                   │
│      - PCR Weighted                                                  │
│    • Max Pain (strike with max option seller pain)                  │
│    • OI Buildup (long/short buildup detection)                      │
│    • OI Unwinding (position unwinding)                              │
│    • OI Short Covering                                              │
│    • GEX (Gamma Exposure)                                           │
│    • IV Rank (Implied Volatility Rank)                             │
│    • Futures Basis (futures premium over spot)                      │
│    • DTE (Days to Expiry)                                           │
│    • Strike Distance (% from ATM)                                   │
│                                                                       │
│ F. MACRO & SENTIMENT (9+ features):                                  │
│    • VIX Level & Change                                             │
│    • FII Net Flow & Rolling Averages                                │
│    • DII Net Flow & Rolling Averages                                │
│    • FII/DII Flow Divergence                                        │
│    • Risk Appetite Composite                                         │
│    • Global Market Correlations                                      │
│                                                                       │
│ TARGET LABEL CREATION:                                               │
│   • 3-Class Classification:                                          │
│     - DOWN (0): Next-day return < -0.5%                             │
│     - FLAT (1): Next-day return between -0.5% and +0.5%            │
│     - UP (2): Next-day return > +0.5%                               │
│   • Target column: "target" (0, 1, or 2)                            │
│                                                                       │
│ OUTPUT:                                                              │
│   • DataFrame with 175+ total features (75 technical + 100 global)  │
│   • All NaN values handled (forward/backward fill)                  │
│   • Features normalized where appropriate                            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     4. TRAINING PIPELINE LAYER                       │
│                     (training_pipeline.py)                           │
├─────────────────────────────────────────────────────────────────────┤
│ VALIDATION STRATEGY: Walk-Forward Cross-Validation                  │
│   • NO random train/test splits (preserves temporal order)          │
│   • Train Window: 504 trading days (~2 years)                       │
│   • Validation Window: 63 trading days (~1 quarter)                 │
│   • Number of Folds: 5                                              │
│   • Roll Forward: Shifts by validation window each fold             │
│                                                                       │
│   Example Timeline:                                                  │
│   Fold 1: Train[Day 1-504]    → Val[Day 505-567]                   │
│   Fold 2: Train[Day 64-567]   → Val[Day 568-630]                   │
│   Fold 3: Train[Day 127-630]  → Val[Day 631-693]                   │
│   ...and so on                                                       │
│                                                                       │
│ MODEL ARCHITECTURE:                                                  │
│                                                                       │
│ ┌───────────────────────────────────────────────────────────────┐  │
│ │           ENSEMBLE PREDICTOR (model_architecture.py)          │  │
│ │                                                                │  │
│ │  ┌─────────────────────────────────────────────────────────┐ │  │
│ │  │  Model 1: Temporal Fusion Transformer (TFT)             │ │  │
│ │  │  ───────────────────────────────────────────            │ │  │
│ │  │  • Architecture: Multi-horizon attention-based          │ │  │
│ │  │  • Input: 20-day lookback window (sequences)            │ │  │
│ │  │  • Hidden Size: 64                                       │ │  │
│ │  │  • LSTM Layers: 2                                        │ │  │
│ │  │  • Attention Heads: 4                                    │ │  │
│ │  │  • Dropout: 0.15                                         │ │  │
│ │  │                                                           │ │  │
│ │  │  Components:                                             │ │  │
│ │  │  1. Variable Selection Networks                          │ │  │
│ │  │     - Selects important static & dynamic features        │ │  │
│ │  │  2. Gated Residual Networks (GRN)                       │ │  │
│ │  │     - Non-linear feature processing                      │ │  │
│ │  │  3. Multi-head Self-Attention                           │ │  │
│ │  │     - Captures temporal dependencies                     │ │  │
│ │  │  4. Temporal Fusion Decoder                             │ │  │
│ │  │     - Combines features across time horizons            │ │  │
│ │  │                                                           │ │  │
│ │  │  Output: 3-class probabilities [P(DOWN), P(FLAT), P(UP)]│ │  │
│ │  └─────────────────────────────────────────────────────────┘ │  │
│ │                            ↓                                  │  │
│ │  ┌─────────────────────────────────────────────────────────┐ │  │
│ │  │  Model 2: LightGBM Classifier                           │ │  │
│ │  │  ─────────────────────────                              │ │  │
│ │  │  • Trees: 500                                            │ │  │
│ │  │  • Max Depth: 10                                         │ │  │
│ │  │  • Learning Rate: 0.05                                   │ │  │
│ │  │  • Num Leaves: 63                                        │ │  │
│ │  │  • Boosting: GBDT with histogram optimization           │ │  │
│ │  │                                                           │ │  │
│ │  │  Output: 3-class probabilities                           │ │  │
│ │  └─────────────────────────────────────────────────────────┘ │  │
│ │                            ↓                                  │  │
│ │  ┌─────────────────────────────────────────────────────────┐ │  │
│ │  │  Model 3: XGBoost Classifier                            │ │  │
│ │  │  ────────────────────────                               │ │  │
│ │  │  • Trees: 500                                            │ │  │
│ │  │  • Max Depth: 8                                          │ │  │
│ │  │  • Learning Rate: 0.05                                   │ │  │
│ │  │  • Subsample: 0.8                                        │ │  │
│ │  │  • Boosting: Gradient boosting                          │ │  │
│ │  │                                                           │ │  │
│ │  │  Output: 3-class probabilities                           │ │  │
│ │  └─────────────────────────────────────────────────────────┘ │  │
│ │                            ↓                                  │  │
│ │  ┌─────────────────────────────────────────────────────────┐ │  │
│ │  │  Model 4: Logistic Regression                           │ │  │
│ │  │  ─────────────────────────                              │ │  │
│ │  │  • Regularization: L2 (C=1.0)                           │ │  │
│ │  │  • Max Iterations: 1000                                  │ │  │
│ │  │  • Multi-class: softmax                                  │ │  │
│ │  │                                                           │ │  │
│ │  │  Output: 3-class probabilities                           │ │  │
│ │  └─────────────────────────────────────────────────────────┘ │  │
│ │                            ↓                                  │  │
│ │  ┌─────────────────────────────────────────────────────────┐ │  │
│ │  │       META-LEARNER (Stacking Layer)                     │ │  │
│ │  │       ────────────────────────────                      │ │  │
│ │  │  Input: Concatenated predictions from all 4 models      │ │  │
│ │  │         [TFT_probs, LGBM_probs, XGB_probs, LR_probs]    │ │  │
│ │  │         Shape: (n_samples, 12) — 4 models × 3 classes   │ │  │
│ │  │                                                           │ │  │
│ │  │  Model: Logistic Regression (default) or Ridge/Lasso    │ │  │
│ │  │                                                           │ │  │
│ │  │  Final Output: Ensemble probabilities [P(0), P(1), P(2)]│ │  │
│ │  └─────────────────────────────────────────────────────────┘ │  │
│ └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│ LOSS FUNCTION: Focal Loss (calibration.py)                         │
│   • Addresses class imbalance in UP/FLAT/DOWN predictions          │
│   • Formula: FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)              │
│   • α (alpha) = [0.25, 0.5, 0.25] — class weights for DOWN/FLAT/UP│
│   • γ (gamma) = 2.0 — focusing parameter (emphasizes hard examples)│
│   • Downweights easy examples, focuses on misclassified samples     │
│                                                                      │
│ HYPERPARAMETER OPTIMIZATION (Optional):                             │
│   • Framework: Optuna with Bayesian TPE sampler                     │
│   • Trials: 100                                                     │
│   • Parallel Jobs: 4                                                │
│   • Pruner: Median Pruner (early stops unpromising trials)         │
│   • Objective: Maximize Sharpe Ratio or F1 Score (macro)           │
│   • Search Space:                                                   │
│     - TFT hidden size: [32, 64, 128]                               │
│     - LGBM/XGB trees: [100, 500, 1000]                             │
│     - Learning rates: [1e-4, 1e-2]                                 │
│     - Dropout rates: [0.1, 0.3]                                    │
│                                                                      │
│ ADVANCED TRAINING TECHNIQUES:                                       │
│   1. Curriculum Learning:                                           │
│      Start with easy examples (high confidence), gradually add all  │
│   2. Snapshot Ensemble:                                             │
│      Save checkpoints during training, average for final prediction │
│   3. Adversarial Training (FGSM):                                   │
│      Add small perturbations to inputs for robustness              │
│                                                                      │
│ OUTPUT:                                                              │
│   • Trained EnsemblePredictor object (saved as .pkl)                │
│   • Saved to: models/ensemble_model.pkl                             │
│   • Model size: ~50-100 MB                                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      5. CALIBRATION LAYER                            │
│                      (calibration.py)                                │
├─────────────────────────────────────────────────────────────────────┤
│ PURPOSE: Convert raw model outputs to reliable confidence estimates │
│                                                                       │
│ A. TEMPERATURE SCALING:                                              │
│    • Problem: Neural networks often produce overconfident predictions│
│    • Solution: Learn temperature parameter T on validation set      │
│    • Formula: p_calibrated = softmax(logits / T)                    │
│    • Optimization: Minimize negative log-likelihood on val set      │
│    • Result: Probabilities match true frequencies (reliability)     │
│                                                                       │
│ B. CONFORMAL PREDICTION:                                             │
│    • Provides prediction intervals with coverage guarantees         │
│    • Methods available:                                              │
│      1. LAC (Least Ambiguous Set)                                   │
│      2. APS (Adaptive Prediction Sets)                              │
│      3. RAPS (Regularized APS) ← Default                            │
│    • Coverage: 90% (configurable via CONFORMAL_ALPHA = 0.1)        │
│    • How it works:                                                   │
│      - Compute non-conformity scores on calibration set             │
│      - Find quantile q such that P(score ≤ q) = 1 - α              │
│      - For new prediction, return set of classes with score ≤ q     │
│    • Guarantees: 90% of true labels will be in prediction set       │
│                                                                       │
│ OUTPUT:                                                              │
│   • Calibrated probabilities [P(DOWN), P(FLAT), P(UP)]             │
│   • Prediction intervals with 90% coverage                          │
│   • Saved: models/calibration_params.pkl                            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     6. PREDICTION PIPELINE                           │
│                  (prediction_pipeline.py)                            │
├─────────────────────────────────────────────────────────────────────┤
│ EXECUTION: Daily at 3:00 PM IST (configurable in config.py)        │
│                                                                       │
│ PROCESS:                                                             │
│   1. Collect latest market data (up to 3:25 PM IST)                │
│   2. Compute all 175+ features for today                            │
│   3. Load trained ensemble model                                    │
│   4. Generate predictions for all F&O instruments                   │
│   5. Apply liquidity filters:                                       │
│      • Minimum OI: 500 lots                                         │
│      • Minimum Volume: 200 contracts                                │
│      • Max Spread: 3%                                               │
│      • Min DTE: 2 days                                              │
│   6. Rank signals by confidence × expected return                   │
│   7. Select top N signals (default: 5)                              │
│                                                                       │
│ OUTPUT: signals_YYYYMMDD_HHMMSS.csv                                 │
│   Columns:                                                           │
│   • DATE, SYMBOL, INSTRUMENT, EXPIRY_DT, STRIKE_PR                 │
│   • direction (UP/FLAT/DOWN)                                        │
│   • confidence (0-1)                                                │
│   • confidence_pct (0-100%)                                         │
│   • pred_up_pct, pred_flat_pct, pred_down_pct                      │
│   • OPEN_INT, CONTRACTS, DTE                                        │
│   • liquidity_pass (TRUE/FALSE)                                     │
│                                                                       │
│ SCHEDULING:                                                          │
│   • Mode: python main.py --mode schedule                            │
│   • Runs in infinite loop                                           │
│   • Checks time every minute                                        │
│   • Executes at PREDICTOR_RUN_TIME (3:00 PM IST)                   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    7. POST-PROCESSING LAYERS                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ A. POSITION SIZING (position_sizing.py)                             │
│    ────────────────────────────────────                             │
│    Kelly Criterion Bet Sizing:                                      │
│      • Formula: f* = (p × b - q) / b                                │
│        where p = win probability (from model)                       │
│              b = odds (expected return / ATR)                       │
│              q = 1 - p (loss probability)                           │
│      • Fractional Kelly: 25% of full Kelly (reduces volatility)    │
│      • Volatility Targeting: 15% annual volatility                  │
│      • Drawdown Circuit Breaker: Stop if DD > 10%                   │
│                                                                       │
│    Position Calculation:                                             │
│      1. Compute Kelly fraction for signal                           │
│      2. Adjust for ATR-based risk (1.5 × ATR stop loss)            │
│      3. Apply volatility scaling                                    │
│      4. Cap at MAX_POSITION_PCT (20% of capital)                   │
│      5. Check circuit breaker                                       │
│      6. Convert to lots (F&O lot sizes)                            │
│                                                                       │
│ B. EXPLAINABILITY (explainability.py)                               │
│    ────────────────────────────                                     │
│    SHAP (SHapley Additive exPlanations):                            │
│      • Global Feature Importance:                                   │
│        - Identifies most influential features across all predictions│
│      • Local Explanations:                                          │
│        - Shows which features drove each specific prediction        │
│      • Background Size: 100 samples                                 │
│      • Computation: 500 samples for SHAP values                     │
│                                                                       │
│    TFT Attention Weights:                                           │
│      • Extracts multi-head attention weights from TFT               │
│      • Shows which time steps model focused on                      │
│      • Interpretable temporal importance                            │
│                                                                       │
│    Reasoning Tags:                                                   │
│      • Auto-generated explanations (e.g., "High VIX, Strong FII")  │
│      • Based on dominant SHAP features                              │
│                                                                       │
│ C. DRIFT DETECTION (drift_detection.py)                             │
│    ────────────────────────────────                                 │
│    Monitors model degradation in production:                        │
│                                                                       │
│    1. ADWIN (Adaptive Windowing):                                   │
│       • Detects changes in prediction accuracy                      │
│       • Sensitivity: ADWIN_DELTA = 0.002                            │
│       • Online algorithm (no retraining needed for detection)       │
│                                                                       │
│    2. Feature Distribution Shift:                                   │
│       • Kolmogorov-Smirnov test for each feature                    │
│       • Compares recent vs. training distribution                   │
│       • Threshold: p-value < 0.05                                   │
│                                                                       │
│    3. Performance Monitoring:                                        │
│       • Tracks: Accuracy, F1, Calibration Error, Sharpe            │
│       • Rolling window: 30 days                                     │
│       • Alerts if metrics drop below thresholds                     │
│                                                                       │
│    4. Auto-Retraining Triggers:                                     │
│       • Drift detected by ADWIN                                     │
│       • Feature shift detected (>10% features drifted)              │
│       • Performance drop (>20% decline in Sharpe)                   │
│       • Time-based: Every 7 days regardless                         │
│                                                                       │
│ OUTPUT: final_signals_YYYYMMDD_HHMMSS.csv                           │
│   All previous columns PLUS:                                        │
│   • position_size (₹)                                               │
│   • position_size_lots (number of lots)                             │
│   • reasoning_tags (comma-separated explanations)                   │
│   • shap_top_features (top 5 features)                              │
│   • drift_warning (TRUE/FALSE)                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download historical data (last 2 years)
python data_downloader.py --all
```

### 2. Train the Model

```bash
# Basic training with walk-forward validation
python main.py --mode train

# Training with hyperparameter optimization (recommended)
python main.py --mode train --optimize
```

**Training Time:**
- Basic: ~2-4 hours (depending on hardware)
- With optimization: ~8-12 hours (100 Optuna trials)

**What Happens During Training:**
1. Data collection from NSE F&O archives (2020-2026)
2. Global market data download (100+ signals)
3. Feature engineering (175+ features per sample)
4. Walk-forward cross-validation (5 folds)
5. Ensemble training (TFT, LightGBM, XGBoost, LogReg)
6. Calibration (temperature scaling + conformal prediction)
7. Model saving to `models/ensemble_model.pkl`

### 3. Generate Predictions

```bash
# One-time prediction
python main.py --mode predict

# Scheduled daily predictions (3:00 PM IST)
python main.py --mode schedule

# Full pipeline with position sizing
python main.py --mode full --check-drift
```

### 4. View Dashboard

```bash
# Start backend + frontend
./startup.sh

# Access at http://localhost:5173
```

---

## 📁 Project Structure

```
btst/
├── config.py                    # All hyperparameters and constants
├── main.py                      # Main orchestrator (4 modes)
│
├── DATA COLLECTION:
├── data_collector.py            # NSE F&O, VIX, FII/DII collection
├── data_downloader.py           # Automated data download script
├── market_data_extended.py      # 100+ global market signals (yfinance)
│
├── FEATURE ENGINEERING:
├── feature_engineering.py       # 75+ technical indicators
├── historical_loader.py         # Legacy data loading utilities
│
├── MODEL & TRAINING:
├── model_architecture.py        # TFT, Ensemble, Focal Loss
├── training_pipeline.py         # Walk-forward CV, Optuna, training loop
│
├── CALIBRATION & INFERENCE:
├── calibration.py               # Temperature scaling, conformal prediction
├── prediction_pipeline.py       # Daily prediction generation (3:00 PM IST)
│
├── RISK & EXPLAINABILITY:
├── position_sizing.py           # Kelly criterion, volatility targeting
├── explainability.py            # SHAP, attention weights
├── drift_detection.py           # ADWIN, feature drift, auto-retraining
│
├── UTILITIES:
├── requirements.txt             # Python dependencies
├── startup.sh                   # Start backend + frontend
├── shutdown.sh                  # Graceful shutdown
│
├── DATA:
├── data/
│   ├── bhavcopy/                # NSE F&O historical data (88 MB)
│   ├── vix/                     # India VIX CSV
│   ├── fii_dii/                 # FII/DII flows
│   └── extended/                # Global market signals (cached)
│
├── MODELS & OUTPUT:
├── models/                      # Saved trained models
├── output/                      # Daily signal CSVs
├── logs/                        # Execution logs
│
├── DASHBOARD:
└── dashboard/
    ├── backend/                 # FastAPI server (port 8000)
    └── frontend/                # React + Vite (port 5173)
```

---

## ⚙️ Configuration

All hyperparameters are in **`config.py`**. Key settings:

### Data & Symbols
```python
SYMBOLS = ["NIFTY", "BANKNIFTY"]      # F&O symbols to trade
TRAIN_START_DATE = "2020-01-01"       # Training data start
VAL_END_DATE = "2026-03-31"           # Validation data end
LOOKBACK_WINDOW = 20                  # Days of history per sample
```

### Model Architecture
```python
TFT_HIDDEN_SIZE = 64                  # TFT hidden layer size
TFT_LSTM_LAYERS = 2                   # Number of LSTM layers
TFT_ATTENTION_HEADS = 4               # Multi-head attention
TFT_DROPOUT = 0.15                    # Dropout rate

LGBM_N_ESTIMATORS = 500               # LightGBM trees
XGB_N_ESTIMATORS = 500                # XGBoost trees
ENSEMBLE_MODELS = ["tft", "lgbm", "xgb", "logreg"]
```

### Training
```python
BATCH_SIZE = 64                       # Training batch size
MAX_EPOCHS = 200                      # Max training epochs
LEARNING_RATE = 1e-3                  # Adam learning rate
TRAIN_WINDOW_DAYS = 504               # 2 years per fold
VAL_WINDOW_DAYS = 63                  # 1 quarter per fold
N_CV_FOLDS = 5                        # Cross-validation folds
```

### Risk Management
```python
CAPITAL = 1_000_000                   # ₹10 lakh trading capital
KELLY_FRACTION = 0.25                 # 25% of full Kelly
VOLATILITY_TARGET = 0.15              # 15% annual volatility
DRAWDOWN_CIRCUIT_BREAKER = 0.10       # Stop if DD > 10%
MAX_POSITION_PCT = 0.20               # Max 20% per trade
```

### Liquidity Filters
```python
MIN_OI_LOTS = 500                     # Minimum open interest
MIN_DAILY_VOLUME = 200                # Minimum contracts/day
MAX_SPREAD_PCT = 0.03                 # Max 3% bid-ask spread
MIN_DTE = 2                           # Min 2 days to expiry
```

### Prediction & Scheduling
```python
PREDICTOR_RUN_TIME = "15:00"          # 3:00 PM IST
TOP_N_SIGNALS = 5                     # Top signals per day
MIN_CONFIDENCE = 0.55                 # Min 55% confidence
```

---

## 📊 Output Format

**File:** `output/final_signals_YYYYMMDD_HHMMSS.csv`

| Column | Description |
|--------|-------------|
| `DATE` | Trading date |
| `SYMBOL` | Underlying (NIFTY, BANKNIFTY) |
| `INSTRUMENT` | Full instrument name |
| `EXPIRY_DT` | Contract expiry date |
| `STRIKE_PR` | Strike price (for options) |
| `OPTION_TYP` | CE/PE/FUT |
| `CLOSE` | Closing price |
| `direction` | Predicted direction: **UP** / **FLAT** / **DOWN** |
| `confidence` | Model confidence (0-1) |
| `confidence_pct` | Confidence percentage (0-100%) |
| `pred_up_pct` | Probability of UP (%) |
| `pred_flat_pct` | Probability of FLAT (%) |
| `pred_down_pct` | Probability of DOWN (%) |
| `OPEN_INT` | Open interest (contracts) |
| `CONTRACTS` | Volume (contracts traded) |
| `DTE` | Days to expiry |
| `liquidity_pass` | Passed liquidity filters (TRUE/FALSE) |
| `position_size` | Recommended position size (₹) |
| `position_size_lots` | Position size in lots |
| `reasoning_tags` | Explanation (e.g., "High FII inflow, Strong momentum") |
| `shap_top_features` | Top 5 influential features |

**Example Row:**
```csv
2026-03-20,NIFTY,NIFTY 20MAR2026 FUT,2026-03-20,,FUT,22500,UP,0.78,78,12,10,78,850000,12500,6,TRUE,180000,4,"Strong FII inflow,High momentum,Low VIX","rsi_14,ema_21,fii_net,vix_level,pcr_oi"
```

---

## 🔬 Key Features Explained

### 1. Walk-Forward Validation (Zero Look-Ahead Bias)

Traditional train/test splits can leak future information. We use **walk-forward validation**:

```
Timeline:
─────────────────────────────────────────────────────────────
[----Train 1 (504d)----][Val 1 (63d)]
                    [----Train 2 (504d)----][Val 2 (63d)]
                                        [----Train 3 (504d)----][Val 3 (63d)]
```

- Train on 2 years, validate on next quarter
- NO random shuffling (preserves temporal order)
- Roll forward by validation window
- Mimics real-world deployment

### 2. Focal Loss for Class Imbalance

F&O markets have imbalanced outcomes (fewer extreme moves). **Focal Loss** addresses this:

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

- **α = [0.25, 0.5, 0.25]**: Weights for DOWN/FLAT/UP classes
- **γ = 2.0**: Focusing parameter
  - γ = 0 → Standard cross-entropy
  - γ > 0 → Focuses on hard examples

**Effect:** Model learns to predict minority classes (UP/DOWN) better.

### 3. Temperature Scaling (Calibration)

Neural networks are often **overconfident**. Temperature scaling fixes this:

```python
# Uncalibrated
p_raw = softmax(logits)

# Calibrated
p_calibrated = softmax(logits / T)
```

- **T < 1**: Sharpens distribution (more confident)
- **T > 1**: Smooths distribution (less confident)
- **T = 1**: No change

Learn optimal T on validation set to match predicted probabilities with true frequencies.

### 4. Conformal Prediction (Uncertainty Quantification)

Provides **prediction intervals with guaranteed coverage**:

```python
# 90% coverage guarantee
conformal_alpha = 0.1

# Output: Set of classes that contain true label 90% of the time
prediction_set = conformal_predictor.predict(x_new)
```

**Methods:**
- **LAC**: Least Ambiguous Set
- **APS**: Adaptive Prediction Sets
- **RAPS**: Regularized APS (default)

### 5. Kelly Criterion Position Sizing

Optimal bet sizing based on edge:

```
f* = (p × b - q) / b

where:
  p = probability of winning (from model)
  b = odds (expected_return / risk)
  q = 1 - p
```

**Fractional Kelly (25%)**: Reduces volatility by using 1/4 of full Kelly.

**Example:**
- Model says 70% chance of 2% return (ATR = 1%)
- b = 2 (2% return / 1% risk)
- f* = (0.7 × 2 - 0.3) / 2 = 0.55 (55% of capital)
- Fractional Kelly: 0.55 × 0.25 = 13.75% allocation

### 6. SHAP Explainability

**SHAP (SHapley Additive exPlanations)** explains predictions:

- **Global Importance**: Which features matter most overall?
- **Local Explanation**: Why did model predict UP for NIFTY today?

**Output:**
```
Top 5 Features for NIFTY UP prediction:
  1. rsi_14 = 68 (bullish momentum) → +0.15 contribution
  2. fii_net = +2000 Cr (strong buying) → +0.12 contribution
  3. vix_level = 12 (low fear) → +0.08 contribution
  4. ema_21_cross = 1 (bullish crossover) → +0.07 contribution
  5. pcr_oi = 0.8 (put writing) → +0.05 contribution
```

### 7. Drift Detection & Auto-Retraining

Markets change. Model must adapt.

**Drift Detection Methods:**
1. **ADWIN**: Detects changes in accuracy over time
2. **KS Test**: Detects feature distribution shifts
3. **Performance Monitoring**: Tracks Sharpe, F1, calibration

**Retraining Triggers:**
- Drift detected (ADWIN alarm)
- >10% of features shifted (KS test)
- Sharpe ratio drops >20%
- Time-based: Every 7 days

---

## 🧪 Testing & Validation

### Unit Tests
Located in `tests/unit/`:
- `test_model_architecture.py`: TFT, ensemble, focal loss
- `test_feature_engineering.py`: All 75+ indicators

```bash
pytest tests/unit/
```

### Module Tests
Each module has a `if __name__ == '__main__'` test block:

```bash
python model_architecture.py       # Test TFT + ensemble
python feature_engineering.py      # Test indicators
python calibration.py              # Test temperature scaling
python position_sizing.py          # Test Kelly criterion
```

### Backtesting

```bash
python main.py --mode train --start-date 2022-01-01 --end-date 2024-12-31
python main.py --mode full --check-drift
```

Evaluate on out-of-sample data (2025+).

---

## 📈 Performance Metrics

The system tracks:

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | Overall prediction accuracy | >55% |
| **F1 Score (Macro)** | Accounts for class imbalance | >0.50 |
| **Sharpe Ratio** | Risk-adjusted returns | >1.5 |
| **Max Drawdown** | Peak-to-trough decline | <15% |
| **Win Rate** | % profitable trades | >60% |
| **Calibration Error** | Negative log-likelihood | <1.0 |
| **Coverage** | Conformal prediction coverage | 90% ±2% |

---

## 🚨 Production Constraints

### 1. Zero Look-Ahead Bias
- Features computed only from data available ≤ 3:25 PM day T
- Walk-forward validation only
- NO random train/test splits
- NO future information leakage

### 2. Liquidity Filters
- Minimum OI: 500 lots
- Minimum volume: 200 contracts/day
- Max spread: 3%
- Min DTE: 2 days

### 3. Risk Controls
- Max position: 20% of capital
- Stop-loss: 1.5 × ATR
- Circuit breaker: 10% max drawdown
- No leverage

---

## 🔧 Troubleshooting

### CUDA/GPU Issues
TFT can use GPU but falls back to CPU:
```python
# In training_pipeline.py, force CPU:
gpus = 0
```

### Memory Issues
Reduce batch size:
```python
# In config.py
BATCH_SIZE = 32  # Default: 64
```

### Import Errors
```bash
pip install -r requirements.txt --upgrade
```

### Data Download Fails
```bash
# Manual download
python data_downloader.py --bhavcopy --start-date 2020-01-01
python data_downloader.py --vix
python data_downloader.py --fii-dii
```

---

## 📚 References

**Academic Papers:**
- Temporal Fusion Transformer: [Lim et al. (2021)](https://arxiv.org/abs/1912.09363)
- Focal Loss: [Lin et al. (2017)](https://arxiv.org/abs/1708.02002)
- SHAP: [Lundberg & Lee (2017)](https://arxiv.org/abs/1705.07874)
- Conformal Prediction: Vovk et al. (2005)
- ADWIN: Bifet & Gavaldà (2007)
- Kelly Criterion: Kelly (1956)

**Libraries:**
- [pytorch-forecasting](https://pytorch-forecasting.readthedocs.io/) (TFT)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)
- [SHAP](https://shap.readthedocs.io/)
- [Optuna](https://optuna.org/)

---

## 📝 License

This is a production-grade quantitative trading system. Use at your own risk. **Not financial advice.**

---

## 🤝 Contributing

Suggested improvements:
1. Add more alternative data (satellite, news sentiment)
2. Multi-asset portfolio optimization
3. Real-time streaming data pipeline
4. Broker API integration for execution
5. Monitoring dashboard (Grafana/Plotly Dash)

---

**Built for production. Ready to trade. 🚀**
