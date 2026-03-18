# F&O Neural Network Predictor

Production-grade Neural Network system for predicting the best F&O instruments to trade near market close (2:30–3:25 PM IST) for maximum next-day return in Indian derivatives markets (NSE F&O).

## 🎯 Overview

This system combines state-of-the-art machine learning models with rigorous financial engineering to predict optimal F&O trading opportunities:

- **Primary Model**: Temporal Fusion Transformer (TFT) — purpose-built for multi-horizon financial time series
- **Ensemble**: TFT + LightGBM + XGBoost + Logistic Regression → Meta-learner stacking
- **75+ Features**: Trend, momentum, volatility, volume, F&O-specific, macro sentiment
- **100+ Global Signals**: Commodities, global indices, currencies, bonds, volatility indices, crypto
- **Production-Ready**: Walk-forward validation, drift detection, automatic retraining

## 🏗️ Architecture

### Models

1. **Temporal Fusion Transformer (TFT)**
   - Multi-head attention for interpretable feature importance
   - Variable selection networks for static and dynamic features
   - Gated residual networks for non-linear processing
   - Handles both static and time-varying features

2. **Ensemble Model**
   - Level 0: TFT, LightGBM, XGBoost, Logistic Regression
   - Level 1: Meta-learner (stacking)
   - Focal loss for class imbalance (UP/FLAT/DOWN)

3. **Calibration**
   - Temperature scaling for reliable confidence estimates
   - Conformal prediction for prediction intervals with coverage guarantees

### Features (75+)

**Trend** (12): EMA 9/21/50/200, MACD, ADX, Supertrend, VWAP, Ichimoku, PSAR

**Momentum** (14): RSI 7/14/21, Stochastic, MFI, Williams %R, ROC, CMO, DPO, CCI

**Volatility** (12): BB squeeze, Keltner, ATR, Donchian, NATR, historical vol 10/20/60d

**Volume** (8): OBV, CMF, AD line, RVOL, volume MA, volume trend

**F&O Specific** (20): PCR (4 variants), max pain, OI buildup/unwinding/short-cover, GEX, IV rank, futures basis, DTE

**Macro & Sentiment** (9+): VIX, FII/DII flows, global indices, USD-INR, risk appetite composite

### Global Market Signals (100+)

- **Commodities**: Gold, Silver, Copper, Crude, Natural Gas
- **Global Indices**: SPX, NDX, Nikkei, Hang Seng, Shanghai, DAX, FTSE
- **Currencies**: DXY, USD-INR, EUR-USD, JPY, CNY, AUD
- **Bonds**: US 10Y/2Y/30Y, India 10Y, yield curve
- **Volatility**: US VIX, VXN, OVX, GVZ, VVIX
- **Crypto**: BTC, ETH (risk-on signal)
- **India Sectors**: IT, Bank, Auto, Metal, Pharma, FMCG

## 📁 Project Structure

```
btst/
├── config.py                    # All constants and hyperparameters
├── data_collector.py            # NSE bhavcopy, option chain, VIX, FII/DII
├── market_data_extended.py      # 100+ global signals (yfinance)
├── feature_engineering.py       # 75+ technical features
├── model_architecture.py        # TFT, ensemble, focal loss
├── training_pipeline.py         # Walk-forward CV, Optuna optimization
├── calibration.py               # Temperature scaling, conformal prediction
├── prediction_pipeline.py       # Daily 3:00 PM IST prediction
├── position_sizing.py           # Kelly criterion, volatility targeting
├── explainability.py            # SHAP, attention weights
├── drift_detection.py           # ADWIN, feature drift, retraining
├── main.py                      # Main orchestrator
├── requirements.txt             # Dependencies
└── README.md                    # This file

data/
├── bhavcopy/                    # NSE F&O historical data
├── vix/                         # India VIX CSV
└── fii_dii/                     # FII/DII daily flows

models/                          # Saved trained models
output/                          # Daily signals CSV
logs/                            # Execution logs
```

## 🚀 Installation

### Requirements

- Python 3.11+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create Data Directories

```bash
mkdir -p data/bhavcopy data/vix data/fii_dii models output logs
```

## 📊 Data Sources

1. **NSE F&O Bhavcopy** (Historical OHLCV + OI)
   - Download from NSE website or use `nsefin`
   - Place in `data/bhavcopy/`

2. **India VIX**
   - Download from NSE
   - Place in `data/vix/india_vix.csv`

3. **FII/DII Flows**
   - Download from NSE or use `nsefin`
   - Place in `data/fii_dii/fii_dii_data.csv`

4. **Global Signals**
   - Automatically fetched via `yfinance`
   - No manual setup required

## 🎓 Usage

### 1. Train Model

Train ensemble model with walk-forward validation:

```bash
python main.py --mode train
```

Train with hyperparameter optimization (100 trials):

```bash
python main.py --mode train --optimize
```

### 2. Generate Predictions

Generate predictions for today:

```bash
python main.py --mode predict
```

Output: `output/signals_YYYYMMDD_HHMMSS.csv`

### 3. Scheduled Predictions

Run daily at 3:00 PM IST:

```bash
python main.py --mode schedule
```

This runs in an infinite loop and executes predictions at 3:00 PM IST every day.

### 4. Full Pipeline

Run complete pipeline with drift detection and position sizing:

```bash
python main.py --mode full --check-drift
```

Output: `output/final_signals_YYYYMMDD_HHMMSS.csv` with position sizes

## 📈 Output Format

Daily signals CSV contains:

| Column | Description |
|--------|-------------|
| `DATE` | Signal date |
| `SYMBOL` | Instrument symbol (NIFTY, BANKNIFTY) |
| `INSTRUMENT` | Full instrument name |
| `EXPIRY_DT` | Expiry date |
| `CLOSE` | Close price |
| `direction` | Predicted direction (UP/FLAT/DOWN) |
| `confidence` | Model confidence (0-1) |
| `confidence_pct` | Confidence percentage |
| `pred_up_pct` | UP probability % |
| `pred_down_pct` | DOWN probability % |
| `OPEN_INT` | Open interest |
| `CONTRACTS` | Volume (contracts) |
| `DTE` | Days to expiry |
| `liquidity_pass` | Liquidity filter pass/fail |
| `position_size` | Recommended position size (₹) |
| `position_size_lots` | Position size in lots |

## ⚙️ Configuration

All hyperparameters are in `config.py`:

### Model Architecture

```python
TFT_HIDDEN_SIZE = 64
TFT_LSTM_LAYERS = 2
TFT_ATTENTION_HEADS = 4
TFT_DROPOUT = 0.15
TFT_MAX_ENCODER_LENGTH = 20  # Lookback window
```

### Ensemble

```python
ENSEMBLE_MODELS = ["tft", "lgbm", "xgb", "logreg"]
LGBM_N_ESTIMATORS = 500
XGB_N_ESTIMATORS = 500
```

### Risk Management

```python
KELLY_FRACTION = 0.25              # 25% of full Kelly
VOLATILITY_TARGET = 0.15           # 15% annual volatility
DRAWDOWN_CIRCUIT_BREAKER = 0.10    # Stop if DD > 10%
```

### Liquidity Filters

```python
MIN_OI_LOTS = 500                  # Minimum OI (lots)
MIN_DAILY_VOLUME = 200             # Minimum contracts
MAX_SPREAD_PCT = 0.03              # Max 3% spread
MIN_DTE = 2                        # Min 2 days to expiry
```

## 🧪 Testing

Each module has a test block at the bottom. Run individual modules:

```bash
python model_architecture.py
python training_pipeline.py
python calibration.py
python position_sizing.py
python explainability.py
python drift_detection.py
```

## 🔬 Key Features

### 1. Walk-Forward Validation

- Train on 2 years (504 trading days)
- Validate on 1 quarter (63 trading days)
- Roll forward by validation window
- **NO random splits** — preserves temporal order

### 2. Optuna Hyperparameter Optimization

- Bayesian optimization with TPE sampler
- 100 trials, 4 parallel jobs
- Median pruner for early stopping
- Optimizes F1 score (macro)

### 3. Focal Loss

Handles class imbalance (UP/FLAT/DOWN):

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

- `alpha = [0.25, 0.5, 0.25]` — class weights
- `gamma = 2.0` — focusing parameter

### 4. Temperature Scaling

Calibrates softmax outputs:

```
p_calibrated = softmax(logits / T)
```

Learns optimal temperature T on validation set.

### 5. Conformal Prediction

Provides prediction intervals with guaranteed coverage (90%):

- LAC (Least Ambiguous Set)
- APS (Adaptive Prediction Sets)
- RAPS (Regularized APS)

### 6. Kelly Criterion Position Sizing

Optimal bet sizing:

```
f* = (p * b - q) / b
```

- `p` = win probability (from model)
- `b` = odds (expected return / risk)
- Uses fractional Kelly (25%) to reduce volatility

### 7. SHAP Explainability

- Global feature importance
- Local explanations per prediction
- Reasoning tags (e.g., "High VIX", "Strong momentum")

### 8. Drift Detection

- **ADWIN**: Detects changes in prediction accuracy
- **Feature Distribution**: KS test for covariate shift
- **Performance Monitoring**: Tracks accuracy, F1, calibration
- **Auto Retraining**: Triggers when drift detected or every 7 days

## 📊 Performance Metrics

The system tracks:

- **Accuracy**: Overall prediction accuracy
- **F1 Score**: Macro F1 (accounts for class imbalance)
- **Calibration Error**: Negative log-likelihood
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Peak-to-trough decline
- **Win Rate**: % of profitable trades

## 🚨 Critical Constraints

1. **Zero Look-Ahead Bias**
   - Features computed only from data available ≤ 3:25 PM day T
   - Walk-forward validation only
   - No random train/test splits

2. **Liquidity Filters**
   - Minimum OI: 500 lots
   - Minimum volume: 200 contracts
   - Max spread: 3%
   - DTE ≥ 2 days

3. **Production Quality**
   - Type hints on all functions
   - Comprehensive docstrings
   - Logging at every step
   - Try/except error handling
   - `if __name__ == '__main__'` test blocks

## 🔧 Troubleshooting

### Import Errors

If you get import errors for `nsefin` or `nsepython`:

```bash
pip install nsefin nsepython
```

### CUDA/GPU Issues

TFT training can use GPU. If CUDA not available, it will fall back to CPU:

```python
# In training_pipeline.py
gpus = 0  # Force CPU
```

### Memory Issues

For large datasets, reduce batch size in `config.py`:

```python
BATCH_SIZE = 32  # Default: 64
```

## 📝 License

This is a production-grade quantitative trading system. Use at your own risk. Not financial advice.

## 🤝 Contributing

This is a complete, production-ready system. Suggested improvements:

1. Add more alternative data sources (satellite imagery, news sentiment)
2. Implement multi-asset portfolio optimization
3. Add real-time streaming data pipeline
4. Integrate with broker APIs for automated execution
5. Build monitoring dashboard (Grafana, Plotly Dash)

## 📚 References

- **Temporal Fusion Transformer**: Lim et al. (2021) — https://arxiv.org/abs/1912.09363
- **Focal Loss**: Lin et al. (2017) — https://arxiv.org/abs/1708.02002
- **Conformal Prediction**: Vovk et al. (2005)
- **SHAP**: Lundberg & Lee (2017) — https://arxiv.org/abs/1705.07874
- **ADWIN**: Bifet & Gavaldà (2007)
- **Kelly Criterion**: Kelly (1956)

## 📧 Support

For issues, please check:
1. Logs in `logs/`
2. Error messages in console output
3. Module test blocks for debugging

---

**Built for production. Ready to trade.**
