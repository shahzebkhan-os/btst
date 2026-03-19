# F&O Neural Network Predictor (BTST)

Production-grade ensemble prediction system for Indian F&O markets, combining Temporal Fusion Transformers (TFT) with Gradient Boosting and advanced risk management.

## Overview
This system identifies high-probability Buy-Today-Sell-Tomorrow (BTST) opportunities in NIFTY and BANKNIFTY instruments by analyzing OHLCV, Open Interest dynamics, Market Regimes, and Global Macro signals.

### Architecture
```ascii
[ NSE / Yahoo / Google ] -> [ data_collector.py ] -> [ feature_engineering.py ]
                                     |                         |
                                     v                         v
[ Backtester ] <---------- [ ensemble.py ] <---------- [ trainer.py (Optuna) ]
      |                 (TFT + LGBM + XGB + LR)                |
      v                              |                         |
[ Signal Report ] <--------- [ predictor.py ] <--------- [ explainability.py ]
      |                        (3:00 PM IST)             (SHAP + Attention)
      v
[ Drift Monitor ] -> [ Auto-Retrain ]
```

## Quick Start (5 Steps)
1. **Clone & Setup**: `mkdir fo_nn_predictor && cd fo_nn_predictor`
2. **Install Deps**: `pip install -r requirements.txt`
3. **Configure**: `cp .env.example .env` (Add your Kite API keys if using live data)
4. **Train**: `python3 train.py --start-date 2022-01-01 --end-date 2024-12-31`
5. **Run Live**: `python3 src/predictor.py --run-now`

## Data Download Guide
- **NSE Bhavcopy**: [NSE Reports](https://www.nseindia.com/all-reports-derivatives) (Daily ZIPs)
- **India VIX**: [NSE VIX](https://www.nseindia.com/market-data/vix-indices)
- **FII/DII**: [Participant Wise OI](https://www.nseindia.com/all-reports-derivatives)

## Feature Reference
| Category | Count | Key Indicators |
|----------|-------|----------------|
| Price/Trend | 25+ | EMA, MACD, Supertrend, Ichimoku |
| Momentum | 15+ | RSI, Stochastic, MFI, Williams %R |
| Volatility | 10+ | BB, Keltner, ATR, NATR |
| F&O Specific | 30+ | PCR Dynamics, OI Flip, Max Pain Shift |
| Macro/Global | 10+ | India VIX, USD-INR, S&P 500, Yields |
| Regimes | 5 | Volatility Regime, Trend Regime |

## Model Architecture Decision
**Why TFT over CNN-LSTM?**
- **Variable Selection**: TFT's gated residual networks automatically prune irrelevant features per timestep.
- **Interpretability**: Native multi-head attention extraction allows us to see *when* the model is looking at specific indicators.
- **Quantile Loss**: Provides prediction intervals (p10, p90) crucial for risk management.

## Running the Predictor
The predictor is designed to run at 3:00 PM IST on trading days.
- **Manual**: `python3 src/predictor.py --run-now`
- **Cron**: `0 15 * * 1-5 /usr/bin/python3 /path/to/src/predictor.py`

## Output Format
Daily signals are saved to `output/signals_YYYYMMDD.csv` with:
- `direction`: 0=DOWN, 1=FLAT, 2=UP
- `confidence`: Probability score (0.0 - 1.0)
- `reasoning_tags`: Human-readable trigger explanations.
- `expected_return_p50`: Median outcome forecast.

## Risk Disclaimer
Trading in F&O involves high risk. This model is for research purposes only. Past performance does not guarantee future results.
