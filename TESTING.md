# Testing Guide - 1-5 Day Trial Period

## Overview

This guide explains how to test the F&O Neural Network Predictor for a short 1-5 day period to verify the system is ready for production training.

## Quick Start

```bash
# 1. Run readiness tests
python test_readiness.py --quick

# 2. Train on 5-day test period
python train_test_period.py --days 5

# 3. Start the dashboard
./startup.sh

# 4. Access the dashboard
# Open browser: http://localhost:5173
```

## Prerequisites

### System Requirements
- Python 3.11+
- Node.js 20+
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

### Data Requirements
- Historical F&O data (2020-2026)
- India VIX data
- FII/DII flow data
- Global market data

## Testing Workflow

### Step 1: Environment Verification

Run the comprehensive readiness test:

```bash
# Quick test (2-3 minutes)
python test_readiness.py --quick

# Full test including training and build (10-15 minutes)
python test_readiness.py --all

# Backend only
python test_readiness.py --backend-only

# Frontend only
python test_readiness.py --frontend-only
```

**Expected Output:**
```
✓ PASS | Python Version
✓ PASS | Dependencies
✓ PASS | Data Availability
✓ PASS | Data Quality
✓ PASS | Module Imports
✓ PASS | DataCollector
✓ PASS | Feature Engineering
✓ PASS | Model Architecture
✓ PASS | Backend API
✓ PASS | Frontend Dependencies

Results: 10/10 tests passed
🎉 ALL TESTS PASSED - Project is ready for training!
```

### Step 2: Test Training (1-5 Days)

Train the model on a short period to verify the pipeline:

```bash
# Train on 5 days (recommended for initial test)
python train_test_period.py --days 5

# Train on 3 days (faster, less data)
python train_test_period.py --days 3

# Train with hyperparameter optimization (slower)
python train_test_period.py --days 5 --optimize

# Generate predictions only (skip training)
python train_test_period.py --predict-only
```

**What This Does:**
1. Collects last 30 days of historical data
2. Engineers 75+ technical features
3. Adds 100+ global market signals
4. Trains ensemble model (LightGBM + XGBoost)
5. Generates predictions for current date
6. Saves outputs to `output_test/`

**Expected Duration:**
- 5-day test without optimization: ~5-10 minutes
- 5-day test with optimization: ~15-20 minutes
- 3-day test: ~3-5 minutes

**Expected Output:**
```
Step 1: Collecting data...
✓ Collected 147 rows
  Symbols: ['NIFTY', 'BANKNIFTY']

Step 2: Engineering features...
✓ Features computed: 89 columns

Step 3: Adding global market features...
✓ Global features added: 127 columns

Step 4: Training model...
✓ Training complete!

Step 5: Generating test predictions...
✓ Generated 3 signals

🎉 TEST PERIOD TRAINING COMPLETE!
```

### Step 3: Start the Dashboard

Launch both backend API and frontend dashboard:

```bash
# Start both services
./startup.sh

# Or start individually
./startup.sh --backend-only   # Only FastAPI
./startup.sh --frontend-only  # Only React
```

**Services Started:**
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Frontend Dashboard: http://localhost:5173

**To Stop:**
```bash
./shutdown.sh
```

### Step 4: Verify Dashboard

Open browser and navigate to http://localhost:5173

**Pages to Check:**

1. **Signals** (`/signals`)
   - Should display today's top signals
   - Verify confidence scores are shown
   - Check signal history appears

2. **Training** (`/training`)
   - Should show training status
   - Loss curves should render (if trained)

3. **Data Pipeline** (`/data-pipeline`)
   - Verify all 8 data sources
   - Check freshness indicators
   - Confirm row counts are displayed

4. **Backtester** (`/backtester`)
   - Should display metrics (may be mock data initially)
   - Verify charts render correctly

5. **Drift Health** (`/drift-health`)
   - Model status should be visible
   - Check drift indicators

6. **Market Snapshot** (`/market-snapshot`)
   - Risk appetite gauge should display
   - Market heatmap should load

7. **Explainability** (`/explainability`)
   - SHAP feature importance chart

## Test Configurations

### Test Config vs Production Config

The `config_test.py` file contains reduced parameters for quick testing:

| Parameter | Production | Test | Reason |
|-----------|-----------|------|--------|
| MAX_EPOCHS | 200 | 5 | Quick validation |
| N_CV_FOLDS | 5 | 2 | Faster training |
| LOOKBACK_WINDOW | 20 | 5 | Less history needed |
| LGBM_N_ESTIMATORS | 500 | 50 | Faster training |
| XGB_N_ESTIMATORS | 500 | 50 | Faster training |
| OPTUNA_N_TRIALS | 100 | 5 | Quick optimization |
| CAPITAL | 1,000,000 | 100,000 | Test amounts |
| MIN_OI_LOTS | 500 | 100 | More lenient filters |

### Switching Between Configs

```python
# Use test config
import config_test as config

# Use production config
import config
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
pip install -r requirements.txt
```

#### 2. Empty Dataset

**Problem:** `DataCollector returned an empty dataset`

**Solution:**
```bash
# Download required data
python data_downloader.py --all

# Or download specific data
python data_downloader.py --bhavcopy --vix --fii-dii
```

#### 3. CUDA/GPU Errors

**Problem:** CUDA not available or version mismatch

**Solution:**
```python
# Force CPU mode in config_test.py
TFT_TRAINER_KWARGS = {"accelerator": "cpu"}
```

#### 4. Memory Issues

**Problem:** Out of memory during training

**Solution:**
```python
# Reduce batch size in config_test.py
BATCH_SIZE = 16  # or 8
M1_BATCH_SIZE = 8
```

#### 5. Port Already in Use

**Problem:** `Port 8000 is already in use`

**Solution:**
```bash
# Stop existing services
./shutdown.sh

# Or kill manually
lsof -ti:8000 | xargs kill -9
lsof -ti:5173 | xargs kill -9
```

#### 6. Frontend Build Fails

**Problem:** Frontend dependencies missing

**Solution:**
```bash
cd dashboard/frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

#### 7. Node.js Not Found

**Problem:** `node: command not found`

**Solution:**
```bash
# Install Node.js
# macOS
brew install node

# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify
node --version
npm --version
```

## Validation Checklist

Use this checklist to ensure the system is ready:

### Backend Validation
- [ ] Python 3.11+ installed
- [ ] All dependencies install without errors
- [ ] Data files present and valid
- [ ] All modules import successfully
- [ ] DataCollector loads data
- [ ] FeatureEngineer computes features
- [ ] Model can be instantiated
- [ ] Training completes without errors
- [ ] Predictions are generated
- [ ] FastAPI starts and responds
- [ ] All API endpoints return 200

### Frontend Validation
- [ ] Node.js 20+ installed
- [ ] npm dependencies install
- [ ] Frontend builds without errors
- [ ] Vite dev server starts
- [ ] All pages load
- [ ] WebSocket connects to backend
- [ ] Charts render correctly
- [ ] Real-time updates work
- [ ] No console errors

### Integration Validation
- [ ] Backend can serve frontend requests
- [ ] CORS is configured correctly
- [ ] WebSocket channels work
- [ ] Data flows from backend to frontend
- [ ] Dashboard displays live data
- [ ] Signals page shows predictions
- [ ] Training status updates in real-time

## Next Steps

After successful testing:

1. **Review Test Results**
   ```bash
   # Check logs
   tail -f logs/train_test.log
   tail -f logs/backend.log
   tail -f logs/frontend.log

   # Review outputs
   ls -lh output_test/
   ```

2. **Full Training**
   ```bash
   # Train on full dataset (2020-2026)
   python main.py --mode train

   # Train with hyperparameter optimization
   python main.py --mode train --optimize
   ```

3. **Production Deployment**
   ```bash
   # Generate predictions
   python main.py --mode predict

   # Schedule daily predictions
   python main.py --mode schedule

   # Run full pipeline with drift detection
   python main.py --mode full --check-drift
   ```

4. **Monitoring**
   - Set up log monitoring
   - Configure alerting for drift detection
   - Monitor model performance
   - Track prediction accuracy

## Performance Benchmarks

Expected performance on different hardware:

### Test Training (5 days, no optimization)

| Hardware | Time | Notes |
|----------|------|-------|
| M1 Mac (8GB) | 5-7 min | Uses unified memory |
| M2 Mac (16GB) | 4-6 min | Faster than M1 |
| Intel i7 + 16GB | 8-10 min | CPU only |
| GPU (NVIDIA 3080) | 3-4 min | With CUDA |

### Full Training (2020-2026, with optimization)

| Hardware | Time | Notes |
|----------|------|-------|
| M1 Mac (8GB) | 4-6 hours | May swap to disk |
| M2 Mac (16GB) | 3-4 hours | Recommended |
| Intel i7 + 16GB | 6-8 hours | CPU only |
| GPU (NVIDIA 3080) | 2-3 hours | Optimal |

## Support

If you encounter issues:

1. Check logs in `logs/` directory
2. Run tests with `--verbose` flag
3. Review error messages carefully
4. Check data availability and quality
5. Verify Python and Node.js versions
6. Ensure all dependencies are installed

## Additional Resources

- [README.md](README.md) - Project overview
- [DATA_SETUP.md](DATA_SETUP.md) - Data download guide
- [QUICK_START.md](QUICK_START.md) - Quick reference
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Architecture details

---

**Last Updated:** 2026-03-20
