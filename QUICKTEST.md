# Quick Start - Training Verification

## For Users Who Want to Quickly Test the System

This is the fastest path to verify the F&O Neural Network Predictor works correctly.

## 5-Minute Quick Test

```bash
# 1. Install dependencies (may take 5-10 minutes)
pip install -r requirements.txt

# 2. Run quick tests (2-3 minutes)
python test_readiness.py --quick

# 3. If all tests pass, you're ready!
```

## 30-Minute Full Test

```bash
# 1. Install all dependencies
pip install -r requirements.txt
cd dashboard/frontend && npm install && cd ../..

# 2. Run comprehensive tests (10-15 minutes)
python test_readiness.py --all

# 3. Train on 5-day period (5-10 minutes)
python train_test_period.py --days 5

# 4. Start dashboard
./startup.sh

# 5. Open browser: http://localhost:5173
```

## What Gets Tested

### Quick Test (`--quick` flag)
✓ Python version
✓ Critical dependencies
✓ Data availability
✓ Data quality
✓ Module imports
✓ Data collection
✓ Feature engineering
✓ Model architecture
✓ Backend API
✓ Frontend dependencies

### Full Test (no flags)
Everything in Quick Test, plus:
✓ Quick training run (2 epochs)
✓ Frontend build process

## Expected Results

### All Tests Pass
```
Results: 10/10 tests passed
🎉 ALL TESTS PASSED - Project is ready for training!
```

**Next Step:** Run `python train_test_period.py --days 5`

### Some Tests Fail

Most common issue: Dependencies not installed

**Solution:**
```bash
pip install -r requirements.txt
```

Second most common: Data not available

**Solution:**
```bash
python data_downloader.py --all
```

## Training Test Output

When you run `python train_test_period.py --days 5`, expect:

```
Step 1: Collecting data...
✓ Collected 147 rows

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

**Duration:** 5-10 minutes on modern hardware

## Dashboard Access

After running `./startup.sh`:

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

**To Stop:** Run `./shutdown.sh`

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| Import errors | `pip install -r requirements.txt` |
| No data | `python data_downloader.py --all` |
| Port in use | `./shutdown.sh` then retry |
| CUDA errors | Add `accelerator: "cpu"` to config |
| Memory issues | Reduce `BATCH_SIZE` in config_test.py |
| Node.js missing | Install from nodejs.org |

## What's Next?

After successful testing:

1. **Review Results**
   - Check `output_test/` for predictions
   - Review `logs/train_test.log` for details

2. **Full Training**
   ```bash
   python main.py --mode train
   ```

3. **Production Use**
   ```bash
   python main.py --mode predict  # Generate signals
   python main.py --mode schedule # Daily automation
   ```

## Time Requirements

| Task | Time | What It Does |
|------|------|--------------|
| Install Python deps | 5-10 min | One-time setup |
| Install Node deps | 2-3 min | One-time setup |
| Quick test | 2-3 min | Verify system ready |
| Full test | 10-15 min | Complete verification |
| 5-day training | 5-10 min | Test ML pipeline |
| Dashboard startup | 30 sec | Launch services |

**Total first-time setup:** ~30-40 minutes

## Need More Details?

- **Comprehensive Guide:** [TESTING.md](TESTING.md)
- **Project Overview:** [README.md](README.md)
- **Data Setup:** [DATA_SETUP.md](DATA_SETUP.md)
- **Production Readiness:** [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md)

---

**TL;DR:** Run `python test_readiness.py --quick` to verify everything works.
