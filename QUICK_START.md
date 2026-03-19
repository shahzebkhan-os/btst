# Quick Reference: Data Download Commands

## Prerequisites
```bash
# Install all dependencies
pip install -r requirements.txt

# Optional: Install specific data libraries for better access
pip install jugaad-data nsefin nsepython
```

## Main Data Download Commands

### Download All Data (Recommended)
```bash
# Download all data for last 2 years
python data_downloader.py --all

# Download all data with custom date range
python data_downloader.py --all --start-date 2022-01-01 --end-date 2024-12-31

# Force refresh (re-download even if exists)
python data_downloader.py --all --force-refresh
```

### Download Specific Data Types
```bash
# NSE F&O Bhavcopy only
python data_downloader.py --bhavcopy

# India VIX only
python data_downloader.py --vix

# FII/DII Flows only
python data_downloader.py --fii-dii

# Combine multiple types
python data_downloader.py --bhavcopy --vix --fii-dii
```

### Information Commands
```bash
# Display help
python data_downloader.py --help

# Show global signals info
python data_downloader.py --global-info

# Verbose logging
python data_downloader.py --all --verbose
```

## After Data Download

### Verify Data
```bash
# Check directory structure
ls -lh data/bhavcopy/ | head -10
head data/vix/india_vix.csv
head data/fii_dii/fii_dii_data.csv
```

### Train Model
```bash
# Basic training
python main.py --mode train

# Training with hyperparameter optimization
python main.py --mode train --optimize
```

### Generate Predictions
```bash
# Generate today's predictions
python main.py --mode predict

# Scheduled predictions (daily at 3:00 PM IST)
python main.py --mode schedule

# Full pipeline with drift detection
python main.py --mode full --check-drift
```

## Data Update Schedule

### Daily Updates (After Market Close)
```bash
# Update all data daily at 4:00 PM IST
python data_downloader.py --all --start-date $(date -d '7 days ago' +%Y-%m-%d) --end-date $(date +%Y-%m-%d)
```

### Cron Job Setup (Linux/Mac)
```bash
# Edit crontab
crontab -e

# Add this line for daily 4:00 PM IST data download
0 16 * * 1-5 cd /path/to/btst && python data_downloader.py --all >> logs/data_download.log 2>&1
```

## Troubleshooting Commands

### Check Python Environment
```bash
python --version
pip list | grep -E "(numpy|pandas|yfinance|nsefin|jugaad)"
```

### Check Directory Permissions
```bash
ls -la data/
chmod -R 755 data/
```

### Clean and Restart
```bash
# Remove existing data (careful!)
rm -rf data/bhavcopy/* data/vix/* data/fii_dii/*

# Re-download everything
python data_downloader.py --all --force-refresh
```

### Check Logs
```bash
# View recent logs
tail -50 logs/data_collector.log
tail -50 logs/main.log
```

## Advanced Usage

### Download Specific Date Range
```bash
# Download January 2024 only
python data_downloader.py --all --start-date 2024-01-01 --end-date 2024-01-31

# Download last 6 months
python data_downloader.py --all --start-date $(date -d '6 months ago' +%Y-%m-%d)

# Download from specific date to today
python data_downloader.py --all --start-date 2023-06-01
```

### Parallel Downloads (if needed)
```bash
# Download different types in parallel (in separate terminals)
python data_downloader.py --bhavcopy &
python data_downloader.py --vix &
python data_downloader.py --fii-dii &
wait
```

## Data Directory Structure
```
data/
├── bhavcopy/              # NSE F&O Bhavcopy files
│   ├── fo01JAN2023bhav.csv
│   ├── fo02JAN2023bhav.csv
│   └── ...
├── vix/
│   └── india_vix.csv      # India VIX time series
├── fii_dii/
│   └── fii_dii_data.csv   # FII/DII flows
└── extended/              # Global signals cache (auto-generated)
    └── market_data_extended.parquet
```

## File Sizes (Approximate)
- Bhavcopy (1 year): ~500 MB
- India VIX: ~5 MB
- FII/DII: ~2 MB
- Global Signals Cache: ~50 MB

## Useful Links
- NSE India: https://www.nseindia.com/
- NSE Archives: https://www.nseindia.com/all-reports-derivatives
- Full Documentation: See DATA_SETUP.md
