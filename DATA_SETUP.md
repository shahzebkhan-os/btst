# Data Setup Guide

This guide explains how to download and prepare all required data for the F&O Neural Network Predictor.

## Overview

The system requires the following data sources:

1. **NSE F&O Bhavcopy** - Historical OHLCV + Open Interest data
2. **India VIX** - Volatility index data
3. **FII/DII Flows** - Foreign and Domestic Institutional Investor flow data
4. **Global Signals** - Automatically fetched via yfinance (no manual setup required)

## Quick Start

### Install Dependencies

First, install all required dependencies:

```bash
pip install -r requirements.txt
```

### Download All Data (Recommended)

To download all required data for the last 2 years:

```bash
python data_downloader.py --all
```

This will:
- Create all necessary data directories
- Download NSE F&O Bhavcopy files to `data/bhavcopy/`
- Download India VIX data to `data/vix/india_vix.csv`
- Download FII/DII flows data to `data/fii_dii/fii_dii_data.csv`
- Display information about global signals (auto-fetched)

## Detailed Usage

### Command-Line Options

```
python data_downloader.py [OPTIONS]

Options:
  --all                 Download all data types
  --bhavcopy            Download NSE F&O Bhavcopy files only
  --vix                 Download India VIX data only
  --fii-dii             Download FII/DII flows data only
  --global-info         Display info about global signals
  --start-date DATE     Start date (YYYY-MM-DD format, default: 2 years ago)
  --end-date DATE       End date (YYYY-MM-DD format, default: today)
  --force-refresh       Re-download even if data exists
  --verbose             Enable verbose logging
```

### Examples

**Download all data for a specific date range:**
```bash
python data_downloader.py --all --start-date 2022-01-01 --end-date 2024-12-31
```

**Download only Bhavcopy and VIX data:**
```bash
python data_downloader.py --bhavcopy --vix
```

**Force refresh existing data:**
```bash
python data_downloader.py --all --force-refresh
```

**Download with verbose logging:**
```bash
python data_downloader.py --all --verbose
```

## Data Sources Details

### 1. NSE F&O Bhavcopy

**What:** Historical OHLCV (Open, High, Low, Close, Volume) and Open Interest data for all F&O contracts

**Storage:** `data/bhavcopy/`

**Format:** CSV files (one per trading day)

**Download Methods:**
- Primary: `jugaad_data` library (for dates before July 2024)
- Fallback: Direct NSE archive downloads
- Historical: `historical_loader.py` module

**Installation:**
```bash
pip install jugaad-data
```

**Manual Download (if needed):**
```bash
python data_downloader.py --bhavcopy --start-date 2023-01-01 --end-date 2024-12-31
```

### 2. India VIX

**What:** India Volatility Index (VIX) - measures market volatility expectations

**Storage:** `data/vix/india_vix.csv`

**Format:** CSV with columns: `DATE`, `VIX`

**Download Methods:**
- Primary: yfinance (`^INDIAVIX` ticker)
- Fallback: nsefin library
- Alternative: NSE direct API

**Installation:**
```bash
pip install yfinance nsefin
```

**Manual Download:**
```bash
python data_downloader.py --vix --start-date 2023-01-01 --end-date 2024-12-31
```

### 3. FII/DII Flows

**What:** Foreign Institutional Investor (FII) and Domestic Institutional Investor (DII) trading flow data

**Storage:** `data/fii_dii/fii_dii_data.csv`

**Format:** CSV with columns: `DATE`, `FII_BUY`, `FII_SELL`, `DII_BUY`, `DII_SELL`

**Download Methods:**
- Primary: nsefin library
- Fallback: NSE direct API
- Manual: NSE website (Participant Wise Open Interest section)

**Installation:**
```bash
pip install nsefin
```

**Manual Download:**
```bash
python data_downloader.py --fii-dii --start-date 2023-01-01 --end-date 2024-12-31
```

**Manual Alternative:**
If automatic download fails, you can manually download from:
- NSE Website: https://www.nseindia.com/
- Navigate to: Reports → Archives → FII/DII Data
- Download CSV files and place in `data/fii_dii/`

### 4. Global Signals

**What:** 100+ global market signals including:
- Global indices (S&P 500, Nasdaq, FTSE, DAX, Nikkei, etc.)
- Commodities (Gold, Silver, Brent, WTI, Copper, etc.)
- Currencies (DXY, USD-INR, EUR-INR, GBP-INR, etc.)
- Bonds & Rates (US 10Y, US 2Y yields, etc.)
- Volatility indices (VIX, VXN, OVX, GVZ)
- India sector indices (Nifty IT, Auto, FMCG, Bank, Metal, etc.)

**Storage:** `data/extended/` (cached parquet files)

**Format:** Parquet (6-hour cache TTL)

**Method:** Automatically fetched via yfinance when running training/prediction pipelines

**Installation:**
```bash
pip install yfinance
```

**Info:**
```bash
python data_downloader.py --global-info
```

**Note:** No manual download required! These signals are automatically fetched when you run:
```bash
python main.py --mode train
python main.py --mode predict
```

## Directory Structure

After running the data downloader, your directory structure should look like:

```
btst/
├── data/
│   ├── bhavcopy/           # NSE F&O Bhavcopy files
│   │   ├── fo01JAN2023bhav.csv
│   │   ├── fo02JAN2023bhav.csv
│   │   └── ...
│   ├── vix/
│   │   └── india_vix.csv   # India VIX time series
│   ├── fii_dii/
│   │   └── fii_dii_data.csv # FII/DII flows
│   └── extended/           # Global signals cache (auto-generated)
│       └── market_data_extended.parquet
├── models/                 # Trained models (auto-generated)
├── output/                 # Daily signal outputs (auto-generated)
└── logs/                   # Execution logs (auto-generated)
```

## Troubleshooting

### Issue: "nsefin not installed"

**Solution:**
```bash
pip install nsefin
```

### Issue: "jugaad_data missing"

**Solution:**
```bash
pip install jugaad-data
```

### Issue: "Failed to download bhavcopy"

**Causes:**
- Network connectivity issues
- NSE server rate limiting
- Date is a weekend or holiday

**Solutions:**
1. Check internet connection
2. Try smaller date ranges
3. Add delays between downloads (built-in)
4. Try manual download from NSE website

### Issue: "VIX download failed"

**Solutions:**
1. Install nsefin: `pip install nsefin`
2. Try yfinance alternative ticker: `^INDIA50VIX` or `INDIAVIX.NS`
3. Manually download from NSE website
4. Check network connectivity

### Issue: "FII/DII data not available"

**Solutions:**
1. Install nsefin: `pip install nsefin`
2. Manually download from NSE website
3. Check NSE API availability
4. Use placeholder data for testing (auto-generated)

### Issue: "Rate limiting / Too many requests"

**Solutions:**
1. The script has built-in rate limiting (1.5s between requests)
2. Download smaller date ranges
3. Wait and retry after some time
4. Use `--force-refresh` sparingly

## Data Quality Checks

After downloading data, verify the quality:

### Check Bhavcopy Files
```bash
ls -lh data/bhavcopy/ | head -10
```

Expected: Multiple CSV files, one per trading day

### Check India VIX
```bash
head data/vix/india_vix.csv
```

Expected output:
```
DATE,VIX
2023-01-02,12.45
2023-01-03,11.89
...
```

### Check FII/DII Flows
```bash
head data/fii_dii/fii_dii_data.csv
```

Expected output:
```
DATE,FII_BUY,FII_SELL,DII_BUY,DII_SELL
2023-01-02,5000.0,4500.0,3000.0,2800.0
...
```

## Next Steps

After downloading all data, you can:

### 1. Train the Model
```bash
python main.py --mode train
```

This will:
- Load all downloaded data
- Fetch global signals automatically
- Engineer features
- Train ensemble model
- Save trained model to `models/`

### 2. Generate Predictions
```bash
python main.py --mode predict
```

This will:
- Load trained model
- Fetch latest market data
- Generate trading signals
- Save signals to `output/signals_YYYYMMDD_HHMMSS.csv`

### 3. Run Scheduled Predictions
```bash
python main.py --mode schedule
```

This will run predictions daily at 3:00 PM IST automatically.

## Data Updates

### How often should I update data?

- **Bhavcopy:** Daily (after market close at 3:30 PM IST)
- **India VIX:** Daily
- **FII/DII Flows:** Daily
- **Global Signals:** Automatically updated (6-hour cache)

### Automated Data Updates

You can set up a cron job to download data daily:

```bash
# Edit crontab
crontab -e

# Add this line to download data at 4:00 PM IST daily (after market close)
0 16 * * 1-5 cd /path/to/btst && python data_downloader.py --all >> logs/data_download.log 2>&1
```

Or use the built-in scheduler:
```bash
# This will download data and run predictions daily
python main.py --mode schedule
```

## Optional Libraries

For enhanced functionality, install these optional libraries:

```bash
# For better data access
pip install nsefin nsepython jugaad-data

# For Google Trends sentiment
pip install pytrends

# For progress bars
pip install tqdm
```

## Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review logs in `logs/` directory
3. Enable verbose logging: `python data_downloader.py --all --verbose`
4. Check NSE website status: https://www.nseindia.com/
5. Verify internet connectivity

## References

- NSE India: https://www.nseindia.com/
- NSE Archives: https://www.nseindia.com/all-reports-derivatives
- jugaad-data: https://github.com/jugaad-py/jugaad-data
- nsefin: https://github.com/BennyThadikaran/NsePython
- yfinance: https://github.com/ranaroussi/yfinance
