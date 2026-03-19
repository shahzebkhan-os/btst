"""
data_downloader.py
==================
Master script to download/load all required data for the F&O Neural Network Predictor.

This script handles:
  1. NSE F&O Bhavcopy (Historical OHLCV + OI) → data/bhavcopy/
  2. India VIX                                  → data/vix/india_vix.csv
  3. FII/DII Flows                              → data/fii_dii/fii_dii_data.csv
  4. Global Signals (via yfinance)              → Automatically fetched, no setup required

Usage:
    # Download all data (last 2 years)
    python data_downloader.py --all

    # Download specific data types
    python data_downloader.py --bhavcopy
    python data_downloader.py --vix
    python data_downloader.py --fii-dii

    # Specify date range (format: YYYY-MM-DD)
    python data_downloader.py --all --start-date 2022-01-01 --end-date 2024-12-31

    # Force refresh (re-download even if data exists)
    python data_downloader.py --all --force-refresh
"""

import os
import sys
import time
import logging
import argparse
import warnings
import datetime as dt
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests

warnings.filterwarnings("ignore")

# ── Configure logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("DataDownloader")

# ── Import project modules ────────────────────────────────────────────────────
try:
    from historical_loader import download_bhavcopy_range, download_spot_prices
    HISTORICAL_LOADER_AVAILABLE = True
except ImportError:
    HISTORICAL_LOADER_AVAILABLE = False
    logger.warning("historical_loader.py not found. Bhavcopy download may be limited.")

try:
    import nsefin
    NSEFIN_AVAILABLE = True
    logger.info("nsefin available ✓")
except ImportError:
    NSEFIN_AVAILABLE = False
    logger.warning("nsefin not installed. Run: pip install nsefin")

try:
    from nsepython import nsefetch
    NSEPYTHON_AVAILABLE = True
    logger.info("nsepython available ✓")
except ImportError:
    NSEPYTHON_AVAILABLE = False
    logger.warning("nsepython not installed. Run: pip install nsepython")

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
BHAVCOPY_DIR = DATA_DIR / "bhavcopy"
VIX_DIR = DATA_DIR / "vix"
FII_DII_DIR = DATA_DIR / "fii_dii"
EXTENDED_DIR = DATA_DIR / "extended"

VIX_FILE = VIX_DIR / "india_vix.csv"
FII_DII_FILE = FII_DII_DIR / "fii_dii_data.csv"

# NSE URLs and headers
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


# ═════════════════════════════════════════════════════════════════════════════
# DIRECTORY SETUP
# ═════════════════════════════════════════════════════════════════════════════

def setup_directories() -> None:
    """Create all required data directories."""
    logger.info("Setting up data directories...")

    dirs = [
        DATA_DIR,
        BHAVCOPY_DIR,
        VIX_DIR,
        FII_DII_DIR,
        EXTENDED_DIR,
        Path("logs"),
        Path("models"),
        Path("output"),
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.debug(f"  ✓ {d}")

    logger.info("✓ Directories created")


# ═════════════════════════════════════════════════════════════════════════════
# 1. NSE F&O BHAVCOPY DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════

def download_bhavcopy(
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
) -> bool:
    """
    Download NSE F&O Bhavcopy files for the specified date range.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        force_refresh: Re-download even if files exist

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("DOWNLOADING NSE F&O BHAVCOPY")
    logger.info("=" * 80)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Target directory: {BHAVCOPY_DIR}")

    if not HISTORICAL_LOADER_AVAILABLE:
        logger.error("historical_loader module not available. Cannot download bhavcopy.")
        logger.info("Attempting fallback method using jugaad_data...")

        try:
            from jugaad_data.nse import bhavcopy_fo_save

            # Parse dates
            start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
            end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()

            # Download each day
            current = start
            success_count = 0
            fail_count = 0

            while current <= end:
                # Skip weekends
                if current.weekday() >= 5:
                    current += dt.timedelta(days=1)
                    continue

                try:
                    logger.info(f"Downloading {current}...")
                    bhavcopy_fo_save(current, str(BHAVCOPY_DIR))
                    success_count += 1
                    time.sleep(1.5)  # Rate limiting
                except Exception as e:
                    logger.warning(f"  Failed for {current}: {e}")
                    fail_count += 1

                current += dt.timedelta(days=1)

            logger.info(f"✓ Bhavcopy download complete: {success_count} successful, {fail_count} failed")
            return success_count > 0

        except ImportError:
            logger.error("jugaad_data not installed. Run: pip install jugaad-data")
            return False

    try:
        # Use historical_loader module
        downloaded_files = download_bhavcopy_range(
            start_date_str=start_date,
            end_date_str=end_date,
            data_dir=str(BHAVCOPY_DIR),
        )

        logger.info(f"✓ Bhavcopy download complete: {len(downloaded_files)} files downloaded")
        return len(downloaded_files) > 0

    except Exception as e:
        logger.error(f"Failed to download bhavcopy: {e}", exc_info=True)
        return False


# ═════════════════════════════════════════════════════════════════════════════
# 2. INDIA VIX DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════

def download_india_vix(
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
) -> bool:
    """
    Download India VIX data and save to CSV.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        force_refresh: Re-download even if file exists

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("DOWNLOADING INDIA VIX")
    logger.info("=" * 80)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Target file: {VIX_FILE}")

    # Check if file exists and skip if not force refresh
    if VIX_FILE.exists() and not force_refresh:
        logger.info(f"India VIX file already exists. Use --force-refresh to re-download.")
        return True

    try:
        # Method 1: Try yfinance (India VIX ticker)
        logger.info("Attempting download via yfinance (^INDIAVIX)...")
        vix_ticker = "^INDIAVIX"

        vix_data = yf.download(
            vix_ticker,
            start=start_date,
            end=end_date,
            progress=False,
        )

        if vix_data.empty:
            logger.warning("yfinance returned empty data. Trying NSE direct API...")

            # Method 2: Try nsefin library
            if NSEFIN_AVAILABLE:
                logger.info("Attempting download via nsefin...")
                nse_client = nsefin.NSEClient()

                # Get historical VIX data
                # Note: nsefin may have limited historical data
                vix_data = nse_client.get_vix_data(start_date, end_date)

                if vix_data is not None and not vix_data.empty:
                    logger.info(f"✓ Downloaded {len(vix_data)} VIX records via nsefin")
                else:
                    logger.warning("nsefin returned no data")
                    return False
            else:
                logger.error("No method available to download India VIX")
                return False
        else:
            logger.info(f"✓ Downloaded {len(vix_data)} VIX records via yfinance")

        # Prepare data for saving
        if not vix_data.empty:
            # Reset index to have date as column
            vix_df = vix_data.reset_index()

            # Rename columns to standard format
            if 'Date' in vix_df.columns:
                vix_df.rename(columns={'Date': 'DATE'}, inplace=True)
            elif vix_df.index.name == 'Date':
                vix_df.reset_index(inplace=True)
                vix_df.rename(columns={'Date': 'DATE'}, inplace=True)

            # Keep only necessary columns
            cols_to_keep = ['DATE', 'Close']
            available_cols = [c for c in cols_to_keep if c in vix_df.columns]

            if 'Close' not in vix_df.columns and 'close' in vix_df.columns:
                vix_df.rename(columns={'close': 'Close'}, inplace=True)

            if 'Close' in vix_df.columns:
                vix_df.rename(columns={'Close': 'VIX'}, inplace=True)

            # Save to CSV
            vix_df.to_csv(VIX_FILE, index=False)
            logger.info(f"✓ India VIX saved to {VIX_FILE}")
            logger.info(f"  Columns: {list(vix_df.columns)}")
            logger.info(f"  Date range: {vix_df['DATE'].min()} to {vix_df['DATE'].max()}")
            return True
        else:
            logger.error("No VIX data available")
            return False

    except Exception as e:
        logger.error(f"Failed to download India VIX: {e}", exc_info=True)
        return False


# ═════════════════════════════════════════════════════════════════════════════
# 3. FII/DII FLOWS DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════

def download_fii_dii_flows(
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
) -> bool:
    """
    Download FII/DII flows data and save to CSV.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        force_refresh: Re-download even if file exists

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("DOWNLOADING FII/DII FLOWS")
    logger.info("=" * 80)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Target file: {FII_DII_FILE}")

    # Check if file exists and skip if not force refresh
    if FII_DII_FILE.exists() and not force_refresh:
        logger.info(f"FII/DII file already exists. Use --force-refresh to re-download.")
        return True

    try:
        if NSEFIN_AVAILABLE:
            logger.info("Attempting download via nsefin...")
            nse_client = nsefin.NSEClient()

            # Parse dates
            start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d")

            # Collect data day by day (NSE API limitation)
            all_data = []
            current_dt = start_dt

            while current_dt <= end_dt:
                # Skip weekends
                if current_dt.weekday() >= 5:
                    current_dt += dt.timedelta(days=1)
                    continue

                try:
                    date_str = current_dt.strftime("%d-%b-%Y")

                    # Get FII/DII data for the date
                    # Note: This is a placeholder - actual nsefin API may vary
                    # Check nsefin documentation for correct method

                    logger.debug(f"  Fetching {date_str}...")

                    # Placeholder for actual nsefin call
                    # data = nse_client.get_fii_dii_data(date_str)
                    # all_data.append(data)

                    time.sleep(1.0)  # Rate limiting

                except Exception as e:
                    logger.debug(f"  Failed for {date_str}: {e}")

                current_dt += dt.timedelta(days=1)

            if all_data:
                fii_dii_df = pd.DataFrame(all_data)
                fii_dii_df.to_csv(FII_DII_FILE, index=False)
                logger.info(f"✓ FII/DII data saved to {FII_DII_FILE}")
                return True
            else:
                logger.warning("No FII/DII data collected")

        else:
            logger.warning("nsefin not available. Trying alternative method...")

        # Alternative method: NSE direct API
        logger.info("Attempting download via NSE direct API...")

        # Create session with proper headers
        session = requests.Session()
        session.headers.update(NSE_HEADERS)

        # First, get the homepage to set cookies
        session.get("https://www.nseindia.com/", timeout=10)
        time.sleep(1)

        # FII/DII data endpoint
        # Note: NSE API structure may change; this is a common pattern
        url = "https://www.nseindia.com/api/fiidiiTradeReact"

        response = session.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Extract relevant data
            # The structure depends on NSE API response format
            if data:
                # Process and save data
                fii_dii_df = pd.DataFrame(data)
                fii_dii_df.to_csv(FII_DII_FILE, index=False)
                logger.info(f"✓ FII/DII data saved to {FII_DII_FILE}")
                return True
        else:
            logger.error(f"NSE API returned status code: {response.status_code}")

        # If all methods fail, create a placeholder file
        logger.warning("Creating placeholder FII/DII file...")
        placeholder_df = pd.DataFrame({
            'DATE': pd.date_range(start=start_date, end=end_date, freq='D'),
            'FII_BUY': 0.0,
            'FII_SELL': 0.0,
            'DII_BUY': 0.0,
            'DII_SELL': 0.0,
        })
        placeholder_df.to_csv(FII_DII_FILE, index=False)
        logger.warning(f"⚠ Placeholder FII/DII file created at {FII_DII_FILE}")
        logger.warning("  Please manually update with actual data or install nsefin")
        return False

    except Exception as e:
        logger.error(f"Failed to download FII/DII data: {e}", exc_info=True)
        return False


# ═════════════════════════════════════════════════════════════════════════════
# 4. GLOBAL SIGNALS (INFO ONLY)
# ═════════════════════════════════════════════════════════════════════════════

def info_global_signals() -> None:
    """Display information about global signals (automatically fetched)."""
    logger.info("=" * 80)
    logger.info("GLOBAL SIGNALS INFO")
    logger.info("=" * 80)
    logger.info("Global market signals are automatically fetched via yfinance")
    logger.info("when running the main training or prediction pipeline.")
    logger.info("")
    logger.info("These signals include:")
    logger.info("  • Global indices (S&P 500, Nasdaq, FTSE, DAX, Nikkei, etc.)")
    logger.info("  • Commodities (Gold, Silver, Brent, WTI, Copper, etc.)")
    logger.info("  • Currencies (DXY, USD-INR, EUR-INR, GBP-INR, etc.)")
    logger.info("  • Bonds & Rates (US 10Y, US 2Y yields, etc.)")
    logger.info("  • Volatility indices (VIX, VXN, OVX, GVZ)")
    logger.info("  • India sector indices (Nifty IT, Auto, FMCG, Bank, Metal, etc.)")
    logger.info("")
    logger.info("No manual setup required ✓")
    logger.info("=" * 80)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point for data downloader."""
    parser = argparse.ArgumentParser(
        description="Download all required data for F&O Neural Network Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all data (last 2 years)
  python data_downloader.py --all

  # Download specific data types
  python data_downloader.py --bhavcopy --vix

  # Specify custom date range
  python data_downloader.py --all --start-date 2022-01-01 --end-date 2024-12-31

  # Force refresh existing data
  python data_downloader.py --all --force-refresh
        """
    )

    # Data type flags
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all data types",
    )
    parser.add_argument(
        "--bhavcopy",
        action="store_true",
        help="Download NSE F&O Bhavcopy files",
    )
    parser.add_argument(
        "--vix",
        action="store_true",
        help="Download India VIX data",
    )
    parser.add_argument(
        "--fii-dii",
        action="store_true",
        help="Download FII/DII flows data",
    )
    parser.add_argument(
        "--global-info",
        action="store_true",
        help="Display info about global signals (automatically fetched)",
    )

    # Date range options
    default_start = (dt.datetime.now() - relativedelta(years=2)).strftime("%Y-%m-%d")
    default_end = dt.datetime.now().strftime("%Y-%m-%d")

    parser.add_argument(
        "--start-date",
        type=str,
        default=default_start,
        help=f"Start date in YYYY-MM-DD format (default: {default_start})",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=default_end,
        help=f"End date in YYYY-MM-DD format (default: {default_end})",
    )

    # Other options
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download even if data exists",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Setup directories
    setup_directories()

    # Track results
    results = {}

    # If no specific flag is set, show help
    if not any([args.all, args.bhavcopy, args.vix, args.fii_dii, args.global_info]):
        parser.print_help()
        logger.info("\n💡 Tip: Use --all to download all data types")
        return

    # Download requested data
    if args.all or args.bhavcopy:
        results['bhavcopy'] = download_bhavcopy(
            start_date=args.start_date,
            end_date=args.end_date,
            force_refresh=args.force_refresh,
        )

    if args.all or args.vix:
        results['vix'] = download_india_vix(
            start_date=args.start_date,
            end_date=args.end_date,
            force_refresh=args.force_refresh,
        )

    if args.all or args.fii_dii:
        results['fii_dii'] = download_fii_dii_flows(
            start_date=args.start_date,
            end_date=args.end_date,
            force_refresh=args.force_refresh,
        )

    if args.all or args.global_info:
        info_global_signals()

    # Print summary
    if results:
        logger.info("")
        logger.info("=" * 80)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 80)

        for data_type, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"  {data_type.upper():15s}: {status}")

        all_success = all(results.values())
        if all_success:
            logger.info("")
            logger.info("🎉 All downloads completed successfully!")
            logger.info("")
            logger.info("Next steps:")
            logger.info("  1. Train model:      python main.py --mode train")
            logger.info("  2. Generate signals: python main.py --mode predict")
        else:
            logger.warning("")
            logger.warning("⚠ Some downloads failed. Check logs above for details.")

        logger.info("=" * 80)


if __name__ == "__main__":
    main()
