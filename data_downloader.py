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
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Configure logging ─────────────────────────────────────────────────────────
# Use a more compact format when tqdm is active
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        TqdmLoggingHandler(),
    ],
)
logger = logging.getLogger("DataDownloader")

# ── Import project modules ────────────────────────────────────────────────────
try:
    from historical_loader import download_bhavcopy_range
    HISTORICAL_LOADER_AVAILABLE = True
except ImportError:
    HISTORICAL_LOADER_AVAILABLE = False
    logger.warning("historical_loader.py not found. Bhavcopy download may be limited.")

try:
    import nsefin
    NSEFIN_AVAILABLE = True
except ImportError:
    NSEFIN_AVAILABLE = False
    logger.warning("nsefin not installed. Some data features will use fallbacks.")

try:
    from nsepython import nsefetch
    NSEPYTHON_AVAILABLE = True
except ImportError:
    NSEPYTHON_AVAILABLE = False
    logger.warning("nsepython not installed. Some data features will use fallbacks.")

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
BHAVCOPY_DIR = DATA_DIR / "bhavcopy"
VIX_DIR = DATA_DIR / "vix"
FII_DII_DIR = DATA_DIR / "fii_dii"
EXTENDED_DIR = DATA_DIR / "extended"
OPTION_CHAIN_DIR = DATA_DIR / "option_chain"
INTRADAY_DIR = DATA_DIR / "intraday"
BULK_DEALS_DIR = DATA_DIR / "bulk_deals"

VIX_FILE = VIX_DIR / "india_vix.csv"
FII_DII_FILE = FII_DII_DIR / "fii_dii_data.csv"

# NSE URLs and headers
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# ═════════════════════════════════════════════════════════════════════════════
# UTILS
# ═════════════════════════════════════════════════════════════════════════════

def init_nse_session() -> requests.Session:
    """Initialize a robust NSE session by visiting relevant pages."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        # Visit home to get initial cookies
        session.get("https://www.nseindia.com/", timeout=15)
        # CRITICAL: Wait for session to solidify
        time.sleep(3)
        # Visit a sub-page to solidify session (XHR Referer source)
        session.get("https://www.nseindia.com/market-data/live-equity-market", timeout=15)
        time.sleep(1)
    except Exception as e:
        logger.warning(f"NSE session init warning: {e}")
    return session

def retry_request(url: str, session: requests.Session = None, retries: int = 3, backoff: float = 2.0, referer: str = None):
    """Enhanced retry logic with mandatory session and referer support."""
    if not session:
        session = init_nse_session()
        
    headers = session.headers.copy()
    if referer: headers["Referer"] = referer
    headers["Accept"] = "application/json, text/javascript, */*; q=0.01"
    headers["X-Requested-With"] = "XMLHttpRequest"
    
    for i in range(retries):
        try:
            resp = session.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                text = resp.text.strip()
                if "Access Denied" in text[:500] or not text:
                    raise Exception("Access Denied HTML or empty response")
                return resp
            
            if resp.status_code in [401, 403]:
                session = init_nse_session()
                headers = session.headers.copy()
                if referer: headers["Referer"] = referer
                time.sleep(backoff * (i + 2))
            
        except Exception as e:
            logger.debug(f"Retry {i} failed for {url}: {e}")
        
        time.sleep(backoff * (i + 1))
    return None

# ═════════════════════════════════════════════════════════════════════════════
# DIRECTORY SETUP
# ═════════════════════════════════════════════════════════════════════════════

def setup_directories() -> None:
    """Create all required data directories."""
    dirs = [
        DATA_DIR, BHAVCOPY_DIR, VIX_DIR, FII_DII_DIR, EXTENDED_DIR,
        OPTION_CHAIN_DIR, INTRADAY_DIR, BULK_DEALS_DIR,
        Path("logs"), Path("models"), Path("output"),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("✓ Data directories initialized")


# ═════════════════════════════════════════════════════════════════════════════
# 1. NSE F&O BHAVCOPY DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════

def _download_single_bhavcopy(date_obj: dt.date, target_dir: str) -> bool:
    """Download a single day's bhavcopy using jugaad_data."""
    from jugaad_data.nse import bhavcopy_fo_save
    
    # Granular check: check if file already exists in targets
    # (Checking both root and 'bhavcopies' subfolder for compatibility)
    fname = f"fo{date_obj.strftime('%d%b%Y').upper()}bhav.csv"
    if (Path(target_dir) / fname).exists() or (Path(target_dir) / "bhavcopies" / fname).exists():
        return True
        
    try:
        bhavcopy_fo_save(date_obj, target_dir)
        return True
    except Exception:
        return False

def download_bhavcopy(
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
    workers: int = 4
) -> bool:
    """Download NSE F&O Bhavcopy files for the specified date range."""
    # Ensure nested dir exists for historical_loader compatibility
    (BHAVCOPY_DIR / "bhavcopies").mkdir(exist_ok=True)
    
    logger.info(f"Downloading NSE F&O Bhavcopy from {start_date} to {end_date}...")

    # Parse dates
    start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    
    # Generate date list (excluding weekends)
    dates_to_download = []
    curr = start
    while curr <= end:
        if curr.weekday() < 5:
            dates_to_download.append(curr)
        curr += dt.timedelta(days=1)

    if not HISTORICAL_LOADER_AVAILABLE:
        logger.info("Using parallel fallback method (jugaad_data)...")
        try:
            import jugaad_data
            
            success_count = 0
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Save directly to bhavcopies subfolder to match historical_loader
                futures = {executor.submit(_download_single_bhavcopy, d, str(BHAVCOPY_DIR / "bhavcopies")): d for d in dates_to_download}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Bhavcopy"):
                    if future.result():
                        success_count += 1
            
            logger.info(f"✓ Bhavcopy complete: {success_count}/{len(dates_to_download)} days acquired")
            return success_count > 0
            
        except ImportError:
            logger.error("jugaad_data not installed. Run: pip install jugaad-data")
            return False

    try:
        # Use robust historical_loader which handles retries and nesting better
        downloaded_files = download_bhavcopy_range(
            start_date_str=start_date,
            end_date_str=end_date,
            data_dir=str(BHAVCOPY_DIR),
        )
        logger.info(f"✓ Bhavcopy complete: {len(downloaded_files)} files processed")
        return len(downloaded_files) > 0
    except Exception as e:
        logger.error(f"Bhavcopy download failed: {e}")
        return False


# ═════════════════════════════════════════════════════════════════════════════
# 2. INDIA VIX DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════

def download_india_vix(
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
) -> bool:
    """Download India VIX data and save to CSV."""
    if VIX_FILE.exists() and not force_refresh:
        logger.info("India VIX file already exists. Skipping.")
        return True

    logger.info(f"Downloading India VIX from {start_date} to {end_date}...")
    
    try:
        vix_data = yf.download("^INDIAVIX", start=start_date, end=end_date, progress=False)
        
        if vix_data.empty and NSEFIN_AVAILABLE:
            logger.info("yfinance failed. Trying nsefin...")
            nse_client = nsefin.NSEClient()
            vix_data = nse_client.get_vix_data(start_date, end_date)

        if not vix_data.empty:
            vix_df = vix_data.reset_index()
            vix_df.columns = [c.upper() for c in vix_df.columns]
            
            if 'DATE' not in vix_df.columns and 'DATETIME' in vix_df.columns:
                vix_df.rename(columns={'DATETIME': 'DATE'}, inplace=True)
            
            close_col = next((c for c in ['CLOSE', 'VIX_CLOSE', 'VIX'] if c in vix_df.columns), None)
            if close_col:
                vix_df.rename(columns={close_col: 'VIX'}, inplace=True)
            
            if 'VIX' not in vix_df.columns:
                numeric_cols = vix_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    vix_df.rename(columns={numeric_cols[0]: 'VIX'}, inplace=True)

            vix_df[['DATE', 'VIX']].to_csv(VIX_FILE, index=False)
            logger.info(f"✓ Saved India VIX to {VIX_FILE} ({len(vix_df)} rows)")
            return True
            
        logger.warning("Failed to acquire India VIX data.")
        return False
    except Exception as e:
        logger.error(f"VIX download error: {e}")
        return False


# ═════════════════════════════════════════════════════════════════════════════
# 3. FII/DII FLOWS DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════

def download_fii_dii_flows(
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
) -> bool:
    """Download FII/DII flows data and save to CSV."""
    if FII_DII_FILE.exists() and not force_refresh:
        logger.info("FII/DII file already exists. Skipping.")
        return True

    logger.info(f"Downloading FII/DII flows (latest snapshot)...")

    try:
        session = requests.Session()
        session.get("https://www.nseindia.com/", headers=NSE_HEADERS, timeout=10)
        
        # Try multiple endpoints
        endpoints = [
            "https://www.nseindia.com/api/fiidiiTradeReact",
            "https://www.nseindia.com/api/fiidii-trade-details"
        ]
        
        for url in endpoints:
            resp = retry_request(url, session)
            if resp:
                data = resp.json()
                df = pd.DataFrame(data)
                df.to_csv(FII_DII_FILE, index=False)
                logger.info(f"✓ Saved FII/DII flows to {FII_DII_FILE} ({len(df)} rows)")
                return True

        logger.warning("FII/DII download failed. Using mock data for pipeline continuity.")
        dates = pd.date_range(end=dt.datetime.now(), periods=5)
        pd.DataFrame({'date': dates, 'category': 'FII', 'buyValue': 1000, 'sellValue': 800, 'netValue': 200}).to_csv(FII_DII_FILE, index=False)
        return False
    except Exception as e:
        logger.error(f"FII/DII error: {e}")
        return False


# ═════════════════════════════════════════════════════════════════════════════
# 4. ADDITIONAL DATA SOURCES (Bulk Deals, Option Chains, Intraday)
# ═════════════════════════════════════════════════════════════════════════════

def download_bulk_deals() -> bool:
    """Download daily Bulk and Block deal reports."""
    logger.info("Downloading Bulk/Block deals (Historical OR)...")
    
    # Use a recent date range (last 7 days)
    end_dt = dt.datetime.now()
    start_dt = end_dt - dt.timedelta(days=7)
    from_date = start_dt.strftime("%d-%m-%Y")
    to_date = end_dt.strftime("%d-%m-%Y")
    
    out_file = BULK_DEALS_DIR / f"bulk_deals_{end_dt.strftime('%Y%m%d')}.csv"
    
    # Verified API URL
    url = f"https://www.nseindia.com/api/historicalOR/bulk-block-short-deals?optionType=bulk_deals&from={from_date}&to={to_date}"
    referer = "https://www.nseindia.com/report-detail/display-bulk-and-block-deals"
    
    try:
        session = init_nse_session()
        resp = retry_request(url, session, referer=referer)
        
        if resp:
            data = resp.json()
            if 'data' in data:
                df = pd.DataFrame(data['data'])
                if not df.empty:
                    df.to_csv(out_file, index=False)
                    logger.info(f"✓ Saved Bulk Deals to {out_file.name} ({len(df)} rows)")
                    return True
        logger.warning("Bulk deals fetch failed or returned no data.")
        return False
    except Exception as e:
        logger.error(f"Bulk deals error: {e}")
        return False

def download_option_chains() -> bool:
    """Download latest Option Chain snapshots using contract-info + v3 API."""
    logger.info("Downloading Option Chain snapshots (v3 + contract-info)...")
    symbols = ["NIFTY", "BANKNIFTY"]
    success = True
    
    session = init_nse_session()
    
    for symbol in symbols:
        try:
            # Step 1: Get Expiry Dates
            info_url = f"https://www.nseindia.com/api/option-chain-contract-info?symbol={symbol}"
            referer = f"https://www.nseindia.com/option-chain"
            
            info_resp = retry_request(info_url, session, referer=referer)
            if not info_resp:
                logger.warning(f"  • {symbol}: Failed to fetch contract info (expiries)")
                success = False
                continue
                
            info_data = info_resp.json()
            exp_dates = info_data.get('expiryDates', [])
            if not exp_dates:
                logger.warning(f"  • {symbol}: No expiry dates found in contract info")
                success = False
                continue
            
            latest_expiry = exp_dates[0]
            
            # Step 2: Get Option Chain v3 for the latest expiry
            # Note: type=Indices is required for NIFTY/BANKNIFTY
            v3_url = f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={symbol}&expiry={latest_expiry}"
            
            resp = retry_request(v3_url, session, referer=referer)
            if resp:
                data = resp.json()
                # Records are usually under 'records' -> 'data'
                records = data.get('records', {}).get('data', [])
                if not records and 'data' in data:
                    records = data['data']
                    
                if records:
                    rows = []
                    for r in records:
                        if 'CE' in r: rows.append({**r['CE'], 'type': 'CE', 'expiry': latest_expiry})
                        if 'PE' in r: rows.append({**r['PE'], 'type': 'PE', 'expiry': latest_expiry})
                    
                    df = pd.DataFrame(rows)
                    ts = dt.datetime.now().strftime('%Y%m%d_%H%M')
                    out_path = OPTION_CHAIN_DIR / f"{symbol.lower()}_{ts}.csv"
                    df.to_csv(out_path, index=False)
                    logger.info(f"  • {symbol} Option Chain ({latest_expiry}) captured ({len(df)} rows)")
                else:
                    logger.warning(f"  • {symbol} records empty for expiry {latest_expiry}")
                    success = False
            else:
                logger.warning(f"  • {symbol} v3 request failed after retries")
                success = False
        except Exception as e:
            logger.error(f"Option Chain ({symbol}) error: {e}")
            success = False
            
    return success

def download_intraday_candles() -> bool:
    """Download recent 5-minute candles for indices."""
    logger.info("Downloading Intraday 5m candles...")
    indices = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
    success = True
    
    for name, ticker in indices.items():
        try:
            df = yf.download(ticker, period="5d", interval="5m", progress=False)
            if not df.empty:
                df.to_csv(INTRADAY_DIR / f"{name.lower()}_5m.csv")
                logger.info(f"  • {name} 5m candles updated")
            else:
                success = False
        except Exception as e:
            logger.error(f"Intraday ({name}) error: {e}")
            success = False
            
    return success


def info_global_signals() -> None:
    """Display information about global signals."""
    logger.info("-" * 40)
    logger.info("GLOBAL SIGNALS (Auto-managed)")
    logger.info("-" * 40)
    logger.info("These are fetched on-the-fly via yfinance:")
    logger.info("  • Indices (S&P500, Nasdaq, India VIX, etc.)")
    logger.info("  • Commodities (Gold, Brent, etc.)")
    logger.info("  • Currencies (DXY, USD-INR)")
    logger.info("  • Yields (US 10Y)")
    logger.info("Managed by market_data_extended.py ✓")
    logger.info("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="NSE F&O Data Downloader")
    parser.add_argument("--all", action="store_true", help="Download all data")
    parser.add_argument("--bhavcopy", action="store_true", help="Download Bhavcopies")
    parser.add_argument("--vix", action="store_true", help="Download VIX")
    parser.add_argument("--fii-dii", action="store_true", help="Download FII/DII")
    parser.add_argument("--bulk-deals", action="store_true", help="Download Bulk/Block deals")
    parser.add_argument("--option-chain", action="store_true", help="Download Option Chain")
    parser.add_argument("--intraday", action="store_true", help="Download Intraday 5m candles")
    parser.add_argument("--global-info", action="store_true", help="Show global signals info")
    
    default_start = (dt.datetime.now() - relativedelta(years=2)).strftime("%Y-%m-%d")
    default_end = dt.datetime.now().strftime("%Y-%m-%d")
    
    parser.add_argument("--start-date", type=str, default=default_start)
    parser.add_argument("--end-date", type=str, default=default_end)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers for Bhavcopy")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    setup_directories()
    
    # If no flags passed, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return

    results = {}
    
    # Run selected tasks
    if args.all or args.bhavcopy:
        results['bhavcopy'] = download_bhavcopy(args.start_date, args.end_date, args.force_refresh, args.workers)
    
    if args.all or args.vix:
        results['vix'] = download_india_vix(args.start_date, args.end_date, args.force_refresh)
        
    if args.all or args.fii_dii:
        results['fii_dii'] = download_fii_dii_flows(args.start_date, args.end_date, args.force_refresh)
        
    if args.all or args.bulk_deals:
        results['bulk_deals'] = download_bulk_deals()
        
    if args.all or args.option_chain:
        results['option_chain'] = download_option_chains()
        
    if args.all or args.intraday:
        results['intraday'] = download_intraday_candles()

    if args.all or args.global_info:
        info_global_signals()

    # Final Summary
    logger.info("\n" + "="*20 + " DATA PIPELINE SUMMARY " + "="*20)
    for k, v in results.items():
        logger.info(f"{k.upper():15}: {'✓ OK' if v else '✗ FAILED'}")
    logger.info("="*63)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
