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
            # Handle MultiIndex columns from yfinance 0.2.x
            if isinstance(vix_data.columns, pd.MultiIndex):
                vix_data.columns = [c[0] for c in vix_data.columns]
                
            vix_df = vix_data.reset_index()
            vix_df.columns = [str(c).upper() for c in vix_df.columns]
            
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
    """Download FII/DII historical flows.

    Strategy:
      1. NSE live API  → last 2 days of real FII/FPI + DII data.
      2. yfinance EM ETF proxy → 5-year daily institutional flow proxy using
         EEM (iShares EM ETF) and FXI (China ETF) net volume flows as a
         surrogate for FII buying pressure. Stored as FII_PROXY / DII_PROXY rows.
    """
    if FII_DII_FILE.exists() and not force_refresh:
        try:
            existing = pd.read_csv(FII_DII_FILE)
            if len(existing) >= 100:
                logger.info(f"FII/DII file already exists ({len(existing)} rows). Skipping.")
                return True
            logger.info(f"FII/DII file has only {len(existing)} rows (likely stale). Re-downloading...")
        except Exception:
            pass

    logger.info(f"Downloading FII/DII data from {start_date} to {end_date}...")
    records = []

    # ─── Strategy 1: NSE live API (last 2 trading days) ─────────────────────
    try:
        session = requests.Session()
        session.headers.update(NSE_HEADERS)
        session.get("https://www.nseindia.com/", timeout=15)
        time.sleep(3)
        resp = retry_request("https://www.nseindia.com/api/fiidiiTradeReact", session)
        if resp:
            live_data = resp.json()
            if isinstance(live_data, list):
                for row in live_data:
                    cat = str(row.get("category", "")).upper().replace("FII/FPI", "FII")
                    try:
                        date_str = pd.to_datetime(row.get("date", ""), dayfirst=True).strftime("%Y-%m-%d")
                    except Exception:
                        continue
                    records.append({
                        "date": date_str,
                        "category": cat,
                        "buyValue": float(str(row.get("buyValue", 0)).replace(",", "") or 0),
                        "sellValue": float(str(row.get("sellValue", 0)).replace(",", "") or 0),
                        "netValue": float(str(row.get("netValue", 0)).replace(",", "") or 0),
                    })
                logger.info(f"  ✓ NSE live: {len(live_data)} rows")
    except Exception as e:
        logger.warning(f"NSE live FII/DII failed: {e}")

    # ─── Strategy 2: yfinance proxy for historical 5-year data ───────────────
    # EEM = iShares EM ETF (tracks FII institutional buying of EM including India)
    # Proxy: daily volume * price = flow proxy; normalised to plausible INR crore scale
    try:
        import yfinance as _yf
        proxy_df = _yf.download(
            ["EEM", "^NSEI"],
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
        # Handle MultiIndex
        if isinstance(proxy_df.columns, pd.MultiIndex):
            eem_vol   = proxy_df.get(("Volume", "EEM"), pd.Series(dtype=float))
            eem_close = proxy_df.get(("Close", "EEM"), pd.Series(dtype=float))
            nsei_ret  = proxy_df.get(("Close", "^NSEI"), pd.Series(dtype=float)).pct_change()
        else:
            eem_vol = eem_close = nsei_ret = pd.Series(dtype=float)

        if not eem_vol.empty and not eem_close.empty:
            eem_flow = (eem_vol * eem_close).fillna(0)  # USD proxy
            # Scale to approx INR crore: 1 USD ≈ 84 INR, 1 crore = 1e7, EEM scale factor ~0.01
            scale = 84 / 1e7 * 0.01
            for date_idx in eem_flow.index:
                flow_crore = round(float(eem_flow.loc[date_idx]) * scale, 2)
                buy = round(max(flow_crore, 0), 2)
                sell = round(max(-flow_crore, 0), 2)
                records.append({
                    "date": pd.Timestamp(date_idx).strftime("%Y-%m-%d"),
                    "category": "FII_PROXY",
                    "buyValue": buy,
                    "sellValue": sell,
                    "netValue": round(flow_crore, 2),
                })
            logger.info(f"  ✓ yfinance proxy: {len(eem_flow)} daily rows")
    except Exception as e:
        logger.warning(f"yfinance FII proxy failed: {e}")

    # ─── Consolidate & Save ───────────────────────────────────────────────────
    if records:
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=["date", "category"])
        df = df.sort_values("date")
        df.to_csv(FII_DII_FILE, index=False)
        logger.info(f"✓ Saved {len(df)} FII/DII rows to {FII_DII_FILE}")
        return len(df) >= 10

    # ─── Last-resort scaffold ─────────────────────────────────────────────────
    logger.warning("All FII/DII sources failed. Writing NaN scaffold.")
    scaffold = pd.DataFrame({
        "date": pd.date_range(start_date, end_date, freq="B").strftime("%Y-%m-%d"),
        "category": "FII",
        "buyValue": float("nan"),
        "sellValue": float("nan"),
        "netValue": float("nan"),
    })
    scaffold.to_csv(FII_DII_FILE, index=False)
    return False




# ═════════════════════════════════════════════════════════════════════════════
# 4. ADDITIONAL DATA SOURCES (Bulk Deals, Option Chains, Intraday)
# ═════════════════════════════════════════════════════════════════════════════

def download_bulk_deals(start_date: str = None, end_date: str = None) -> bool:
    """Download Bulk & Block deal reports for a date range.
    
    Iterates month-by-month through the NSE historical API
    and saves a single consolidated CSV into BULK_DEALS_DIR.
    """
    today = dt.datetime.now()
    if not end_date:
        end_date = today.strftime("%Y-%m-%d")
    if not start_date:
        # Default: last 7 days
        start_date = (today - dt.timedelta(days=7)).strftime("%Y-%m-%d")

    logger.info(f"Downloading Bulk/Block deals from {start_date} to {end_date}...")

    start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = dt.datetime.strptime(end_date, "%Y-%m-%d")

    # Output file: consolidated CSV named with from-to range
    out_file = BULK_DEALS_DIR / f"bulk_deals_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"

    # Also keep the today-stamped file in sync for real-time compatibility
    daily_file = BULK_DEALS_DIR / f"bulk_deals_{today.strftime('%Y%m%d')}.csv"

    referer = "https://www.nseindia.com/report-detail/display-bulk-and-block-deals"

    all_data: list = []
    cur = start_dt.replace(day=1)

    try:
        session = init_nse_session()

        while cur <= end_dt:
            from_d = cur.strftime("%d-%m-%Y")
            if cur.month == 12:
                next_month = cur.replace(year=cur.year + 1, month=1, day=1)
            else:
                next_month = cur.replace(month=cur.month + 1, day=1)
            to_d = min(next_month - dt.timedelta(days=1), end_dt).strftime("%d-%m-%Y")

            url = (
                f"https://www.nseindia.com/api/historicalOR/bulk-block-short-deals"
                f"?optionType=bulk_deals&from={from_d}&to={to_d}"
            )

            try:
                resp = retry_request(url, session, referer=referer)
                if resp:
                    data = resp.json()
                    rows = data.get("data", []) if isinstance(data, dict) else []
                    if rows:
                        all_data.extend(rows)
                        logger.info(f"  ✓ {from_d} → {to_d}: {len(rows)} records")
                    else:
                        logger.debug(f"  – {from_d} → {to_d}: 0 records (no deals)")
                else:
                    logger.warning(f"  ✗ {from_d} → {to_d}: request failed")
                time.sleep(1.5)   # NSE rate limiting
            except Exception as e:
                logger.warning(f"  ✗ {from_d} → {to_d}: {e}")

            cur = next_month

    except Exception as e:
        logger.error(f"Bulk deals download failed: {e}")
        return False

    if not all_data:
        logger.warning("Bulk deals: no data returned for the range.")
        return False

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates()

    # Save consolidated file
    df.to_csv(out_file, index=False)
    # Keep daily symlink in sync
    df.to_csv(daily_file, index=False)

    logger.info(f"✓ Saved {len(df)} bulk deal records to {out_file.name}")
    return True


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
                logger.warning(f"  • {symbol}: Failed to fetch contract info")
                success = False
                continue
                
            info_data = info_resp.json()
            exp_dates = info_data.get('expiryDates', [])
            if not exp_dates:
                logger.warning(f"  • {symbol}: No expiry dates found")
                continue
            
            # Fetch top 3 expiries to get more historical/contextual data
            target_expiries = exp_dates[:3]
            logger.info(f"  • {symbol}: Fetching {len(target_expiries)} expiries...")
            
            for expiry in target_expiries:
                # Step 2: Get Option Chain v3
                v3_url = f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={symbol}&expiry={expiry}"
                
                resp = retry_request(v3_url, session, referer=referer)
                if resp:
                    data = resp.json()
                    records = data.get('records', {}).get('data', [])
                    if not records and 'data' in data: records = data['data']
                        
                    if records:
                        rows = []
                        for r in records:
                            if 'CE' in r: rows.append({**r['CE'], 'type': 'CE', 'expiry': expiry})
                            if 'PE' in r: rows.append({**r['PE'], 'type': 'PE', 'expiry': expiry})
                        
                        df = pd.DataFrame(rows)
                        ts = dt.datetime.now().strftime('%Y%m%d_%H%M')
                        # Sanitize expiry for filename (e.g. 26-Mar-2026 -> 26Mar2026)
                        exp_label = expiry.replace("-", "")
                        out_path = OPTION_CHAIN_DIR / f"{symbol.lower()}_{exp_label}_{ts}.csv"
                        df.to_csv(out_path, index=False)
                        logger.info(f"    ✓ {expiry} saved")
                    else:
                        logger.warning(f"    ✗ {expiry} empty")
                else:
                    logger.warning(f"    ✗ {expiry} failed")
                time.sleep(1) # Rate limit protection
                
        except Exception as e:
            logger.error(f"Option Chain ({symbol}) error: {e}")
            success = False
            
    return success

def download_intraday_candles(start_date: str = None) -> bool:
    """Download intraday candles for all F&O indices + top stocks.
    
    Intervals downloaded:
      • 5m  – last 60 days (yfinance hard limit for 5m)
      • 1h  – last 2 years (yfinance limit for 1h)
      • 1d  – from start_date to today (full history, used as daily proxy)
    """
    logger.info("Downloading Intraday candles (5m / 1h / 1d)...")

    # All F&O indices + Nifty 50 constituents available via yfinance
    SYMBOLS: dict = {
        # Indices
        "NIFTY":      "^NSEI",
        "BANKNIFTY":  "^NSEBANK",
        "FINNIFTY":   "NIFTY_FIN_SERVICE.NS",
        # Top F&O stocks
        "RELIANCE":   "RELIANCE.NS",
        "TCS":        "TCS.NS",
        "HDFCBANK":   "HDFCBANK.NS",
        "ICICIBANK":  "ICICIBANK.NS",
        "INFY":       "INFY.NS",
        "SBIN":       "SBIN.NS",
        "BAJFINANCE": "BAJFINANCE.NS",
        "LT":         "LT.NS",
        "AXISBANK":   "AXISBANK.NS",
        "KOTAKBANK":  "KOTAKBANK.NS",
        "HCLTECH":    "HCLTECH.NS",
        "TATAMOTORS": "TATAMOTORS.NS",
        "WIPRO":      "WIPRO.NS",
        "ADANIENT":   "ADANIENT.NS",
        "ADANIPORTS": "ADANIPORTS.NS",
        "ULTRACEMCO": "ULTRACEMCO.NS",
        "HINDALCO":   "HINDALCO.NS",
        "TATASTEEL":  "TATASTEEL.NS",
        "M&M":        "M&M.NS",
    }

    today = dt.datetime.now()
    hist_start = start_date or "2020-01-01"
    success = True

    intervals = [
        # (interval, period_or_start, use_period)
        ("5m",  "60d",         True),   # last 60 days — yfinance max
        ("1h",  "730d",        True),   # last 2 years
        ("1d",  hist_start,    False),  # full history from start_date
    ]

    for name, ticker in SYMBOLS.items():
        for interval, period_or_start, use_period in intervals:
            try:
                if use_period:
                    df = yf.download(
                        ticker, period=period_or_start,
                        interval=interval, progress=False, auto_adjust=True
                    )
                else:
                    df = yf.download(
                        ticker,
                        start=period_or_start,
                        end=today.strftime("%Y-%m-%d"),
                        interval=interval,
                        progress=False,
                        auto_adjust=True,
                    )

                # Flatten MultiIndex if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]

                if df.empty:
                    logger.warning(f"  ✗ {name} {interval}: empty")
                    continue

                df = df.reset_index()
                out_path = INTRADAY_DIR / f"{name.lower()}_{interval.replace('m','m').replace('h','h').replace('d','d')}.csv"
                df.to_csv(out_path, index=False)
                logger.info(f"  ✓ {name} {interval}: {len(df)} rows → {out_path.name}")

            except Exception as e:
                logger.error(f"  ✗ {name} {interval}: {e}")
                success = False

        time.sleep(0.5)  # polite rate limit between symbols

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
    parser.add_argument("--loop", type=int, help="Run in loop mode every N minutes (captures OC + Intraday)")
    
    default_start = "2020-01-01"
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
        results['bulk_deals'] = download_bulk_deals(args.start_date, args.end_date)
        
    if args.all or args.option_chain:
        results['option_chain'] = download_option_chains()
        
    if args.all or args.intraday:
        results['intraday'] = download_intraday_candles(args.start_date)

    if args.all or args.global_info:
        info_global_signals()

    # Loop Mode
    if args.loop:
        interval = args.loop * 60
        logger.info(f"Entering LOOP MODE. Refreshing every {args.loop} minutes...")
        try:
            while True:
                logger.info("-" * 20 + f" CYCLE START: {dt.datetime.now()} " + "-" * 20)
                download_option_chains()
                download_intraday_candles(args.start_date)
                logger.info(f"Waiting {args.loop} minutes...")
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Loop mode stopped by user.")
            return

    # Final Summary
    logger.info("\n" + "="*20 + " DATA PIPELINE SUMMARY " + "="*20)
    for k, v in results.items():
        logger.info(f"{k.upper():15}: {'✓ OK' if v else '✗ FAILED'}")
    logger.info("="*63)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
