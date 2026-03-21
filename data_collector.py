"""
data_collector.py
=================
Primary data collection module for the F&O Neural Network Predictor.

Data Sources (in priority order):
  1. NSE F&O Bhavcopy CSV files      — historical OHLCV + OI for all F&O contracts
  2. nsefin library                   — live option chain, Greeks, FII/DII, VIX
  3. nsepython library                — PCR, participant OI, illiquid options check
  4. yfinance                         — Nifty / BankNifty OHLCV, India VIX, USD-INR, global indices
  5. NSE direct API (requests)        — fallback for option chain if nsefin unavailable

Usage:
    collector = DataCollector()
    df = collector.get_full_dataset(start_date="2022-01-01", end_date="2024-12-31")
"""

import os
import time
import logging
import warnings
import datetime as dt
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import yfinance as yf
import requests
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/data_collector.log", mode="a"),
    ],
)
logger = logging.getLogger("DataCollector")

# ── Try importing optional NSE libraries ──────────────────────────────────────
try:
    import nsefin
    NSEFIN_AVAILABLE = True
    logger.info("nsefin available ✓")
except ImportError:
    NSEFIN_AVAILABLE = False
    logger.warning("nsefin not installed. Run: pip install nsefin")

try:
    from nsepython import nsefetch, pcr
    NSEPYTHON_AVAILABLE = True
    logger.info("nsepython available ✓")
except ImportError:
    NSEPYTHON_AVAILABLE = False
    logger.warning("nsepython not installed. Run: pip install nsepython")

# ── Constants ─────────────────────────────────────────────────────────────────
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
NSE_BASE = "https://www.nseindia.com/api"
NIFTY_STRIKE_STEP = 50
BANKNIFTY_STRIKE_STEP = 100
ATM_RANGE_STRIKES = 10   # strikes above and below ATM to aggregate PCR/OI

# ─────────────────────────────────────────────────────────────────────────────
class DataCollector:
    """
    Master data collector — merges all sources into a single clean DataFrame
    ready for feature engineering.
    """

    def __init__(
        self,
        bhavcopy_dir: str = "data/bhavcopy",
        vix_file: str = "data/vix/india_vix.csv",
        fii_file: str = "data/fii_dii/fii_dii_data.csv",
    ):
        self.bhavcopy_dir = Path(bhavcopy_dir)
        self.vix_file = Path(vix_file)
        self.fii_file = Path(fii_file)
        self._nse_session: Optional[requests.Session] = None

        if NSEFIN_AVAILABLE:
            self.nse_client = nsefin.NSEClient()
        else:
            self.nse_client = None

        # Create output dirs
        for d in ["logs", "data/bhavcopy", "data/vix", "data/fii_dii",
                  "output", "models"]:
            Path(d).mkdir(parents=True, exist_ok=True)

    # ─── NSE Session ──────────────────────────────────────────────────────────
    def _get_nse_session(self) -> requests.Session:
        """Create a warm NSE session (required to bypass NSE bot-check)."""
        if self._nse_session is not None:
            return self._nse_session
        session = requests.Session()
        session.headers.update(NSE_HEADERS)
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
        self._nse_session = session
        logger.info("NSE session initialized")
        return session

    def load_bhavcopy_range(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load NSE F&O Bhavcopy CSV files for a date range in parallel.
        """
        import concurrent.futures

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end)

        paths_to_load = []
        missing = []

        # 1. Identify files
        for date in dates:
            formats = [
                f"fo{date.strftime('%d%m%Y')}bhav.csv",
                f"fo{date.strftime('%d%b%Y').upper()}bhav.csv",
                f"fo{date.strftime('%d%b%Y')}bhav.csv",
                f"fo{date.strftime('%d-%m-%Y')}bhav.csv",
            ]
            
            fname = None
            for fmt in formats:
                test_path = self.bhavcopy_dir / fmt
                if test_path.exists():
                    fname = test_path
                    break
                test_path_nested = self.bhavcopy_dir / "bhavcopies" / fmt
                if test_path_nested.exists():
                    fname = test_path_nested
                    break

            if fname:
                paths_to_load.append((fname, date))
            elif date.weekday() < 5:
                missing.append(date.strftime("%d-%m-%Y"))

        # 2. Parallel read and parse
        def _read_file(args):
            path, dt_date = args
            try:
                df = pd.read_csv(path, low_memory=False)
                df = self._clean_bhavcopy(df, dt_date)
                df = df.dropna(subset=["CLOSE"])
                return df if not df.empty else None
            except Exception as e:
                logger.warning(f"Failed to parse {path.name}: {e}")
                return None

        frames = []
        if paths_to_load:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                results = list(tqdm(executor.map(_read_file, paths_to_load), 
                                    total=len(paths_to_load), 
                                    desc="Loading bhavcopies", unit="day"))
            frames = [df for df in results if df is not None]

        if missing:
            logger.warning(f"Missing bhavcopy files ({len(missing)}): {missing[:5]}...")

        if not frames:
            logger.error("No bhavcopy data loaded. Check data/bhavcopy/ directory.")
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)

        if symbols:
            result = result[result["SYMBOL"].isin(symbols)]

        # --- OPTIMIZATION: Filter for Near-Month Future Only ---
        # F&O bhavcopy contains thousands of option rows per symbol per day.
        # Downstream features expect a daily time-series.
        # We pick the nearest-expiry future contract (INSTRUMENT = FUTIDX/FUTSTK or OptnTp = XX)
        if not result.empty:
            # Detect futures: OPTION_TYP is often 'XX', 'nan', or empty for futures
            if "OPTION_TYP" in result.columns:
                # Handle different formats for future type
                is_future = result["OPTION_TYP"].astype(str).str.upper().isin(["XX", "NAN", "", "NONE", "-"])
                result = result[is_future]
            
            # Additional instrument filter if available (Old: FUTIDX, New: IDF/STF)
            if "INSTRUMENT" in result.columns:
                is_fut_inst = (
                    result["INSTRUMENT"].str.contains("FUT", na=False) |
                    result["INSTRUMENT"].isin(["IDF", "STF", "IDX", "STK"])
                )
                result = result[is_fut_inst]

            # Pick near-month (min DTE) for each Date and Symbol
            if "DTE" in result.columns:
                # Sort and drop duplicates to get near-month
                result = result.sort_values(["DATE", "SYMBOL", "DTE"])
                result = result.drop_duplicates(subset=["DATE", "SYMBOL"], keep="first")
        # -----------------------------------------------------

        logger.info(
            f"Loaded bhavcopy: {len(result):,} rows | "
            f"{result['DATE'].nunique()} trading days | "
            f"{result['SYMBOL'].nunique()} symbols"
        )
        return result

    def _clean_bhavcopy(self, df: pd.DataFrame, target_date: str = None) -> pd.DataFrame:
        """Standardize column names and types for bhavcopy data."""
        if df.empty:
            return df

        col_map = {
            # Legacy Format
            "SYMBOL":     "SYMBOL",
            "CLOSE":      "CLOSE",
            "OPEN":       "OPEN",
            "HIGH":       "HIGH",
            "LOW":        "LOW",
            "OPEN_INT":   "OPEN_INT",
            "CHG_IN_OI":  "CHG_IN_OI",
            "STRIKE_PR":  "STRIKE_PR",
            "OPTION_TYP": "OPTION_TYP",
            "EXPIRY_DT":  "EXPIRY_DT",
            # New Format (effective late 2024/2025)
            "TradDt":     "DATE",
            "TckrSymb":   "SYMBOL",
            "ClsPric":    "CLOSE",
            "OpnPric":    "OPEN",
            "HghPric":    "HIGH",
            "LwPric":     "LOW",
            "OpnIntrst":  "OPEN_INT",
            "ChngInOpnIntrst": "CHG_IN_OI",
            "StrkPric":   "STRIKE_PR",
            "OptnTp":     "OPTION_TYP",
            "XpryDt":     "EXPIRY_DT",
            "TtlTradgVol": "CONTRACTS",
            "TtlTrfVal":  "VAL_INLAKH",
            "SttlmPric":  "SETTLE_PR",
            "FinInstrmTp": "INSTRUMENT",
            "UndrlygPric": "UNDERLYING",
            # Standard mappings
            "INSTRUMENT": "INSTRUMENT",
            "STRIKE_PR":  "STRIKE_PR",
            "OPTION_TYP": "OPTION_TYP",
            "SETTLE_PR":  "SETTLE_PR",
            "CONTRACTS":  "CONTRACTS",
            "VAL_INLAKH": "VAL_INLAKH",
            "TIMESTAMP":  "TIMESTAMP",
        }

        # Handle col name whitespace and rename
        df = df.rename(columns={c: col_map.get(c.strip(), c.strip()) for c in df.columns})

        # --- DEDUPLICATE COLUMNS ---
        # If multiple original columns mapped to the same name (e.g. TradDt -> DATE and DATE exists)
        if df.columns.duplicated().any():
            # Keep the first occurrence of each column name
            df = df.loc[:, ~df.columns.duplicated()]
        # ---------------------------

        # Add DATE if provided and missing
        if target_date and "DATE" not in df.columns:
            df["DATE"] = pd.to_datetime(target_date)

        # Standardize Dates
        for col in ["DATE", "EXPIRY_DT", "TIMESTAMP"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Drop rows where no price data
        if "CLOSE" in df.columns:
            df = df.dropna(subset=["CLOSE"])

        # Keep only required columns to save memory
        # Use a set to ensure unique columns in the filter list
        target_cols = set(col_map.values()) | {"DATE", "DTE", "NEAR_EXPIRY"}
        keep_cols = [c for c in target_cols if c in df.columns]
        df = df[keep_cols]

        # Compute DTE
        if "EXPIRY_DT" in df.columns and "DATE" in df.columns:
            # Ensure both are datetimes for subtraction
            dte_val = (df["EXPIRY_DT"] - df["DATE"]).dt.days
            df["DTE"] = dte_val.clip(lower=0)
            # Near-expiry flag (weekly)
            df["NEAR_EXPIRY"] = (df["DTE"] <= 2).astype(int)

        return df

    def download_bhavcopy(self, date: dt.datetime) -> Optional[pd.DataFrame]:
        """
        Attempt to download today's bhavcopy directly from NSE via nsefin,
        fallback to direct URL download.
        """
        if NSEFIN_AVAILABLE:
            try:
                df = self.nse_client.get_fno_bhav_copy(date)
                df = self.nse_client.format_fo_data(df)
                logger.info(f"Downloaded bhavcopy via nsefin for {date.date()}")
                return df
            except Exception as e:
                logger.warning(f"nsefin bhavcopy failed: {e}. Trying direct download...")

        # Direct NSE URL fallback
        url = (
            f"https://www.nseindia.com/content/historical/DERIVATIVES/"
            f"{date.year}/{date.strftime('%b').upper()}/"
            f"fo{date.strftime('%d%b%Y').upper()}bhav.csv.zip"
        )
        try:
            session = self._get_nse_session()
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                import io, zipfile
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    fname = z.namelist()[0]
                    df = pd.read_csv(z.open(fname))
                # Save locally
                save_path = self.bhavcopy_dir / f"fo{date.strftime('%d%m%Y')}bhav.csv"
                df.to_csv(save_path, index=False)
                logger.info(f"Downloaded and saved bhavcopy: {save_path}")
                return self._clean_bhavcopy(df, date)
        except Exception as e:
            logger.error(f"Direct bhavcopy download failed for {date.date()}: {e}")
        return None

    # ─── 2. LIVE OPTION CHAIN ─────────────────────────────────────────────────
    def get_live_option_chain(
        self,
        symbol: str = "NIFTY",
        expiry: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch live option chain snapshot for a symbol.
        Returns full chain with columns: strike, CE_OI, CE_chg_OI, CE_IV,
        CE_delta, CE_LTP, PE_OI, PE_chg_OI, PE_IV, PE_delta, PE_LTP,
        total_OI, pcr_strike

        Priority: nsefin → nsepython → direct NSE API
        """
        # --- nsefin ---
        if NSEFIN_AVAILABLE:
            try:
                oc = self.nse_client.get_option_chain(symbol)
                df = self._parse_nsefin_option_chain(oc, symbol)
                logger.info(f"Live option chain fetched via nsefin: {symbol} ({len(df)} strikes)")
                return df
            except Exception as e:
                logger.warning(f"nsefin option chain failed: {e}")

        # --- nsepython PCR + direct chain ---
        if NSEPYTHON_AVAILABLE:
            try:
                oc_url = (
                    f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
                    if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"]
                    else f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
                )
                raw = nsefetch(oc_url)
                df = self._parse_raw_option_chain(raw, symbol)
                logger.info(f"Live option chain fetched via nsepython: {symbol}")
                return df
            except Exception as e:
                logger.warning(f"nsepython option chain failed: {e}")

        # --- Direct NSE API ---
        try:
            session = self._get_nse_session()
            url = (
                f"{NSE_BASE}/option-chain-indices?symbol={symbol}"
                if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"]
                else f"{NSE_BASE}/option-chain-equities?symbol={symbol}"
            )
            resp = session.get(url, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
            df = self._parse_raw_option_chain(raw, symbol)
            logger.info(f"Live option chain fetched via direct API: {symbol}")
            return df
        except Exception as e:
            logger.error(f"All option chain sources failed for {symbol}: {e}")
            return pd.DataFrame()

    def _parse_nsefin_option_chain(self, oc: dict, symbol: str) -> pd.DataFrame:
        """Parse nsefin option chain dict into a clean strike-wise DataFrame."""
        rows = []
        data = oc.get("records", {}).get("data", [])
        underlying = oc.get("records", {}).get("underlyingValue", np.nan)

        for item in data:
            strike = item.get("strikePrice", np.nan)
            ce = item.get("CE", {})
            pe = item.get("PE", {})
            rows.append({
                "STRIKE":        strike,
                "CE_OI":         ce.get("openInterest", 0),
                "CE_CHG_OI":     ce.get("changeinOpenInterest", 0),
                "CE_IV":         ce.get("impliedVolatility", np.nan),
                "CE_LTP":        ce.get("lastPrice", np.nan),
                "CE_VOLUME":     ce.get("totalTradedVolume", 0),
                "CE_DELTA":      ce.get("delta", np.nan),
                "CE_THETA":      ce.get("theta", np.nan),
                "CE_GAMMA":      ce.get("gamma", np.nan),
                "CE_VEGA":       ce.get("vega", np.nan),
                "PE_OI":         pe.get("openInterest", 0),
                "PE_CHG_OI":     pe.get("changeinOpenInterest", 0),
                "PE_IV":         pe.get("impliedVolatility", np.nan),
                "PE_LTP":        pe.get("lastPrice", np.nan),
                "PE_VOLUME":     pe.get("totalTradedVolume", 0),
                "PE_DELTA":      pe.get("delta", np.nan),
                "PE_THETA":      pe.get("theta", np.nan),
                "PE_GAMMA":      pe.get("gamma", np.nan),
                "PE_VEGA":       pe.get("vega", np.nan),
                "UNDERLYING":    underlying,
            })

        df = pd.DataFrame(rows).sort_values("STRIKE").reset_index(drop=True)
        df["PCR_STRIKE"] = np.where(df["CE_OI"] > 0, df["PE_OI"] / df["CE_OI"].replace(0, np.nan), np.nan)
        return df

    def _parse_raw_option_chain(self, raw: dict, symbol: str) -> pd.DataFrame:
        """Parse raw NSE JSON option chain response."""
        rows = []
        underlying = raw.get("records", {}).get("underlyingValue", np.nan)
        for item in raw.get("records", {}).get("data", []):
            strike = item.get("strikePrice", np.nan)
            ce = item.get("CE", {})
            pe = item.get("PE", {})
            rows.append({
                "STRIKE":    strike,
                "CE_OI":     ce.get("openInterest", 0),
                "CE_CHG_OI": ce.get("changeinOpenInterest", 0),
                "CE_IV":     ce.get("impliedVolatility", np.nan),
                "CE_LTP":    ce.get("lastPrice", np.nan),
                "CE_VOLUME": ce.get("totalTradedVolume", 0),
                "PE_OI":     pe.get("openInterest", 0),
                "PE_CHG_OI": pe.get("changeinOpenInterest", 0),
                "PE_IV":     pe.get("impliedVolatility", np.nan),
                "PE_LTP":    pe.get("lastPrice", np.nan),
                "PE_VOLUME": pe.get("totalTradedVolume", 0),
                "UNDERLYING": underlying,
            })
        df = pd.DataFrame(rows).sort_values("STRIKE").reset_index(drop=True)
        df["PCR_STRIKE"] = np.where(df["CE_OI"] > 0, df["PE_OI"] / df["CE_OI"].replace(0, np.nan), np.nan)
        return df

    # ─── 3. INDIA VIX ─────────────────────────────────────────────────────────
    def load_india_vix(self) -> pd.DataFrame:
        """
        Load India VIX data.
        Sources: local CSV file → yfinance (^INDIAVIX) → NSE direct
        """
        # Try local CSV first
        if self.vix_file.exists():
            df = pd.read_csv(self.vix_file)
            df = self._clean_vix_df(df)
            logger.info(f"VIX loaded from local file: {len(df)} rows")
            return df

        # Try yfinance
        try:
            df = yf.download("^INDIAVIX", start="2010-01-01",
                             progress=False, auto_adjust=True)
            df = df.reset_index()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df.rename(columns={"Date": "DATE", "Close": "VIX_CLOSE",
                                    "Open": "VIX_OPEN", "High": "VIX_HIGH",
                                    "Low": "VIX_LOW"})
            df["DATE"] = pd.to_datetime(df["DATE"]).dt.normalize()
            df = df[["DATE", "VIX_OPEN", "VIX_HIGH", "VIX_LOW", "VIX_CLOSE"]].dropna()
            # Save locally for next run
            df.to_csv(self.vix_file, index=False)
            logger.info(f"VIX downloaded from yfinance: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"yfinance VIX failed: {e}")

        logger.error("Could not load India VIX from any source.")
        return pd.DataFrame()

    def _clean_vix_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise NSE VIX CSV columns (NSE naming varies)."""
        col_map = {}
        for col in df.columns:
            lo = col.strip().lower()
            if "date" in lo:
                col_map[col] = "DATE"
            elif lo in ("close", "eod_price", "vix_close", "vix"):
                col_map[col] = "VIX_CLOSE"
            elif lo == "open":
                col_map[col] = "VIX_OPEN"
            elif lo == "high":
                col_map[col] = "VIX_HIGH"
            elif lo == "low":
                col_map[col] = "VIX_LOW"
        df = df.rename(columns=col_map)
        df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce").dt.normalize()
        for c in ["VIX_CLOSE", "VIX_OPEN", "VIX_HIGH", "VIX_LOW"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["DATE", "VIX_CLOSE"]).sort_values("DATE").reset_index(drop=True)
        return df

    # ─── 4. FII / DII DATA ────────────────────────────────────────────────────
    def load_fii_dii(self) -> pd.DataFrame:
        """
        Load FII/DII net flow data.
        Sources: nsefin → local CSV → NSE direct
        """
        if NSEFIN_AVAILABLE:
            try:
                df = self.nse_client.get_fii_dii_data()
                df = self._clean_fii_df(df)
                df.to_csv(self.fii_file, index=False)
                logger.info(f"FII/DII loaded via nsefin: {len(df)} rows")
                return df
            except Exception as e:
                logger.warning(f"nsefin FII/DII failed: {e}")

        if self.fii_file.exists():
            df = pd.read_csv(self.fii_file)
            df = self._clean_fii_df(df)
            logger.info(f"FII/DII loaded from local CSV: {len(df)} rows")
            return df
            
        logger.error("Could not load FII/DII data.")
        return pd.DataFrame()

    # ─── 5. EXTENDED MARKET DATA (Global Signals) ──────────────────────────
    def load_extended_market_data(self) -> pd.DataFrame:
        """
        Load global market signals (risk, sentiment, macro) from processed cache.
        These are generated via market_data_extended.py.
        """
        cache_path = Path("data/extended/market_data_extended.parquet")
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            # Ensure DATE is in datetime format and is a column, not index
            if df.index.name == "DATE":
                df = df.reset_index()
            df["DATE"] = pd.to_datetime(df["DATE"]).dt.normalize()
            logger.info(f"Extended market data loaded: {len(df)} rows")
            return df
        else:
            logger.warning("Extended market data cache not found. Run market_data_extended.py first.")
            return pd.DataFrame()

        # NSE direct API
        try:
            session = self._get_nse_session()
            url = f"{NSE_BASE}/fiidiiTradeReact"
            resp = session.get(url, timeout=10)
            data = resp.json()
            df = pd.DataFrame(data)
            df = self._clean_fii_df(df)
            df.to_csv(self.fii_file, index=False)
            logger.info(f"FII/DII downloaded from NSE API: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"All FII/DII sources failed: {e}")
            return pd.DataFrame()

    def _clean_fii_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise FII/DII columns across data sources."""
        col_map = {}
        for col in df.columns:
            lo = col.strip().lower().replace(" ", "_")
            if "date" in lo:
                col_map[col] = "DATE"
            elif "fii" in lo and "cash" in lo and ("buy" in lo or "purchase" in lo):
                col_map[col] = "FII_CASH_BUY"
            elif "fii" in lo and "cash" in lo and "sell" in lo:
                col_map[col] = "FII_CASH_SELL"
            elif "fii" in lo and ("fut" in lo or "deriv" in lo) and ("buy" in lo or "purchase" in lo):
                col_map[col] = "FII_FNO_BUY"
            elif "fii" in lo and ("fut" in lo or "deriv" in lo) and "sell" in lo:
                col_map[col] = "FII_FNO_SELL"
            elif "dii" in lo and ("buy" in lo or "purchase" in lo):
                col_map[col] = "DII_BUY"
            elif "dii" in lo and "sell" in lo:
                col_map[col] = "DII_SELL"

        df = df.rename(columns=col_map)
        if "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce").dt.normalize()

        # Compute net flows
        for kind in [("FII_CASH", "FII_CASH_BUY", "FII_CASH_SELL"),
                     ("FII_FNO",  "FII_FNO_BUY",  "FII_FNO_SELL"),
                     ("DII",      "DII_BUY",       "DII_SELL")]:
            net_col, buy_col, sell_col = kind
            if buy_col in df.columns and sell_col in df.columns:
                df[f"{net_col}_NET"] = (
                    pd.to_numeric(df[buy_col], errors="coerce") -
                    pd.to_numeric(df[sell_col], errors="coerce")
                )

        df = df.sort_values("DATE").reset_index(drop=True)
        return df

    # ─── 5. EQUITY OHLCV (yfinance) ───────────────────────────────────────────
    def load_equity_ohlcv(
        self,
        tickers: Optional[Dict[str, str]] = None,
        start_date: str = "2018-01-01",
    ) -> pd.DataFrame:
        """
        Download OHLCV for underlying indices and macro assets via yfinance.

        Default tickers:
          ^NSEI        = Nifty 50
          ^NSEBANK     = Bank Nifty
          ^NSMIDCP     = Nifty Midcap
          ^INDIAVIX    = India VIX
          INR=X        = USD/INR
          GC=F         = Gold futures
          CL=F         = Crude oil
          ^GSPC        = S&P 500 (overnight signal)
          ^NDX         = Nasdaq 100
          USDINR=X     = USD-INR
        """
        if tickers is None:
            # We skip downloading these as market_data_extended.parquet
            # already covers NIFTY, BANKNIFTY, and MIDCAP natively.
            tickers = {}

        frames = []
        for ticker, name in tickers.items():
            try:
                raw = yf.download(ticker, start=start_date,
                                  progress=False, auto_adjust=True)
                if raw.empty:
                    continue
                raw = raw.reset_index()
                raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
                raw = raw.rename(columns={
                    "Date":   "DATE",
                    "Open":   f"{name}_OPEN",
                    "High":   f"{name}_HIGH",
                    "Low":    f"{name}_LOW",
                    "Close":  f"{name}_CLOSE",
                    "Volume": f"{name}_VOLUME",
                })
                raw["DATE"] = pd.to_datetime(raw["DATE"]).dt.normalize()
                keep = ["DATE"] + [c for c in raw.columns if c.startswith(name)]
                frames.append(raw[keep])
                logger.info(f"Downloaded {name} ({ticker}): {len(raw)} rows")
            except Exception as e:
                logger.warning(f"yfinance failed for {ticker}: {e}")

        if not frames:
            return pd.DataFrame()

        result = frames[0]
        for f in frames[1:]:
            result = result.merge(f, on="DATE", how="outer")
        result = result.sort_values("DATE").reset_index(drop=True)
        return result

    # ─── 6. NSE PARTICIPANT-WISE OI (FII Long/Short ratio) ────────────────────
    def get_participant_oi(self) -> pd.DataFrame:
        """
        Fetch NSE participant-wise OI data.
        Shows FII, DII, PRO, CLIENT positions in futures (long/short).
        Very useful: FII long/short ratio in index futures is a key signal.
        """
        if NSEPYTHON_AVAILABLE:
            try:
                url = f"{NSE_BASE}/equity-stockIndices?index=PARTICIPANT_OI"
                raw = nsefetch(url)
                df = pd.DataFrame(raw.get("data", []))
                logger.info(f"Participant OI fetched: {len(df)} rows")
                return df
            except Exception as e:
                logger.warning(f"nsepython participant OI failed: {e}")

        try:
            session = self._get_nse_session()
            url = f"{NSE_BASE}/fii-dii"
            resp = session.get(url, timeout=10)
            data = resp.json()
            df = pd.DataFrame(data if isinstance(data, list) else [])
            logger.info(f"Participant OI from direct API: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Participant OI fetch failed: {e}")
            return pd.DataFrame()

    # ─── 7. BULK & BLOCK DEALS ───────────────────────────────────────────────
    def get_bulk_block_deals(self, date: Optional[dt.datetime] = None) -> pd.DataFrame:
        """
        Fetch NSE bulk and block deal data and aggregate by symbol using nsepython.
        """
        if not NSEPYTHON_AVAILABLE:
            logger.warning("nsepython not available. Skipping bulk/block deals.")
            return pd.DataFrame()
            
        try:
            # nsepython returns DataFrames directly
            from nsepython import get_bulkdeals, get_blockdeals
            
            df_bulk = get_bulkdeals()
            df_block = get_blockdeals()
            
            frames = []
            if isinstance(df_bulk, pd.DataFrame) and not df_bulk.empty:
                frames.append(df_bulk)
            if isinstance(df_block, pd.DataFrame) and not df_block.empty:
                frames.append(df_block)
                
            if not frames:
                return pd.DataFrame()
                
            df = pd.concat(frames, ignore_index=True)
            
            # Map columns: ['Date', 'Symbol', 'Security Name', 'Client Name', 'Buy/Sell', 'Quantity Traded', 'Trade Price / Wght. Avg. Price']
            rename_map = {
                'Symbol': 'SYMBOL',
                'Client Name': 'CLIENT_NAME',
                'Buy/Sell': 'DEAL_TYPE',
                'Quantity Traded': 'QTY',
                'Trade Price / Wght. Avg. Price': 'PRICE',
                'Date': 'DATE_STR'
            }
            df = df.rename(columns=rename_map)
            
            # Conversion
            for col in ['QTY', 'PRICE']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            df['VALUE_CR'] = (df['QTY'] * df['PRICE']) / 1e7
            df['DATE'] = pd.to_datetime(df['DATE_STR'], errors='coerce').dt.normalize()
            
            # Institutional flag: 1 if client name contains institutional keywords
            inst_keywords = ['FUND', 'CAPITAL', 'INVEST', 'ASSET', 'BANK', 'INSURANCE', 'MF', 'AMC', 'TRUST', 'ADVISOR']
            df['INSTITUTIONAL_FLAG'] = df['CLIENT_NAME'].str.upper().apply(
                lambda x: 1 if any(kw in str(x) for kw in inst_keywords) else 0
            ) if 'CLIENT_NAME' in df.columns else 0
            
            # Aggregate by SYMBOL and DATE
            summary = []
            # We filter for the requested date if provided
            if date:
                target_date = pd.to_datetime(date).normalize()
                df = df[df['DATE'] == target_date]
                
            for (symbol, d), group in df.groupby(['SYMBOL', 'DATE']):
                buy_mask = group['DEAL_TYPE'].str.upper() == 'BUY'
                sell_mask = group['DEAL_TYPE'].str.upper() == 'SELL'
                
                net_qty = group.loc[buy_mask, 'QTY'].sum() - group.loc[sell_mask, 'QTY'].sum()
                net_val = group.loc[buy_mask, 'VALUE_CR'].sum() - group.loc[sell_mask, 'VALUE_CR'].sum()
                inst_flag = 1 if group['INSTITUTIONAL_FLAG'].max() == 1 else 0
                
                summary.append({
                    'SYMBOL': symbol,
                    'DATE': d,
                    'NET_BULK_QTY': net_qty,
                    'NET_BULK_VALUE_CR': net_val,
                    'INSTITUTIONAL_DEAL': inst_flag
                })
                
            return pd.DataFrame(summary)
        except Exception as e:
            logger.warning(f"Bulk/Block deals fetch failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Bulk/Block deals fetch failed: {e}")
            return pd.DataFrame()

    # ─── 8. CORPORATE ANNOUNCEMENTS SENTIMENT ────────────────────────────────
    def get_nse_announcements_sentiment(self, days_back: int = 5) -> pd.DataFrame:
        """
        Fetch NSE corporate announcements and score sentiment.
        """
        try:
            session = self._get_nse_session()
            url = f"{NSE_BASE}/corporate-announcements?index=equities"
            resp = session.get(url, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"Announcements fetch failed: {resp.status_code}")
                return pd.DataFrame()
                
            data = resp.json() 
            if not isinstance(data, list):
                data = data.get('data', [])
                
            if not data:
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            # Map columns based on debug: ['symbol', 'desc', 'dt', 'attchmntText', 'sort_date', ...]
            rename_map = {
                'symbol': 'SYMBOL',
                'desc': 'TITLE',
                'attchmntText': 'DESC',
                'sort_date': 'DATE_STR'
            }
            df = df.rename(columns=rename_map)
            df['DATE'] = pd.to_datetime(df['DATE_STR'], errors='coerce').dt.normalize()
            
            # Filter by date range
            cutoff = pd.to_datetime(dt.datetime.now() - dt.timedelta(days=days_back)).normalize()
            df = df[df['DATE'].notna() & (df['DATE'] >= cutoff)]
            
            pos_kw = ['dividend', 'bonus', 'buyback', 'profit', 'growth', 'results beat', 'acquisition', 'order', 'contract']
            neg_kw = ['loss', 'default', 'penalty', 'sebi notice', 'fraud', 'delay', 'fire', 'shutdown', 'resignation']
            
            def score_text(row):
                text = str(row['TITLE'] or '') + " " + str(row['DESC'] or '')
                text = text.lower()
                score = 0
                for kw in pos_kw:
                    if kw in text: score += 1
                for kw in neg_kw:
                    if kw in text: score -= 1
                has_results = 'results' in text or 'financial statement' in text
                return pd.Series([score, has_results])

            df[['SCORE', 'HAS_RESULTS']] = df.apply(score_text, axis=1)
            
            # Aggregate per symbol per date
            agg = df.groupby(['SYMBOL', 'DATE']).agg(
                announcement_count=('SYMBOL', 'count'),
                sentiment_score=('SCORE', 'mean'),
                has_results=('HAS_RESULTS', 'max')
            ).reset_index()
            
            # Normalize sentiment score to -1 to +1
            agg['sentiment_score'] = agg['sentiment_score'].clip(-1, 1)
            
            return agg
        except Exception as e:
            logger.warning(f"NSE announcements sentiment failed: {e}")
            return pd.DataFrame()

    # ─── 9. GOOGLE TRENDS PROXY ──────────────────────────────────────────────
    def get_google_trends_proxy(self, keywords: List[str] = None) -> pd.DataFrame:
        """
        Fetch Google Trends data for stock market sentiment in India.
        """
        if not PYTRENDS_AVAILABLE:
            logger.warning("pytrends not installed. Skipping Google Trends.")
            return pd.DataFrame()
            
        if keywords is None:
            keywords = ["Nifty crash", "stock market crash India", "buy Nifty", "market rally India"]
            
        try:
            pytrends = TrendReq(hl='en-IN', tz=330)
            pytrends.build_payload(keywords, timeframe='today 3-m', geo='IN')
            df = pytrends.interest_over_time()
            
            if df.empty:
                return pd.DataFrame()
                
            if 'isPartial' in df.columns:
                df = df.drop(columns=['isPartial'])
                
            # Weekly to daily
            df = df.resample('D').ffill()
            df = df.reset_index().rename(columns={'date': 'DATE'})
            df['DATE'] = pd.to_datetime(df['DATE']).dt.normalize()
            
            # Normalize 0-100
            for kw in keywords:
                if kw in df.columns:
                    mx = df[kw].max()
                    if mx > 0:
                        df[kw] = (df[kw] / mx) * 100
            
            # Indices
            panic_terms = [k for k in keywords if "crash" in k.lower()]
            greed_terms = [k for k in keywords if "buy" in k.lower() or "rally" in k.lower()]
            
            if panic_terms:
                df['panic_index'] = df[panic_terms].mean(axis=1)
            if greed_terms:
                df['greed_index'] = df[greed_terms].mean(axis=1)
                
            return df[['DATE', 'panic_index', 'greed_index']]
        except Exception as e:
            logger.warning(f"Google Trends proxy failed: {e}")
            return pd.DataFrame()

    # ─── 10. PCR AGGREGATION ──────────────────────────────────────────────────
    def compute_aggregate_pcr(
        self,
        chain_df: pd.DataFrame,
        underlying_price: float,
        n_strikes: int = ATM_RANGE_STRIKES,
        step: float = NIFTY_STRIKE_STEP,
    ) -> Dict[str, float]:
        """
        Compute multiple PCR variants from an option chain DataFrame.

        Returns:
            pcr_atm:          PCR using ATM ± n_strikes strikes
            pcr_full:         PCR using full chain
            pcr_otm:          PCR of OTM options only
            put_call_ratio_oi: standard OI-based PCR
            put_call_ratio_vol: volume-based PCR
            max_call_oi_strike: strike with highest call OI (resistance)
            max_put_oi_strike:  strike with highest put OI (support)
            max_pain:           max pain strike price
        """
        if chain_df.empty:
            return {}

        # ATM-range PCR
        atm = round(underlying_price / step) * step
        atm_range = chain_df[
            (chain_df["STRIKE"] >= atm - n_strikes * step) &
            (chain_df["STRIKE"] <= atm + n_strikes * step)
        ]

        def safe_pcr(df, oi_col_put="PE_OI", oi_col_call="CE_OI"):
            total_call = df[oi_col_call].sum()
            return df[oi_col_put].sum() / total_call if total_call > 0 else np.nan

        # OTM only: puts below ATM, calls above ATM
        otm_puts = chain_df[chain_df["STRIKE"] < atm]["PE_OI"].sum()
        otm_calls = chain_df[chain_df["STRIKE"] > atm]["CE_OI"].sum()
        pcr_otm = otm_puts / otm_calls if otm_calls > 0 else np.nan

        # Max pain
        max_pain = self._compute_max_pain(chain_df)

        # GEX (Gamma Exposure) if gamma available
        gex = np.nan
        if "CE_GAMMA" in chain_df.columns and "PE_GAMMA" in chain_df.columns:
            gex = (
                (chain_df["CE_GAMMA"] * chain_df["CE_OI"]).sum() -
                (chain_df["PE_GAMMA"] * chain_df["PE_OI"]).sum()
            ) * underlying_price ** 2 * 0.01

        return {
            "pcr_atm":            safe_pcr(atm_range),
            "pcr_full":           safe_pcr(chain_df),
            "pcr_otm":            pcr_otm,
            "pcr_volume":         safe_pcr(chain_df, "PE_VOLUME", "CE_VOLUME"),
            "max_call_oi_strike": chain_df.loc[chain_df["CE_OI"].idxmax(), "STRIKE"]
                                  if not chain_df.empty else np.nan,
            "max_put_oi_strike":  chain_df.loc[chain_df["PE_OI"].idxmax(), "STRIKE"]
                                  if not chain_df.empty else np.nan,
            "max_pain":           max_pain,
            "gex":                gex,
            "atm_iv_call":        chain_df.loc[chain_df["STRIKE"] == atm, "CE_IV"].values[0]
                                  if atm in chain_df["STRIKE"].values else np.nan,
            "atm_iv_put":         chain_df.loc[chain_df["STRIKE"] == atm, "PE_IV"].values[0]
                                  if atm in chain_df["STRIKE"].values else np.nan,
            "iv_skew":            (
                chain_df.loc[chain_df["STRIKE"] == atm, "CE_IV"].values[0] -
                chain_df.loc[chain_df["STRIKE"] == atm, "PE_IV"].values[0]
            ) if atm in chain_df["STRIKE"].values else np.nan,
        }

    def _compute_max_pain(self, chain_df: pd.DataFrame) -> float:
        """
        Compute max pain strike: the strike at which total option buyer losses
        (i.e., total option writer profit) are maximized.
        """
        if chain_df.empty:
            return np.nan
        strikes = chain_df["STRIKE"].unique()
        min_pain = float("inf")
        max_pain_strike = strikes[0]

        for s in strikes:
            # Call buyer losses at expiry = sum of (strike - S) * OI for strikes < S
            call_loss = sum(
                max(0, k - s) * chain_df.loc[chain_df["STRIKE"] == k, "CE_OI"].values[0]
                for k in strikes
            )
            # Put buyer losses at expiry = sum of (S - strike) * OI for strikes > S
            put_loss = sum(
                max(0, s - k) * chain_df.loc[chain_df["STRIKE"] == k, "PE_OI"].values[0]
                for k in strikes
            )
            total_pain = call_loss + put_loss
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = s

        return float(max_pain_strike)

    def load_option_chain_history(self) -> pd.DataFrame:
        """
        Aggregate and load historical option chain data from data/option_chain/*.csv.
        Returns a daily average of PCR and Max Pain per symbol.
        """
        path = Path("data/option_chain")
        if not path.exists():
            return pd.DataFrame()
            
        all_data = []
        files = list(path.glob("*.csv"))
        if not files:
            return pd.DataFrame()
            
        logger.info(f"Processing {len(files)} historical option chain files...")
        for f in tqdm(files, desc="Parsing Option Chain data", unit="file"):
            try:
                # Format could be symbol_YYYYMMDD_HHMM.csv OR symbol_expiry_YYYYMMDD_HHMM.csv
                parts = f.stem.split("_")
                if len(parts) < 2: continue
                
                symbol = parts[0].upper()
                if symbol == "NIFTY": symbol = "NIFTY"
                elif symbol == "BANKNIFTY": symbol = "BANKNIFTY"
                
                # The date is usually the 2nd to last part if HHMM is last, or last part if YYYYMMDD is last
                # Let's look for an 8-digit part
                date_str = None
                for p in parts:
                    if len(p) == 8 and p.isdigit():
                        date_str = p
                        break
                
                if not date_str: continue
                date = pd.to_datetime(date_str, format="%Y%m%d").normalize()
                
                df = pd.read_csv(f)
                if all(c in df.columns for c in ["CE_OI", "PE_OI", "STRIKE"]):
                    pcr = df["PE_OI"].sum() / (df["CE_OI"].sum() + 1e-9)
                    max_pain = self._compute_max_pain(df)
                    all_data.append({
                        "DATE": date,
                        "SYMBOL": symbol,
                        "pcr_full": pcr,
                        "max_pain": max_pain
                    })
            except Exception:
                continue
                
        if not all_data:
            return pd.DataFrame()
            
        df_oc = pd.DataFrame(all_data)
        # Aggregate by day
        df_oc = df_oc.groupby(["DATE", "SYMBOL"]).agg({
            "pcr_full": "mean",
            "max_pain": "mean"
        }).reset_index()
        
        return df_oc

    # ─── 8. MASTER MERGE ──────────────────────────────────────────────────────
    def get_full_dataset(
        self,
        start_date: str = "2018-01-01",
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Build the complete merged dataset ready for feature engineering.

        Returns a DataFrame indexed by (DATE, SYMBOL) with bhavcopy,
        VIX, FII/DII, and equity OHLCV columns merged.
        """
        if end_date is None:
            end_date = dt.date.today().strftime("%Y-%m-%d")

        if symbols is None:
            symbols = ["NIFTY", "BANKNIFTY"]

        logger.info(f"Building full dataset: {start_date} → {end_date} | symbols: {symbols}")

        # Load all sources
        bhav    = self.load_bhavcopy_range(start_date, end_date, symbols)
        vix     = self.load_india_vix()
        fii_dii = self.load_fii_dii()
        equity  = self.load_equity_ohlcv(start_date=start_date)
        
        # New sources (optional)
        deals = self.get_bulk_block_deals()
        sentiment = self.get_nse_announcements_sentiment()
        trends = self.get_google_trends_proxy()
        oc_hist = self.load_option_chain_history()

        if bhav.empty:
            logger.error("Bhavcopy data is empty — cannot build dataset.")
            return pd.DataFrame()

        # Merge on DATE
        df = bhav.copy()

        if not vix.empty:
            vix_cols = ["DATE"] + [c for c in vix.columns if c.startswith("VIX")]
            df = df.merge(vix[vix_cols], on="DATE", how="left")

        if not fii_dii.empty:
            net_cols = ["DATE"] + [c for c in fii_dii.columns if c.endswith("_NET")]
            df = df.merge(fii_dii[net_cols], on="DATE", how="left")

        if not equity.empty:
            df = df.merge(equity, on="DATE", how="left")
            
        if not deals.empty:
            df = df.merge(deals, on=["DATE", "SYMBOL"], how="left")
            
        if not sentiment.empty:
            df = df.merge(sentiment, on=["DATE", "SYMBOL"], how="left")
            
        if not trends.empty:
            df = df.merge(trends, on="DATE", how="left")
            
        if not oc_hist.empty:
            df = df.merge(oc_hist, on=["DATE", "SYMBOL"], how="left")
            logger.info(f"Merged Option Chain history: {len(oc_hist)} rows")

        # ── 9. EXTENDED MARKET DATA MERGE ──
        extended = self.load_extended_market_data()
        ext_cols = []
        if not extended.empty:
            df = df.merge(extended, on="DATE", how="left")
            ext_cols = [c for c in extended.columns if c != "DATE"]
            logger.info(f"Merged {len(extended.columns)-1} extended global signals")

        # Forward-fill macro data for non-trading days
        macro_cols = [c for c in df.columns
                      if any(c.startswith(p) for p in
                             ["VIX", "FII", "DII", "NIFTY_", "BANKNIFTY_",
                              "SPX_", "NDX_", "USDINR_", "GOLD_", "CRUDE_",
                              "RISK_", "DXY", "BRENT", "WTI", "US10Y"])]
        macro_cols = list(set(macro_cols + ext_cols))
        
        df[macro_cols] = df.groupby("SYMBOL")[macro_cols].ffill()

        # ── 10. FINAL DATA SANITIZATION ──
        # Resolve any _x/_y suffix collisions from multiple merges
        cols_to_drop = [c for c in df.columns if c.endswith("_y")]
        if cols_to_drop:
            logger.info(f"Dropping {len(cols_to_drop)} duplicate columns from merge collisions")
            df = df.drop(columns=cols_to_drop)
            # Rename _x to original
            df.columns = [c.replace("_x", "") for c in df.columns]

        # Final fill for any remaining NaNs (TFT requirement)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df.groupby("SYMBOL")[numeric_cols].ffill().fillna(0)

        df = df.sort_values(["DATE", "SYMBOL"]).reset_index(drop=True)

        logger.info(
            f"Full dataset ready: {len(df):,} rows | "
            f"{df['DATE'].nunique()} trading days | "
            f"columns: {df.shape[1]}"
        )
        return df


# ─── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    collector = DataCollector()

    # Quick connectivity check
    print("\n=== DATA COLLECTOR TEST ===\n")

    # Test VIX
    vix = collector.load_india_vix()
    if not vix.empty:
        print(f"✓ India VIX: {len(vix)} rows | latest: {vix.iloc[-1]['DATE'].date()} = {vix.iloc[-1]['VIX_CLOSE']:.2f}")

    # Test FII/DII
    fii = collector.load_fii_dii()
    if not fii.empty:
        net_cols = [c for c in fii.columns if "NET" in c]
        print(f"✓ FII/DII:   {len(fii)} rows | net flow cols: {net_cols}")

    # Test equity OHLCV
    eq = collector.load_equity_ohlcv(start_date="2024-01-01")
    if not eq.empty:
        print(f"✓ Equity OHLCV: {len(eq)} rows | {eq.shape[1]} columns")

    # Test live option chain
    print("\nFetching live NIFTY option chain...")
    oc = collector.get_live_option_chain("NIFTY")
    if not oc.empty:
        atm_pcr = collector.compute_aggregate_pcr(oc, oc["UNDERLYING"].iloc[0])
        print(f"✓ Option Chain: {len(oc)} strikes")
        print(f"  PCR (ATM):     {atm_pcr.get('pcr_atm', 'N/A'):.3f}")
        print(f"  PCR (full):    {atm_pcr.get('pcr_full', 'N/A'):.3f}")
        print(f"  Max Pain:      {atm_pcr.get('max_pain', 'N/A')}")
        print(f"  ATM Call IV:   {atm_pcr.get('atm_iv_call', 'N/A')}")
        print(f"  IV Skew:       {atm_pcr.get('iv_skew', 'N/A')}")

    # Test New Features
    print("\n=== NEW FEATURES TEST ===\n")
    
    print("Testing Bulk/Block Deals...")
    deals = collector.get_bulk_block_deals()
    if not deals.empty:
        print(f"✓ Bulk/Block Deals: {len(deals)} symbols | cols: {deals.columns.tolist()}")
    else:
        print("⚠ Bulk/Block Deals: No data fetched")
        
    print("\nTesting Announcements Sentiment...")
    sent = collector.get_nse_announcements_sentiment()
    if not sent.empty:
        print(f"✓ Announcements: {len(sent)} rows | avg sentiment: {sent['sentiment_score'].mean():.2f}")
    else:
        print("⚠ Announcements: No data fetched")
        
    print("\nTesting Google Trends...")
    trends = collector.get_google_trends_proxy()
    if not trends.empty:
        print(f"✓ Google Trends: {len(trends)} days | latest panic: {trends.iloc[-1]['panic_index']:.2f}")
    else:
        print("⚠ Google Trends: No data fetched")

    print("\n=== TEST COMPLETE ===")
