"""
market_data_extended.py
========================
Fetches every external market signal that meaningfully affects the
Indian F&O market for a buy-today / sell-tomorrow strategy.

Signal categories (100+ features):
  A. Precious Metals     — Gold, Silver, Platinum, Palladium
  B. Industrial Metals   — Copper, Aluminium, Zinc, Nickel, Lead
  C. Energy              — Brent, WTI, Natural Gas, Heating Oil
  D. Global Indices      — S&P500, Nasdaq, FTSE, DAX, Nikkei, Hang Seng, Shanghai, KOSPI, GIFT Nifty
  E. Currencies          — DXY, USD-INR, EUR-INR, GBP-INR, JPY-INR, EUR-USD, Yen
  F. Bonds & Rates       — US 10Y yield, US 2Y yield, India 10Y yield, yield curve, US-India spread
  G. Volatility Indices  — VIX (US), VXN (Nasdaq), OVX (Oil VIX), GVZ (Gold VIX)
  H. Market Sentiment    — CNN Fear & Greed proxy, BTC (risk-on signal), put-call ratio trends
  I. India Macro Events  — RBI meeting dates, F&O expiry flags, budget/result season, holidays
  J. Sector Rotation     — Nifty IT, Auto, FMCG, Bank, Metal, Pharma, Realty, PSU, Infra
  K. Cross-Asset Signals — Gold/Oil ratio, Copper/Gold ratio, Silver/Gold ratio, yield curve steepness

All data via yfinance (free, no API key).
Event flags via hard-coded calendar (updated quarterly).
"""

import logging
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("MarketDataExtended")

# ─── TICKER MASTER MAP ────────────────────────────────────────────────────────
TICKERS: Dict[str, str] = {

    # ── A. Precious Metals ────────────────────────────────────────────────
    "GC=F":     "GOLD",          # Gold Futures (USD/oz)
    "SI=F":     "SILVER",        # Silver Futures (USD/oz)
    "PL=F":     "PLATINUM",      # Platinum Futures
    "PA=F":     "PALLADIUM",     # Palladium Futures
    "GLD":      "GOLD_ETF",      # SPDR Gold ETF (proxy for institutional gold)

    # ── B. Industrial / Base Metals ───────────────────────────────────────
    "HG=F":     "COPPER",        # Copper Futures — leading economic indicator
    "ALI=F":    "ALUMINIUM",     # Aluminium Futures
    "ZNC=F":    "ZINC",          # Zinc Futures
    "NI=F":     "NICKEL",        # Nickel Futures
    "PB=F":     "LEAD",          # Lead Futures

    # ── C. Energy ─────────────────────────────────────────────────────────
    "BZ=F":     "BRENT",         # Brent Crude (India imports Brent) — HIGH IMPACT
    "CL=F":     "WTI",           # WTI Crude Oil
    "NG=F":     "NATGAS",        # Natural Gas
    "HO=F":     "HEATING_OIL",   # Heating Oil (distillate proxy)
    "USO":      "OIL_ETF",       # Oil ETF — institutional flows

    # ── D. Global Equity Indices ──────────────────────────────────────────
    "^GSPC":    "SPX",           # S&P 500
    "^NDX":     "NDX",           # Nasdaq 100
    "^DJI":     "DJIA",          # Dow Jones Industrial
    "^RUT":     "RUSSELL",       # Russell 2000 (risk appetite)
    "^FTSE":    "FTSE",          # FTSE 100 (UK)
    "^GDAXI":   "DAX",           # DAX (Germany)
    "^FCHI":    "CAC",           # CAC 40 (France)
    "^N225":    "NIKKEI",        # Nikkei 225 (Japan) — Asia open signal
    "^HSI":     "HANGSENG",      # Hang Seng (Hong Kong) — Asia open signal
    "000001.SS":"SHANGHAI",      # Shanghai Composite — China demand proxy
    "^KS11":    "KOSPI",         # KOSPI (South Korea)
    "^STI":     "STI",           # Straits Times Index (Singapore / SGX proxy)
    "EEM":      "EM_ETF",        # iShares EM ETF — emerging market flows
    "FXI":      "CHINA_ETF",     # China large-cap ETF

    # ── E. Indian Indices (underlying) ────────────────────────────────────
    "^NSEI":    "NIFTY",         # Nifty 50
    "^NSEBANK": "BANKNIFTY",     # Bank Nifty
    "^NSMIDCP": "MIDCAP",        # Nifty Midcap 100
    "^CNXIT":   "NIFTY_IT",      # Nifty IT — high weight, USD earner
    "^CNXAUTO": "NIFTY_AUTO",    # Nifty Auto — oil-sensitive
    "^CNXFMCG": "NIFTY_FMCG",   # Nifty FMCG — defensive
    "^CNXPHARMA":"NIFTY_PHARMA", # Nifty Pharma
    "^CNXMETAL": "NIFTY_METAL",  # Nifty Metal — commodities sensitive
    "^CNXREALTY":"NIFTY_REALTY", # Nifty Realty — rate sensitive
    "^CNXINFRA": "NIFTY_INFRA",  # Nifty Infra
    "^CNXPSE":  "NIFTY_PSE",    # Nifty PSE (Public Sector)
    "^CNXENERGY":"NIFTY_ENERGY", # Nifty Energy

    # ── F. Currencies ─────────────────────────────────────────────────────
    "DX-Y.NYB": "DXY",          # US Dollar Index — master currency signal
    "INR=X":    "USDINR",        # USD/INR — direct INR rate
    "EURINR=X": "EURINR",        # EUR/INR
    "GBPINR=X": "GBPINR",        # GBP/INR
    "JPYINR=X": "JPYINR",        # JPY/INR
    "EURUSD=X": "EURUSD",        # EUR/USD
    "JPY=X":    "USDJPY",        # USD/JPY (Yen carry trade)
    "CNY=X":    "USDCNY",        # USD/CNY (China FX)
    "AUDUSD=X": "AUDUSD",        # AUD/USD (commodity currency)

    # ── G. Bonds & Rates ──────────────────────────────────────────────────
    "^TNX":     "US10Y",         # US 10-Year Treasury Yield — most watched
    "^IRX":     "US3M",          # US 3-Month T-Bill
    "^TYX":     "US30Y",         # US 30-Year Yield
    "^FVX":     "US5Y",          # US 5-Year Yield
    "TLT":      "BOND_ETF",      # 20Y+ Treasury ETF (inverse yield signal)

    # ── H. Volatility Indices ─────────────────────────────────────────────
    "^VIX":     "USVIX",         # CBOE VIX — global fear gauge
    "^VXN":     "VXN",           # Nasdaq Volatility Index
    "OVX":      "OIL_VIX",       # Oil Volatility Index
    "GVZ":      "GOLD_VIX",      # Gold Volatility Index
    "^VVIX":    "VVIX",          # VIX of VIX (volatility of volatility)
    "^INDIAVIX":"INDIAVIX",      # India VIX (if not already loaded)

    # ── I. Crypto (risk-on / risk-off signal) ─────────────────────────────
    "BTC-USD":  "BITCOIN",       # Bitcoin — leading risk appetite indicator
    "ETH-USD":  "ETHEREUM",      # Ethereum

    # ── J. Commodities (agriculture) ─────────────────────────────────────
    "ZW=F":     "WHEAT",         # Wheat — food inflation
    "ZC=F":     "CORN",          # Corn
    "CC=F":     "COCOA",         # Cocoa
    "CT=F":     "COTTON",        # Cotton — textile sector India

    # ── K. Bonds / Rate Sensitive ETFs ────────────────────────────────────
    "HYG":      "HIGH_YIELD",    # High-Yield Bond ETF — credit risk signal
    "LQD":      "CORP_BOND",     # Investment Grade Corporate Bonds
    "SHY":      "SHORT_BOND",    # Short-term Treasury ETF
}

# ─── INDIA MACRO EVENT CALENDAR ───────────────────────────────────────────────
# Update quarterly. Format: "YYYY-MM-DD": "event_name"
INDIA_EVENTS: Dict[str, str] = {
    # RBI MPC meetings (typically 8 per year)
    "2024-02-08": "RBI_MPC", "2024-04-05": "RBI_MPC", "2024-06-07": "RBI_MPC",
    "2024-08-08": "RBI_MPC", "2024-10-09": "RBI_MPC", "2024-12-06": "RBI_MPC",
    "2025-02-07": "RBI_MPC", "2025-04-09": "RBI_MPC", "2025-06-06": "RBI_MPC",
    "2025-08-07": "RBI_MPC", "2025-10-08": "RBI_MPC", "2025-12-05": "RBI_MPC",
    "2026-02-06": "RBI_MPC", "2026-04-03": "RBI_MPC",
    # Union Budget
    "2024-02-01": "UNION_BUDGET", "2024-07-23": "UNION_BUDGET_FULL",
    "2025-02-01": "UNION_BUDGET", "2026-02-01": "UNION_BUDGET",
    # US FOMC meetings
    "2024-01-31": "US_FOMC", "2024-03-20": "US_FOMC", "2024-05-01": "US_FOMC",
    "2024-06-12": "US_FOMC", "2024-07-31": "US_FOMC", "2024-09-18": "US_FOMC",
    "2024-11-07": "US_FOMC", "2024-12-18": "US_FOMC",
    "2025-01-29": "US_FOMC", "2025-03-19": "US_FOMC", "2025-05-07": "US_FOMC",
    "2025-06-18": "US_FOMC", "2025-07-30": "US_FOMC", "2025-09-17": "US_FOMC",
    "2025-11-05": "US_FOMC", "2025-12-17": "US_FOMC",
    "2026-01-28": "US_FOMC", "2026-03-18": "US_FOMC",
    # India General Elections / Major Events
    "2024-06-04": "INDIA_ELECTION_RESULT",
    # US CPI releases (high-impact for global risk)
    # India CPI releases (2nd week of month, Wednesday)
}

# NSE Weekly expiry: every Thursday
# NSE Monthly expiry: last Thursday of month
NSE_MONTHLY_EXPIRY_2024_25 = [
    "2024-01-25","2024-02-29","2024-03-28","2024-04-25","2024-05-30",
    "2024-06-27","2024-07-25","2024-08-29","2024-09-26","2024-10-31",
    "2024-11-28","2024-12-26","2025-01-30","2025-02-27","2025-03-27",
    "2025-04-24","2025-05-29","2025-06-26","2025-07-31","2025-08-28",
    "2025-09-25","2025-10-30","2025-11-27","2025-12-25",
    "2026-01-29","2026-02-26","2026-03-26",
]


# ─────────────────────────────────────────────────────────────────────────────
class ExtendedMarketData:
    """
    Downloads and computes all external market signals.
    Produces a date-indexed DataFrame ready to merge into the main feature set.
    """

    def __init__(self, cache_dir: str = "data/extended", start_date: str = "2019-01-01"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = start_date

    # ─── MASTER DOWNLOAD ─────────────────────────────────────────────────────
    def download_all(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Downloads all tickers and returns a single wide DataFrame indexed by DATE.
        Uses local parquet cache to avoid re-downloading on every run.
        """
        cache_file = self.cache_dir / "market_data_extended.parquet"

        if cache_file.exists() and not force_refresh:
            age_hours = (dt.datetime.now().timestamp() -
                         cache_file.stat().st_mtime) / 3600
            if age_hours < 6:
                logger.info(f"Using cached extended data ({age_hours:.1f}h old)")
                return pd.read_parquet(cache_file)

        logger.info(f"Downloading {len(TICKERS)} tickers from yfinance...")

        # Download in batches to avoid rate limiting
        batch_size = 20
        ticker_list = list(TICKERS.keys())
        frames = []

        for i in range(0, len(ticker_list), batch_size):
            batch = ticker_list[i:i + batch_size]
            try:
                raw = yf.download(
                    batch,
                    start=self.start_date,
                    progress=False,
                    auto_adjust=True,
                    group_by="ticker",
                )
                df_batch = self._extract_close_series(raw, batch)
                frames.append(df_batch)
                logger.info(f"Batch {i//batch_size + 1}: downloaded {len(batch)} tickers")
            except Exception as e:
                logger.warning(f"Batch download failed: {e} — retrying individually")
                df_batch = self._download_individually(batch)
                frames.append(df_batch)

        if not frames:
            logger.error("All downloads failed")
            return pd.DataFrame()

        # Merge all on date
        result = frames[0]
        for f in frames[1:]:
            result = result.join(f, how="outer")

        result.index = pd.to_datetime(result.index).normalize()
        result.index.name = "DATE"
        result = result.sort_index()

        # Forward-fill weekends/holidays (markets closed)
        result = result.ffill(limit=3)

        # Add computed signals
        result = self._add_computed_signals(result)

        # Add event flags
        result = self._add_event_flags(result)

        # Add India 10Y yield (if available via NSE bond data)
        result = self._add_india_bond_yield(result)

        result.to_parquet(cache_file)
        logger.info(f"Extended data saved: {len(result)} rows × {len(result.columns)} columns")
        return result

    def _extract_close_series(self, raw: pd.DataFrame, tickers: list) -> pd.DataFrame:
        """Extract Close prices from multi-ticker yfinance download."""
        result = pd.DataFrame(index=raw.index)

        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in tickers:
                name = TICKERS.get(ticker, ticker.replace("=F","").replace("=X",""))
                try:
                    col = (ticker, "Close") if (ticker, "Close") in raw.columns else None
                    if col:
                        result[f"{name}_CLOSE"] = raw[col]
                    # Also grab volume where meaningful
                    vcol = (ticker, "Volume") if (ticker, "Volume") in raw.columns else None
                    if vcol and name in ["SPX", "NDX", "GOLD_ETF", "OIL_ETF", "EM_ETF"]:
                        result[f"{name}_VOL"] = raw[vcol]
                except Exception:
                    pass
        else:
            # Single ticker fallback
            if "Close" in raw.columns:
                ticker = tickers[0]
                name = TICKERS.get(ticker, ticker)
                result[f"{name}_CLOSE"] = raw["Close"]
        return result

    def _download_individually(self, tickers: list) -> pd.DataFrame:
        """Fallback: download one ticker at a time."""
        result = pd.DataFrame()
        for ticker in tickers:
            name = TICKERS.get(ticker, ticker)
            try:
                raw = yf.download(ticker, start=self.start_date,
                                  progress=False, auto_adjust=True)
                if not raw.empty:
                    s = raw["Close"].rename(f"{name}_CLOSE")
                    result = result.join(s, how="outer") if not result.empty else s.to_frame()
            except Exception as e:
                logger.debug(f"Skip {ticker}: {e}")
        return result

    # ─── COMPUTED SIGNALS ────────────────────────────────────────────────────
    def _add_computed_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive cross-asset ratios, spreads, and overnight return signals.
        These are among the most predictive features for next-day Nifty direction.
        """
        c = df.copy()

        # ── Overnight / pre-market signals (shift by 1 = yesterday's close) ──
        # When Indian market opens at 9:15 AM, these are already known
        for col in ["SPX_CLOSE", "NDX_CLOSE", "DJIA_CLOSE", "NIKKEI_CLOSE",
                    "HANGSENG_CLOSE", "SHANGHAI_CLOSE", "FTSE_CLOSE", "DAX_CLOSE"]:
            if col not in c.columns:
                continue
            name = col.replace("_CLOSE", "")
            # 1-day return (today's change, available before Indian open)
            c[f"{name}_1D_RET"] = c[col].pct_change(1) * 100
            # 5-day return
            c[f"{name}_5D_RET"] = c[col].pct_change(5) * 100
            # Gap to 20-day EMA
            ema20 = c[col].ewm(span=20, adjust=False).mean()
            c[f"{name}_EMA_DEV"] = (c[col] - ema20) / ema20 * 100
            # Above/below 50-day EMA (regime)
            ema50 = c[col].ewm(span=50, adjust=False).mean()
            c[f"{name}_ABOVE_EMA50"] = (c[col] > ema50).astype(int)

        # ── US Market Overnight Return (most impactful signal for Indian open) ──
        if "SPX_CLOSE" in c.columns:
            c["SPX_OVERNIGHT_RET"] = c["SPX_CLOSE"].pct_change(1).shift(1) * 100
            c["SPX_STRONG_UP"]   = (c["SPX_OVERNIGHT_RET"] > 1.0).astype(int)
            c["SPX_STRONG_DOWN"] = (c["SPX_OVERNIGHT_RET"] < -1.0).astype(int)

        if "NDX_CLOSE" in c.columns:
            c["NDX_OVERNIGHT_RET"] = c["NDX_CLOSE"].pct_change(1).shift(1) * 100

        # ── Asia-Pacific Morning Signal (Nikkei + Hang Seng before India opens) ─
        apac_cols = [col for col in ["NIKKEI_1D_RET", "HANGSENG_1D_RET", "KOSPI_1D_RET"]
                     if col in c.columns]
        if apac_cols:
            c["APAC_AVG_RET"] = c[apac_cols].mean(axis=1)
            c["APAC_ALL_UP"]  = (c[apac_cols] > 0).all(axis=1).astype(int)
            c["APAC_ALL_DOWN"]= (c[apac_cols] < 0).all(axis=1).astype(int)

        # ── Dollar Index (DXY) — strong dollar = pressure on Indian markets ────
        if "DXY_CLOSE" in c.columns:
            c["DXY_1D_RET"] = c["DXY_CLOSE"].pct_change(1) * 100
            c["DXY_5D_RET"] = c["DXY_CLOSE"].pct_change(5) * 100
            c["DXY_ABOVE_103"] = (c["DXY_CLOSE"] > 103).astype(int)
            ema20 = c["DXY_CLOSE"].ewm(span=20, adjust=False).mean()
            c["DXY_EMA_DEV"] = (c["DXY_CLOSE"] - ema20) / ema20 * 100

        # ── USD-INR dynamics ──────────────────────────────────────────────────
        if "USDINR_CLOSE" in c.columns:
            c["USDINR_1D_CHG"]  = c["USDINR_CLOSE"].pct_change(1) * 100
            c["USDINR_5D_CHG"]  = c["USDINR_CLOSE"].pct_change(5) * 100
            c["USDINR_ABOVE_84"] = (c["USDINR_CLOSE"] > 84).astype(int)
            c["INR_WEAKNESS"]   = (c["USDINR_1D_CHG"] > 0.3).astype(int)
            c["INR_STRENGTH"]   = (c["USDINR_1D_CHG"] < -0.3).astype(int)

        # ── Brent Crude — high impact on India (largest import bill) ──────────
        if "BRENT_CLOSE" in c.columns:
            c["BRENT_1D_RET"] = c["BRENT_CLOSE"].pct_change(1) * 100
            c["BRENT_5D_RET"] = c["BRENT_CLOSE"].pct_change(5) * 100
            c["BRENT_20D_RET"]= c["BRENT_CLOSE"].pct_change(20) * 100
            c["BRENT_ABOVE_90"]= (c["BRENT_CLOSE"] > 90).astype(int)
            c["BRENT_SPIKE"]   = (c["BRENT_1D_RET"] > 3).astype(int)
            c["BRENT_CRASH"]   = (c["BRENT_1D_RET"] < -3).astype(int)

        # ── Gold signals ──────────────────────────────────────────────────────
        if "GOLD_CLOSE" in c.columns:
            c["GOLD_1D_RET"]  = c["GOLD_CLOSE"].pct_change(1) * 100
            c["GOLD_5D_RET"]  = c["GOLD_CLOSE"].pct_change(5) * 100
            c["GOLD_SAFE_HAVEN"] = (c["GOLD_1D_RET"] > 1.0).astype(int)  # fear buying
            # Gold/Silver ratio (high = risk-off, low = risk-on)
            if "SILVER_CLOSE" in c.columns:
                c["GOLD_SILVER_RATIO"] = c["GOLD_CLOSE"] / c["SILVER_CLOSE"].replace(0, np.nan)
                c["GSR_5D_CHG"] = c["GOLD_SILVER_RATIO"].pct_change(5) * 100

        # ── Silver signals ─────────────────────────────────────────────────────
        if "SILVER_CLOSE" in c.columns:
            c["SILVER_1D_RET"] = c["SILVER_CLOSE"].pct_change(1) * 100
            c["SILVER_5D_RET"] = c["SILVER_CLOSE"].pct_change(5) * 100

        # ── Copper — "Doctor Copper" = global growth indicator ────────────────
        if "COPPER_CLOSE" in c.columns:
            c["COPPER_1D_RET"] = c["COPPER_CLOSE"].pct_change(1) * 100
            c["COPPER_5D_RET"] = c["COPPER_CLOSE"].pct_change(5) * 100
            c["COPPER_20D_RET"]= c["COPPER_CLOSE"].pct_change(20) * 100
            # Copper/Gold ratio — best real-rate proxy (risk-on when rising)
            if "GOLD_CLOSE" in c.columns:
                c["COPPER_GOLD_RATIO"] = c["COPPER_CLOSE"] / c["GOLD_CLOSE"].replace(0, np.nan)
                c["CGR_5D_CHG"] = c["COPPER_GOLD_RATIO"].pct_change(5) * 100
                c["CGR_RISING"] = (c["CGR_5D_CHG"] > 0).astype(int)

        # ── US Bond Yields ─────────────────────────────────────────────────────
        if "US10Y_CLOSE" in c.columns:
            c["US10Y_1D_CHG"]  = c["US10Y_CLOSE"].diff(1)
            c["US10Y_5D_CHG"]  = c["US10Y_CLOSE"].diff(5)
            c["US10Y_ABOVE_4"] = (c["US10Y_CLOSE"] > 4.0).astype(int)
            c["US10Y_RISING"]  = (c["US10Y_1D_CHG"] > 0.05).astype(int)

        # ── Yield Curve (2Y-10Y spread) — inversion = recession signal ────────
        if "US10Y_CLOSE" in c.columns and "US3M_CLOSE" in c.columns:
            c["YIELD_CURVE_SPREAD"] = c["US10Y_CLOSE"] - c["US3M_CLOSE"]
            c["YIELD_CURVE_INVERTED"] = (c["YIELD_CURVE_SPREAD"] < 0).astype(int)

        if "US10Y_CLOSE" in c.columns and "US5Y_CLOSE" in c.columns:
            c["YIELD_5_10_SPREAD"] = c["US10Y_CLOSE"] - c["US5Y_CLOSE"]

        # ── US VIX signals ─────────────────────────────────────────────────────
        if "USVIX_CLOSE" in c.columns:
            c["USVIX_1D_CHG"]  = c["USVIX_CLOSE"].pct_change(1) * 100
            c["USVIX_ABOVE_20"] = (c["USVIX_CLOSE"] > 20).astype(int)
            c["USVIX_ABOVE_30"] = (c["USVIX_CLOSE"] > 30).astype(int)
            c["USVIX_SPIKE"]   = (c["USVIX_1D_CHG"] > 20).astype(int)
            vix_52hi = c["USVIX_CLOSE"].rolling(252).max()
            vix_52lo = c["USVIX_CLOSE"].rolling(252).min()
            c["USVIX_RANK"]    = (c["USVIX_CLOSE"] - vix_52lo) / (vix_52hi - vix_52lo + 1e-9)
            # VIX-India VIX convergence (both high = panic)
            if "INDIAVIX_CLOSE" in c.columns:
                c["VIX_SPREAD"] = c["INDIAVIX_CLOSE"] - c["USVIX_CLOSE"]
                c["DUAL_VIX_ELEVATED"] = (
                    (c["USVIX_CLOSE"] > 20) & (c["INDIAVIX_CLOSE"] > 18)
                ).astype(int)

        # ── Bitcoin — risk-on/off proxy ────────────────────────────────────────
        if "BITCOIN_CLOSE" in c.columns:
            c["BTC_1D_RET"]   = c["BITCOIN_CLOSE"].pct_change(1) * 100
            c["BTC_5D_RET"]   = c["BITCOIN_CLOSE"].pct_change(5) * 100
            c["BTC_RISK_ON"]  = (c["BTC_1D_RET"] > 3).astype(int)
            c["BTC_RISK_OFF"] = (c["BTC_1D_RET"] < -3).astype(int)

        # ── EM ETF flows — FII behaviour proxy ────────────────────────────────
        if "EM_ETF_CLOSE" in c.columns:
            c["EM_ETF_1D_RET"] = c["EM_ETF_CLOSE"].pct_change(1) * 100
            c["EM_ETF_5D_RET"] = c["EM_ETF_CLOSE"].pct_change(5) * 100
            c["EM_ETF_ABOVE_EMA20"] = (
                c["EM_ETF_CLOSE"] > c["EM_ETF_CLOSE"].ewm(span=20, adjust=False).mean()
            ).astype(int)

        # ── High-Yield credit spread proxy ────────────────────────────────────
        if "HIGH_YIELD_CLOSE" in c.columns:
            c["HYG_1D_RET"] = c["HIGH_YIELD_CLOSE"].pct_change(1) * 100
            c["CREDIT_STRESS"] = (c["HYG_1D_RET"] < -0.5).astype(int)

        # ── India Sector Rotation features ────────────────────────────────────
        sector_cols = {
            "NIFTY_IT":     "IT",
            "NIFTY_AUTO":   "AUTO",
            "NIFTY_BANK":   "BANK",
            "NIFTY_METAL":  "METAL",
            "NIFTY_PHARMA": "PHARMA",
            "NIFTY_FMCG":   "FMCG",
        }
        for raw_name, alias in sector_cols.items():
            col = f"{raw_name}_CLOSE"
            nifty_col = "NIFTY_CLOSE"
            if col in c.columns and nifty_col in c.columns:
                # Relative strength vs Nifty (5-day)
                sec_5d   = c[col].pct_change(5)
                nifty_5d = c[nifty_col].pct_change(5)
                c[f"{alias}_REL_STRENGTH_5D"] = sec_5d - nifty_5d
                c[f"{alias}_LEADING"]  = (c[f"{alias}_REL_STRENGTH_5D"] > 0).astype(int)
                # 1-day return
                c[f"{alias}_1D_RET"] = c[col].pct_change(1) * 100

        # ── Nifty IT special: IT down + strong USD = sector drag ──────────────
        if "NIFTY_IT_CLOSE" in c.columns and "DXY_CLOSE" in c.columns:
            it_down = (c["NIFTY_IT_CLOSE"].pct_change(5) < -0.02)
            dxy_up  = (c.get("DXY_1D_RET", pd.Series(0, index=c.index)) > 0.3)
            c["IT_DXY_DRAG"] = (it_down & dxy_up).astype(int)

        # ── Composite Risk Appetite Score (0-10) ─────────────────────────────
        risk_signals = []
        for col, direction in [
            ("SPX_OVERNIGHT_RET",  +1),
            ("APAC_AVG_RET",       +1),
            ("BTC_1D_RET",         +1),
            ("EM_ETF_1D_RET",      +1),
            ("USVIX_1D_CHG",       -1),
            ("DXY_1D_RET",         -1),
            ("GOLD_1D_RET",        -1),
            ("BRENT_1D_RET",       -1),
        ]:
            if col in c.columns:
                z = (c[col] - c[col].rolling(60).mean()) / (c[col].rolling(60).std() + 1e-9)
                risk_signals.append(z * direction)

        if risk_signals:
            raw_score = pd.concat(risk_signals, axis=1).mean(axis=1)
            c["RISK_APPETITE_SCORE"] = (
                (raw_score - raw_score.rolling(252).min()) /
                (raw_score.rolling(252).max() - raw_score.rolling(252).min() + 1e-9)
            ) * 10
            c["HIGH_RISK_APPETITE"] = (c["RISK_APPETITE_SCORE"] > 6).astype(int)
            c["LOW_RISK_APPETITE"]  = (c["RISK_APPETITE_SCORE"] < 4).astype(int)

        return c

    # ─── EVENT FLAGS ──────────────────────────────────────────────────────────
    def _add_event_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary flags for high-impact macro events."""
        c = df.copy()

        # Known event days
        event_df = pd.DataFrame([
            {"DATE": pd.to_datetime(d), "EVENT": v}
            for d, v in INDIA_EVENTS.items()
        ])

        c["IS_RBI_DAY"]    = c.index.isin(
            event_df[event_df["EVENT"] == "RBI_MPC"]["DATE"]
        ).astype(int)
        c["IS_FOMC_DAY"]   = c.index.isin(
            event_df[event_df["EVENT"] == "US_FOMC"]["DATE"]
        ).astype(int)
        c["IS_BUDGET_DAY"] = c.index.isin(
            event_df[event_df["EVENT"].str.contains("BUDGET", na=False)]["DATE"]
        ).astype(int)
        c["IS_ELECTION_DAY"] = c.index.isin(
            event_df[event_df["EVENT"].str.contains("ELECTION", na=False)]["DATE"]
        ).astype(int)

        # Pre-event days (day before high-impact event)
        for flag, src in [("PRE_RBI", "IS_RBI_DAY"), ("PRE_FOMC", "IS_FOMC_DAY")]:
            c[flag] = c[src].shift(-1).fillna(0).astype(int)

        # F&O monthly expiry flags
        monthly_exp = pd.to_datetime(NSE_MONTHLY_EXPIRY_2024_25)
        c["IS_MONTHLY_EXPIRY"]  = c.index.isin(monthly_exp).astype(int)
        c["PRE_MONTHLY_EXPIRY"] = c["IS_MONTHLY_EXPIRY"].shift(-1).fillna(0).astype(int)
        c["POST_MONTHLY_EXPIRY"]= c["IS_MONTHLY_EXPIRY"].shift(1).fillna(0).astype(int)

        # Weekly expiry: every Thursday (weekday == 3)
        c["IS_WEEKLY_EXPIRY"]  = (pd.to_datetime(c.index).dayofweek == 3).astype(int)
        c["PRE_WEEKLY_EXPIRY"] = c["IS_WEEKLY_EXPIRY"].shift(-1).fillna(0).astype(int)

        # India result season (April-May, October-November)
        month = pd.to_datetime(c.index).month
        c["IS_RESULT_SEASON"] = month.isin([4, 5, 10, 11]).astype(int)

        # Month-end / quarter-end (institutional rebalancing)
        c["IS_MONTH_END"]    = pd.to_datetime(c.index).is_month_end.astype(int)
        c["IS_QUARTER_END"]  = pd.to_datetime(c.index).is_quarter_end.astype(int)

        return c

    # ─── INDIA 10Y BOND YIELD (via RBI / investing.com proxy) ────────────────
    def _add_india_bond_yield(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Try to fetch India 10Y Government Bond yield.
        yfinance ticker: ^CRBCM10Y (not always available)
        Fallback: compute proxy from Nifty and VIX.
        """
        try:
            raw = yf.download("^CRBCM10Y", start=self.start_date,
                              progress=False, auto_adjust=True)
            if not raw.empty:
                df["INDIA10Y_CLOSE"] = raw["Close"].reindex(df.index).ffill()
                df["INDIA10Y_1D_CHG"] = df["INDIA10Y_CLOSE"].diff(1)
                # US-India yield spread (FII carry trade signal)
                if "US10Y_CLOSE" in df.columns:
                    df["US_INDIA_YIELD_SPREAD"] = df["INDIA10Y_CLOSE"] - df["US10Y_CLOSE"]
                logger.info("India 10Y bond yield loaded")
        except Exception as e:
            logger.debug(f"India 10Y yield unavailable: {e}")
        return df

    # ─── MERGE INTO MAIN DATASET ──────────────────────────────────────────────
    def merge_with_fno(self, fno_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge extended market data into the main F&O feature DataFrame.
        fno_df must have a DATE column.
        """
        ext = self.download_all()
        ext_reset = ext.reset_index()  # DATE becomes a column

        merged = fno_df.merge(ext_reset, on="DATE", how="left")

        # Forward-fill extended data (covers weekends, holidays)
        ext_cols = [c for c in ext_reset.columns if c != "DATE"]
        merged[ext_cols] = merged[ext_cols].ffill()

        logger.info(
            f"Merged extended data: {merged.shape[1]} total columns "
            f"({len(ext_cols)} extended features)"
        )
        return merged

    # ─── FEATURE NAME LIST ────────────────────────────────────────────────────
    def get_extended_feature_names(self) -> List[str]:
        """
        Returns the full list of derived/computed feature column names
        (excludes raw close price columns — only returns signals for the model).
        """
        raw_close_suffixes = ["_CLOSE", "_VOL"]
        # We keep all computed columns (returns, ratios, flags, scores)
        # The model uses these, not the raw prices
        return [
            # Global returns
            "SPX_OVERNIGHT_RET","SPX_STRONG_UP","SPX_STRONG_DOWN",
            "NDX_OVERNIGHT_RET","APAC_AVG_RET","APAC_ALL_UP","APAC_ALL_DOWN",
            # Per-index
            *[f"{n}_{s}" for n in
              ["SPX","NDX","DJIA","NIKKEI","HANGSENG","SHANGHAI","FTSE","DAX","KOSPI"]
              for s in ["1D_RET","5D_RET","EMA_DEV","ABOVE_EMA50"]
              if True],
            # Dollar
            "DXY_1D_RET","DXY_5D_RET","DXY_ABOVE_103","DXY_EMA_DEV",
            # INR
            "USDINR_1D_CHG","USDINR_5D_CHG","USDINR_ABOVE_84","INR_WEAKNESS","INR_STRENGTH",
            # Crude
            "BRENT_1D_RET","BRENT_5D_RET","BRENT_20D_RET","BRENT_ABOVE_90","BRENT_SPIKE","BRENT_CRASH",
            "WTI_1D_RET",
            # Metals
            "GOLD_1D_RET","GOLD_5D_RET","GOLD_SAFE_HAVEN",
            "SILVER_1D_RET","SILVER_5D_RET",
            "COPPER_1D_RET","COPPER_5D_RET","COPPER_20D_RET",
            "PLATINUM_1D_RET","PALLADIUM_1D_RET",
            "ALUMINIUM_1D_RET","ZINC_1D_RET","NICKEL_1D_RET",
            # Ratios
            "GOLD_SILVER_RATIO","GSR_5D_CHG",
            "COPPER_GOLD_RATIO","CGR_5D_CHG","CGR_RISING",
            # Bonds
            "US10Y_1D_CHG","US10Y_5D_CHG","US10Y_ABOVE_4","US10Y_RISING",
            "YIELD_CURVE_SPREAD","YIELD_CURVE_INVERTED","YIELD_5_10_SPREAD",
            "INDIA10Y_1D_CHG","US_INDIA_YIELD_SPREAD",
            # Volatility
            "USVIX_1D_CHG","USVIX_ABOVE_20","USVIX_ABOVE_30","USVIX_SPIKE","USVIX_RANK",
            "VIX_SPREAD","DUAL_VIX_ELEVATED",
            "OIL_VIX_1D_RET","GOLD_VIX_1D_RET",
            # Crypto
            "BTC_1D_RET","BTC_5D_RET","BTC_RISK_ON","BTC_RISK_OFF",
            # EM flows
            "EM_ETF_1D_RET","EM_ETF_5D_RET","EM_ETF_ABOVE_EMA20",
            "HYG_1D_RET","CREDIT_STRESS",
            # Sectors
            *[f"{s}_{m}" for s in ["IT","AUTO","BANK","METAL","PHARMA","FMCG"]
              for m in ["REL_STRENGTH_5D","LEADING","1D_RET"]],
            "IT_DXY_DRAG",
            # Composite
            "RISK_APPETITE_SCORE","HIGH_RISK_APPETITE","LOW_RISK_APPETITE",
            # Events
            "IS_RBI_DAY","IS_FOMC_DAY","IS_BUDGET_DAY","IS_ELECTION_DAY",
            "PRE_RBI","PRE_FOMC",
            "IS_MONTHLY_EXPIRY","PRE_MONTHLY_EXPIRY","POST_MONTHLY_EXPIRY",
            "IS_WEEKLY_EXPIRY","PRE_WEEKLY_EXPIRY",
            "IS_RESULT_SEASON","IS_MONTH_END","IS_QUARTER_END",
            # Natural gas, cotton
            "NATGAS_1D_RET","WHEAT_1D_RET","COTTON_1D_RET",
        ]


# ─── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("\n=== EXTENDED MARKET DATA TEST ===\n")
    emd = ExtendedMarketData(start_date="2023-01-01")

    df = emd.download_all(force_refresh=True)
    print(f"✓ Shape: {df.shape}")
    print(f"✓ Date range: {df.index.min().date()} → {df.index.max().date()}")

    cats = {
        "Precious metals":  [c for c in df.columns if any(x in c for x in ["GOLD","SILVER","PLATINUM","PALLADIUM"])],
        "Base metals":      [c for c in df.columns if any(x in c for x in ["COPPER","ALUMINIUM","ZINC","NICKEL"])],
        "Energy":           [c for c in df.columns if any(x in c for x in ["BRENT","WTI","NATGAS","OIL"])],
        "Global indices":   [c for c in df.columns if any(x in c for x in ["SPX","NDX","NIKKEI","HANGSENG","FTSE","DAX"])],
        "Currencies":       [c for c in df.columns if any(x in c for x in ["DXY","INR","EURUSD","USDJPY"])],
        "Bonds / yields":   [c for c in df.columns if any(x in c for x in ["US10Y","US3M","YIELD","BOND"])],
        "Volatility":       [c for c in df.columns if any(x in c for x in ["VIX","OVX","GVZ"])],
        "Crypto":           [c for c in df.columns if any(x in c for x in ["BITCOIN","ETHEREUM","BTC"])],
        "Sectors":          [c for c in df.columns if any(x in c for x in ["IT_","AUTO_","BANK_","METAL_","PHARMA_","FMCG_"])],
        "Events / flags":   [c for c in df.columns if any(x in c for x in ["IS_","PRE_","POST_","RESULT"])],
        "Risk composite":   [c for c in df.columns if "RISK" in c or "APAC" in c],
    }

    total = 0
    print("\nFeature count by category:")
    for cat, cols in cats.items():
        print(f"  {cat:20s}: {len(cols):3d}")
        total += len(cols)
    print(f"  {'TOTAL':20s}: {total:3d}")

    # Show latest values
    print("\nLatest signal snapshot:")
    key_cols = ["RISK_APPETITE_SCORE","SPX_OVERNIGHT_RET","USVIX_CLOSE",
                "DXY_1D_RET","USDINR_1D_CHG","BRENT_1D_RET",
                "GOLD_1D_RET","COPPER_GOLD_RATIO","YIELD_CURVE_SPREAD",
                "IS_WEEKLY_EXPIRY","IS_MONTHLY_EXPIRY"]
    for col in key_cols:
        if col in df.columns:
            val = df[col].dropna().iloc[-1]
            print(f"  {col:30s}: {val:.4f}")

    print("\n=== TEST COMPLETE ===")
