"""
feature_engineering.py
=======================
Computes 75+ features across 7 categories for the F&O Neural Network model.

Categories:
  1.  Price / Trend             (EMA, MACD, Supertrend, VWAP, Ichimoku)
  2.  Momentum                  (RSI, Stochastic, MFI, ROC, Williams %R, CMO, DPO)
  3.  Volatility                (BB, Keltner, ATR, Donchian, NATR, historical vol)
  4.  Volume & Breadth          (OBV, VWAP, AD line, CMF, RVOL)
  5.  F&O Specific              (PCR, Max Pain, OI buildup, Futures basis, IV rank, GEX)
  6.  Macro & Sentiment         (VIX, FII/DII net flows, global indices, USD-INR)
  7.  Derived / Cross-asset     (rolling returns, regime, intermarket divergence)

No look-ahead bias: all features computable before 3:25 PM on day T.
Target: next-day return and direction (computed AFTER the fact for training).
"""

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# pandas-ta for 50+ technical indicators (pip install pandas-ta)
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logging.warning("pandas-ta not installed. Run: pip install pandas-ta")

warnings.filterwarnings("ignore")
logger = logging.getLogger("FeatureEngineering")

# ── CONFIG ────────────────────────────────────────────────────────────────────
CLOSE_COL = "CLOSE"
HIGH_COL  = "HIGH"
LOW_COL   = "LOW"
OPEN_COL  = "OPEN"
VOL_COL   = "CONTRACTS"  # F&O bhavcopy uses CONTRACTS as volume proxy
OI_COL    = "OPEN_INT"
CHG_OI    = "CHG_IN_OI"

# Rolling window presets (trading days)
W_SHORT  = 5
W_MID    = 20
W_LONG   = 60
W_YEARLY = 252


# ─────────────────────────────────────────────────────────────────────────────
class FeatureEngineer:
    """
    Computes all features for a single symbol's time series.
    Call compute_all(df) where df is sorted ascending by DATE.
    """

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point. Takes a single-symbol DataFrame sorted by DATE.
        Returns the same DataFrame enriched with all feature columns.
        """
        df = df.copy().sort_values("DATE").reset_index(drop=True)

        logger.debug(f"Computing features: {df['SYMBOL'].iloc[0]} | {len(df)} rows")

        df = self._add_price_trend_features(df)
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_volume_features(df)
        df = self._add_fno_features(df)
        df = self._add_macro_sentiment_features(df)
        df = self._add_derived_features(df)
        df = self._add_pcr_dynamics(df)
        df = self._add_regime_features(df)
        df = self._add_intermarket_features(df)
        df = self._add_statistical_features(df)
        df = self._add_cyclical_features(df)
        df = self._add_price_gap_features(df)
        df = self._add_velocity_acceleration_features(df)
        df = self._add_target(df)

        # Drop rows with insufficient history (first 60 rows)
        df = df.iloc[W_LONG:].reset_index(drop=True)

        feature_cols = [c for c in df.columns if c not in
                        ["DATE", "SYMBOL", "EXPIRY_DT", "INSTRUMENT",
                         "OPTION_TYP", "TIMESTAMP", "NEAR_EXPIRY"]]
        logger.debug(f"Features computed: {len(feature_cols)} columns")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # 1. PRICE / TREND
    # ─────────────────────────────────────────────────────────────────────
    def _add_price_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df[CLOSE_COL]
        h = df[HIGH_COL]
        lo = df[LOW_COL]
        o = df[OPEN_COL]

        # ── EMAs ──────────────────────────────────────────────────────────
        for period in [9, 21, 50, 200]:
            df[f"ema_{period}"] = c.ewm(span=period, adjust=False).mean()

        df["ema_cross_9_21"]   = (df["ema_9"] > df["ema_21"]).astype(int)
        df["ema_cross_21_50"]  = (df["ema_21"] > df["ema_50"]).astype(int)
        df["ema_cross_50_200"] = (df["ema_50"] > df["ema_200"]).astype(int)

        # Distance from key EMAs (normalised)
        df["price_to_ema21"]  = (c - df["ema_21"])  / df["ema_21"]
        df["price_to_ema50"]  = (c - df["ema_50"])  / df["ema_50"]
        df["price_to_ema200"] = (c - df["ema_200"]) / df["ema_200"]

        # ── MACD ──────────────────────────────────────────────────────────
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df["macd_line"]      = ema12 - ema26
        df["macd_signal"]    = df["macd_line"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
        df["macd_cross"]     = (df["macd_line"] > df["macd_signal"]).astype(int)

        # ── ADX (Average Directional Index) ───────────────────────────────
        if PANDAS_TA_AVAILABLE:
            adx_df = ta.adx(h, lo, c, length=14)
            if adx_df is not None and not adx_df.empty:
                df["adx"]  = adx_df.get("ADX_14", np.nan)
                df["dmp"]  = adx_df.get("DMP_14", np.nan)  # +DI
                df["dmn"]  = adx_df.get("DMN_14", np.nan)  # -DI
                df["di_diff"] = df["dmp"] - df["dmn"]
        else:
            df["adx"] = self._manual_adx(h, lo, c, 14)

        # ── Supertrend ────────────────────────────────────────────────────
        if PANDAS_TA_AVAILABLE:
            st = ta.supertrend(h, lo, c, length=10, multiplier=3)
            if st is not None and not st.empty:
                col = [c for c in st.columns if "SUPERTd" in c]
                df["supertrend_dir"] = st[col[0]] if col else np.nan

        # ── Parabolic SAR ─────────────────────────────────────────────────
        if PANDAS_TA_AVAILABLE:
            psar = ta.psar(h, lo, c)
            if psar is not None and not psar.empty:
                bsr_cols = [c for c in psar.columns if "PSARr" in c]
                df["psar_reversal"] = psar[bsr_cols[0]] if bsr_cols else np.nan

        # ── VWAP deviation (proxy using OHLC4 and volume) ─────────────────
        typical = (h + lo + c) / 3
        vol_proxy = df[VOL_COL].clip(lower=1)
        cum_vol = vol_proxy.rolling(W_MID).sum()
        cum_tp_vol = (typical * vol_proxy).rolling(W_MID).sum()
        df["vwap_20d"] = cum_tp_vol / cum_vol
        df["vwap_dev"] = (c - df["vwap_20d"]) / df["vwap_20d"]

        # ── Ichimoku Cloud ────────────────────────────────────────────────
        nine_high  = h.rolling(9).max()
        nine_low   = lo.rolling(9).min()
        df["ichi_tenkan"]  = (nine_high + nine_low) / 2
        twenty_six_high = h.rolling(26).max()
        twenty_six_low  = lo.rolling(26).min()
        df["ichi_kijun"]   = (twenty_six_high + twenty_six_low) / 2
        df["ichi_senkou_a"] = ((df["ichi_tenkan"] + df["ichi_kijun"]) / 2).shift(26)
        df["ichi_cloud_signal"] = (
            (c > df["ichi_senkou_a"]) & (c > df["ichi_kijun"])
        ).astype(int)

        # ── Candle patterns ───────────────────────────────────────────────
        df["body_size"]    = (c - o).abs() / c
        df["upper_shadow"] = (h - c.clip(upper=h)) / c
        df["lower_shadow"] = (lo.clip(upper=c) - lo) / c
        df["doji"]         = (df["body_size"] < 0.001).astype(int)
        df["gap_up"]       = ((o - c.shift(1)) / c.shift(1)).clip(lower=0)
        df["gap_down"]     = ((c.shift(1) - o) / c.shift(1)).clip(lower=0)

        return df

    # ─────────────────────────────────────────────────────────────────────
    # 2. MOMENTUM
    # ─────────────────────────────────────────────────────────────────────
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c  = df[CLOSE_COL]
        h  = df[HIGH_COL]
        lo = df[LOW_COL]
        v  = df[VOL_COL].clip(lower=1)

        # ── RSI ───────────────────────────────────────────────────────────
        df["rsi_14"]  = self._rsi(c, 14)
        df["rsi_7"]   = self._rsi(c, 7)
        df["rsi_21"]  = self._rsi(c, 21)
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
        df["rsi_oversold"]   = (df["rsi_14"] < 30).astype(int)

        # ── Stochastic ────────────────────────────────────────────────────
        if PANDAS_TA_AVAILABLE:
            stoch = ta.stoch(h, lo, c, k=14, d=3)
            if stoch is not None and not stoch.empty:
                df["stoch_k"] = stoch.get("STOCHk_14_3_3", np.nan)
                df["stoch_d"] = stoch.get("STOCHd_14_3_3", np.nan)
                df["stoch_cross"] = (df["stoch_k"] > df["stoch_d"]).astype(int)
        else:
            low14  = lo.rolling(14).min()
            high14 = h.rolling(14).max()
            df["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-9)
            df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # ── MFI (Money Flow Index) ────────────────────────────────────────
        typical = (h + lo + c) / 3
        raw_mf = typical * v
        pos_mf = raw_mf.where(typical > typical.shift(1), 0).rolling(14).sum()
        neg_mf = raw_mf.where(typical < typical.shift(1), 0).rolling(14).sum()
        df["mfi_14"] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-9)))
        df["mfi_overbought"] = (df["mfi_14"] > 80).astype(int)
        df["mfi_oversold"]   = (df["mfi_14"] < 20).astype(int)

        # ── Williams %R ───────────────────────────────────────────────────
        high14 = h.rolling(14).max()
        low14  = lo.rolling(14).min()
        df["williams_r"] = -100 * (high14 - c) / (high14 - low14 + 1e-9)

        # ── Rate of Change ────────────────────────────────────────────────
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = c.pct_change(period) * 100

        # ── Chande Momentum Oscillator (CMO) ──────────────────────────────
        delta = c.diff()
        up   = delta.clip(lower=0).rolling(14).sum()
        down = (-delta).clip(lower=0).rolling(14).sum()
        df["cmo_14"] = 100 * (up - down) / (up + down + 1e-9)

        # ── Detrended Price Oscillator (DPO) ──────────────────────────────
        sma20 = c.rolling(20).mean()
        df["dpo_20"] = c.shift(11) - sma20

        # ── Commodity Channel Index (CCI) ─────────────────────────────────
        tp = (h + lo + c) / 3
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df["cci_20"] = (tp - tp.rolling(20).mean()) / (0.015 * mad + 1e-9)

        return df

    # ─────────────────────────────────────────────────────────────────────
    # 3. VOLATILITY
    # ─────────────────────────────────────────────────────────────────────
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c  = df[CLOSE_COL]
        h  = df[HIGH_COL]
        lo = df[LOW_COL]

        # ── ATR ───────────────────────────────────────────────────────────
        tr = pd.concat([
            (h - lo),
            (h - c.shift(1)).abs(),
            (lo - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        df["atr_14"]  = tr.ewm(span=14, adjust=False).mean()
        df["natr_14"] = df["atr_14"] / c * 100   # normalised ATR %

        # ── Bollinger Bands ───────────────────────────────────────────────
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        df["bb_upper"]  = sma20 + 2 * std20
        df["bb_lower"]  = sma20 - 2 * std20
        df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / sma20
        df["bb_pct_b"]  = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
        df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(W_YEARLY).quantile(0.20)).astype(int)

        # ── Keltner Channel ───────────────────────────────────────────────
        ema20 = c.ewm(span=20, adjust=False).mean()
        df["kc_upper"] = ema20 + 2 * df["atr_14"]
        df["kc_lower"] = ema20 - 2 * df["atr_14"]
        df["kc_pct"]   = (c - df["kc_lower"]) / (df["kc_upper"] - df["kc_lower"] + 1e-9)

        # ── BB inside KC → Squeeze Momentum ───────────────────────────────
        df["sqz_on"] = (
            (df["bb_lower"] > df["kc_lower"]) &
            (df["bb_upper"] < df["kc_upper"])
        ).astype(int)

        # ── Donchian Channel ──────────────────────────────────────────────
        df["don_high_20"] = h.rolling(20).max()
        df["don_low_20"]  = lo.rolling(20).min()
        df["don_width"]   = (df["don_high_20"] - df["don_low_20"]) / c

        # ── Historical Volatility (annualised) ────────────────────────────
        log_ret = np.log(c / c.shift(1))
        df["hvol_10"]  = log_ret.rolling(10).std() * np.sqrt(252) * 100
        df["hvol_20"]  = log_ret.rolling(20).std() * np.sqrt(252) * 100
        df["hvol_60"]  = log_ret.rolling(60).std() * np.sqrt(252) * 100
        df["hvol_ratio"] = df["hvol_10"] / (df["hvol_20"] + 1e-9)   # vol term structure

        # ── VIX-based features (if available) ─────────────────────────────
        if "VIX_CLOSE" in df.columns:
            vix = df["VIX_CLOSE"]
            vix_52w_hi = vix.rolling(W_YEARLY).max()
            vix_52w_lo = vix.rolling(W_YEARLY).min()
            df["vix_rank"]  = (vix - vix_52w_lo) / (vix_52w_hi - vix_52w_lo + 1e-9)
            df["vix_pct_chg_1d"] = vix.pct_change(1) * 100
            df["vix_pct_chg_5d"] = vix.pct_change(5) * 100
            df["vix_above_20"]   = (vix > 20).astype(int)
            df["vix_spike"]      = (df["vix_pct_chg_1d"] > 10).astype(int)

        # ── IV Rank and IV Skew (from bhavcopy or option chain) ───────────
        # These are computed in the F&O section below; placeholders here
        return df

    # ─────────────────────────────────────────────────────────────────────
    # 4. VOLUME & BREADTH
    # ─────────────────────────────────────────────────────────────────────
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c  = df[CLOSE_COL]
        h  = df[HIGH_COL]
        lo = df[LOW_COL]
        v  = df[VOL_COL].clip(lower=1)

        # ── OBV (On-Balance Volume) ───────────────────────────────────────
        direction = np.sign(c.diff().fillna(0))
        df["obv"] = (v * direction).cumsum()
        df["obv_ema_20"] = df["obv"].ewm(span=20, adjust=False).mean()
        df["obv_trend"]  = (df["obv"] > df["obv_ema_20"]).astype(int)

        # ── CMF (Chaikin Money Flow) ──────────────────────────────────────
        mfm = ((c - lo) - (h - c)) / (h - lo + 1e-9)
        df["cmf_20"] = (mfm * v).rolling(20).sum() / v.rolling(20).sum()

        # ── Accum/Distribution Line ───────────────────────────────────────
        clv = ((c - lo) - (h - c)) / (h - lo + 1e-9)
        df["ad_line"] = (clv * v).cumsum()

        # ── Relative Volume ───────────────────────────────────────────────
        df["rvol_20"] = v / (v.rolling(20).mean() + 1e-9)
        df["high_volume"] = (df["rvol_20"] > 2.0).astype(int)

        # ── Volume trend ──────────────────────────────────────────────────
        df["vol_ema_10"] = v.ewm(span=10, adjust=False).mean()
        df["vol_ratio"]  = v / (df["vol_ema_10"] + 1e-9)

        return df

    # ─────────────────────────────────────────────────────────────────────
    # 5. F&O SPECIFIC
    # ─────────────────────────────────────────────────────────────────────
    def _add_fno_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c  = df[CLOSE_COL]
        oi = df[OI_COL].clip(lower=0) if OI_COL in df.columns else pd.Series(0, index=df.index)
        chg_oi = df[CHG_OI] if CHG_OI in df.columns else pd.Series(0, index=df.index)

        # ── OI Change Rate ────────────────────────────────────────────────
        df["oi_chg_pct"]   = chg_oi / (oi.shift(1).clip(lower=1))
        df["oi_buildup"]   = ((chg_oi > 0) & (c > c.shift(1))).astype(int)   # long buildup
        df["oi_unwinding"] = ((chg_oi < 0) & (c > c.shift(1))).astype(int)   # long unwinding
        df["short_buildup"]= ((chg_oi > 0) & (c < c.shift(1))).astype(int)
        df["short_cover"]  = ((chg_oi < 0) & (c < c.shift(1))).astype(int)

        # ── OI momentum ───────────────────────────────────────────────────
        df["oi_5d_chg"]  = oi.pct_change(5)
        df["oi_20d_chg"] = oi.pct_change(20)
        df["oi_zscore"]  = (oi - oi.rolling(20).mean()) / (oi.rolling(20).std() + 1e-9)

        # ── Futures Basis (if SETTLE_PR available) ────────────────────────
        if "SETTLE_PR" in df.columns and "NIFTY_CLOSE" in df.columns:
            df["futures_basis"] = (df["SETTLE_PR"] - df["NIFTY_CLOSE"]) / df["NIFTY_CLOSE"]
        elif "SETTLE_PR" in df.columns:
            df["futures_basis"] = (df["SETTLE_PR"] - c) / c

        # ── DTE features ──────────────────────────────────────────────────
        if "DTE" in df.columns:
            df["dte_log"] = np.log1p(df["DTE"])
            df["expiry_week"] = (df["DTE"] <= 7).astype(int)
            df["expiry_day"]  = (df["DTE"] == 0).astype(int)

        # ── IV Rank and IV Percentile (computed from historical IVs) ──────
        # These columns get populated from option chain data if available
        for col in ["atm_iv_call", "atm_iv_put", "iv_skew", "pcr_atm",
                    "pcr_full", "pcr_otm", "pcr_volume",
                    "max_call_oi_strike", "max_put_oi_strike", "max_pain",
                    "gex"]:
            if col in df.columns:
                # Compute rolling rank for IV-related features
                if "iv" in col.lower():
                    vals = df[col]
                    iv_52w_hi = vals.rolling(W_YEARLY).max()
                    iv_52w_lo = vals.rolling(W_YEARLY).min()
                    df[f"{col}_rank"] = (vals - iv_52w_lo) / (iv_52w_hi - iv_52w_lo + 1e-9)

        # ── PCR momentum (rolling) ─────────────────────────────────────────
        if "pcr_atm" in df.columns:
            df["pcr_5d_avg"]  = df["pcr_atm"].rolling(5).mean()
            df["pcr_trend"]   = (df["pcr_atm"] > df["pcr_5d_avg"]).astype(int)
            df["pcr_extreme_bull"] = (df["pcr_atm"] > 1.5).astype(int)
            df["pcr_extreme_bear"] = (df["pcr_atm"] < 0.5).astype(int)

        # ── Max Pain distance ─────────────────────────────────────────────
        if "max_pain" in df.columns:
            df["dist_from_max_pain"] = (c - df["max_pain"]) / c

        return df

    # ─────────────────────────────────────────────────────────────────────
    # 6. MACRO & SENTIMENT
    # ─────────────────────────────────────────────────────────────────────
    def _add_macro_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # ── FII/DII net flows ─────────────────────────────────────────────
        for col in ["FII_CASH_NET", "FII_FNO_NET", "DII_NET"]:
            if col in df.columns:
                df[f"{col}_5d"] = df[col].rolling(5).sum()
                df[f"{col}_20d"] = df[col].rolling(20).sum()
                # Normalise by rolling std
                std20 = df[col].rolling(20).std() + 1e-9
                df[f"{col}_zscore"] = df[col] / std20
                # Consecutive buying/selling streak
                df[f"{col}_streak"] = (
                    df[col].gt(0).astype(int)
                    .groupby((df[col].gt(0) != df[col].gt(0).shift()).cumsum())
                    .cumsum()
                )

        # ── FII Long/Short ratio in index futures ─────────────────────────
        if "FII_FNO_NET" in df.columns:
            fii_net = df["FII_FNO_NET"]
            df["fii_fno_bullish"] = (fii_net > 0).astype(int)
            df["fii_fno_20d_trend"] = np.sign(df["FII_FNO_NET_20d"]) if "FII_FNO_NET_20d" in df.columns else 0

        # ── US Markets overnight return ────────────────────────────────────
        for idx_col in ["SPX_CLOSE", "NDX_CLOSE"]:
            if idx_col in df.columns:
                name = idx_col.replace("_CLOSE", "")
                df[f"{name}_overnight_ret"] = df[idx_col].pct_change(1).shift(1) * 100
                df[f"{name}_5d_ret"]        = df[idx_col].pct_change(5) * 100

        # ── USD-INR ───────────────────────────────────────────────────────
        if "USDINR_CLOSE" in df.columns:
            df["usdinr_1d_chg"] = df["USDINR_CLOSE"].pct_change(1) * 100
            df["usdinr_5d_chg"] = df["USDINR_CLOSE"].pct_change(5) * 100
            df["inr_weakening"] = (df["usdinr_1d_chg"] > 0.5).astype(int)

        # ── Gold & Crude ──────────────────────────────────────────────────
        if "GOLD_CLOSE" in df.columns:
            df["gold_1d_ret"] = df["GOLD_CLOSE"].pct_change(1) * 100
            df["gold_5d_ret"] = df["GOLD_CLOSE"].pct_change(5) * 100

        if "CRUDE_CLOSE" in df.columns:
            df["crude_1d_ret"] = df["CRUDE_CLOSE"].pct_change(1) * 100
            df["crude_5d_ret"] = df["CRUDE_CLOSE"].pct_change(5) * 100

        # ── India VIX features (if not already in volatility section) ─────
        if "VIX_CLOSE" in df.columns and "vix_rank" not in df.columns:
            vix = df["VIX_CLOSE"]
            vix_52w_hi = vix.rolling(W_YEARLY).max()
            vix_52w_lo = vix.rolling(W_YEARLY).min()
            df["vix_rank"]      = (vix - vix_52w_lo) / (vix_52w_hi - vix_52w_lo + 1e-9)
            df["vix_pct_chg_1d"] = vix.pct_change(1) * 100
            df["vix_pct_chg_5d"] = vix.pct_change(5) * 100
            df["vix_above_20"]   = (vix > 20).astype(int)
            df["vix_above_25"]   = (vix > 25).astype(int)
            df["vix_spike"]      = (df["vix_pct_chg_1d"] > 10).astype(int)

        return df

    # ─────────────────────────────────────────────────────────────────────
    # 7. DERIVED / CROSS-ASSET / REGIME
    # ─────────────────────────────────────────────────────────────────────
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df[CLOSE_COL]

        # ── Rolling returns ───────────────────────────────────────────────
        for p in [1, 3, 5, 10, 20, 60]:
            df[f"ret_{p}d"] = c.pct_change(p) * 100

        # ── Return z-scores (relative to history) ─────────────────────────
        for p in [5, 20]:
            col = f"ret_{p}d"
            if col in df.columns:
                mu = df[col].rolling(W_YEARLY).mean()
                sigma = df[col].rolling(W_YEARLY).std()
                df[f"ret_{p}d_zscore"] = (df[col] - mu) / (sigma + 1e-9)

        # ── Drawdown from 52-week high ─────────────────────────────────────
        rolling_hi = c.rolling(W_YEARLY).max()
        df["drawdown_from_hi"] = (c - rolling_hi) / rolling_hi * 100

        # ── Regime detection (simple HMM-proxy) ───────────────────────────
        # Bull = price > 200 EMA, Vol < median, Trend strong
        if "ema_200" in df.columns and "hvol_20" in df.columns:
            above_ema200 = (c > df["ema_200"]).astype(int)
            low_vol = (df["hvol_20"] < df["hvol_20"].rolling(W_YEARLY).median()).astype(int)
            df["bull_regime"] = (above_ema200 + low_vol == 2).astype(int)
            df["bear_regime"] = (above_ema200 == 0).astype(int)

        # ── Day of week / Month effects ────────────────────────────────────
        df["dow"] = df["DATE"].dt.dayofweek         # 0=Mon, 4=Fri
        df["dom"] = df["DATE"].dt.day               # day of month
        df["month"] = df["DATE"].dt.month
        df["is_month_end"]   = df["DATE"].dt.is_month_end.astype(int)
        df["is_month_start"] = df["DATE"].dt.is_month_start.astype(int)

        # ── Midcap vs Nifty divergence ────────────────────────────────────
        if "MIDCAP_CLOSE" in df.columns and "NIFTY_CLOSE" in df.columns:
            df["midcap_rel"] = df["MIDCAP_CLOSE"].pct_change(5) - df["NIFTY_CLOSE"].pct_change(5)

        # ── Rank-transform continuous features (0-1 over rolling window) ──
        # This helps the neural network handle outliers
        for col in ["ret_5d", "ret_20d", "vix_rank", "oi_zscore"]:
            if col in df.columns:
                df[f"{col}_pctrank"] = df[col].rolling(W_YEARLY).rank(pct=True)

        return df

    # ─────────────────────────────────────────────────────────────────────
    # 8. PCR DYNAMICS
    # ─────────────────────────────────────────────────────────────────────
    def _add_pcr_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect put/call OI structural shifts."""
        if "pcr_full" not in df.columns:
            return df
            
        pcr = df["pcr_full"]
        df["pcr_1d_chg"] = pcr.diff()
        df["pcr_5d_zscore"] = (pcr - pcr.rolling(5).mean()) / (pcr.rolling(5).std() + 1e-9)
        df["pcr_20d_zscore"] = (pcr - pcr.rolling(20).mean()) / (pcr.rolling(20).std() + 1e-9)
        
        # Flips
        df["pcr_flip_bullish"] = ((pcr.shift(1) > 1.2) & (pcr < 1.0)).astype(int)
        df["pcr_flip_bearish"] = ((pcr.shift(1) < 0.8) & (pcr > 1.0)).astype(int)
        
        # OI Change %
        if "CE_OI" in df.columns:
            df["call_oi_pct_chg"] = df["CE_OI"].pct_change()
        if "PE_OI" in df.columns:
            df["put_oi_pct_chg"] = df["PE_OI"].pct_change()
            
        # OI Ratio Trend: 5-day EMA of put_OI/call_OI ratio slope
        if "CE_OI" in df.columns and "PE_OI" in df.columns:
            ratio = df["PE_OI"] / (df["CE_OI"] + 1e-9)
            slope = ratio.diff()
            df["oi_ratio_trend"] = slope.ewm(span=5, adjust=False).mean()
            
        # Call writing signal: call OI increases + PCR rises + price falls
        if all(c in df.columns for c in ["CE_OI", "pcr_full", "CLOSE"]):
            ce_oi_inc = df["CE_OI"].diff() > 0
            pcr_inc = df["pcr_full"].diff() > 0
            price_dec = df["CLOSE"].diff() < 0
            df["call_writing_signal"] = (ce_oi_inc & pcr_inc & price_dec).astype(int)
            
        return df

    # ─────────────────────────────────────────────────────────────────────
    # 9. MARKET REGIMES
    # ─────────────────────────────────────────────────────────────────────
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market regime detection (low/high vol, trend, etc.)"""
        c = df["CLOSE"]
        
        # Vol Regime
        if "hvol_20" in df.columns:
            p15 = df["hvol_20"].rolling(252).quantile(0.15)
            p85 = df["hvol_20"].rolling(252).quantile(0.85)
            df["vol_regime"] = 1 # Normal
            df.loc[df["hvol_20"] < p15, "vol_regime"] = 0 # Low vol
            df.loc[df["hvol_20"] > p85, "vol_regime"] = 2 # High vol
        
        # Trend Regime
        if all(f"ema_{p}" in df.columns for p in [50, 200]):
            df["trend_regime"] = 0
            df.loc[(c > df["ema_50"]) & (df["ema_50"] > df["ema_200"]), "trend_regime"] = 1
            df.loc[(c < df["ema_50"]) & (df["ema_50"] < df["ema_200"]), "trend_regime"] = -1
            
        # Mean Rev Regime (RSI > 65 AND BB%B > 0.9)
        if "rsi_14" in df.columns and "bb_pct_b" in df.columns:
            df["mean_rev_regime"] = ((df["rsi_14"] > 65) & (df["bb_pct_b"] > 0.9)).astype(int)
            
        # Momentum Regime (RSI > 55 AND MACD > signal AND price > VWAP)
        if all(c in df.columns for c in ["rsi_14", "macd_line", "macd_signal", "vwap_20d"]):
            bull = (df["rsi_14"] > 55) & (df["macd_line"] > df["macd_signal"]) & (c > df["vwap_20d"])
            df["momentum_regime"] = bull.astype(int)
            df["bull_regime"] = bull.astype(int)
            
        # VIX Threshold
        if "VIX_CLOSE" in df.columns:
            df["vix_above_25"] = (df["VIX_CLOSE"] > 25).astype(int)
        else:
            df["vix_above_25"] = 0
            
        # Composite Regime
        df["regime_composite"] = "choppy"
        if "trend_regime" in df.columns and "vol_regime" in df.columns:
            df.loc[(df["trend_regime"] == 1) & (df["vol_regime"] == 0), "regime_composite"] = "bull_low_vol"
            df.loc[(df["trend_regime"] == 1) & (df["vol_regime"] >= 1), "regime_composite"] = "bull_high_vol"
            df.loc[(df["trend_regime"] == -1) & (df["vol_regime"] == 0), "regime_composite"] = "bear_low_vol"
            df.loc[(df["trend_regime"] == -1) & (df["vol_regime"] >= 1), "regime_composite"] = "bear_high_vol"
            
        # Regime Change
        if "trend_regime" in df.columns:
            df["regime_change"] = (df["trend_regime"] != df["trend_regime"].shift(1)).astype(int)
            # Days in regime
            df["days_in_regime"] = df.groupby((df["trend_regime"] != df["trend_regime"].shift()).cumsum()).cumcount() + 1
            
        return df

    # ─────────────────────────────────────────────────────────────────────
    # 10. INTERMARKET & SMART MONEY
    # ─────────────────────────────────────────────────────────────────────
    def _add_intermarket_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-asset signals."""
        # nifty_vs_em_rel: Nifty 5d return minus EM ETF 5d return
        if "NIFTY_CLOSE" in df.columns and "EEM_CLOSE" in df.columns:
            df["nifty_vs_em_rel"] = df["NIFTY_CLOSE"].pct_change(5) - df["EEM_CLOSE"].pct_change(5)
            
        # vix_nifty_diverge: 1 if VIX rises but Nifty also rises
        if "VIX_CLOSE" in df.columns and "NIFTY_CLOSE" in df.columns:
            df["vix_nifty_diverge"] = ((df["VIX_CLOSE"].diff() > 0) & (df["NIFTY_CLOSE"].diff() > 0)).astype(int)
            
        # bond_equity_signal: 1 if US 10Y yield rises AND EM ETF falls
        if "US10Y_CLOSE" in df.columns and "EEM_CLOSE" in df.columns:
            df["bond_equity_signal"] = ((df["US10Y_CLOSE"].diff() > 0) & (df["EEM_CLOSE"].diff() < 0)).astype(int)
            
        # copper_nifty_lead: 3-day lag correlation of copper return and Nifty return
        if "COPPER_CLOSE" in df.columns and "NIFTY_CLOSE" in df.columns:
            ret_c = df["COPPER_CLOSE"].pct_change()
            ret_n = df["NIFTY_CLOSE"].pct_change()
            df["copper_nifty_lead"] = ret_c.shift(3).rolling(20).corr(ret_n)
            
        # crude_auto_drag: 1 if crude > 3% 5d move AND Nifty Auto sector lagging
        if "CRUDE_CLOSE" in df.columns and "AUTO_CLOSE" in df.columns and "NIFTY_CLOSE" in df.columns:
            crude_up = df["CRUDE_CLOSE"].pct_change(5) > 0.03
            auto_lag = df["AUTO_CLOSE"].pct_change(5) < df["NIFTY_CLOSE"].pct_change(5)
            df["crude_auto_drag"] = (crude_up & auto_lag).astype(int)
            
        # dxy_it_headwind: 1 if DXY 5d return > 0.5% AND Nifty IT sector underperforming
        if "DXY_CLOSE" in df.columns and "IT_CLOSE" in df.columns and "NIFTY_CLOSE" in df.columns:
            dxy_up = df["DXY_CLOSE"].pct_change(5) > 0.005
            it_lag = df["IT_CLOSE"].pct_change(5) < df["NIFTY_CLOSE"].pct_change(5)
            df["dxy_it_headwind"] = (dxy_up & it_lag).astype(int)
            
        # smart_money_flow: FII_FNO_NET + DII_NET normalised z-score
        if "FII_FNO_NET" in df.columns and "DII_NET" in df.columns:
            total_flow = df["FII_FNO_NET"] + df["DII_NET"]
            df["smart_money_flow"] = (total_flow - total_flow.rolling(20).mean()) / (total_flow.rolling(20).std() + 1e-9)
            
        # retail_vs_smart: bulk_deal_net (retail proxy) minus FII_FNO_NET
        if "NET_BULK_VALUE_CR" in df.columns and "FII_FNO_NET" in df.columns:
            bulk_z = (df["NET_BULK_VALUE_CR"] - df["NET_BULK_VALUE_CR"].rolling(20).mean()) / (df["NET_BULK_VALUE_CR"].rolling(20).std() + 1e-9)
            smart_z = (df["FII_FNO_NET"] - df["FII_FNO_NET"].rolling(20).mean()) / (df["FII_FNO_NET"].rolling(20).std() + 1e-9)
            df["retail_vs_smart"] = bulk_z - smart_z
            
        return df

    # ─────────────────────────────────────────────────────────────────────
    # 11. STATISTICAL FEATURES
    # ─────────────────────────────────────────────────────────────────────
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling statistical moments and autocorrelations."""
        ret = df["CLOSE"].pct_change()
        
        for p in [20, 60]:
            df[f"skew_{p}d"] = ret.rolling(p).skew()
            df[f"kurt_{p}d"] = ret.rolling(p).kurt()
            
        # Autocorrelation (Lags 1, 3, 5)
        for lag in [1, 3, 5]:
            df[f"autocorr_ret_l{lag}"] = ret.rolling(20).apply(lambda x: x.autocorr(lag=lag) if len(x) > lag else 0, raw=False)
            
        # sharpe_20d: Rolling 20d risk-adj return
        df["sharpe_20d"] = ret.rolling(20).mean() / (ret.rolling(20).std() + 1e-9)
        
        # sortino_20d
        neg_ret = ret.where(ret < 0, 0)
        df["sortino_20d"] = ret.rolling(20).mean() / (neg_ret.rolling(20).std() + 1e-9)
        
        return df

    # ─────────────────────────────────────────────────────────────────────
    # 12. CYCLICAL & SEASONAL
    # ─────────────────────────────────────────────────────────────────────
    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sine/Cosine encoding for time features."""
        # Day of week
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 5)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 5)
        
        # Month
        df["month_sin"] = np.sin(2 * np.pi * (df["month"]-1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"]-1) / 12)
        
        return df

    # ─────────────────────────────────────────────────────────────────────
    # 13. PRICE GAPS & INTENSITY
    # ─────────────────────────────────────────────────────────────────────
    def _add_price_gap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gap analysis and intraday location."""
        # Gap %: (Open - Prev Close) / Prev Close
        df["gap_pct"] = (df["OPEN"] - df["CLOSE"].shift(1)) / (df["CLOSE"].shift(1) + 1e-9)
        df["gap_5d_avg"] = df["gap_pct"].rolling(5).mean()
        
        # Intraday Intensity Proxy: how close is close to the high/low?
        df["intraday_intensity"] = (2*df["CLOSE"] - df["HIGH"] - df["LOW"]) / (df["HIGH"] - df["LOW"] + 1e-9)
        
        # Price location in N-day range
        for p in [5, 10, 20]:
            df[f"price_loc_{p}d"] = (df["CLOSE"] - df["LOW"].rolling(p).min()) / (df["HIGH"].rolling(p).max() - df["LOW"].rolling(p).min() + 1e-9)
            
        # Distance from moving averages (Percentage)
        for p in [20, 50, 200]:
            ema_col = f"ema_{p}"
            if ema_col in df.columns:
                df[f"dist_ema_{p}"] = (df["CLOSE"] / df[ema_col]) - 1
                
        return df

    # ─────────────────────────────────────────────────────────────────────
    # 14. VELOCITY & ACCELERATION
    # ─────────────────────────────────────────────────────────────────────
    def _add_velocity_acceleration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Price velocity (first derivative) and acceleration (second derivative).
        These capture momentum and momentum changes.
        """
        c = df["CLOSE"]

        # Velocity: Rate of price change (first derivative)
        for p in [1, 3, 5, 10]:
            df[f"velocity_{p}d"] = c.diff(p) / p

        # Acceleration: Rate of velocity change (second derivative)
        for p in [3, 5, 10]:
            vel_col = f"velocity_{p}d"
            if vel_col in df.columns:
                df[f"accel_{p}d"] = df[vel_col].diff()

        # Normalized velocity (relative to ATR)
        if "atr_14" in df.columns:
            df["velocity_norm_5d"] = df["velocity_5d"] / (df["atr_14"] + 1e-9)

        # Momentum change detection
        df["velocity_reversal"] = (
            (df["velocity_5d"].shift(1) > 0) & (df["velocity_5d"] < 0)
        ).astype(int) - (
            (df["velocity_5d"].shift(1) < 0) & (df["velocity_5d"] > 0)
        ).astype(int)

        # Jerk (third derivative) for very short-term changes
        if "accel_5d" in df.columns:
            df["jerk_5d"] = df["accel_5d"].diff()

        # Velocity correlation with volume
        if VOL_COL in df.columns:
            vol = df[VOL_COL].clip(lower=1)
            for p in [5, 20]:
                df[f"velocity_vol_corr_{p}d"] = df[f"velocity_{p}d"].rolling(p).corr(vol)

        return df

    # ─────────────────────────────────────────────────────────────────────
    # TARGET VARIABLE
    # ─────────────────────────────────────────────────────────────────────
    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute next-day return and direction label.
        These are ONLY used during training — never at inference time.
        """
        c = df[CLOSE_COL]

        df["next_day_return"]    = c.shift(-1) / c - 1          # float
        df["next_day_return_pct"] = df["next_day_return"] * 100

        # Classification: 1 = UP (>+0.5%), -1 = DOWN (<-0.5%), 0 = FLAT
        df["label"] = 0
        df.loc[df["next_day_return_pct"] >  0.5, "label"] =  1
        df.loc[df["next_day_return_pct"] < -0.5, "label"] = -1

        # 3-class encoded for softmax: 0=DOWN, 1=FLAT, 2=UP
        df["label_3c"] = df["label"] + 1

        return df

    # ─────────────────────────────────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────────────────────────────────
    def _rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI without pandas-ta dependency."""
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta).clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)

    def _manual_adx(self, high, low, close, period=14) -> pd.Series:
        """Manual ADX fallback."""
        tr = pd.concat([
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        up_move   = high - high.shift(1)
        down_move = low.shift(1) - low
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        pos_di = 100 * pos_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
        neg_di = 100 * neg_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
        dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-9)
        return dx.ewm(span=period, adjust=False).mean()

    def get_feature_names(self, df: pd.DataFrame) -> list:
        """Return list of all feature column names (excludes metadata and target)."""
        skip = {
            "DATE", "SYMBOL", "EXPIRY_DT", "INSTRUMENT", "OPTION_TYP",
            "TIMESTAMP", "NEAR_EXPIRY", "OPEN", "HIGH", "LOW", "CLOSE",
            "SETTLE_PR", "CONTRACTS", "VAL_INLAKH", "OPEN_INT", "CHG_IN_OI",
            "STRIKE_PR", "DTE", "next_day_return", "next_day_return_pct",
            "label", "label_3c", "regime_composite", # categorical
            "CE_OI", "PE_OI", "TOTAL_CE_OI", "TOTAL_PE_OI", # raw input cols
        }
        # Also skip raw macro columns
        macro_raw = [c for c in df.columns if any(
            c.startswith(p) for p in
            ["NIFTY_", "BANKNIFTY_", "SPX_", "NDX_", "USDINR_",
             "GOLD_", "CRUDE_", "MIDCAP_", "VIX_"]
        )]
        skip.update(macro_raw)
        return [c for c in df.columns if c not in skip]


# ─────────────────────────────────────────────────────────────────────────────
def compute_features(
    df: pd.DataFrame,
    symbol_col: str = "SYMBOL",
) -> Tuple[pd.DataFrame, list]:
    """
    Top-level function: compute features for all symbols in a merged DataFrame.

    Args:
        df:         Output of DataCollector.get_full_dataset()
        symbol_col: Column name for symbol grouping

    Returns:
        (features_df, feature_names): enriched DataFrame + list of feature column names
    """
    engineer = FeatureEngineer()
    frames = []

    for symbol, group in df.groupby(symbol_col):
        try:
            enriched = engineer.compute_all(group)
            frames.append(enriched)
        except Exception as e:
            logger.error(f"Feature engineering failed for {symbol}: {e}")

    if not frames:
        return pd.DataFrame(), []

    result = pd.concat(frames, ignore_index=True)

    # Drop rows with NaN in any feature column
    feature_names = engineer.get_feature_names(result)
    before = len(result)
    result = result.dropna(subset=feature_names)
    after = len(result)
    logger.info(
        f"Features computed: {len(feature_names)} features | "
        f"{after:,} clean rows (dropped {before - after:,} NaN rows)"
    )
    return result, feature_names


# ─── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_collector import DataCollector

    print("\n=== FEATURE ENGINEERING TEST ===\n")
    collector = DataCollector()
    raw = collector.get_full_dataset(start_date="2023-01-01", end_date="2024-01-01")

    if not raw.empty:
        feat_df, feat_cols = compute_features(raw)
        print(f"✓ Total features: {len(feat_cols)}")
        print(f"✓ Total rows:     {len(feat_df)}")
        print(f"\nFeature categories:")

        categories = {
            "Trend":     [f for f in feat_cols if any(x in f.lower() for x in ["ema","macd","adx","vwap","ichi","psar","super"])],
            "Momentum":  [f for f in feat_cols if any(x in f.lower() for x in ["rsi","stoch","mfi","williams","roc","cmo","dpo","cci"])],
            "Volatility":[f for f in feat_cols if any(x in f.lower() for x in ["bb","kc","atr","natr","don","hvol","sqz","vix"])],
            "Volume":    [f for f in feat_cols if any(x in f.lower() for x in ["obv","cmf","ad_","rvol","vol_"])],
            "F&O / PCR": [f for f in feat_cols if any(x in f.lower() for x in ["oi","pcr","max_pain","futures","dte","iv","gex","writing"])],
            "Macro":     [f for f in feat_cols if any(x in f.lower() for x in ["fii","dii","spx","ndx","usdinr","gold","crude"])],
            "Intermarket":[f for f in feat_cols if any(x in f.lower() for x in ["rel","diverge","lead","drag","headwind","smart","retail"])],
            "Derived/Reg":[f for f in feat_cols if any(x in f.lower() for x in ["ret_","regime","dow","dom","month","drawdown","midcap","skew","kurt","autocorr","sharpe","sortino","gap","intensity","loc","velocity","accel","corr"])],
        }
        for cat, cols in categories.items():
            print(f"  {cat:12s}: {len(cols):3d} features")

        print(f"\nTarget distribution:")
        if "label" in feat_df.columns:
            counts = feat_df["label"].value_counts()
            total = len(feat_df)
            for lbl, cnt in sorted(counts.items()):
                name = {-1: "DOWN", 0: "FLAT", 1: "UP"}.get(lbl, str(lbl))
                print(f"  {name}: {cnt:,} ({cnt/total*100:.1f}%)")
    print("\n=== TEST COMPLETE ===")
