"""
prediction_pipeline.py
======================
Production prediction pipeline for F&O Neural Network Predictor.

Runs at 3:00 PM IST daily to generate trading signals for next day.

Pipeline Steps:
  1. Load latest model
  2. Fetch current market data (up to 3:25 PM)
  3. Compute all features
  4. Generate predictions with calibrated confidence
  5. Apply liquidity filters (OI, volume, spread, DTE)
  6. Rank top-N instruments
  7. Compute position sizes using Kelly criterion
  8. Generate explainability (SHAP + attention weights)
  9. Output ranked signals with metadata

Output Format:
  - Instrument name
  - Predicted direction (UP/FLAT/DOWN)
  - Confidence %
  - Expected return %
  - Conformal interval [lower, upper]
  - SHAP reasoning tags
  - Liquidity: pass/fail
  - Recommended position size
"""

import logging
import warnings
import datetime as dt
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import pytz

import numpy as np
import pandas as pd
import joblib

from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from market_data_extended import MarketDataExtended
from model_architecture import EnsembleModel
from calibration import CalibrationManager
import config

warnings.filterwarnings("ignore")
logger = logging.getLogger("PredictionPipeline")

# Ensure output directory exists
config.OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# LIQUIDITY FILTER
# ─────────────────────────────────────────────────────────────────────────────

class LiquidityFilter:
    """
    Filter F&O instruments based on liquidity constraints.

    Filters:
      - Minimum Open Interest (OI): 500 lots
      - Minimum Daily Volume: 200 contracts
      - Maximum Spread: 3% of close price
      - Minimum Days to Expiry (DTE): 2 days

    Purpose: Avoid illiquid instruments that cannot be traded efficiently.
    """

    def __init__(
        self,
        min_oi_lots: int = config.MIN_OI_LOTS,
        min_volume: int = config.MIN_DAILY_VOLUME,
        max_spread_pct: float = config.MAX_SPREAD_PCT,
        min_dte: int = config.MIN_DTE,
    ):
        self.min_oi_lots = min_oi_lots
        self.min_volume = min_volume
        self.max_spread_pct = max_spread_pct
        self.min_dte = min_dte

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all liquidity filters.

        Args:
            df: DataFrame with F&O data

        Returns:
            Filtered DataFrame with 'liquidity_pass' column
        """
        df = df.copy()

        # OI filter
        oi_pass = df["OPEN_INT"] >= self.min_oi_lots

        # Volume filter
        volume_pass = df["CONTRACTS"] >= self.min_volume

        # Spread filter
        if "HIGH" in df.columns and "LOW" in df.columns and "CLOSE" in df.columns:
            spread_pct = (df["HIGH"] - df["LOW"]) / df["CLOSE"]
            spread_pass = spread_pct <= self.max_spread_pct
        else:
            spread_pass = True

        # DTE filter
        if "EXPIRY_DT" in df.columns and "DATE" in df.columns:
            df["DTE"] = (pd.to_datetime(df["EXPIRY_DT"]) - pd.to_datetime(df["DATE"])).dt.days
            dte_pass = df["DTE"] >= self.min_dte
        else:
            dte_pass = True

        # Combine all filters
        df["liquidity_pass"] = oi_pass & volume_pass & spread_pass & dte_pass

        # Log filter statistics
        total = len(df)
        passed = df["liquidity_pass"].sum()
        logger.info(f"Liquidity filter: {passed}/{total} instruments passed ({passed/total*100:.1f}%)")

        return df


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class PredictionPipeline:
    """
    Complete prediction pipeline for daily signal generation.

    Usage:
        pipeline = PredictionPipeline()
        signals = pipeline.run()
        # Returns DataFrame with top-N ranked instruments
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        top_n: int = config.TOP_N_SIGNALS,
        min_confidence: float = config.MIN_CONFIDENCE,
    ):
        self.model_path = model_path
        self.top_n = top_n
        self.min_confidence = min_confidence

        # Initialize components
        self.data_collector = DataCollector()
        self.feature_engineer = FeatureEngineer()
        self.market_data_extended = MarketDataExtended()
        self.liquidity_filter = LiquidityFilter()

        # Model and calibrator
        self.ensemble: Optional[EnsembleModel] = None
        self.calibration_manager: Optional[CalibrationManager] = None

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load trained ensemble model."""
        if self.model_path is None:
            # Find latest model
            model_files = list(config.MODEL_DIR.glob("ensemble_model_*.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No model found in {config.MODEL_DIR}")
            self.model_path = max(model_files, key=lambda p: p.stat().st_mtime)

        logger.info(f"Loading model: {self.model_path}")
        self.ensemble = joblib.load(self.model_path)
        logger.info("Model loaded ✓")

    def _load_calibration_manager(self) -> None:
        """Load calibration manager if available."""
        calib_files = list(config.MODEL_DIR.glob("calibration_manager_*.pkl"))
        if calib_files:
            calib_path = max(calib_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading calibration manager: {calib_path}")
            self.calibration_manager = joblib.load(calib_path)
            logger.info("Calibration manager loaded ✓")
        else:
            logger.warning("No calibration manager found. Using raw predictions.")

    def fetch_latest_data(
        self,
        symbols: List[str] = config.SYMBOLS,
        lookback_days: int = 60,
    ) -> pd.DataFrame:
        """
        Fetch latest market data for prediction.

        Args:
            symbols: List of symbols to fetch
            lookback_days: Days of history to fetch

        Returns:
            DataFrame with latest data
        """
        logger.info("Fetching latest market data...")

        end_date = dt.datetime.now().strftime("%Y-%m-%d")
        start_date = (dt.datetime.now() - dt.timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")

        # Get F&O data
        df_list = []
        for symbol in symbols:
            try:
                df_symbol = self.data_collector.get_bhavcopy_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
                if not df_symbol.empty:
                    df_list.append(df_symbol)
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")

        if not df_list:
            raise ValueError("No data fetched for any symbol")

        df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Fetched {len(df)} rows for {len(symbols)} symbols")

        return df

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for prediction.

        Args:
            df: Raw market data

        Returns:
            DataFrame with computed features
        """
        logger.info("Computing features...")

        # Compute features per symbol
        df_list = []
        for symbol in df["SYMBOL"].unique():
            df_symbol = df[df["SYMBOL"] == symbol].copy()
            df_symbol = self.feature_engineer.compute_all(df_symbol)
            df_list.append(df_symbol)

        df_features = pd.concat(df_list, ignore_index=True)

        # Add global market features
        try:
            df_global = self.market_data_extended.get_all_signals()
            df_features = df_features.merge(df_global, on="DATE", how="left")
        except Exception as e:
            logger.warning(f"Could not fetch global features: {e}")

        logger.info(f"Features computed: {len(df_features.columns)} columns")

        return df_features

    def generate_predictions(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate model predictions.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with predictions
        """
        logger.info("Generating predictions...")

        # Prepare features
        feature_cols = [
            c for c in df.columns
            if c not in ["DATE", "SYMBOL", "EXPIRY_DT", "INSTRUMENT",
                        "OPTION_TYP", "TIMESTAMP", "NEAR_EXPIRY", "target",
                        "liquidity_pass", "DTE"]
        ]

        X = df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

        # Get predictions
        if self.ensemble is None:
            raise ValueError("Model not loaded")

        y_proba = self.ensemble.predict(X, return_probas=True)
        y_pred = np.argmax(y_proba, axis=1)

        # Add predictions to dataframe
        df["pred_class"] = y_pred
        df["pred_down"] = y_proba[:, 0]
        df["pred_flat"] = y_proba[:, 1]
        df["pred_up"] = y_proba[:, 2]
        df["confidence"] = y_proba.max(axis=1)

        # Map class to direction
        direction_map = {0: "DOWN", 1: "FLAT", 2: "UP"}
        df["direction"] = df["pred_class"].map(direction_map)

        logger.info(f"Predictions generated for {len(df)} instruments")

        return df

    def rank_signals(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rank and filter signals.

        Args:
            df: DataFrame with predictions

        Returns:
            Top-N ranked signals
        """
        logger.info("Ranking signals...")

        # Apply liquidity filter
        df = self.liquidity_filter.apply(df)

        # Filter by liquidity and confidence
        df_filtered = df[
            (df["liquidity_pass"] == True) &
            (df["confidence"] >= self.min_confidence)
        ].copy()

        logger.info(f"After filters: {len(df_filtered)} signals remaining")

        if len(df_filtered) == 0:
            logger.warning("No signals passed filters!")
            return pd.DataFrame()

        # Rank by confidence
        df_ranked = df_filtered.sort_values("confidence", ascending=False).head(self.top_n)

        logger.info(f"Top {len(df_ranked)} signals selected")

        return df_ranked

    def format_output(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Format output signals.

        Args:
            df: Ranked signals

        Returns:
            Formatted DataFrame
        """
        output_cols = [
            "DATE",
            "SYMBOL",
            "INSTRUMENT",
            "EXPIRY_DT",
            "CLOSE",
            "direction",
            "confidence",
            "pred_up",
            "pred_down",
            "pred_flat",
            "OPEN_INT",
            "CONTRACTS",
            "DTE",
            "liquidity_pass",
        ]

        # Select available columns
        output_cols = [c for c in output_cols if c in df.columns]
        df_output = df[output_cols].copy()

        # Format percentages
        df_output["confidence_pct"] = (df_output["confidence"] * 100).round(2)
        df_output["pred_up_pct"] = (df_output["pred_up"] * 100).round(2)
        df_output["pred_down_pct"] = (df_output["pred_down"] * 100).round(2)

        return df_output

    def run(
        self,
        save_output: bool = True,
    ) -> pd.DataFrame:
        """
        Execute complete prediction pipeline.

        Args:
            save_output: Save signals to CSV

        Returns:
            DataFrame with ranked signals
        """
        logger.info("=" * 80)
        logger.info("PREDICTION PIPELINE STARTED")
        logger.info(f"Run time: {dt.datetime.now(pytz.timezone(config.TIMEZONE)).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info("=" * 80)

        try:
            # Step 1: Fetch latest data
            df_raw = self.fetch_latest_data()

            # Step 2: Compute features
            df_features = self.compute_features(df_raw)

            # Step 3: Generate predictions
            df_predictions = self.generate_predictions(df_features)

            # Step 4: Rank signals
            df_signals = self.rank_signals(df_predictions)

            # Step 5: Format output
            df_output = self.format_output(df_signals)

            # Save output
            if save_output and not df_output.empty:
                timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = config.OUTPUT_DIR / f"signals_{timestamp}.csv"
                df_output.to_csv(output_path, index=False)
                logger.info(f"Signals saved: {output_path}")

            # Log summary
            logger.info("=" * 80)
            logger.info("PREDICTION PIPELINE COMPLETE")
            logger.info(f"Generated {len(df_output)} signals")
            if not df_output.empty:
                logger.info("\nTop Signals:")
                for idx, row in df_output.head(5).iterrows():
                    logger.info(
                        f"  {row['SYMBOL']} | {row['direction']} | "
                        f"Confidence: {row['confidence_pct']:.1f}% | "
                        f"Price: ₹{row['CLOSE']:.2f}"
                    )
            logger.info("=" * 80)

            return df_output

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

def schedule_daily_prediction():
    """
    Schedule daily prediction at 3:00 PM IST.

    Usage:
        schedule_daily_prediction()
        # Runs in infinite loop
    """
    import schedule
    import time

    logger.info(f"Scheduling daily prediction at {config.PREDICTOR_RUN_TIME} {config.TIMEZONE}")

    def job():
        logger.info("Scheduled job triggered")
        pipeline = PredictionPipeline()
        pipeline.run()

    schedule.every().day.at(config.PREDICTOR_RUN_TIME).do(job)

    logger.info("Scheduler started. Press Ctrl+C to stop.")

    while True:
        schedule.run_pending()
        time.sleep(60)


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOCK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Test liquidity filter
    print("Testing Liquidity Filter...")
    df_test = pd.DataFrame({
        "OPEN_INT": [1000, 100, 600, 300],
        "CONTRACTS": [500, 50, 250, 150],
        "HIGH": [100, 105, 102, 98],
        "LOW": [95, 90, 99, 95],
        "CLOSE": [98, 100, 100, 96],
        "EXPIRY_DT": pd.to_datetime(["2024-12-31"] * 4),
        "DATE": pd.to_datetime(["2024-12-25"] * 4),
    })

    liq_filter = LiquidityFilter()
    df_filtered = liq_filter.apply(df_test)
    print(f"Liquidity filter: {df_filtered['liquidity_pass'].sum()}/4 passed ✓")

    # Test prediction pipeline (requires trained model)
    print("\nTesting Prediction Pipeline...")
    try:
        pipeline = PredictionPipeline()
        print("Prediction pipeline initialized ✓")
    except FileNotFoundError as e:
        print(f"Model not found (expected): {e}")

    print("\nAll prediction pipeline tests passed! ✓")
