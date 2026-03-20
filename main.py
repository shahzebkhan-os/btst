"""
main.py
=======
Main orchestrator for F&O Neural Network Predictor.

Usage:
    # Train model
    python main.py --mode train

    # Generate predictions
    python main.py --mode predict

    # Run scheduled predictions (daily at 3:00 PM IST)
    python main.py --mode schedule

    # Full pipeline with retraining
    python main.py --mode full --check-drift
"""

import argparse
import logging
from typing import Optional
import datetime as dt
from pathlib import Path
import sys
from tqdm import tqdm

# --- Compatibility Patch for pytorch-forecasting metadata bug ---
import importlib.metadata
_orig_version = importlib.metadata.version
def _patched_version(name):
    """Fallback for hyphen/dot naming mismatches (e.g. zope.interface vs zope-interface)"""
    try:
        return _orig_version(name)
    except importlib.metadata.PackageNotFoundError:
        alt_name = name.replace(".", "-") if "." in name else name.replace("-", ".")
        try:
            return _orig_version(alt_name)
        except importlib.metadata.PackageNotFoundError:
            raise importlib.metadata.PackageNotFoundError(name)
importlib.metadata.version = _patched_version
# -------------------------------------------------------------

import pandas as pd
import numpy as np

from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from market_data_extended import MarketDataExtended
from training_pipeline import TrainingPipeline
from prediction_pipeline import PredictionPipeline, schedule_daily_prediction
from calibration import CalibrationManager
from position_sizing import PositionSizingManager
from explainability import ModelExplainer
from drift_detection import DriftDetectionManager
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_DIR / "main.log", mode="a"),
    ],
)
logger = logging.getLogger("Main")


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN MODE
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    optimize: bool = False,
    use_walk_forward: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> None:
    """
    Train ensemble model from scratch.

    Args:
        optimize: Run Optuna hyperparameter optimization
        use_walk_forward: Use walk-forward validation
    """
    logger.info("=" * 80)
    logger.info("MODE: TRAIN")
    logger.info("=" * 80)

    # Step 1: Collect data
    logger.info("Step 1: Collecting data...")
    collector = DataCollector()
    df = collector.get_full_dataset(
        start_date=start_date or config.TRAIN_START_DATE,
        end_date=end_date or config.VAL_END_DATE,
        symbols=config.SYMBOLS,
    )
    logger.info(f"Collected {len(df)} rows")

    if df.empty:
        logger.error("DataCollector returned an empty dataset. Check your date ranges and data directories.")
        return

    # Step 2: Engineer features
    logger.info("Step 2: Engineering features...")
    engineer = FeatureEngineer()
    df_list = []
    
    unique_symbols = df["SYMBOL"].unique()
    for symbol in tqdm(unique_symbols, desc="Engineering features", unit="symbol"):
        df_symbol = df[df["SYMBOL"] == symbol].copy()
        df_symbol = engineer.compute_all(df_symbol)
        df_list.append(df_symbol)
    
    df_features = pd.concat(df_list, ignore_index=True)
    logger.info(f"Features computed: {len(df_features.columns)} columns")

    # Step 3: Add global market features
    logger.info("Step 3: Adding global market features...")
    try:
        market_data = MarketDataExtended()
        df_global = market_data.download_all()
        df_features = df_features.merge(df_global, on="DATE", how="left")
    except Exception as e:
        logger.warning(f"Could not fetch global features: {e}")

    # Step 4: Train model
    logger.info("Step 4: Training model...")
    pipeline = TrainingPipeline(
        optimize_hyperparams=optimize,
        use_walk_forward=use_walk_forward,
    )
    ensemble = pipeline.train(df_features, target_col="target")

    logger.info("Training complete ✓")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT MODE
# ─────────────────────────────────────────────────────────────────────────────

def predict_signals(
    save_output: bool = True,
) -> pd.DataFrame:
    """
    Generate predictions for today.

    Args:
        save_output: Save signals to CSV

    Returns:
        DataFrame with ranked signals
    """
    logger.info("=" * 80)
    logger.info("MODE: PREDICT")
    logger.info("=" * 80)

    pipeline = PredictionPipeline()
    df_signals = pipeline.run(save_output=save_output)

    return df_signals


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULE MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_scheduled():
    """Run scheduled predictions (daily at 3:00 PM IST)."""
    logger.info("=" * 80)
    logger.info("MODE: SCHEDULE")
    logger.info("=" * 80)

    schedule_daily_prediction()


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE WITH DRIFT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    check_drift: bool = True,
) -> None:
    """
    Run complete pipeline with drift detection and retraining.

    Args:
        check_drift: Check for drift and retrain if needed
    """
    logger.info("=" * 80)
    logger.info("MODE: FULL PIPELINE")
    logger.info("=" * 80)

    # Step 1: Generate predictions
    logger.info("Step 1: Generating predictions...")
    pipeline = PredictionPipeline()
    df_signals = pipeline.run(save_output=True)

    if df_signals.empty:
        logger.warning("No signals generated. Exiting.")
        return

    # Step 2: Check for drift (if enabled)
    if check_drift:
        logger.info("Step 2: Checking for drift...")

        # Load drift detection manager
        try:
            import joblib
            drift_manager_path = config.MODEL_DIR / "drift_manager.pkl"

            if drift_manager_path.exists():
                drift_manager = joblib.load(drift_manager_path)
                logger.info("Drift manager loaded")
            else:
                logger.warning("Drift manager not found. Skipping drift check.")
                return

            # Check drift (requires ground truth, so this is a placeholder)
            # In production, you would collect actual outcomes and check drift daily
            logger.info("Drift checking complete (placeholder)")

        except Exception as e:
            logger.error(f"Drift checking failed: {e}")

    # Step 3: Compute position sizes
    logger.info("Step 3: Computing position sizes...")
    position_manager = PositionSizingManager(capital=config.CAPITAL)
    df_signals = position_manager.calculate_for_signals(df_signals)

    # Step 4: Generate explanations
    logger.info("Step 4: Generating explanations...")
    # This requires the trained model and feature data
    # Placeholder for now
    logger.info("Explanations generated (placeholder)")

    # Save final signals with position sizes
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = config.OUTPUT_DIR / f"final_signals_{timestamp}.csv"
    df_signals.to_csv(output_path, index=False)
    logger.info(f"Final signals saved: {output_path}")

    # Print summary
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Generated {len(df_signals)} signals")
    logger.info(f"Total capital allocation: ₹{df_signals['position_size'].sum():,.0f}")
    logger.info("\nTop 3 Signals:")
    for idx, row in df_signals.head(3).iterrows():
        logger.info(
            f"  {row['SYMBOL']} | {row['direction']} | "
            f"Confidence: {row.get('confidence_pct', 0):.1f}% | "
            f"Position: ₹{row.get('position_size', 0):,.0f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="F&O Neural Network Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model with hyperparameter optimization
  python main.py --mode train --optimize

  # Generate predictions
  python main.py --mode predict

  # Run scheduled predictions
  python main.py --mode schedule

  # Full pipeline with drift detection
  python main.py --mode full --check-drift
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "predict", "schedule", "full"],
        help="Operation mode",
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization (train mode only)",
    )

    parser.add_argument(
        "--check-drift",
        action="store_true",
        help="Check for drift and retrain if needed (full mode only)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for training (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for training (YYYY-MM-DD)",
    )

    args = parser.parse_args()

    try:
        if args.mode == "train":
            train_model(
                optimize=args.optimize,
                start_date=args.start_date,
                end_date=args.end_date
            )

        elif args.mode == "predict":
            predict_signals()

        elif args.mode == "schedule":
            run_scheduled()

        elif args.mode == "full":
            run_full_pipeline(check_drift=args.check_drift)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
