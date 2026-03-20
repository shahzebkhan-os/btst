#!/usr/bin/env python3
"""
train_test_period.py
====================
Train the model on a short test period (1-5 days) to verify the pipeline works.
Uses config_test.py with reduced hyperparameters for quick validation.

Usage:
    python train_test_period.py --days 5
    python train_test_period.py --days 3 --optimize
"""

import sys
import os
import argparse
import logging
import datetime as dt
from pathlib import Path

# Force use of test config
os.environ["CONFIG_MODE"] = "test"

# Use test config instead of main config
import config_test as config

from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from market_data_extended import MarketDataExtended
from training_pipeline import TrainingPipeline
from prediction_pipeline import PredictionPipeline
from position_sizing import PositionSizingManager
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_DIR / "train_test.log", mode="a"),
    ],
)
logger = logging.getLogger("TrainTestPeriod")


def train_for_test_period(num_days: int, optimize: bool = False):
    """
    Train model on a short test period.

    Args:
        num_days: Number of days to test (1-5)
        optimize: Whether to run hyperparameter optimization
    """
    logger.info("=" * 80)
    logger.info(f"TEST TRAINING MODE - {num_days} DAY PERIOD")
    logger.info("=" * 80)

    # Calculate date range
    today = dt.datetime.now()
    end_date = today.strftime("%Y-%m-%d")

    # For training, we need more history for features
    # Use 30 days of history + test period
    history_days = 30 + num_days
    start_date = (today - dt.timedelta(days=history_days)).strftime("%Y-%m-%d")

    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Training history: {history_days} days")
    logger.info(f"Test period: {num_days} days")

    # Step 1: Collect data
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Collecting data...")
    logger.info("=" * 80)

    collector = DataCollector()
    df = collector.get_full_dataset(
        start_date=start_date,
        end_date=end_date,
        symbols=config.SYMBOLS,
    )

    if df.empty:
        logger.error("DataCollector returned an empty dataset. Check your date ranges and data directories.")
        logger.error("Possible solutions:")
        logger.error("  1. Run: python data_downloader.py --all")
        logger.error("  2. Check that data/historical_data/reconstructed.csv exists")
        logger.error("  3. Verify date range is within available data")
        return False

    logger.info(f"✓ Collected {len(df)} rows")
    logger.info(f"  Symbols: {df['SYMBOL'].unique().tolist()}")
    logger.info(f"  Date range: {df['DATE'].min()} to {df['DATE'].max()}")

    # Step 2: Engineer features
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Engineering features...")
    logger.info("=" * 80)

    engineer = FeatureEngineer()
    df_list = []

    unique_symbols = df["SYMBOL"].unique()
    for symbol in tqdm(unique_symbols, desc="Engineering features", unit="symbol"):
        df_symbol = df[df["SYMBOL"] == symbol].copy()
        df_symbol = engineer.compute_all(df_symbol)
        df_list.append(df_symbol)

    df_features = pd.concat(df_list, ignore_index=True)
    logger.info(f"✓ Features computed: {len(df_features.columns)} columns")
    logger.info(f"  Feature rows: {len(df_features)}")

    # Step 3: Add global market features
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Adding global market features...")
    logger.info("=" * 80)

    try:
        market_data = MarketDataExtended()
        df_global = market_data.download_all()
        df_features = df_features.merge(df_global, on="DATE", how="left")
        logger.info(f"✓ Global features added: {len(df_global.columns)} columns")
    except Exception as e:
        logger.warning(f"⚠ Could not fetch global features: {e}")
        logger.warning("Continuing without global features...")

    # Step 4: Train model
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Training model...")
    logger.info("=" * 80)

    # Override config for test
    config.MAX_EPOCHS = 5 if not optimize else 10
    config.N_CV_FOLDS = 2
    config.OPTUNA_N_TRIALS = 3 if optimize else 0

    logger.info(f"  Max epochs: {config.MAX_EPOCHS}")
    logger.info(f"  CV folds: {config.N_CV_FOLDS}")
    logger.info(f"  Optuna trials: {config.OPTUNA_N_TRIALS if optimize else 'Disabled'}")

    pipeline = TrainingPipeline(
        optimize_hyperparams=optimize,
        use_walk_forward=False,  # Use simple split for testing
    )

    try:
        ensemble = pipeline.train(df_features, target_col="target")
        logger.info("✓ Training complete!")
        return True
    except Exception as e:
        logger.error(f"✗ Training failed: {e}", exc_info=True)
        return False


def generate_test_predictions():
    """Generate predictions for the test period"""
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Generating test predictions...")
    logger.info("=" * 80)

    try:
        pipeline = PredictionPipeline()
        df_signals = pipeline.run(save_output=True)

        if df_signals.empty:
            logger.warning("⚠ No signals generated")
            return False

        logger.info(f"✓ Generated {len(df_signals)} signals")
        logger.info("\nTop 3 Signals:")
        for idx, row in df_signals.head(3).iterrows():
            logger.info(
                f"  {idx+1}. {row['SYMBOL']} | {row['direction']} | "
                f"Confidence: {row.get('confidence_pct', 0):.1f}%"
            )

        return True
    except Exception as e:
        logger.error(f"✗ Prediction failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train model on test period (1-5 days)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--days",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="Number of days to test (1-5)"
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization (takes longer)"
    )

    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Only generate predictions (skip training)"
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 80)
    print("F&O NEURAL NETWORK PREDICTOR - TEST PERIOD TRAINING")
    print("=" * 80)
    print(f"Test period: {args.days} day(s)")
    print(f"Optimization: {'Enabled' if args.optimize else 'Disabled'}")
    print(f"Start time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    try:
        if not args.predict_only:
            # Train model
            success = train_for_test_period(args.days, args.optimize)

            if not success:
                logger.error("\n✗ Training failed. Please check logs above.")
                sys.exit(1)

        # Generate predictions
        success = generate_test_predictions()

        # Final summary
        print("\n" + "=" * 80)
        if success:
            print("🎉 TEST PERIOD TRAINING COMPLETE!")
            print("=" * 80)
            print(f"\nModel trained on {args.days} day test period")
            print(f"Models saved to: {config.MODEL_DIR}")
            print(f"Outputs saved to: {config.OUTPUT_DIR}")
            print(f"\nNext steps:")
            print("  1. Review logs in: logs/train_test.log")
            print("  2. Check model performance in: output_test/")
            print("  3. If satisfied, train on full dataset: python main.py --mode train")
            print("=" * 80)
            sys.exit(0)
        else:
            print("✗ TEST TRAINING COMPLETED WITH WARNINGS")
            print("=" * 80)
            print("Check logs for details")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Test training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
