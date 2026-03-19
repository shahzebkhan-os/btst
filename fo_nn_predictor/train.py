"""
train.py
========
End-to-end training runner for the F&O project.
"""

import argparse
import logging
import pandas as pd
from src.data_collector import DataCollector
from src.feature_engineering import FeatureEngineer
from src.trainer import TrainingPipeline
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainRunner")

def main():
    parser = argparse.ArgumentParser(description="F&O Model Training Runner")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--skip-optuna", action="store_true")
    parser.add_argument("--symbols", nargs="+", default=["NIFTY", "BANKNIFTY"])
    args = parser.parse_args()

    # 1. Data Collection
    logger.info(f"Step 1: Collecting data from {args.start_date} to {args.end_date}")
    collector = DataCollector()
    df_raw = collector.get_full_dataset(args.start_date, args.end_date)
    
    # 2. Feature Engineering
    logger.info("Step 2: Engineering 200+ features...")
    engineer = FeatureEngineer()
    df_features = pd.concat([engineer.compute_all(df_raw[df_raw["SYMBOL"] == s]) for s in args.symbols])
    
    # 3. Training
    logger.info("Step 3: Training Ensemble Predictor...")
    pipeline = TrainingPipeline(optimize=not args.skip_optuna)
    ensemble = pipeline.train_full(df_features)
    
    # 4. Save
    model_path = config.MODEL_DIR / "ensemble_latest.pkl"
    ensemble.save(model_path)
    logger.info(f"Project Scaffold Training Pipeline Completed. Model: {model_path} ✓")

if __name__ == "__main__":
    main()
