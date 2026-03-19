"""
predictor.py
============
Live end-of-day prediction engine for F&O BTST signals.
Orchestrates collection, engineering, prediction, explainability, and drift monitoring.
"""

import os
import sys
import logging
import time
import datetime as dt
import argparse
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import pytz
import schedule
from tabulate import tabulate

import config
from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from model_architecture import EnsemblePredictor
from explainability import ModelExplainer
from drift_monitor import DriftMonitor
from backtester import Backtester

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_DIR / f"predictor_{dt.datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger("LivePredictor")

class LivePredictor:
    """
    Final orchestration layer for live daily predictions.
    """

    def __init__(self, model_tag: str = "latest"):
        logger.info(f"Initializing LivePredictor (Model: {model_tag})")
        
        # Load components
        self.collector = DataCollector()
        self.engineer = FeatureEngineer()
        self.ensemble = EnsemblePredictor() # In real usage: self.ensemble.load(...)
        self.explainer = ModelExplainer()
        
        # Identify feature names for monitor
        dummy_df = pd.DataFrame(columns=["CLOSE", "HIGH", "LOW", "VOLUME", "OPEN_INT"])
        # self.feature_names = self.engineer.get_feature_names(dummy_df)
        self.feature_names = ["pcr_1d_chg", "vix_close", "rsi_14"] # Mock for now
        
        self.monitor = DriftMonitor(str(config.MODEL_DIR), self.feature_names)
        self.backtester = Backtester()
        
        self.version = "2.0.0-ensemble"
        self.last_train_date = dt.datetime(2026, 3, 12) # Mock
        
        logger.info(f"System Ready. Model Version: {self.version} | Last Train: {self.last_train_date.date()}")

    def collect_eod_snapshot(self) -> pd.DataFrame:
        """
        Collect all data for today ready by 3:00 PM IST.
        """
        logger.info("Collecting EOD snapshot data...")
        try:
            # 1. Get last 60 days of historical data for lookback
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=90)
            
            df = self.collector.get_full_dataset(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            
            # 2. In real scenario, we'd fetch live bits (Option Chain, Intraday) 
            # and merge them into the last row of df.
            # self.collector.get_live_option_chain("NIFTY") ...
            
            return df
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
            # Graceful degradation: try previous day's data if today fails
            return pd.DataFrame()

    def run_prediction(self, target_date: Optional[str] = None) -> pd.DataFrame:
        """
        Full prediction pipeline orchestration.
        """
        logger.info(f"--- Starting Prediction Run for {target_date or 'TODAY'} ---")
        
        # STEP 1: Collect
        snapshot = self.collect_eod_snapshot()
        if snapshot.empty:
            logger.error("Data collection failed. Aborting run.")
            return pd.DataFrame()
            
        # STEP 2: Engineer
        features = self.engineer.compute_all(snapshot)
        last_features = features.groupby("SYMBOL").tail(1)
        
        # STEP 3: Drift Check
        drift_results = self.monitor.check_feature_drift(last_features)
        
        # STEP 4: Predict
        predictions = self.ensemble.predict_with_intervals(last_features)
        
        # STEP 5: Liquidity Filter (via Backtester code or wrapper)
        final_signals = []
        for _, row in predictions.iterrows():
            # Merging prediction info with feature info for filter
            full_row = pd.concat([row, last_features[last_features["SYMBOL"] == row["SYMBOL"]].iloc[0]])
            passes, reason = self.backtester.liquidity_filter(full_row)
            if passes:
                final_signals.append(row)
            else:
                logger.info(f"Symbol {row['SYMBOL']} filtered out: {reason}")
                
        if not final_signals:
            logger.warning("No instruments passed the liquidity filter today.")
            return pd.DataFrame()
            
        signals_df = pd.DataFrame(final_signals)
        
        # STEP 6: Explain & Report
        # Note: Explainer needs the LGBM model from the ensemble
        report_df = self.explainer.generate_signal_report(signals_df, last_features, self.ensemble.lgbm)
        
        # STEP 7: Check Prediction Drift (accuracy of yesterday's signals)
        # prediction_drift = self.monitor.check_prediction_drift(...)
        
        # STEP 8: Decision on Retraining
        should_retrain, reason = self.monitor.should_retrain(drift_results, False, self.last_train_date)
        if should_retrain:
            logger.warning(f"RETRAINING RECOMMENDED: {reason}")
            # Trigger overnight retrain...
            
        self.format_terminal_output(report_df)
        return report_df

    def format_terminal_output(self, signals: pd.DataFrame):
        """
        Print formatted signal report to console.
        """
        if signals.empty:
            print("\n[!] NO VALID TRADING SIGNALS FOR TODAY [!]\n")
            return

        date_str = dt.datetime.now().strftime("%Y-%m-%d")
        print(f"\n┌───────────────────────────────────────────────────────────────────────────────┐")
        print(f"│  F&O SIGNAL REPORT — {date_str}   Model: {self.version}             │")
        print(f"├──────────┬────────┬──────────┬────────┬──────────────────┬────────────────────┤")
        
        display_df = signals.copy()
        display_df["Dir"] = display_df["direction"].map({0: "DOWN", 1: "FLAT", 2: "UP"})
        display_df["Conf%"] = (display_df["confidence"] * 100).map("{:.1f}%".format)
        display_df["Return"] = (display_df["expected_return_p50"] * 100).map("{:+.1f}%".format)
        display_df["Interval"] = display_df.apply(
            lambda x: f"{x['interval_low_p10']*100:+.1f} to {x['interval_high_p90']*100:+.1f}%", axis=1
        )
        
        cols = ["SYMBOL", "Dir", "Conf%", "Return", "Interval", "reasoning_tags"]
        table = tabulate(display_df[cols], headers=["Symbol", "Dir", "Conf%", "Ret", "Interval", "Reasoning"], tablefmt="presto")
        print(table)
        print(f"└───────────────────────────────────────────────────────────────────────────────┘")
        
        # Health summary
        health = self.monitor.generate_health_report()
        print(f"System Health: Drift={health['adwin_drift_detected']} | Predictions={health['total_predictions']}")

def schedule_runner():
    """
    Main entry point for scheduled and CLI execution.
    """
    parser = argparse.ArgumentParser(description="Live F&O Predictor Engine")
    parser.add_argument("--run-now", action="store_true", help="Run prediction immediately")
    parser.add_argument("--backfill", nargs=2, metavar=('START', 'END'), help="Run backfill for date range (YYYY-MM-DD)")
    args = parser.parse_args()

    predictor = LivePredictor()
    ist = pytz.timezone('Asia/Kolkata')

    if args.run_now:
        predictor.run_prediction()
    elif args.backfill:
        logger.info(f"Backfilling from {args.backfill[0]} to {args.backfill[1]}")
        # Logic for historical loop...
    else:
        # Schedule for weekdays at 3:00 PM IST
        logger.info("Starting scheduler. Waiting for 3:00 PM IST on weekdays...")
        for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
            getattr(schedule.every(), day).at("15:00").do(predictor.run_prediction)

        while True:
            schedule.run_pending()
            time.sleep(30)

if __name__ == "__main__":
    schedule_runner()
