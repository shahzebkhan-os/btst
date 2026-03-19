"""
trainer.py
==========
Modular training pipeline components.
"""

import logging
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.metrics import accuracy_score, f1_score

import config
from .ensemble import EnsemblePredictor

logger = logging.getLogger("Trainer")

class WalkForwardValidator:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, df):
        indices = np.arange(len(df))
        split_size = len(df) // (self.n_splits + 1)
        splits = []
        for i in range(self.n_splits):
            train_idx = indices[: (i + 1) * split_size]
            val_idx = indices[(i + 1) * split_size : (i + 2) * split_size]
            splits.append((train_idx, val_idx))
        return splits

class TrainingPipeline:
    def __init__(self, optimize=False):
        self.optimize = optimize
        self.ensemble = EnsemblePredictor()

    def run_optuna(self, X, y):
        logger.info("Starting Optuna HPO...")
        def objective(trial):
            # Param suggestions...
            return np.random.uniform(0.5, 0.7) # Mock f1
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=config.OPTUNA_TRIALS)
        return study.best_params

    def train_full(self, df: pd.DataFrame):
        logger.info("Starting full training...")
        # Prepare X, y
        target = "label_3c"
        features = [c for c in df.columns if c not in ["DATE", "SYMBOL", target, "time_idx"]]
        X, y = df[features].values, df[target].values
        
        # Split (Walk-forward)
        split_idx = int(0.8 * len(df))
        X_tr, X_val = X[:split_idx], X[split_idx:]
        y_tr, y_val = y[:split_idx], y[split_idx:]
        
        # Fit
        self.ensemble.fit(X_tr, y_tr, X_val, y_val)
        
        # Eval
        preds = self.ensemble.predict_proba(X_val)
        acc = accuracy_score(y_val, preds["direction"])
        f1 = f1_score(y_val, preds["direction"], average="macro")
        
        logger.info(f"Model Trained. Acc: {acc:.4f}, F1: {f1:.4f}")
        return self.ensemble
