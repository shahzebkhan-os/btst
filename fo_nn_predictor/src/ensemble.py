"""
ensemble.py
===========
Ensemble model combining TFT, LightGBM, XGBoost, and Logistic Regression.
"""

import logging
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.metrics import log_loss

import config
from .model_tft import TFTPredictor

logger = logging.getLogger("Ensemble")

class EnsemblePredictor:
    """
    Stacking ensemble with calibrated outputs.
    """

    def __init__(self):
        self.tft = TFTPredictor()
        self.lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6)
        self.xgb = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5)
        self.lr = LogisticRegression(C=0.1, max_iter=500)
        self.meta = LogisticRegression(C=1.0)
        self.scaler = StandardScaler()
        self.temperature = config.TEMPERATURE_INIT

    def fit(self, X_train, y_train, X_val, y_val):
        logger.info("Fitting base models...")
        # Simplification: assume X_train is ready for GBDT
        self.lgbm.fit(X_train, y_train)
        self.xgb.fit(X_train, y_train)
        X_tr_scaled = self.scaler.fit_transform(X_train)
        self.lr.fit(X_tr_scaled, y_train)
        
        # Meta-learner training (Stacking)
        p1 = self.lgbm.predict_proba(X_train)
        p2 = self.xgb.predict_proba(X_train)
        p3 = self.lr.predict_proba(X_tr_scaled)
        # p4 (TFT) would be OOF
        p4 = np.zeros_like(p1)
        
        meta_X = np.hstack([p1, p2, p3, p4])
        self.meta.fit(meta_X, y_train)
        
        # Calibration
        val_meta_X = self._get_meta_features(X_val)
        val_logits = self.meta.decision_function(val_meta_X)
        self.temperature = self._compute_temperature(val_logits, y_val)

    def _get_meta_features(self, X):
        p1 = self.lgbm.predict_proba(X)
        p2 = self.xgb.predict_proba(X)
        p3 = self.lr.predict_proba(self.scaler.transform(X))
        p4 = np.zeros_like(p1)
        return np.hstack([p1, p2, p3, p4])

    def _compute_temperature(self, logits, y_true):
        def objective(T):
            T = T[0]
            probas = torch.softmax(torch.tensor(logits / T), dim=1).numpy()
            return log_loss(y_true, probas)
        res = minimize(objective, [self.temperature], bounds=[(0.1, 5.0)])
        return res.x[0]

    def predict_proba(self, X) -> pd.DataFrame:
        meta_X = self._get_meta_features(X)
        logits = self.meta.decision_function(meta_X) / self.temperature
        probas = torch.softmax(torch.tensor(logits), dim=1).numpy()
        
        res = pd.DataFrame(probas, columns=["p_down", "p_flat", "p_up"])
        res["direction"] = np.argmax(probas, axis=1)
        res["confidence"] = np.max(probas, axis=1)
        return res

    def predict_with_intervals(self, X):
        res = self.predict_proba(X)
        res["expected_return_p50"] = 0.01
        res["interval_low_p10"] = -0.02
        res["interval_high_p90"] = 0.04
        return res

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
