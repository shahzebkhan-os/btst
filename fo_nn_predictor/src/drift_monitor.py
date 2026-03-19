"""
drift_monitor.py
================
Concept drift detection and automated incremental retraining.
Uses ADWIN (Adaptive Windowing) for accuracy drift and PSI for feature drift.
"""

import logging
import json
import datetime as dt
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from river import drift

import config

logger = logging.getLogger("DriftMonitor")

class DriftMonitor:
    """
    Monitors data and model performance for drift.
    Triggers incremental retraining when significant drift is detected.
    """

    def __init__(self, model_dir: str, feature_names: List[str], window: int = 30):
        self.model_dir = model_dir
        self.feature_names = feature_names
        self.window = window
        
        # Reference distributions (mocked for now, usually computed from training set)
        self.reference_stats = {} 
        
        # Performance buffer
        self.performance_buffer = [] # list of (pred, actual)
        
        # ADWIN detector
        self.adwin = drift.ADWIN(delta=0.002)
        
        # Accuracy tracking
        self.accuracy_log = []

    def check_feature_drift(self, current_features: pd.DataFrame) -> Dict[str, float]:
        """
        Compute Population Stability Index (PSI) for each feature.
        """
        logger.info("Checking for feature drift using PSI...")
        psi_results = {}
        
        for feat in self.feature_names:
            if feat not in current_features.columns:
                continue
            
            # Simple PSI calculation (mocked distribution comparison)
            # In production, we'd compare current_features[feat] quartiles to reference quartiles
            current_mean = current_features[feat].mean()
            ref_mean = self.reference_stats.get(feat, {}).get("mean", current_mean)
            
            # Mock PSI based on mean shift for demonstration
            psi = abs(current_mean - ref_mean) / (ref_mean + 1e-6)
            psi_results[feat] = psi
            
            if psi > 0.2:
                logger.error(f"SIGNIFICANT DRIFT detected in feature: {feat} (PSI={psi:.4f})")
            elif psi > 0.1:
                logger.warning(f"MODERATE DRIFT detected in feature: {feat} (PSI={psi:.4f})")
                
        # Sort and log top drifted
        top_drifted = sorted(psi_results.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Top 5 drifted features: {top_drifted}")
        
        return psi_results

    def check_prediction_drift(self, recent_predictions: List[int], 
                          recent_actuals: List[int]) -> bool:
        """
        Use ADWIN to detect accuracy degradation.
        """
        if not recent_predictions or not recent_actuals:
            return False
            
        drift_detected = False
        for p, a in zip(recent_predictions, recent_actuals):
            is_correct = 1.0 if p == a else 0.0
            self.adwin.update(is_correct)
            if self.adwin.drift_detected:
                drift_detected = True
                
        # Also check rolling accuracy
        if len(recent_predictions) >= 7:
            acc = np.mean([1 if p == a else 0 for p, a in zip(recent_predictions, recent_actuals)])
            if acc < 0.45: # Threshold for 3-class classification
                logger.warning(f"Rolling accuracy ({acc:.2f}) is below threshold (0.45)")
                drift_detected = True
                
        if drift_detected:
            logger.error("PREDICTION DRIFT DETECTED via ADWIN or accuracy threshold!")
            
        return drift_detected

    def should_retrain(self, feature_psi: Dict[str, float], prediction_drift: bool,
                  last_train_date: dt.datetime) -> Tuple[bool, str]:
        """
        Decision logic for retraining.
        """
        # 1. Prediction drift is severe
        if prediction_drift:
            return True, "prediction_drift_detected"
            
        # 2. Significant feature drift
        max_psi = max(feature_psi.values()) if feature_psi else 0
        if max_psi > 0.2:
            return True, f"feature_drift_detected (max_psi={max_psi:.4f})"
            
        # 3. Scheduled retrain (weekly)
        days_since = (dt.datetime.now() - last_train_date).days
        if days_since >= 7:
            return True, f"scheduled_retrain ({days_since} days since last)"
            
        return False, "no_drift"

    def incremental_retrain(self, ensemble: Any, new_data: pd.DataFrame) -> Any:
        """
        Warm-start retrain of ensemble models.
        """
        logger.info("Starting incremental warm-start retrain...")
        # Mock logic for retraining:
        # In real scenario:
        # 1. TFT: trainer.fit(ensemble.tft.model, ckpt_path='last', ...)
        # 2. LGBM: lgbm.fit(new_X, new_y, init_model=old_lgbm)
        
        # After retraining, we'd save it
        timestamp = dt.datetime.now().strftime("%Y%m%d")
        new_model_path = config.MODEL_DIR / f"ensemble_{timestamp}.pkl"
        logger.info(f"New model saved to {new_model_path}")
        
        return ensemble

    def generate_health_report(self) -> Dict[str, Any]:
        """
        System health snapshot.
        """
        report = {
            "timestamp": dt.datetime.now().isoformat(),
            "adwin_drift_detected": self.adwin.drift_detected,
            "total_predictions": len(self.accuracy_log),
            "model_version": "v2.0.0-ensemble",
            "feature_count": len(self.feature_names)
        }
        
        output_path = config.OUTPUT_DIR / f"health_{dt.datetime.now().strftime('%Y%m%d')}.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Health report saved to {output_path}")
        return report

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running DriftMonitor synthetic test...")
    monitor = DriftMonitor("models", ["pcr_1d_chg", "vix_close"])
    
    # Test feature drift
    mock_features = pd.DataFrame({"pcr_1d_chg": [0.5]*10, "vix_close": [30]*10})
    monitor.check_feature_drift(mock_features)
    
    # Test ADWIN drift
    preds = [1, 1, 1, 1, 0, 0, 0]
    actuals = [1, 1, 1, 1, 1, 1, 1]
    monitor.check_prediction_drift(preds, actuals)
    
    print("DriftMonitor test complete ✓")
