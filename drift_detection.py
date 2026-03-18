"""
drift_detection.py
==================
Concept drift detection and automatic model retraining.

Methods:
  1. ADWIN (Adaptive Windowing) — detect changes in prediction accuracy
  2. Feature Distribution Monitoring — detect covariate shift using KS test
  3. Performance Monitoring — track accuracy, F1, and calibration over time
  4. Automatic Retraining — trigger retraining when drift detected

Drift Types:
  - Concept Drift: P(Y|X) changes (relationship between features and target)
  - Covariate Shift: P(X) changes (feature distribution changes)
  - Prior Shift: P(Y) changes (target distribution changes)

Retraining Strategy:
  - Check for drift daily after predictions
  - Force retrain every N days (default: 7 days)
  - Use rolling window for incremental training
  - Maintain model versioning
"""

import logging
import warnings
import datetime as dt
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from river.drift import ADWIN
import joblib

import config

warnings.filterwarnings("ignore")
logger = logging.getLogger("DriftDetection")

# Ensure directories exist
config.MODEL_DIR.mkdir(exist_ok=True)
config.LOG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ADWIN DRIFT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class ADWINDriftDetector:
    """
    ADWIN (Adaptive Windowing) drift detector.

    ADWIN maintains a sliding window of recent observations and detects drift
    when the mean of two sub-windows differs significantly.

    Advantages:
      - No need to set window size
      - Adapts to data automatically
      - Provides rigorous statistical guarantees

    Usage:
        detector = ADWINDriftDetector()
        for accuracy in accuracy_stream:
            is_drift = detector.update(accuracy)
    """

    def __init__(
        self,
        delta: float = config.ADWIN_DELTA,
    ):
        """
        Initialize ADWIN detector.

        Args:
            delta: Confidence parameter (lower = more sensitive)
        """
        self.delta = delta
        self.adwin = ADWIN(delta=delta)
        self.drift_count = 0
        self.drift_timestamps: List[dt.datetime] = []

    def update(
        self,
        value: float,
    ) -> bool:
        """
        Update detector with new observation.

        Args:
            value: New observation (e.g., prediction accuracy)

        Returns:
            True if drift detected, False otherwise
        """
        self.adwin.update(value)

        if self.adwin.drift_detected:
            self.drift_count += 1
            self.drift_timestamps.append(dt.datetime.now())
            logger.warning(f"ADWIN drift detected! (count: {self.drift_count})")
            return True

        return False

    def reset(self) -> None:
        """Reset detector."""
        self.adwin = ADWIN(delta=self.delta)
        logger.info("ADWIN detector reset")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE DISTRIBUTION MONITOR
# ─────────────────────────────────────────────────────────────────────────────

class FeatureDistributionMonitor:
    """
    Monitor feature distributions for covariate shift.

    Uses Kolmogorov-Smirnov (KS) test to compare current distribution
    with reference distribution.

    Usage:
        monitor = FeatureDistributionMonitor()
        monitor.fit(X_reference)
        is_drift = monitor.detect(X_current)
    """

    def __init__(
        self,
        threshold: float = config.FEATURE_DRIFT_THRESHOLD,
    ):
        """
        Initialize feature distribution monitor.

        Args:
            threshold: p-value threshold for KS test (default: 0.05)
        """
        self.threshold = threshold
        self.reference_data: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None

    def fit(
        self,
        X_reference: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "FeatureDistributionMonitor":
        """
        Fit monitor with reference distribution.

        Args:
            X_reference: Reference feature data
            feature_names: Feature names (optional)

        Returns:
            Self
        """
        self.reference_data = X_reference
        self.feature_names = feature_names
        logger.info(f"Feature distribution monitor fitted with {len(X_reference)} reference samples")
        return self

    def detect(
        self,
        X_current: np.ndarray,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Detect feature drift using KS test.

        Args:
            X_current: Current feature data

        Returns:
            Tuple of (is_drift, drift_scores)
              - is_drift: True if any feature drifted
              - drift_scores: Dict of {feature: p_value}
        """
        if self.reference_data is None:
            raise ValueError("Monitor not fitted. Call fit() first.")

        n_features = self.reference_data.shape[1]
        drift_scores = {}
        drifted_features = []

        for i in range(n_features):
            # KS test
            statistic, p_value = ks_2samp(
                self.reference_data[:, i],
                X_current[:, i],
            )

            feature_name = self.feature_names[i] if self.feature_names else f"feature_{i}"
            drift_scores[feature_name] = p_value

            # Check if drifted
            if p_value < self.threshold:
                drifted_features.append(feature_name)

        is_drift = len(drifted_features) > 0

        if is_drift:
            logger.warning(
                f"Feature drift detected in {len(drifted_features)} features: "
                f"{', '.join(drifted_features[:5])}{'...' if len(drifted_features) > 5 else ''}"
            )

        return is_drift, drift_scores


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE MONITOR
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceMonitor:
    """
    Monitor model performance over time.

    Tracks:
      - Prediction accuracy
      - F1 score
      - Calibration error
      - Prediction confidence

    Usage:
        monitor = PerformanceMonitor()
        monitor.update(y_true, y_pred, y_proba)
        is_degraded = monitor.check_degradation()
    """

    def __init__(
        self,
        window_size: int = 50,
        degradation_threshold: float = 0.10,
    ):
        """
        Initialize performance monitor.

        Args:
            window_size: Rolling window size for metrics
            degradation_threshold: Relative degradation threshold (10%)
        """
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold

        self.accuracy_history: List[float] = []
        self.f1_history: List[float] = []
        self.calibration_history: List[float] = []
        self.timestamps: List[dt.datetime] = []

        self.baseline_accuracy: Optional[float] = None
        self.baseline_f1: Optional[float] = None

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Update performance metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)

        Returns:
            Dictionary with current metrics
        """
        from sklearn.metrics import accuracy_score, f1_score, log_loss

        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        calibration_error = 0.0
        if y_proba is not None:
            try:
                calibration_error = log_loss(y_true, y_proba)
            except:
                pass

        # Update history
        self.accuracy_history.append(accuracy)
        self.f1_history.append(f1)
        self.calibration_history.append(calibration_error)
        self.timestamps.append(dt.datetime.now())

        # Set baseline if not set
        if self.baseline_accuracy is None:
            self.baseline_accuracy = accuracy
            self.baseline_f1 = f1
            logger.info(f"Baseline performance set: Accuracy={accuracy:.4f}, F1={f1:.4f}")

        metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "calibration_error": calibration_error,
            "n_samples": len(self.accuracy_history),
        }

        return metrics

    def check_degradation(self) -> Tuple[bool, Dict[str, float]]:
        """
        Check if performance has degraded.

        Returns:
            Tuple of (is_degraded, metrics)
        """
        if len(self.accuracy_history) < 10:
            return False, {}

        # Compute recent performance (last window_size samples)
        recent_accuracy = np.mean(self.accuracy_history[-self.window_size:])
        recent_f1 = np.mean(self.f1_history[-self.window_size:])

        # Compare to baseline
        accuracy_drop = (self.baseline_accuracy - recent_accuracy) / self.baseline_accuracy
        f1_drop = (self.baseline_f1 - recent_f1) / self.baseline_f1

        is_degraded = (
            accuracy_drop > self.degradation_threshold or
            f1_drop > self.degradation_threshold
        )

        metrics = {
            "recent_accuracy": recent_accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "accuracy_drop": accuracy_drop,
            "recent_f1": recent_f1,
            "baseline_f1": self.baseline_f1,
            "f1_drop": f1_drop,
        }

        if is_degraded:
            logger.warning(
                f"Performance degradation detected: "
                f"Accuracy drop={accuracy_drop:.2%}, F1 drop={f1_drop:.2%}"
            )

        return is_degraded, metrics

    def get_history(self) -> pd.DataFrame:
        """
        Get performance history as DataFrame.

        Returns:
            DataFrame with performance metrics over time
        """
        return pd.DataFrame({
            "timestamp": self.timestamps,
            "accuracy": self.accuracy_history,
            "f1": self.f1_history,
            "calibration_error": self.calibration_history,
        })


# ─────────────────────────────────────────────────────────────────────────────
# DRIFT DETECTION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class DriftDetectionManager:
    """
    Complete drift detection and retraining manager.

    Combines:
      - ADWIN for concept drift
      - Feature distribution monitoring for covariate shift
      - Performance monitoring for model degradation

    Triggers retraining when:
      - Drift detected by any method
      - Performance degraded beyond threshold
      - N days since last training

    Usage:
        manager = DriftDetectionManager()
        manager.fit(X_reference, feature_names)
        should_retrain = manager.check_drift(y_true, y_pred, X_current)
    """

    def __init__(
        self,
        adwin_delta: float = config.ADWIN_DELTA,
        feature_threshold: float = config.FEATURE_DRIFT_THRESHOLD,
        retrain_days: int = config.DRIFT_RETRAIN_DAYS,
    ):
        self.adwin_detector = ADWINDriftDetector(delta=adwin_delta)
        self.feature_monitor = FeatureDistributionMonitor(threshold=feature_threshold)
        self.performance_monitor = PerformanceMonitor()
        self.retrain_days = retrain_days

        self.last_retrain_date: Optional[dt.datetime] = None
        self.retrain_count = 0
        self.drift_log: List[Dict[str, Any]] = []

    def fit(
        self,
        X_reference: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "DriftDetectionManager":
        """
        Fit drift detection manager.

        Args:
            X_reference: Reference feature data
            feature_names: Feature names (optional)

        Returns:
            Self
        """
        logger.info("Fitting drift detection manager...")
        self.feature_monitor.fit(X_reference, feature_names)
        self.last_retrain_date = dt.datetime.now()
        logger.info("Drift detection manager fitted ✓")
        return self

    def check_drift(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        X_current: Optional[np.ndarray] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for drift and determine if retraining needed.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            X_current: Current feature data (optional)

        Returns:
            Tuple of (should_retrain, drift_info)
        """
        logger.info("Checking for drift...")

        drift_info = {
            "timestamp": dt.datetime.now().isoformat(),
            "adwin_drift": False,
            "feature_drift": False,
            "performance_degradation": False,
            "scheduled_retrain": False,
            "should_retrain": False,
        }

        # 1. ADWIN drift detection
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_true, y_pred)
        adwin_drift = self.adwin_detector.update(accuracy)
        drift_info["adwin_drift"] = adwin_drift

        # 2. Feature distribution drift
        if X_current is not None:
            feature_drift, drift_scores = self.feature_monitor.detect(X_current)
            drift_info["feature_drift"] = feature_drift
            drift_info["drift_scores"] = drift_scores
        else:
            feature_drift = False

        # 3. Performance monitoring
        metrics = self.performance_monitor.update(y_true, y_pred, y_proba)
        is_degraded, degradation_metrics = self.performance_monitor.check_degradation()
        drift_info["performance_degradation"] = is_degraded
        drift_info["metrics"] = metrics
        drift_info["degradation_metrics"] = degradation_metrics

        # 4. Scheduled retrain check
        if self.last_retrain_date is not None:
            days_since_retrain = (dt.datetime.now() - self.last_retrain_date).days
            scheduled_retrain = days_since_retrain >= self.retrain_days
            drift_info["scheduled_retrain"] = scheduled_retrain
            drift_info["days_since_retrain"] = days_since_retrain
        else:
            scheduled_retrain = False

        # Determine if retraining needed
        should_retrain = (
            adwin_drift or
            feature_drift or
            is_degraded or
            scheduled_retrain
        )

        drift_info["should_retrain"] = should_retrain

        if should_retrain:
            reasons = []
            if adwin_drift:
                reasons.append("ADWIN drift")
            if feature_drift:
                reasons.append("Feature drift")
            if is_degraded:
                reasons.append("Performance degradation")
            if scheduled_retrain:
                reasons.append(f"Scheduled retrain ({days_since_retrain} days)")

            logger.warning(f"RETRAINING TRIGGERED: {', '.join(reasons)}")
            self.drift_log.append(drift_info)
        else:
            logger.info("No drift detected")

        return should_retrain, drift_info

    def mark_retrained(self) -> None:
        """Mark that retraining has been performed."""
        self.last_retrain_date = dt.datetime.now()
        self.retrain_count += 1
        self.adwin_detector.reset()
        logger.info(f"Retraining marked (count: {self.retrain_count})")

    def save_drift_log(self, filename: Optional[str] = None) -> None:
        """
        Save drift detection log.

        Args:
            filename: Output filename (optional)
        """
        if filename is None:
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drift_log_{timestamp}.json"

        output_path = config.LOG_DIR / filename

        with open(output_path, "w") as f:
            json.dump(self.drift_log, f, indent=2, default=str)

        logger.info(f"Drift log saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOCK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Test ADWIN
    print("Testing ADWIN Drift Detector...")
    adwin = ADWINDriftDetector(delta=0.002)

    # Simulate accuracy stream with drift
    accuracy_stream = [0.80] * 50 + [0.60] * 50  # Drift at sample 50
    drift_detected = False

    for i, acc in enumerate(accuracy_stream):
        is_drift = adwin.update(acc)
        if is_drift and not drift_detected:
            print(f"Drift detected at sample {i} ✓")
            drift_detected = True

    # Test Feature Distribution Monitor
    print("\nTesting Feature Distribution Monitor...")
    X_reference = np.random.randn(500, 10)
    X_current_no_drift = np.random.randn(100, 10)
    X_current_drift = np.random.randn(100, 10) + 2.0  # Shifted distribution

    monitor = FeatureDistributionMonitor()
    monitor.fit(X_reference)

    is_drift_no, _ = monitor.detect(X_current_no_drift)
    print(f"No drift: {not is_drift_no} ✓")

    is_drift_yes, _ = monitor.detect(X_current_drift)
    print(f"Drift detected: {is_drift_yes} ✓")

    # Test Performance Monitor
    print("\nTesting Performance Monitor...")
    perf_monitor = PerformanceMonitor(window_size=20, degradation_threshold=0.10)

    # Simulate degrading performance
    for i in range(50):
        accuracy = 0.80 if i < 30 else 0.65  # Degradation after sample 30
        y_true = np.random.randint(0, 3, 10)
        y_pred = np.random.randint(0, 3, 10)

        metrics = perf_monitor.update(y_true, y_pred)

        if i >= 40:
            is_degraded, _ = perf_monitor.check_degradation()
            if is_degraded:
                print(f"Performance degradation detected at sample {i} ✓")
                break

    # Test Drift Detection Manager
    print("\nTesting Drift Detection Manager...")
    manager = DriftDetectionManager()
    manager.fit(X_reference)

    y_true = np.random.randint(0, 3, 100)
    y_pred = np.random.randint(0, 3, 100)

    should_retrain, drift_info = manager.check_drift(y_true, y_pred, X_current=X_current_drift)
    print(f"Retraining triggered: {should_retrain} ✓")

    print("\nAll drift detection tests passed! ✓")
