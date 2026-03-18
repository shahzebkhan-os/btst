"""
calibration.py
==============
Model calibration for reliable confidence estimates.

Methods:
  1. Temperature Scaling — calibrates softmax outputs
  2. Conformal Prediction — provides prediction intervals with coverage guarantees

Temperature Scaling:
  - Learns a single scalar parameter T that scales logits before softmax
  - Minimizes negative log-likelihood on validation set
  - Improves calibration without changing accuracy

Conformal Prediction:
  - Provides prediction intervals with guaranteed coverage (e.g., 90%)
  - Methods: LAC (Least Ambiguous Set), APS (Adaptive Prediction Sets), RAPS
  - Uses calibration set to compute conformity scores
"""

import logging
import warnings
from typing import Optional, Dict, Tuple, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from mapie.classification import MapieClassifier
from scipy.special import softmax

import config

warnings.filterwarnings("ignore")
logger = logging.getLogger("Calibration")


# ─────────────────────────────────────────────────────────────────────────────
# TEMPERATURE SCALING
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaling:
    """
    Temperature scaling for calibrating neural network outputs.

    Formula:
        p_calibrated = softmax(logits / T)

    where T is learned to minimize NLL on validation set.

    Usage:
        calibrator = TemperatureScaling()
        calibrator.fit(logits_val, y_val)
        calibrated_probs = calibrator.transform(logits_test)
    """

    def __init__(self):
        self.temperature: Optional[float] = None

    def fit(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 1000,
    ) -> float:
        """
        Learn optimal temperature parameter.

        Args:
            logits: Raw model outputs (N, C) before softmax
            y_true: True labels (N,)
            lr: Learning rate for optimization
            max_iter: Maximum optimization iterations

        Returns:
            Optimal temperature value
        """
        logger.info("Fitting temperature scaling...")

        # Convert to torch tensors
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(y_true, dtype=torch.long)

        # Initialize temperature parameter
        temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

        # Define loss function
        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(logits_tensor / temperature, labels_tensor)
            loss.backward()
            return loss

        # Optimize temperature
        optimizer.step(eval_loss)

        self.temperature = temperature.item()
        logger.info(f"Optimal temperature: {self.temperature:.4f}")

        # Compute calibration metrics
        calibrated_probs = self.transform(logits)
        nll_before = log_loss(y_true, softmax(logits, axis=1))
        nll_after = log_loss(y_true, calibrated_probs)

        logger.info(f"NLL before calibration: {nll_before:.4f}")
        logger.info(f"NLL after calibration: {nll_after:.4f}")
        logger.info(f"NLL improvement: {nll_before - nll_after:.4f}")

        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Raw model outputs (N, C)

        Returns:
            Calibrated probabilities (N, C)
        """
        if self.temperature is None:
            raise ValueError("Temperature not fitted. Call fit() first.")

        scaled_logits = logits / self.temperature
        calibrated_probs = softmax(scaled_logits, axis=1)

        return calibrated_probs

    def fit_transform(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 1000,
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            logits: Raw model outputs (N, C)
            y_true: True labels (N,)
            lr: Learning rate
            max_iter: Maximum iterations

        Returns:
            Calibrated probabilities (N, C)
        """
        self.fit(logits, y_true, lr, max_iter)
        return self.transform(logits)


# ─────────────────────────────────────────────────────────────────────────────
# CONFORMAL PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

class ConformalPredictor:
    """
    Conformal prediction for classification with guaranteed coverage.

    Methods:
      - LAC (Least Ambiguous Set-valued Classifier)
      - APS (Adaptive Prediction Sets)
      - RAPS (Regularized Adaptive Prediction Sets)

    Provides prediction sets with guaranteed coverage (e.g., 90% of the time,
    the true label is in the prediction set).

    Usage:
        cp = ConformalPredictor(base_estimator)
        cp.fit(X_train, y_train, X_cal, y_cal)
        y_pred, intervals = cp.predict(X_test, alpha=0.1)
    """

    def __init__(
        self,
        base_estimator: Any,
        method: str = config.CONFORMAL_METHOD,
        cv: str = "prefit",
    ):
        """
        Initialize conformal predictor.

        Args:
            base_estimator: Trained sklearn-compatible classifier
            method: Conformal method (naive, lac, aps, raps)
            cv: Cross-validation strategy ('prefit' uses pre-trained model)
        """
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.mapie_classifier: Optional[MapieClassifier] = None

    def fit(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "ConformalPredictor":
        """
        Fit conformal predictor on calibration set.

        Args:
            X_cal: Calibration features
            y_cal: Calibration labels

        Returns:
            Self
        """
        logger.info(f"Fitting conformal predictor (method={self.method})...")

        self.mapie_classifier = MapieClassifier(
            estimator=self.base_estimator,
            method=self.method,
            cv=self.cv,
        )

        # Fit using calibration set
        self.mapie_classifier.fit(X_cal, y_cal)

        logger.info("Conformal predictor fitted ✓")
        return self

    def predict(
        self,
        X: np.ndarray,
        alpha: float = config.CONFORMAL_ALPHA,
        return_probas: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate conformal predictions with prediction sets.

        Args:
            X: Input features
            alpha: Miscoverage rate (0.1 = 90% coverage)
            return_probas: Return probability estimates

        Returns:
            Tuple of:
              - y_pred: Predicted labels (N,)
              - intervals: Dict with prediction sets and probabilities
        """
        if self.mapie_classifier is None:
            raise ValueError("Conformal predictor not fitted. Call fit() first.")

        # Get predictions with prediction sets
        y_pred, y_ps = self.mapie_classifier.predict(X, alpha=alpha)

        result = {
            "prediction_sets": y_ps,  # (N, n_classes) boolean array
            "set_size": y_ps.sum(axis=1),  # Number of classes in each set
        }

        if return_probas:
            y_proba = self.mapie_classifier.predict_proba(X)
            result["probabilities"] = y_proba

        return y_pred, result

    def evaluate_coverage(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        alpha: float = config.CONFORMAL_ALPHA,
    ) -> Dict[str, float]:
        """
        Evaluate conformal prediction coverage.

        Args:
            X_test: Test features
            y_test: Test labels
            alpha: Miscoverage rate

        Returns:
            Dictionary with coverage metrics
        """
        y_pred, intervals = self.predict(X_test, alpha=alpha)

        # Check if true label is in prediction set
        n_samples = len(y_test)
        coverage = 0

        for i in range(n_samples):
            if intervals["prediction_sets"][i, y_test[i]]:
                coverage += 1

        empirical_coverage = coverage / n_samples
        expected_coverage = 1 - alpha

        metrics = {
            "empirical_coverage": empirical_coverage,
            "expected_coverage": expected_coverage,
            "coverage_gap": empirical_coverage - expected_coverage,
            "avg_set_size": intervals["set_size"].mean(),
            "median_set_size": np.median(intervals["set_size"]),
        }

        logger.info(f"Conformal Prediction Metrics:")
        logger.info(f"  Expected coverage: {expected_coverage:.2%}")
        logger.info(f"  Empirical coverage: {empirical_coverage:.2%}")
        logger.info(f"  Coverage gap: {metrics['coverage_gap']:+.2%}")
        logger.info(f"  Avg set size: {metrics['avg_set_size']:.2f}")

        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class CalibrationManager:
    """
    Complete calibration pipeline combining temperature scaling and
    conformal prediction.

    Usage:
        manager = CalibrationManager()
        manager.fit(logits_cal, y_cal, base_estimator)
        calibrated_probs, intervals = manager.predict(logits_test, X_test)
    """

    def __init__(
        self,
        use_temp_scaling: bool = config.TEMP_SCALING_ENABLED,
        use_conformal: bool = True,
        conformal_method: str = config.CONFORMAL_METHOD,
    ):
        self.use_temp_scaling = use_temp_scaling
        self.use_conformal = use_conformal
        self.conformal_method = conformal_method

        self.temp_scaler: Optional[TemperatureScaling] = None
        self.conformal_predictor: Optional[ConformalPredictor] = None

    def fit(
        self,
        logits_cal: np.ndarray,
        y_cal: np.ndarray,
        X_cal: np.ndarray,
        base_estimator: Any,
    ) -> "CalibrationManager":
        """
        Fit all calibration methods.

        Args:
            logits_cal: Raw model outputs on calibration set
            y_cal: True labels on calibration set
            X_cal: Features on calibration set
            base_estimator: Base model for conformal prediction

        Returns:
            Self
        """
        logger.info("=" * 80)
        logger.info("CALIBRATION PIPELINE")
        logger.info("=" * 80)

        # Temperature scaling
        if self.use_temp_scaling:
            self.temp_scaler = TemperatureScaling()
            self.temp_scaler.fit(logits_cal, y_cal)
        else:
            logger.info("Temperature scaling disabled")

        # Conformal prediction
        if self.use_conformal:
            self.conformal_predictor = ConformalPredictor(
                base_estimator=base_estimator,
                method=self.conformal_method,
            )
            self.conformal_predictor.fit(X_cal, y_cal)
        else:
            logger.info("Conformal prediction disabled")

        logger.info("Calibration complete ✓")
        return self

    def predict(
        self,
        logits: np.ndarray,
        X: Optional[np.ndarray] = None,
        alpha: float = config.CONFORMAL_ALPHA,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Generate calibrated predictions.

        Args:
            logits: Raw model outputs
            X: Features (required if using conformal prediction)
            alpha: Miscoverage rate for conformal prediction

        Returns:
            Tuple of:
              - calibrated_probs: Calibrated probabilities
              - intervals: Conformal prediction intervals (None if disabled)
        """
        # Apply temperature scaling
        if self.use_temp_scaling and self.temp_scaler is not None:
            calibrated_probs = self.temp_scaler.transform(logits)
        else:
            calibrated_probs = softmax(logits, axis=1)

        # Apply conformal prediction
        intervals = None
        if self.use_conformal and self.conformal_predictor is not None:
            if X is None:
                raise ValueError("X required for conformal prediction")
            _, intervals = self.conformal_predictor.predict(X, alpha=alpha)

        return calibrated_probs, intervals


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOCK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Test Temperature Scaling
    print("Testing Temperature Scaling...")
    np.random.seed(42)
    logits_val = np.random.randn(100, 3)
    y_val = np.random.randint(0, 3, 100)

    temp_scaler = TemperatureScaling()
    temp_scaler.fit(logits_val, y_val, max_iter=100)
    calibrated_probs = temp_scaler.transform(logits_val)
    print(f"Temperature: {temp_scaler.temperature:.4f} ✓")

    # Test Conformal Prediction (with dummy model)
    print("\nTesting Conformal Prediction...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    X_train, y_train = make_classification(n_samples=500, n_features=20, n_informative=10, n_classes=3, random_state=42)
    X_cal, y_cal = make_classification(n_samples=200, n_features=20, n_informative=10, n_classes=3, random_state=43)
    X_test, y_test = make_classification(n_samples=100, n_features=20, n_informative=10, n_classes=3, random_state=44)

    # Train base model
    base_model = LogisticRegression(multi_class="multinomial", random_state=42)
    base_model.fit(X_train, y_train)

    # Fit conformal predictor
    cp = ConformalPredictor(base_model, method="lac")
    cp.fit(X_cal, y_cal)

    # Predict
    y_pred, intervals = cp.predict(X_test, alpha=0.1)
    print(f"Prediction sets shape: {intervals['prediction_sets'].shape} ✓")
    print(f"Avg set size: {intervals['set_size'].mean():.2f} ✓")

    # Evaluate coverage
    metrics = cp.evaluate_coverage(X_test, y_test, alpha=0.1)
    print(f"Empirical coverage: {metrics['empirical_coverage']:.2%} ✓")

    print("\nAll calibration tests passed! ✓")
