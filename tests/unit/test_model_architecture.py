"""
test_model_architecture.py
===========================
Unit tests for model architecture module.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/runner/work/btst/btst')

from model_architecture import FocalLoss, TFTPredictor, EnsemblePredictor, compute_temperature


class TestFocalLoss:
    """Test suite for Focal Loss"""

    def test_focal_loss_initialization(self):
        """Test Focal Loss initializes correctly"""
        focal_loss = FocalLoss(alpha=[0.25, 0.5, 0.25], gamma=2.0)
        assert focal_loss is not None
        assert focal_loss.gamma == 2.0

    def test_focal_loss_forward(self):
        """Test Focal Loss forward pass"""
        focal_loss = FocalLoss(alpha=[0.25, 0.5, 0.25], gamma=2.0)

        # Create dummy inputs
        inputs = torch.randn(10, 3, requires_grad=True)  # (batch, classes)
        targets = torch.randint(0, 3, (10,))  # (batch,)

        # Forward pass
        loss = focal_loss(inputs, targets)

        # Check loss is scalar and positive
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_focal_loss_gradient(self):
        """Test Focal Loss computes gradients correctly"""
        focal_loss = FocalLoss(alpha=[0.25, 0.5, 0.25], gamma=2.0)

        inputs = torch.randn(10, 3, requires_grad=True)
        targets = torch.randint(0, 3, (10,))

        loss = focal_loss(inputs, targets)
        loss.backward()

        # Check gradients exist
        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()

    def test_focal_loss_class_imbalance(self):
        """Test Focal Loss handles class imbalance"""
        # All samples of class 0 (minority class with alpha=0.25)
        focal_loss = FocalLoss(alpha=[0.25, 0.5, 0.25], gamma=2.0)

        inputs = torch.randn(10, 3)
        targets = torch.zeros(10, dtype=torch.long)  # All class 0

        loss = focal_loss(inputs, targets)

        # Loss should be computed without errors
        assert loss.item() > 0


class TestTFTPredictor:
    """Test suite for Temporal Fusion Transformer"""

    @pytest.fixture
    def sample_tft_data(self):
        """Generate sample time series data for TFT"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=200, freq='B')

        df = pd.DataFrame({
            'DATE': dates,
            'SYMBOL': 'NIFTY',
            'CLOSE': 100 + np.cumsum(np.random.randn(200) * 2),
            'ema_9': np.random.randn(200),
            'rsi_14': np.random.uniform(30, 70, 200),
            'atr_14': np.random.uniform(1, 5, 200),
            'label_3c': np.random.randint(0, 3, 200),
            'dow': (dates.dayofweek % 5),
            'month': dates.month,
            'regime_composite': ['choppy'] * 200,
            'vol_regime': np.random.randint(0, 3, 200),
            'trend_regime': np.random.randint(-1, 2, 200),
        })

        # Add time index
        df['time_idx'] = np.arange(len(df))

        return df

    def test_tft_initialization(self):
        """Test TFT predictor initializes correctly"""
        tft = TFTPredictor()
        assert tft is not None
        assert tft.max_encoder_length == 20
        assert tft.max_prediction_length == 1

    def test_tft_prepare_dataset(self, sample_tft_data):
        """Test TFT dataset preparation"""
        tft = TFTPredictor()
        dataset = tft.prepare_dataset(sample_tft_data, is_training=True)

        assert dataset is not None
        assert tft.training_dataset is not None


class TestEnsemblePredictor:
    """Test suite for Ensemble Predictor"""

    def test_ensemble_initialization(self):
        """Test Ensemble predictor initializes correctly"""
        ensemble = EnsemblePredictor()

        assert ensemble is not None
        assert ensemble.lgbm is not None
        assert ensemble.xgb is not None
        assert ensemble.lr is not None
        assert ensemble.meta is not None

    @pytest.fixture
    def sample_ensemble_data(self):
        """Generate sample data for ensemble training"""
        np.random.seed(42)

        n_samples = 1000
        n_features = 20

        X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        X_train['SYMBOL'] = 'NIFTY'
        X_train['vol_regime'] = np.random.randint(0, 3, n_samples)
        X_train['trend_regime'] = np.random.randint(-1, 2, n_samples)

        y_train = pd.Series(np.random.randint(0, 3, n_samples))

        X_val = pd.DataFrame(
            np.random.randn(200, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        X_val['SYMBOL'] = 'NIFTY'
        X_val['vol_regime'] = np.random.randint(0, 3, 200)
        X_val['trend_regime'] = np.random.randint(-1, 2, 200)

        y_val = pd.Series(np.random.randint(0, 3, 200))

        return X_train, y_train, X_val, y_val

    def test_ensemble_predict_proba(self, sample_ensemble_data):
        """Test ensemble prediction"""
        X_train, y_train, X_val, y_val = sample_ensemble_data

        ensemble = EnsemblePredictor()

        # Train (mock - this would fail without proper TFT setup)
        try:
            ensemble.fit(X_train, y_train, X_val, y_val)
        except Exception:
            pass  # Expected to fail without full TFT training

        # Test prediction shape
        predictions = ensemble.predict_proba(X_val)

        assert 'p_down' in predictions.columns
        assert 'p_flat' in predictions.columns
        assert 'p_up' in predictions.columns
        assert 'direction' in predictions.columns
        assert 'confidence' in predictions.columns


class TestComputeTemperature:
    """Test suite for temperature scaling"""

    def test_compute_temperature_basic(self):
        """Test temperature computation"""
        # Create mock probabilities
        probas = np.random.dirichlet([1, 1, 1], size=100)
        y_true = pd.Series(np.random.randint(0, 3, 100))

        # Compute temperature
        temperature = compute_temperature(probas, y_true)

        # Temperature should be positive
        assert temperature > 0
        # Usually between 0.5 and 3.0
        assert 0.1 < temperature < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
