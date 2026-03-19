"""
test_feature_engineering.py
============================
Unit tests for feature engineering module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/home/runner/work/btst/btst')

from feature_engineering import FeatureEngineer, compute_features


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data for testing"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=300, freq='B')

        # Generate realistic price data
        close_prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.02))

        df = pd.DataFrame({
            'DATE': dates,
            'SYMBOL': 'NIFTY',
            'OPEN': close_prices * (1 + np.random.randn(300) * 0.005),
            'HIGH': close_prices * (1 + np.abs(np.random.randn(300) * 0.01)),
            'LOW': close_prices * (1 - np.abs(np.random.randn(300) * 0.01)),
            'CLOSE': close_prices,
            'CONTRACTS': np.random.randint(1000, 10000, 300),
            'OPEN_INT': np.random.randint(50000, 200000, 300),
            'CHG_IN_OI': np.random.randint(-10000, 10000, 300),
        })

        return df

    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer initializes correctly"""
        engineer = FeatureEngineer()
        assert engineer is not None

    def test_add_price_trend_features(self, sample_data):
        """Test price and trend feature generation"""
        engineer = FeatureEngineer()
        result = engineer._add_price_trend_features(sample_data.copy())

        # Check EMAs are added
        assert 'ema_9' in result.columns
        assert 'ema_21' in result.columns
        assert 'ema_50' in result.columns
        assert 'ema_200' in result.columns

        # Check MACD is added
        assert 'macd_line' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_histogram' in result.columns

        # Check crossovers
        assert 'ema_cross_9_21' in result.columns
        assert result['ema_cross_9_21'].isin([0, 1]).all()

    def test_add_momentum_features(self, sample_data):
        """Test momentum indicator generation"""
        engineer = FeatureEngineer()
        result = engineer._add_momentum_features(sample_data.copy())

        # Check RSI
        assert 'rsi_14' in result.columns
        assert 'rsi_7' in result.columns
        assert result['rsi_14'].between(0, 100).all()

        # Check ROC
        assert 'roc_5' in result.columns
        assert 'roc_10' in result.columns
        assert 'roc_20' in result.columns

    def test_add_volatility_features(self, sample_data):
        """Test volatility indicator generation"""
        engineer = FeatureEngineer()
        result = engineer._add_volatility_features(sample_data.copy())

        # Check ATR
        assert 'atr_14' in result.columns
        assert 'natr_14' in result.columns
        assert (result['atr_14'] > 0).all()

        # Check Bollinger Bands
        assert 'bb_upper' in result.columns
        assert 'bb_lower' in result.columns
        assert 'bb_width' in result.columns
        assert (result['bb_upper'] > result['bb_lower']).all()

    def test_add_volume_features(self, sample_data):
        """Test volume indicator generation"""
        engineer = FeatureEngineer()
        result = engineer._add_volume_features(sample_data.copy())

        # Check OBV
        assert 'obv' in result.columns
        assert 'obv_ema_20' in result.columns

        # Check CMF
        assert 'cmf_20' in result.columns

        # Check RVOL
        assert 'rvol_20' in result.columns

    def test_add_fno_features(self, sample_data):
        """Test F&O specific feature generation"""
        engineer = FeatureEngineer()
        result = engineer._add_fno_features(sample_data.copy())

        # Check OI features
        assert 'oi_chg_pct' in result.columns
        assert 'oi_buildup' in result.columns
        assert 'oi_unwinding' in result.columns
        assert 'short_buildup' in result.columns
        assert 'short_cover' in result.columns

    def test_add_velocity_acceleration_features(self, sample_data):
        """Test velocity and acceleration feature generation"""
        engineer = FeatureEngineer()
        result = engineer._add_velocity_acceleration_features(sample_data.copy())

        # Check velocity features
        assert 'velocity_1d' in result.columns
        assert 'velocity_3d' in result.columns
        assert 'velocity_5d' in result.columns
        assert 'velocity_10d' in result.columns

        # Check acceleration features
        assert 'accel_3d' in result.columns
        assert 'accel_5d' in result.columns
        assert 'accel_10d' in result.columns

        # Check jerk
        assert 'jerk_5d' in result.columns

        # Check velocity reversal
        assert 'velocity_reversal' in result.columns
        assert result['velocity_reversal'].isin([-1, 0, 1]).all()

    def test_add_target(self, sample_data):
        """Test target variable generation"""
        engineer = FeatureEngineer()
        result = engineer._add_target(sample_data.copy())

        # Check target columns exist
        assert 'next_day_return' in result.columns
        assert 'next_day_return_pct' in result.columns
        assert 'label' in result.columns
        assert 'label_3c' in result.columns

        # Check label values are correct
        assert result['label'].isin([-1, 0, 1]).all()
        assert result['label_3c'].isin([0, 1, 2]).all()

        # Check label logic
        positive_mask = result['next_day_return_pct'] > 0.5
        assert (result.loc[positive_mask, 'label'] == 1).all()

        negative_mask = result['next_day_return_pct'] < -0.5
        assert (result.loc[negative_mask, 'label'] == -1).all()

    def test_compute_all(self, sample_data):
        """Test full feature computation pipeline"""
        engineer = FeatureEngineer()
        result = engineer.compute_all(sample_data.copy())

        # Check result is not empty
        assert len(result) > 0
        assert len(result) < len(sample_data)  # Some rows dropped

        # Check basic columns still exist
        assert 'DATE' in result.columns
        assert 'SYMBOL' in result.columns
        assert 'CLOSE' in result.columns

        # Check features were added
        feature_names = engineer.get_feature_names(result)
        assert len(feature_names) > 50  # Should have 75+ features

    def test_rsi_calculation(self, sample_data):
        """Test RSI calculation accuracy"""
        engineer = FeatureEngineer()
        close = sample_data['CLOSE']
        rsi = engineer._rsi(close, 14)

        # Check RSI is in valid range
        valid_rsi = rsi.dropna()
        assert valid_rsi.between(0, 100).all()

        # Check RSI is not constant
        assert valid_rsi.std() > 0

    def test_get_feature_names(self, sample_data):
        """Test feature name extraction"""
        engineer = FeatureEngineer()
        result = engineer.compute_all(sample_data.copy())

        feature_names = engineer.get_feature_names(result)

        # Check metadata columns are excluded
        assert 'DATE' not in feature_names
        assert 'SYMBOL' not in feature_names
        assert 'CLOSE' not in feature_names

        # Check target columns are excluded
        assert 'label' not in feature_names
        assert 'label_3c' not in feature_names
        assert 'next_day_return' not in feature_names

    def test_no_nan_in_features(self, sample_data):
        """Test that features don't have NaN values after cleanup"""
        engineer = FeatureEngineer()
        result = engineer.compute_all(sample_data.copy())

        feature_names = engineer.get_feature_names(result)

        # Check for NaN values
        nan_counts = result[feature_names].isna().sum()
        assert nan_counts.sum() == 0, f"Found NaN values in features: {nan_counts[nan_counts > 0]}"

    def test_feature_dtypes(self, sample_data):
        """Test that features have correct data types"""
        engineer = FeatureEngineer()
        result = engineer.compute_all(sample_data.copy())

        feature_names = engineer.get_feature_names(result)

        # All features should be numeric
        for col in feature_names:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} is not numeric"


class TestComputeFeatures:
    """Test suite for compute_features function"""

    @pytest.fixture
    def multi_symbol_data(self):
        """Generate data for multiple symbols"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=300, freq='B')

        dfs = []
        for symbol in ['NIFTY', 'BANKNIFTY']:
            close_prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.02))

            df = pd.DataFrame({
                'DATE': dates,
                'SYMBOL': symbol,
                'OPEN': close_prices * (1 + np.random.randn(300) * 0.005),
                'HIGH': close_prices * (1 + np.abs(np.random.randn(300) * 0.01)),
                'LOW': close_prices * (1 - np.abs(np.random.randn(300) * 0.01)),
                'CLOSE': close_prices,
                'CONTRACTS': np.random.randint(1000, 10000, 300),
                'OPEN_INT': np.random.randint(50000, 200000, 300),
                'CHG_IN_OI': np.random.randint(-10000, 10000, 300),
            })
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def test_compute_features_multi_symbol(self, multi_symbol_data):
        """Test feature computation for multiple symbols"""
        result, feature_names = compute_features(multi_symbol_data)

        # Check both symbols are present
        assert 'NIFTY' in result['SYMBOL'].values
        assert 'BANKNIFTY' in result['SYMBOL'].values

        # Check features were computed
        assert len(feature_names) > 50

        # Check no NaN values
        nan_counts = result[feature_names].isna().sum()
        assert nan_counts.sum() == 0

    def test_compute_features_empty_input(self):
        """Test handling of empty input"""
        empty_df = pd.DataFrame()
        result, feature_names = compute_features(empty_df)

        assert len(result) == 0
        assert len(feature_names) == 0


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformance:
    """Performance benchmarks for feature engineering"""

    def test_feature_computation_speed(self, benchmark):
        """Benchmark feature computation speed"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='B')
        close_prices = 100 * np.exp(np.cumsum(np.random.randn(1000) * 0.02))

        df = pd.DataFrame({
            'DATE': dates,
            'SYMBOL': 'NIFTY',
            'OPEN': close_prices * (1 + np.random.randn(1000) * 0.005),
            'HIGH': close_prices * (1 + np.abs(np.random.randn(1000) * 0.01)),
            'LOW': close_prices * (1 - np.abs(np.random.randn(1000) * 0.01)),
            'CLOSE': close_prices,
            'CONTRACTS': np.random.randint(1000, 10000, 1000),
            'OPEN_INT': np.random.randint(50000, 200000, 1000),
            'CHG_IN_OI': np.random.randint(-10000, 10000, 1000),
        })

        engineer = FeatureEngineer()

        # Benchmark
        result = benchmark(engineer.compute_all, df)

        # Should complete in reasonable time
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
