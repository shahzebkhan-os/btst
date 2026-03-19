"""
test_alternative_data.py
=========================
Unit tests for alternative data sources module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
sys.path.insert(0, '/home/runner/work/btst/btst')

from alternative_data_sources import AlternativeDataCollector


class TestAlternativeDataCollector:
    """Test suite for AlternativeDataCollector"""

    @pytest.fixture
    def collector(self):
        """Create collector instance"""
        return AlternativeDataCollector()

    @pytest.fixture
    def date_range(self):
        """Standard date range for testing"""
        return ("2024-01-01", "2024-03-31")

    def test_initialization(self, collector):
        """Test collector initializes correctly"""
        assert collector is not None
        assert hasattr(collector, 'cache')

    def test_get_news_sentiment(self, collector, date_range):
        """Test news sentiment data collection"""
        result = collector.get_news_sentiment("NIFTY", *date_range)

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert 'DATE' in result.columns
        assert 'sentiment_score' in result.columns
        assert 'news_count' in result.columns

        # Check data validity
        assert len(result) > 0
        assert result['sentiment_score'].between(-1, 1).all()
        assert (result['news_count'] >= 0).all()

    def test_get_order_book_metrics(self, collector, date_range):
        """Test order book metrics collection"""
        result = collector.get_order_book_metrics("NIFTY", *date_range)

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert 'DATE' in result.columns
        assert 'bid_ask_spread' in result.columns
        assert 'order_book_imbalance' in result.columns

        # Check data validity
        assert len(result) > 0
        assert (result['bid_ask_spread'] >= 0).all()

    def test_get_volatility_surface(self, collector, date_range):
        """Test volatility surface collection"""
        result = collector.get_volatility_surface("NIFTY", *date_range)

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert 'DATE' in result.columns
        assert 'atm_iv' in result.columns
        assert 'iv_skew' in result.columns
        assert 'iv_term_slope' in result.columns

        # Check data validity
        assert len(result) > 0
        assert (result['atm_iv'] > 0).all()

    def test_get_mutual_fund_flows(self, collector, date_range):
        """Test mutual fund flows collection"""
        result = collector.get_mutual_fund_flows(*date_range)

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert 'DATE' in result.columns
        assert 'equity_fund_net_flow_cr' in result.columns
        assert 'sip_flow_cr' in result.columns

        # Check data validity
        assert len(result) > 0

    def test_get_corporate_events(self, collector, date_range):
        """Test corporate events collection"""
        result = collector.get_corporate_events(*date_range)

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert 'DATE' in result.columns
        assert 'earnings_in_5d' in result.columns
        assert 'dividend_upcoming' in result.columns

        # Check data validity
        assert len(result) > 0
        assert result['earnings_in_5d'].isin([0, 1]).all()

    def test_get_block_bulk_deals(self, collector, date_range):
        """Test block/bulk deals collection"""
        result = collector.get_block_bulk_deals(*date_range)

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert 'DATE' in result.columns
        assert 'block_deal_count' in result.columns
        assert 'bulk_deal_count' in result.columns

        # Check data validity
        assert len(result) > 0
        assert (result['block_deal_count'] >= 0).all()

    def test_get_options_greeks(self, collector, date_range):
        """Test options Greeks collection"""
        result = collector.get_options_greeks("NIFTY", *date_range)

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert 'DATE' in result.columns
        assert 'total_call_delta' in result.columns
        assert 'gex_level' in result.columns

        # Check data validity
        assert len(result) > 0

    def test_get_economic_calendar(self, collector, date_range):
        """Test economic calendar collection"""
        result = collector.get_economic_calendar(*date_range)

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert 'DATE' in result.columns
        assert 'rbi_policy_today' in result.columns
        assert 'us_fed_meeting_today' in result.columns

        # Check data validity
        assert len(result) > 0
        assert result['rbi_policy_today'].isin([0, 1]).all()

    def test_collect_all_alternative_data(self, collector, date_range):
        """Test collection of all alternative data sources"""
        result = collector.collect_all_alternative_data("NIFTY", *date_range)

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Check all data sources are merged
        assert 'sentiment_score' in result.columns  # News
        assert 'bid_ask_spread' in result.columns   # Order book
        assert 'atm_iv' in result.columns            # Volatility surface
        assert 'earnings_in_5d' in result.columns    # Corporate events
        assert 'block_deal_count' in result.columns  # Block deals
        assert 'total_gamma' in result.columns       # Greeks
        assert 'rbi_policy_today' in result.columns  # Economic calendar

        # Check no excessive NaN values
        nan_pct = result.isna().sum() / len(result)
        assert (nan_pct < 0.5).all(), "Too many NaN values in alternative data"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
