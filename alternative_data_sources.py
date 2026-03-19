"""
alternative_data_sources.py
============================
Alternative data sources to enhance F&O prediction accuracy.

New Data Sources:
  1. News Sentiment Analysis (from news APIs)
  2. Order Book Depth & Microstructure
  3. Volatility Surface & Implied Volatility Term Structure
  4. Social Media Sentiment (Twitter/Reddit for market sentiment)
  5. Mutual Fund Flows (AMFI data)
  6. Earnings Calendar & Corporate Actions
  7. Weather Data (for commodity-sensitive sectors)
  8. Economic Calendar Events
  9. Block & Bulk Deal Activity
  10. Options Greeks Time Series (Delta, Gamma, Vega, Theta)
"""

import logging
import warnings
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")
logger = logging.getLogger("AlternativeData")


class AlternativeDataCollector:
    """
    Collects alternative data sources to enhance prediction accuracy.
    """

    def __init__(self):
        self.cache = {}
        logger.info("AlternativeDataCollector initialized")

    # ─────────────────────────────────────────────────────────────────────
    # 1. NEWS SENTIMENT ANALYSIS
    # ─────────────────────────────────────────────────────────────────────
    def get_news_sentiment(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch news headlines and compute sentiment scores.

        Data Sources:
          - NewsAPI (requires API key)
          - MoneyControl RSS feeds
          - Economic Times RSS
          - NSE announcements

        Returns:
            DataFrame with columns:
              - DATE
              - sentiment_score (-1 to +1)
              - news_count
              - headline_polarity_avg
              - headline_subjectivity_avg
        """
        logger.info(f"Fetching news sentiment for {symbol} from {start_date} to {end_date}")

        # Placeholder for actual implementation
        # In production, integrate with:
        # - NewsAPI: https://newsapi.org/
        # - FinBERT for financial sentiment: https://huggingface.co/ProsusAI/finbert

        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        sentiment_data = []

        for date in dates:
            # Mock sentiment data (replace with actual API calls)
            sentiment_data.append({
                'DATE': date,
                'sentiment_score': np.random.uniform(-0.5, 0.5),
                'news_count': np.random.randint(5, 50),
                'headline_polarity_avg': np.random.uniform(-0.3, 0.3),
                'headline_subjectivity_avg': np.random.uniform(0.3, 0.7),
                'bullish_news_count': np.random.randint(0, 20),
                'bearish_news_count': np.random.randint(0, 20),
            })

        df = pd.DataFrame(sentiment_data)
        logger.info(f"News sentiment data: {len(df)} days")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # 2. ORDER BOOK DEPTH & MICROSTRUCTURE
    # ─────────────────────────────────────────────────────────────────────
    def get_order_book_metrics(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch order book depth and microstructure metrics.

        Features:
          - Bid-ask spread
          - Order book imbalance (bid volume / ask volume)
          - Depth imbalance (5-level weighted)
          - Large order flow (block trades)
          - Price impact of trades

        Data Source:
          - NSE Market Depth API (requires authentication)
          - Historical order book snapshots
        """
        logger.info(f"Fetching order book metrics for {symbol}")

        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        order_book_data = []

        for date in dates:
            # Mock order book data
            bid_ask_spread = np.random.uniform(0.0001, 0.005)
            order_book_data.append({
                'DATE': date,
                'bid_ask_spread': bid_ask_spread,
                'bid_ask_spread_pct': bid_ask_spread * 100,
                'order_book_imbalance': np.random.uniform(0.8, 1.2),
                'depth_imbalance_5level': np.random.uniform(0.9, 1.1),
                'large_buy_orders': np.random.randint(0, 10),
                'large_sell_orders': np.random.randint(0, 10),
                'avg_trade_size': np.random.uniform(10000, 100000),
                'order_flow_toxicity': np.random.uniform(0, 0.3),
            })

        df = pd.DataFrame(order_book_data)
        logger.info(f"Order book data: {len(df)} days")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # 3. VOLATILITY SURFACE & IV TERM STRUCTURE
    # ─────────────────────────────────────────────────────────────────────
    def get_volatility_surface(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Construct volatility surface from option chain data.

        Features:
          - IV skew (ATM vs OTM)
          - IV term structure (near vs far month)
          - Volatility smile asymmetry
          - IV rank percentile
          - IV term structure slope

        Data Source:
          - NSE option chain (nsefin/nsepython)
          - Historical IV data
        """
        logger.info(f"Fetching volatility surface for {symbol}")

        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        vol_surface_data = []

        for date in dates:
            # Mock volatility surface metrics
            vol_surface_data.append({
                'DATE': date,
                'atm_iv': np.random.uniform(15, 35),
                'otm_call_iv_5pct': np.random.uniform(16, 38),
                'otm_put_iv_5pct': np.random.uniform(16, 38),
                'iv_skew': np.random.uniform(-5, 5),
                'near_month_iv': np.random.uniform(15, 40),
                'far_month_iv': np.random.uniform(14, 35),
                'iv_term_slope': np.random.uniform(-2, 2),
                'vol_smile_asymmetry': np.random.uniform(-0.1, 0.1),
                'iv_rank_252d': np.random.uniform(0.2, 0.8),
            })

        df = pd.DataFrame(vol_surface_data)
        logger.info(f"Volatility surface data: {len(df)} days")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # 4. MUTUAL FUND FLOWS (AMFI DATA)
    # ─────────────────────────────────────────────────────────────────────
    def get_mutual_fund_flows(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch mutual fund flow data from AMFI.

        Features:
          - Equity fund net flows
          - Debt fund net flows
          - Sectoral fund flows (IT, Banking, etc.)
          - SIP flows (systematic investment plan)

        Data Source:
          - AMFI website: https://www.amfiindia.com/
        """
        logger.info(f"Fetching mutual fund flows from {start_date} to {end_date}")

        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        mf_data = []

        for date in dates:
            # Mock mutual fund flow data
            mf_data.append({
                'DATE': date,
                'equity_fund_net_flow_cr': np.random.uniform(-5000, 15000),
                'debt_fund_net_flow_cr': np.random.uniform(-2000, 10000),
                'sip_flow_cr': np.random.uniform(5000, 15000),
                'bank_fund_flow_cr': np.random.uniform(-500, 2000),
                'it_fund_flow_cr': np.random.uniform(-300, 1500),
            })

        df = pd.DataFrame(mf_data)
        logger.info(f"Mutual fund flow data: {len(df)} months")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # 5. EARNINGS CALENDAR & CORPORATE ACTIONS
    # ─────────────────────────────────────────────────────────────────────
    def get_corporate_events(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch upcoming earnings announcements and corporate actions.

        Features:
          - Days until earnings
          - Dividend announcement dates
          - Stock splits
          - Bonus issues
          - Rights issues

        Data Source:
          - NSE corporate actions: https://www.nseindia.com/
          - MoneyControl earnings calendar
        """
        logger.info(f"Fetching corporate events from {start_date} to {end_date}")

        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        events_data = []

        for date in dates:
            # Mock corporate events
            has_earnings = np.random.random() < 0.05  # 5% chance
            has_dividend = np.random.random() < 0.02  # 2% chance

            events_data.append({
                'DATE': date,
                'earnings_in_5d': 1 if has_earnings else 0,
                'earnings_in_10d': 1 if np.random.random() < 0.1 else 0,
                'dividend_upcoming': 1 if has_dividend else 0,
                'major_nifty_earnings_today': np.random.randint(0, 5),
                'major_banknifty_earnings_today': np.random.randint(0, 3),
            })

        df = pd.DataFrame(events_data)
        logger.info(f"Corporate events data: {len(df)} days")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # 6. BLOCK & BULK DEAL ACTIVITY
    # ─────────────────────────────────────────────────────────────────────
    def get_block_bulk_deals(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch block and bulk deal activity.

        Features:
          - Daily block deal count
          - Daily bulk deal count
          - Net block deal value
          - Net bulk deal value
          - Insider buying/selling patterns

        Data Source:
          - NSE Block Deals: https://www.nseindia.com/report-detail/eq_block
          - NSE Bulk Deals: https://www.nseindia.com/report-detail/eq_bulk
        """
        logger.info(f"Fetching block/bulk deals from {start_date} to {end_date}")

        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        deals_data = []

        for date in dates:
            # Mock block/bulk deal data
            deals_data.append({
                'DATE': date,
                'block_deal_count': np.random.randint(0, 20),
                'bulk_deal_count': np.random.randint(5, 50),
                'net_block_value_cr': np.random.uniform(-500, 1000),
                'net_bulk_value_cr': np.random.uniform(-200, 500),
                'insider_buying_count': np.random.randint(0, 10),
                'insider_selling_count': np.random.randint(0, 10),
            })

        df = pd.DataFrame(deals_data)
        logger.info(f"Block/bulk deals data: {len(df)} days")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # 7. OPTIONS GREEKS TIME SERIES
    # ─────────────────────────────────────────────────────────────────────
    def get_options_greeks(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Calculate and fetch options Greeks time series.

        Features:
          - Total portfolio delta
          - Total portfolio gamma (GEX)
          - Vega exposure
          - Theta decay
          - Gamma squeeze zones

        Data Source:
          - Calculated from option chain data
          - nsefin for option chain
        """
        logger.info(f"Fetching options Greeks for {symbol}")

        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        greeks_data = []

        for date in dates:
            # Mock Greeks data
            greeks_data.append({
                'DATE': date,
                'total_call_delta': np.random.uniform(0.3, 0.7),
                'total_put_delta': np.random.uniform(-0.7, -0.3),
                'net_delta': np.random.uniform(-0.2, 0.2),
                'total_gamma': np.random.uniform(0, 0.1),
                'gex_level': np.random.uniform(-1e9, 1e9),
                'total_vega': np.random.uniform(0, 1000000),
                'total_theta': np.random.uniform(-100000, 0),
                'gamma_squeeze_risk': np.random.uniform(0, 1),
            })

        df = pd.DataFrame(greeks_data)
        logger.info(f"Options Greeks data: {len(df)} days")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # 8. ECONOMIC CALENDAR EVENTS
    # ─────────────────────────────────────────────────────────────────────
    def get_economic_calendar(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch scheduled economic events.

        Features:
          - RBI policy meeting dates
          - GDP release dates
          - CPI/WPI inflation dates
          - IIP (Industrial Production) dates
          - US Fed meetings

        Data Source:
          - Investing.com Economic Calendar
          - RBI website
        """
        logger.info(f"Fetching economic calendar from {start_date} to {end_date}")

        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        calendar_data = []

        for date in dates:
            # Mock economic calendar
            has_rbi = np.random.random() < 0.02  # ~6 meetings per year
            has_fed = np.random.random() < 0.03  # ~8 meetings per year

            calendar_data.append({
                'DATE': date,
                'rbi_policy_today': 1 if has_rbi else 0,
                'rbi_policy_in_5d': 1 if np.random.random() < 0.05 else 0,
                'us_fed_meeting_today': 1 if has_fed else 0,
                'us_fed_meeting_in_5d': 1 if np.random.random() < 0.08 else 0,
                'india_gdp_release': 1 if np.random.random() < 0.01 else 0,
                'india_cpi_release': 1 if np.random.random() < 0.04 else 0,
                'us_nonfarm_payrolls': 1 if np.random.random() < 0.04 else 0,
            })

        df = pd.DataFrame(calendar_data)
        logger.info(f"Economic calendar data: {len(df)} days")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # MASTER FUNCTION TO COLLECT ALL ALTERNATIVE DATA
    # ─────────────────────────────────────────────────────────────────────
    def collect_all_alternative_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Collect all alternative data sources and merge into single DataFrame.

        Returns:
            Merged DataFrame with all alternative features
        """
        logger.info("=" * 80)
        logger.info(f"Collecting ALL alternative data for {symbol}")
        logger.info("=" * 80)

        # Collect all data sources
        news_df = self.get_news_sentiment(symbol, start_date, end_date)
        orderbook_df = self.get_order_book_metrics(symbol, start_date, end_date)
        vol_surface_df = self.get_volatility_surface(symbol, start_date, end_date)
        mf_df = self.get_mutual_fund_flows(start_date, end_date)
        corporate_df = self.get_corporate_events(start_date, end_date)
        deals_df = self.get_block_bulk_deals(start_date, end_date)
        greeks_df = self.get_options_greeks(symbol, start_date, end_date)
        calendar_df = self.get_economic_calendar(start_date, end_date)

        # Merge all dataframes on DATE
        merged_df = news_df
        for df in [orderbook_df, vol_surface_df, corporate_df, deals_df, greeks_df, calendar_df]:
            merged_df = pd.merge(merged_df, df, on='DATE', how='left')

        # Mutual fund data is monthly, so forward-fill
        merged_df = pd.merge(merged_df, mf_df, on='DATE', how='left')
        merged_df = merged_df.sort_values('DATE').fillna(method='ffill')

        logger.info(f"✓ Alternative data collected: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        return merged_df


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOCK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    print("\n=== ALTERNATIVE DATA SOURCES TEST ===\n")

    collector = AlternativeDataCollector()

    # Test collection
    alt_data = collector.collect_all_alternative_data(
        symbol="NIFTY",
        start_date="2024-01-01",
        end_date="2024-03-31"
    )

    print(f"\n✓ Collected {len(alt_data)} days of alternative data")
    print(f"✓ Total features: {len(alt_data.columns)}")
    print(f"\nSample columns:")
    for col in alt_data.columns[:20]:
        print(f"  - {col}")

    print("\n=== TEST COMPLETE ===")
