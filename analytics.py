"""
analytics.py — Project-wide analytics and scoring logic.
"""

import numpy as np

def black_scholes_greeks(S, K, T, r, sigma, opt_type="CE"):
    """Placeholder for Black-Scholes greeks."""
    return {"delta": 0.5, "gamma": 0.01, "theta": -0.05, "vega": 0.1}

def compute_stock_score_v2(data_row):
    """Placeholder for stock scoring logic."""
    return 50

def nearest_atm(spot, symbol):
    """Find nearest ATM strike."""
    if "NIFTY" in symbol:
        return round(spot / 50) * 50
    if "BANKNIFTY" in symbol:
        return round(spot / 100) * 100
    return round(spot / 5) * 5
