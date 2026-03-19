"""
backtester.py
=============
Production-grade backtesting engine for F&O strategies.

Features:
  - Liquidity filtering
  - Fractional Kelly position sizing
  - Volatility-targeted sizing
  - Stop-loss and circuit breaker logic
  - Comprehensive performance metrics
  - Monte Carlo robustness analysis
"""

import logging
import datetime as dt
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

import config

logger = logging.getLogger("Backtester")

class Backtester:
    """
    Production-grade backtester for F&O prediction strategies.
    """

    def __init__(self, capital: float = config.CAPITAL, cfg: Any = config):
        self.capital = capital
        self.initial_capital = capital
        self.config = cfg
        
        # Load constraints
        self.brokerage_per_leg = getattr(cfg, "BROKERAGE_PER_LEG", 20)
        self.slippage_pct = getattr(cfg, "SLIPPAGE_PCT", 0.0005)
        self.stt_pct = getattr(cfg, "STT_PCT", 0.0001)
        self.min_oi_lots = getattr(cfg, "MIN_OI_LOTS", 500)
        self.min_volume = getattr(cfg, "MIN_DAILY_VOLUME", 200)
        self.max_spread_pct = getattr(cfg, "MAX_SPREAD_PCT", 0.03)
        self.min_dte = getattr(cfg, "MIN_DTE", 2)
        self.risk_free_rate = getattr(cfg, "RISK_FREE_RATE", 0.065)
        self.atr_stop_mult = getattr(cfg, "ATR_STOP_MULT", 1.5)
        self.max_position_pct = getattr(cfg, "MAX_POSITION_PCT", 0.20)
        self.circuit_breaker_dd = getattr(cfg, "DRAWDOWN_CIRCUIT_BREAKER", 0.05)
        
        self.trade_log = []
        self.circuit_breaker_active = False

    def liquidity_filter(self, row: pd.Series) -> Tuple[bool, str]:
        """
        Check if an instrument passes liquidity and confidence constraints.
        """
        if row.get("OPEN_INT", 0) < self.min_oi_lots:
            return False, "low_OI"
        if row.get("CONTRACTS", 0) < self.min_volume:
            return False, "low_volume"
        
        if "HIGH" in row and "LOW" in row and "CLOSE" in row:
            spread = (row["HIGH"] - row["LOW"]) / row["CLOSE"]
            if spread > self.max_spread_pct:
                return False, "wide_spread"
        
        if "DTE" in row and row["DTE"] < self.min_dte:
            return False, "near_expiry"
            
        if row.get("confidence", 0) < 0.55:
            return False, "low_confidence"
            
        return True, "pass"

    def fractional_kelly_size(self, win_prob: float, avg_win: float, avg_loss: float,
                          capital: float, kelly_fraction: float = 0.25) -> float:
        """
        Compute position size using Fractional Kelly Criterion.
        """
        if avg_loss == 0: return 0.0
        b = abs(avg_win / avg_loss)
        if b == 0: return 0.0
        
        f = (win_prob * b - (1 - win_prob)) / b
        f = max(0, f) # No shorting with negative Kelly
        
        size = f * kelly_fraction * capital
        # Cap at MAX_POSITION_PCT
        size = min(size, self.max_position_pct * capital)
        return size

    def volatility_targeted_size(self, base_size: float, atr: float, close: float,
                             vix: float, target_vol: float = 0.15) -> float:
        """
        Adjust position size based on asset volatility and market fear (VIX).
        """
        if close == 0: return 0.0
        current_vol = (atr / close) * np.sqrt(252)
        if current_vol == 0: return base_size
        
        scaling_factor = target_vol / current_vol
        size = base_size * scaling_factor
        
        # Penalize in high VIX regimes
        if vix > 25:
            size *= 0.5
            
        return min(size, self.max_position_pct * self.capital)

    def run_backtest(self, predictions_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute sequential backtest.
        """
        logger.info("Starting backtest...")
        predictions_df = predictions_df.sort_values("DATE")
        dates = predictions_df["DATE"].unique()
        
        current_capital = self.initial_capital
        equity_curve = []
        
        for date in dates:
            day_preds = predictions_df[predictions_df["DATE"] == date]
            
            # Circuit breaker check (rolling 10d PnL)
            if len(self.trade_log) >= 1:
                recent_pnl = sum([t['net_pnl'] for t in self.trade_log[-10:]])
                if recent_pnl < -self.circuit_breaker_dd * self.initial_capital:
                    logger.warning(f"CIRCUIT BREAKER triggered on {date}")
                    self.circuit_breaker_active = True
                    continue
            
            for _, row in day_preds.iterrows():
                passes, reason = self.liquidity_filter(row)
                if not passes:
                    continue
                
                symbol = row["SYMBOL"]
                # Fetch next-day price data for execution
                # Entry: next-day open + slippage
                # Exit: next-day close or stop-loss
                entry_price = row.get("NEXT_OPEN", row["CLOSE"]) * (1 + self.slippage_pct)
                stop_loss = entry_price - self.atr_stop_mult * row.get("atr_14", 0)
                
                # Sizing
                base_size = self.fractional_kelly_size(row["confidence"], 0.02, 0.01, current_capital)
                final_size = self.volatility_targeted_size(base_size, row.get("atr_14", 0), row["CLOSE"], row.get("VIX_CLOSE", 20))
                
                lots = int(final_size / (row["CLOSE"] * 50)) # Mock lot size = 50
                if lots == 0: continue
                
                exit_price = row.get("NEXT_CLOSE", row["CLOSE"])
                exit_reason = "EOD"
                
                if row.get("NEXT_LOW", exit_price) <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "STOP_LOSS"
                
                gross_pnl = (exit_price - entry_price) * lots * 50
                net_pnl = gross_pnl - (2 * self.brokerage_per_leg) - (abs(gross_pnl) * self.stt_pct)
                
                current_capital += net_pnl
                
                self.trade_log.append({
                    "date": date,
                    "symbol": symbol,
                    "direction": row["direction"],
                    "entry": entry_price,
                    "exit": exit_price,
                    "lots": lots,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "exit_reason": exit_reason,
                    "position_size_method": "kelly_vol_targeted"
                })
                
        return pd.DataFrame(self.trade_log)

    def compute_metrics(self, trade_log: pd.DataFrame) -> dict:
        """
        Compute performance metrics from trade log.
        """
        if trade_log.empty: return {}
        
        net_pnls = trade_log["net_pnl"]
        total_net_pnl = net_pnls.sum()
        total_return_pct = (total_net_pnl / self.initial_capital) * 100
        
        # Approximation for annualized metrics
        days = (pd.to_datetime(trade_log["date"].max()) - pd.to_datetime(trade_log["date"].min())).days
        years = max(days / 365, 0.1)
        cagr = ((1 + total_return_pct/100) ** (1/years) - 1) * 100
        
        # Sharpe
        daily_returns = net_pnls / self.initial_capital
        sharpe = (daily_returns.mean() * 252 - self.risk_free_rate) / (daily_returns.std() * np.sqrt(252))
        
        # Drawdown
        cum_pnl = net_pnls.cumsum() + self.initial_capital
        running_max = cum_pnl.cummax()
        drawdowns = (cum_pnl - running_max) / running_max
        max_dd = drawdowns.min() * 100
        
        return {
            "total_return_pct": total_return_pct,
            "cagr": cagr,
            "annualized_sharpe": sharpe,
            "max_drawdown_pct": max_dd,
            "calmar_ratio": cagr / abs(max_dd) if max_dd != 0 else 0,
            "win_rate": (net_pnls > 0).mean() * 100,
            "total_trades": len(trade_log),
            "profit_factor": abs(net_pnls[net_pnls > 0].sum() / net_pnls[net_pnls < 0].sum()) if any(net_pnls < 0) else np.inf
        }

    def monte_carlo_analysis(self, trade_log: pd.DataFrame, n_simulations: int = 1000) -> dict:
        """
        Bootstrap simulation to estimate range of outcomes.
        """
        if trade_log.empty: return {}
        pnls = trade_log["net_pnl"].values
        sim_results = []
        
        for _ in range(n_simulations):
            sampled_pnls = np.random.choice(pnls, size=len(pnls), replace=True)
            total_ret = np.sum(sampled_pnls) / self.initial_capital
            sim_results.append(total_ret)
            
        return {
            "median_return": np.median(sim_results),
            "p5_return": np.percentile(sim_results, 5),
            "p95_return": np.percentile(sim_results, 95),
            "prob_positive": (np.array(sim_results) > 0).mean()
        }

    def plot_results(self, trade_log: pd.DataFrame):
        """
        Generate 4-panel performance plot.
        """
        if trade_log.empty: return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Panel 1: Cumulative P&L
        trade_log["cum_pnl"] = trade_log["net_pnl"].cumsum()
        axes[0, 0].plot(trade_log["date"], trade_log["cum_pnl"])
        axes[0, 0].set_title("Cumulative Net P&L")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Panel 2: P&L Distribution
        sns.histplot(trade_log["net_pnl"], kde=True, ax=axes[0, 1])
        axes[0, 1].set_title("Trade P&L Distribution")
        
        # Panel 3: Drawdown
        cum_pnl = trade_log["net_pnl"].cumsum() + self.initial_capital
        running_max = cum_pnl.cummax()
        dd = (cum_pnl - running_max) / running_max
        axes[1, 0].fill_between(trade_log["date"], dd, color='red', alpha=0.3)
        axes[1, 0].set_title("Drawdown Curve")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Panel 4: Win Rate by Direction
        win_by_dir = trade_log.groupby("direction").apply(lambda x: (x["net_pnl"] > 0).mean())
        win_by_dir.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title("Win Rate by Predicted Direction")
        
        plt.tight_layout()
        output_path = config.OUTPUT_DIR / f"backtest_results_{dt.datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(output_path)
        logger.info(f"Backtest plots saved to {output_path}")
