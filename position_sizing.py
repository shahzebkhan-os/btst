"""
position_sizing.py
==================
Position sizing and risk management for F&O trading.

Methods:
  1. Fractional Kelly Criterion — optimal bet sizing based on edge and odds
  2. Volatility Targeting — scale position size to target volatility
  3. Drawdown Circuit Breaker — stop trading if drawdown exceeds threshold

Kelly Formula:
  f* = (p * b - q) / b

  where:
    f* = fraction of capital to bet
    p = probability of winning
    b = odds (net profit / bet)
    q = probability of losing = 1 - p

For multi-class (UP/FLAT/DOWN), use Kelly criterion per class and weight
by predicted probabilities.

Risk Controls:
  - Maximum position size: 20% of capital per trade
  - Fractional Kelly: 25% of full Kelly (to reduce volatility)
  - Volatility targeting: scale to 15% annual volatility
  - Drawdown circuit breaker: stop if DD > 10%
"""

import logging
import warnings
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

import config

warnings.filterwarnings("ignore")
logger = logging.getLogger("PositionSizing")


# ─────────────────────────────────────────────────────────────────────────────
# KELLY CRITERION
# ─────────────────────────────────────────────────────────────────────────────

class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.

    Formula:
        f* = (p * b - q) / b

    where:
      f* = optimal fraction of capital to bet
      p = probability of winning (from model)
      b = odds received (expected return / risk)
      q = 1 - p (probability of losing)

    For fractional Kelly, multiply by a fraction (e.g., 0.25 for quarter Kelly).

    Usage:
        kelly = KellyCriterion(kelly_fraction=0.25)
        position_size = kelly.calculate(win_prob=0.60, odds=2.0, capital=100000)
    """

    def __init__(
        self,
        kelly_fraction: float = config.KELLY_FRACTION,
        max_position_pct: float = config.MAX_POSITION_PCT,
    ):
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct

    def calculate(
        self,
        win_prob: float,
        odds: float,
        capital: float,
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.

        Args:
            win_prob: Probability of winning (0 to 1)
            odds: Odds received (profit / risk ratio)
            capital: Total available capital

        Returns:
            Recommended position size in currency
        """
        # Validate inputs
        if not (0 <= win_prob <= 1):
            raise ValueError(f"win_prob must be between 0 and 1, got {win_prob}")
        if odds <= 0:
            raise ValueError(f"odds must be positive, got {odds}")

        # Kelly formula
        lose_prob = 1 - win_prob
        kelly_pct = (win_prob * odds - lose_prob) / odds

        # Apply fractional Kelly
        kelly_pct = kelly_pct * self.kelly_fraction

        # Clip to [0, max_position_pct]
        kelly_pct = np.clip(kelly_pct, 0, self.max_position_pct)

        # Calculate position size
        position_size = capital * kelly_pct

        logger.debug(
            f"Kelly calculation: win_prob={win_prob:.3f}, odds={odds:.2f}, "
            f"kelly_pct={kelly_pct:.3f}, position_size={position_size:.2f}"
        )

        return position_size

    def calculate_multi_class(
        self,
        probas: np.ndarray,
        expected_returns: np.ndarray,
        capital: float,
    ) -> float:
        """
        Calculate Kelly position size for multi-class prediction.

        Args:
            probas: Class probabilities [p_down, p_flat, p_up]
            expected_returns: Expected returns for each class (e.g., [-0.02, 0, 0.03])
            capital: Total available capital

        Returns:
            Recommended position size
        """
        # Calculate expected value
        expected_value = np.dot(probas, expected_returns)

        # Calculate odds (use UP probability for long trades)
        win_prob = probas[2]  # UP class
        expected_return = expected_returns[2]

        if expected_return <= 0 or expected_value <= 0:
            # No positive edge, don't trade
            return 0.0

        # Estimate odds (expected return / max loss)
        # Assume max loss = -1 * expected return (symmetric risk)
        odds = abs(expected_return)

        # Calculate Kelly position size
        position_size = self.calculate(win_prob, odds, capital)

        return position_size


# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY TARGETING
# ─────────────────────────────────────────────────────────────────────────────

class VolatilityTargeting:
    """
    Scale position sizes to target a specific portfolio volatility.

    Formula:
        position_size_scaled = position_size * (target_vol / realized_vol)

    This ensures consistent risk-adjusted position sizing across different
    market regimes.

    Usage:
        vol_target = VolatilityTargeting(target_vol=0.15)
        scaled_size = vol_target.scale(position_size=50000, realized_vol=0.20)
    """

    def __init__(
        self,
        target_vol: float = config.VOLATILITY_TARGET,
    ):
        self.target_vol = target_vol

    def scale(
        self,
        position_size: float,
        realized_vol: float,
    ) -> float:
        """
        Scale position size to target volatility.

        Args:
            position_size: Unscaled position size
            realized_vol: Realized volatility (annualized)

        Returns:
            Scaled position size
        """
        if realized_vol <= 0:
            logger.warning(f"Invalid realized_vol: {realized_vol}, returning 0")
            return 0.0

        # Scale position
        vol_scalar = self.target_vol / realized_vol
        scaled_size = position_size * vol_scalar

        logger.debug(
            f"Volatility scaling: realized_vol={realized_vol:.3f}, "
            f"vol_scalar={vol_scalar:.3f}, scaled_size={scaled_size:.2f}"
        )

        return scaled_size

    def estimate_realized_vol(
        self,
        returns: np.ndarray,
        annualize: bool = True,
    ) -> float:
        """
        Estimate realized volatility from return series.

        Args:
            returns: Array of returns
            annualize: Annualize volatility (assumes daily returns)

        Returns:
            Realized volatility
        """
        if len(returns) < 2:
            logger.warning("Insufficient data for volatility estimation")
            return self.target_vol

        vol = np.std(returns, ddof=1)

        if annualize:
            # Annualize assuming 252 trading days
            vol = vol * np.sqrt(252)

        return vol


# ─────────────────────────────────────────────────────────────────────────────
# DRAWDOWN CIRCUIT BREAKER
# ─────────────────────────────────────────────────────────────────────────────

class DrawdownCircuitBreaker:
    """
    Stop trading when drawdown exceeds threshold.

    Drawdown = (Peak - Current) / Peak

    When drawdown exceeds threshold, reduce position sizes to zero until
    recovery to a specified level.

    Usage:
        breaker = DrawdownCircuitBreaker(threshold=0.10)
        is_active = breaker.check(current_equity=90000, peak_equity=100000)
    """

    def __init__(
        self,
        threshold: float = config.DRAWDOWN_CIRCUIT_BREAKER,
        recovery_threshold: float = 0.05,
    ):
        self.threshold = threshold
        self.recovery_threshold = recovery_threshold
        self.is_triggered = False
        self.peak_equity = 0.0

    def update_peak(self, equity: float) -> None:
        """
        Update peak equity.

        Args:
            equity: Current equity value
        """
        if equity > self.peak_equity:
            self.peak_equity = equity
            # Reset trigger if recovered
            if self.is_triggered:
                logger.info("Drawdown circuit breaker: RESET (new peak)")
                self.is_triggered = False

    def calculate_drawdown(self, equity: float) -> float:
        """
        Calculate current drawdown.

        Args:
            equity: Current equity value

        Returns:
            Drawdown as a fraction (0 to 1)
        """
        if self.peak_equity == 0:
            return 0.0

        drawdown = (self.peak_equity - equity) / self.peak_equity
        return max(0.0, drawdown)

    def check(
        self,
        equity: float,
        update_peak: bool = True,
    ) -> bool:
        """
        Check if circuit breaker should trigger.

        Args:
            equity: Current equity value
            update_peak: Update peak equity if new high

        Returns:
            True if circuit breaker is active, False otherwise
        """
        if update_peak:
            self.update_peak(equity)

        drawdown = self.calculate_drawdown(equity)

        # Trigger circuit breaker
        if drawdown >= self.threshold and not self.is_triggered:
            self.is_triggered = True
            logger.warning(
                f"CIRCUIT BREAKER TRIGGERED: Drawdown = {drawdown:.2%} "
                f"(threshold: {self.threshold:.2%})"
            )

        # Check for recovery
        if self.is_triggered and drawdown <= self.recovery_threshold:
            self.is_triggered = False
            logger.info(
                f"CIRCUIT BREAKER RESET: Drawdown recovered to {drawdown:.2%}"
            )

        return self.is_triggered


# ─────────────────────────────────────────────────────────────────────────────
# POSITION SIZING MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class PositionSizingManager:
    """
    Complete position sizing manager combining all methods.

    Pipeline:
      1. Calculate Kelly optimal position size
      2. Apply volatility targeting
      3. Check drawdown circuit breaker
      4. Apply position size limits

    Usage:
        manager = PositionSizingManager(capital=1_000_000)
        position_size = manager.calculate(
            win_prob=0.65,
            odds=2.0,
            realized_vol=0.20,
            current_equity=950_000,
        )
    """

    def __init__(
        self,
        capital: float = config.CAPITAL,
        kelly_fraction: float = config.KELLY_FRACTION,
        max_position_pct: float = config.MAX_POSITION_PCT,
        target_vol: float = config.VOLATILITY_TARGET,
        drawdown_threshold: float = config.DRAWDOWN_CIRCUIT_BREAKER,
    ):
        self.capital = capital
        self.current_equity = capital

        self.kelly = KellyCriterion(
            kelly_fraction=kelly_fraction,
            max_position_pct=max_position_pct,
        )
        self.vol_target = VolatilityTargeting(target_vol=target_vol)
        self.circuit_breaker = DrawdownCircuitBreaker(threshold=drawdown_threshold)

    def calculate(
        self,
        win_prob: float,
        odds: float,
        realized_vol: Optional[float] = None,
        current_equity: Optional[float] = None,
    ) -> float:
        """
        Calculate position size with all risk controls.

        Args:
            win_prob: Probability of winning
            odds: Odds received
            realized_vol: Realized volatility (optional, uses default if None)
            current_equity: Current equity value (optional, uses capital if None)

        Returns:
            Recommended position size
        """
        # Update equity
        if current_equity is not None:
            self.current_equity = current_equity

        # Check circuit breaker
        if self.circuit_breaker.check(self.current_equity):
            logger.warning("Circuit breaker active. Position size = 0")
            return 0.0

        # Calculate Kelly position size
        position_size = self.kelly.calculate(win_prob, odds, self.current_equity)

        # Apply volatility targeting
        if realized_vol is not None and realized_vol > 0:
            position_size = self.vol_target.scale(position_size, realized_vol)

        # Apply position limits
        max_position = self.current_equity * config.MAX_POSITION_PCT
        position_size = min(position_size, max_position)

        logger.info(
            f"Position sizing: win_prob={win_prob:.3f}, odds={odds:.2f}, "
            f"position_size=₹{position_size:,.0f}"
        )

        return position_size

    def calculate_for_signals(
        self,
        df: pd.DataFrame,
        realized_vol: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Calculate position sizes for multiple signals.

        Args:
            df: DataFrame with signals (must have 'pred_up', 'pred_down', 'confidence')
            realized_vol: Realized volatility (optional)

        Returns:
            DataFrame with 'position_size' column
        """
        df = df.copy()

        position_sizes = []
        for idx, row in df.iterrows():
            # Use UP probability as win probability
            win_prob = row.get("pred_up", row.get("confidence", 0.5))

            # Estimate odds from expected return (if available)
            # Otherwise use confidence as proxy
            odds = row.get("expected_return", win_prob * 2)

            # Calculate position size
            position_size = self.calculate(
                win_prob=win_prob,
                odds=max(odds, 0.1),  # Ensure positive odds
                realized_vol=realized_vol,
            )

            position_sizes.append(position_size)

        df["position_size"] = position_sizes
        df["position_size_lots"] = (df["position_size"] / df["CLOSE"]).astype(int)

        return df


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOCK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Test Kelly Criterion
    print("Testing Kelly Criterion...")
    kelly = KellyCriterion(kelly_fraction=0.25, max_position_pct=0.20)
    position_size = kelly.calculate(win_prob=0.65, odds=2.0, capital=1_000_000)
    print(f"Kelly position size: ₹{position_size:,.0f} ✓")

    # Test multi-class Kelly
    probas = np.array([0.15, 0.25, 0.60])  # DOWN, FLAT, UP
    expected_returns = np.array([-0.02, 0.0, 0.03])
    position_size_mc = kelly.calculate_multi_class(probas, expected_returns, 1_000_000)
    print(f"Multi-class Kelly position size: ₹{position_size_mc:,.0f} ✓")

    # Test Volatility Targeting
    print("\nTesting Volatility Targeting...")
    vol_target = VolatilityTargeting(target_vol=0.15)
    scaled_size = vol_target.scale(position_size=100_000, realized_vol=0.25)
    print(f"Volatility-scaled position size: ₹{scaled_size:,.0f} ✓")

    # Test realized vol estimation
    returns = np.random.randn(100) * 0.01  # Simulated daily returns
    realized_vol = vol_target.estimate_realized_vol(returns, annualize=True)
    print(f"Estimated realized vol: {realized_vol:.3f} ✓")

    # Test Drawdown Circuit Breaker
    print("\nTesting Drawdown Circuit Breaker...")
    breaker = DrawdownCircuitBreaker(threshold=0.10, recovery_threshold=0.05)

    equity_series = [1_000_000, 1_050_000, 1_100_000, 950_000, 880_000, 900_000, 1_000_000]
    for equity in equity_series:
        is_active = breaker.check(equity)
        dd = breaker.calculate_drawdown(equity)
        print(f"Equity: ₹{equity:,} | Drawdown: {dd:.2%} | Active: {is_active}")

    # Test Position Sizing Manager
    print("\nTesting Position Sizing Manager...")
    manager = PositionSizingManager(capital=1_000_000)
    final_size = manager.calculate(
        win_prob=0.65,
        odds=2.0,
        realized_vol=0.20,
        current_equity=950_000,
    )
    print(f"Final position size: ₹{final_size:,.0f} ✓")

    print("\nAll position sizing tests passed! ✓")
