import pandas as pd
import numpy as np

def run_checks():
    df = pd.read_parquet("data/extended/market_data_extended.parquet")

    # Should all pass
    assert not any(c.endswith('_y') for c in df.columns), "Duplicate columns!"
    assert "KOSPI_1D_RET" in df.columns, "KOSPI missing"
    assert "BANK_1D_RET" in df.columns, "BANK missing"
    
    # Cleanups check
    assert "GOLD_REALIZED_VOL" in df.columns, "GOLD_REALIZED_VOL missing"
    assert "OIL_REALIZED_VOL" in df.columns, "OIL_REALIZED_VOL missing"
    
    broken_cols = ["NICKEL_CLOSE", "LEAD_CLOSE", "OIL_VIX_CLOSE", "GOLD_VIX_CLOSE"]
    for bc in broken_cols:
        assert bc not in df.columns, f"Broken column {bc} still present!"

    # Event flag spot check
    jan2_2020 = df.loc["2020-01-02"]
    assert jan2_2020["IS_WEEKLY_EXPIRY"] == 1, "Thursday should be weekly expiry"

    # RISK_APPETITE_SCORE should have range, not all near-zero
    score = df["RISK_APPETITE_SCORE"].dropna()
    assert score.max() > 5, f"Score never above 5 — check normalisation (max={score.max():.2f})"

    print("All checks passed ✓")
    print(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
    print(f"Total signal columns (no _CLOSE): {sum(1 for c in df.columns if '_CLOSE' not in c and c != 'DATE')}")

if __name__ == "__main__":
    try:
        run_checks()
    except Exception as e:
        print(f"Verification failed: {e}")
        import sys
        sys.exit(1)
