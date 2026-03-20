import pandas as pd
from pathlib import Path

def verify_parquet(path="data/extended/market_data_extended.parquet"):
    if not Path(path).exists():
        print(f"File not found: {path}")
        return

    df = pd.read_parquet(path)

    print("\n=== PARQUET VERIFICATION ===")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} → {df.index.max()}")
    
    # Check for duplicates
    dup_y = [c for c in df.columns if c.endswith('_y')]
    print(f"Duplicate _y cols: {dup_y}")
    
    # Check for all-zero/all-NaN cols
    zero_cols = [c for c in df.columns if df[c].eq(0).all()]
    nan_cols = [c for c in df.columns if df[c].isna().all()]
    print(f"All-zero cols: {zero_cols}")
    print(f"All-NaN cols: {nan_cols}")
    
    # Check for raw close columns (should be dropped in merge_with_fno, but let's see in parquet)
    # The user said FIXED 1 — Drop raw close prices before they enter the model
    # Wait! If they are in the parquet, they might be dropped in merge_with_fno.
    # But Fix 3 says: "Raw close cols (should be dropped)"
    raw_cols = [c for c in df.columns if '_CLOSE' in c or '_VOL' in c]
    print(f"\nRaw columns in parquet: {raw_cols[:5]} ... ({len(raw_cols)} total)")
    
    if len(raw_cols) > 0:
        print("(!) WARNING: Raw columns still present in parquet. They should be dropped in merge_with_fno().")

    print("\n=== VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    verify_parquet()
