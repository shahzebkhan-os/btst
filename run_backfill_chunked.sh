#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_backfill_chunked.sh
# Processes NSE F&O historical data ONE YEAR AT A TIME to avoid OOM.
# Merges all yearly chunks into a single reconstructed.csv at the end.
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

YEARS=("2020" "2021" "2022" "2023" "2024" "2025" "2026")
START_DATES=("2020-01-01" "2021-01-01" "2022-01-01" "2023-01-01" "2024-01-01" "2025-01-01" "2026-01-01")
END_DATES=("2020-12-31" "2021-12-31" "2022-12-31" "2023-12-31" "2024-12-31" "2025-12-31" "2026-03-20")

HIST_DIR="${HOME}/Desktop/btst/historical_data"
CHUNKS_DIR="${HIST_DIR}/chunks"
mkdir -p "$CHUNKS_DIR"

echo "════════════════════════════════════════════════════"
echo "  NSE F&O CHUNKED BACKFILL — Year-by-Year"
echo "════════════════════════════════════════════════════"

for i in "${!YEARS[@]}"; do
    YEAR="${YEARS[$i]}"
    START="${START_DATES[$i]}"
    END="${END_DATES[$i]}"
    CHUNK_FILE="${CHUNKS_DIR}/reconstructed_${YEAR}.csv"

    if [ -f "$CHUNK_FILE" ] && [ -s "$CHUNK_FILE" ]; then
        echo "✓ $YEAR: already processed ($CHUNK_FILE exists). Skipping."
        continue
    fi

    echo ""
    echo "▶ Processing $YEAR ($START → $END)..."
    python3 historical_loader.py process --start "$START" --end "$END"

    # Move the generated reconstructed.csv to the annual chunk
    RECON="${HIST_DIR}/reconstructed.csv"
    if [ -f "$RECON" ]; then
        mv "$RECON" "$CHUNK_FILE"
        echo "  ✓ Saved chunk: $CHUNK_FILE"
    else
        echo "  ✗ No reconstructed.csv generated for $YEAR. Skipping."
    fi

    # Small pause between years to let memory settle
    sleep 2
done

echo ""
echo "▶ Merging all yearly chunks into reconstructed.csv..."
python3 - <<'PYEOF'
import os, glob, pandas as pd

hist_dir = os.path.expanduser("~/Desktop/btst/historical_data")
chunks_dir = os.path.join(hist_dir, "chunks")
chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "reconstructed_*.csv")))

if not chunk_files:
    print("❌ No chunk files found.")
    exit(1)

dfs = []
for f in chunk_files:
    print(f"  Reading {os.path.basename(f)}...")
    try:
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
        print(f"    → {len(df)} rows")
    except Exception as e:
        print(f"    ✗ Error: {e}")

if not dfs:
    print("❌ No data to merge.")
    exit(1)

merged = pd.concat(dfs, ignore_index=True)
merged = merged.drop_duplicates(subset=["trade_date", "symbol"]) if "trade_date" in merged.columns else merged.drop_duplicates()
merged = merged.sort_values("trade_date") if "trade_date" in merged.columns else merged

out = os.path.join(hist_dir, "reconstructed.csv")
merged.to_csv(out, index=False)
print(f"\n✅ Merged {len(dfs)} chunks → {len(merged)} total rows → {out}")
PYEOF

echo ""
echo "════════════════════════════════════════════════════"
echo "  BACKFILL COMPLETE"
echo "════════════════════════════════════════════════════"
echo ""
echo "Now run:"
echo "  python3 historical_loader.py load-db --start 2020-01-01 --end 2026-03-20"
