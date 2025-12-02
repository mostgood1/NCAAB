"""Compare two stake sheets and write per-pick deltas.

Usage (PowerShell):
  .\.venv\Scripts\python.exe scripts/compare_stake_sheets.py \
      --orig outputs/stake_sheet_today.csv \
      --cal outputs/stake_sheet_today_cal.csv \
      --out outputs/stake_sheet_today_compare.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


KEYS = [
    "date","game_id","book","market","period","selection","line"
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", type=Path, required=True)
    ap.add_argument("--cal", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    o = pd.read_csv(args.orig)
    c = pd.read_csv(args.cal)

    def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
        # Coerce date to uniform string YYYY-MM-DD if present
        if "date" in df.columns:
            # Handle various representations; errors='coerce' turns invalid to NaT
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        # Ensure game_id always string
        if "game_id" in df.columns:
            df["game_id"] = df["game_id"].astype(str)
        # Other merge keys may differ in dtype (e.g., line numeric vs object); cast both sides to string
        for k in ("book","market","period","selection","line"):
            if k in df.columns:
                df[k] = df[k].astype(str)
        return df

    o = normalize_keys(o)
    c = normalize_keys(c)

    cols_common = [k for k in KEYS if k in o.columns and k in c.columns]
    # Safe outer merge now that dtypes are harmonized
    merged = o.merge(c, on=cols_common, how="outer", suffixes=("_orig","_cal"))
    # Compute deltas for key numeric columns present
    num_cols = [
        ("kelly","kelly"),
        ("prob","prob"),
        ("ev","ev"),
        ("stake","stake"),
    ]
    for a, b in num_cols:
        a_col = f"{a}_orig"
        b_col = f"{b}_cal"
        if a_col in merged.columns and b_col in merged.columns:
            merged[f"delta_{a}"] = merged[b_col] - merged[a_col]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"Wrote comparison -> {args.out} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
