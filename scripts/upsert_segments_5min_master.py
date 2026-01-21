from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLS = ["date", "game_id", "end_min"]


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df

    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column '{c}': {path}")

    df = df.copy()
    df["date"] = df["date"].astype(str)
    df["game_id"] = df["game_id"].astype(str)
    df["end_min"] = pd.to_numeric(df["end_min"], errors="coerce")
    df = df.dropna(subset=["date", "game_id", "end_min"])
    df["end_min"] = df["end_min"].astype(int)

    return df


def upsert_segments_master(daily_csv: Path, master_csv: Path) -> dict:
    daily_csv = Path(daily_csv)
    master_csv = Path(master_csv)

    if not daily_csv.exists():
        raise FileNotFoundError(f"Daily CSV not found: {daily_csv}")

    daily = _read_csv(daily_csv)
    if daily.empty:
        return {
            "status": "noop",
            "reason": "daily_csv_empty",
            "daily_csv": str(daily_csv),
            "master_csv": str(master_csv),
            "rows_master_before": int(0 if not master_csv.exists() else len(pd.read_csv(master_csv))),
            "rows_master_after": int(0 if not master_csv.exists() else len(pd.read_csv(master_csv))),
        }

    if master_csv.exists():
        master = _read_csv(master_csv)
    else:
        master = daily.head(0).copy()

    key_cols = ["date", "game_id", "end_min"]

    combined = pd.concat([master, daily], ignore_index=True, sort=False)
    combined["date"] = combined["date"].astype(str)
    combined["game_id"] = combined["game_id"].astype(str)
    combined["end_min"] = pd.to_numeric(combined["end_min"], errors="coerce")
    combined = combined.dropna(subset=key_cols)
    combined["end_min"] = combined["end_min"].astype(int)

    # Prefer newest rows for the same key (daily CSV last).
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(key_cols, kind="mergesort")

    master_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(master_csv, index=False)

    return {
        "status": "ok",
        "daily_csv": str(daily_csv),
        "master_csv": str(master_csv),
        "rows_daily": int(len(daily)),
        "rows_master_before": int(len(master)),
        "rows_master_after": int(len(combined)),
        "unique_keys": int(len(combined[key_cols].drop_duplicates())),
        "date_min": str(combined["date"].min()) if not combined.empty else None,
        "date_max": str(combined["date"].max()) if not combined.empty else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Upsert a daily 5-min segments backtest CSV into a master CSV")
    ap.add_argument("--daily", required=True, help="Daily backtest CSV (e.g., outputs/backtests/segments_5min_daily_<d>_to_<d>.csv)")
    ap.add_argument(
        "--master",
        default=str(Path("outputs") / "backtests" / "segments_5min_master.csv"),
        help="Master CSV to upsert into (default: outputs/backtests/segments_5min_master.csv)",
    )
    args = ap.parse_args()

    res = upsert_segments_master(Path(args.daily), Path(args.master))
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
