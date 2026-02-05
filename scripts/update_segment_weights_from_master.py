from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _norm_probs(arr: np.ndarray) -> list[float]:
    a = np.asarray(arr, dtype=float).reshape(-1)
    a = np.where(np.isfinite(a), a, 0.0)
    a = np.clip(a, 0.0, None)
    s = float(a.sum())
    if s <= 0:
        return [0.25, 0.25, 0.25, 0.25]
    return [float(x) for x in (a / s)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Derive 5-min segment allocation weights from outputs/backtests/segments_5min_master.csv"
    )
    ap.add_argument(
        "--master",
        default=str(Path("outputs") / "backtests" / "segments_5min_master.csv"),
        help="Path to segments_5min_master.csv",
    )
    ap.add_argument(
        "--out",
        default=str(Path("outputs") / "segment_weights.json"),
        help="Output JSON path consumed by the simulator (default outputs/segment_weights.json)",
    )
    ap.add_argument(
        "--shrink-to-uniform",
        type=float,
        default=0.10,
        help="Shrinkage toward uniform weights (0..1)",
    )
    ap.add_argument(
        "--min-games",
        type=int,
        default=200,
        help="Minimum complete games required to write weights",
    )
    args = ap.parse_args()

    master_path = Path(args.master)
    out_path = Path(args.out)
    shrink = float(args.shrink_to_uniform)
    shrink = float(np.clip(shrink, 0.0, 1.0))

    if not master_path.exists():
        raise SystemExit(f"Master CSV not found: {master_path}")

    df = pd.read_csv(master_path)
    needed = {"date", "game_id", "end_min", "actual_total"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise SystemExit(f"Master CSV missing required columns: {missing}")

    df = df[["date", "game_id", "end_min", "actual_total"]].copy()
    df["date"] = df["date"].astype(str)
    df["game_id"] = df["game_id"].astype(str)
    df["end_min"] = pd.to_numeric(df["end_min"], errors="coerce")
    df["actual_total"] = pd.to_numeric(df["actual_total"], errors="coerce")
    df = df.dropna(subset=["end_min", "actual_total"]).copy()

    endpoints = [5, 10, 15, 20, 25, 30, 35, 40]
    wide = (
        df.pivot_table(
            index=["date", "game_id"],
            columns="end_min",
            values="actual_total",
            aggfunc="mean",
        )
        .reset_index()
        .copy()
    )

    # Filter to complete games with all endpoints present
    have_cols = [c for c in endpoints if c in wide.columns]
    for c in have_cols:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")

    if any(c not in wide.columns for c in endpoints):
        missing_ep = [c for c in endpoints if c not in wide.columns]
        raise SystemExit(f"Master CSV does not contain required endpoints: {missing_ep}")

    mask_complete = wide[endpoints].notna().all(axis=1)
    wide = wide[mask_complete].copy()

    n_games = int(len(wide))
    if n_games < int(args.min_games):
        raise SystemExit(f"Not enough complete games to derive weights: n_games={n_games} < {args.min_games}")

    m5, m10, m15, m20, m25, m30, m35, m40 = (wide[c].to_numpy(dtype=float) for c in endpoints)

    # Segment point totals
    s1 = np.vstack([m5 - 0.0, m10 - m5, m15 - m10, m20 - m15]).T
    s2 = np.vstack([m25 - m20, m30 - m25, m35 - m30, m40 - m35]).T

    # Convert to shares within each half
    h1_tot = np.clip(m20, 1e-9, None)
    h2_tot = np.clip(m40 - m20, 1e-9, None)
    sh1 = s1 / h1_tot.reshape(-1, 1)
    sh2 = s2 / h2_tot.reshape(-1, 1)

    # Robust cleanup
    sh1 = np.where(np.isfinite(sh1), sh1, 0.0)
    sh2 = np.where(np.isfinite(sh2), sh2, 0.0)
    sh1 = np.clip(sh1, 0.0, None)
    sh2 = np.clip(sh2, 0.0, None)

    w1 = _norm_probs(np.nanmean(sh1, axis=0))
    w2 = _norm_probs(np.nanmean(sh2, axis=0))

    # Shrink to uniform
    uni = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    w1 = _norm_probs((1.0 - shrink) * np.asarray(w1) + shrink * uni)
    w2 = _norm_probs((1.0 - shrink) * np.asarray(w2) + shrink * uni)

    payload = {
        "half1": w1,
        "half2": w2,
        "source": str(master_path).replace("\\\\", "/"),
        "n_games": n_games,
        "shrink_to_uniform": shrink,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "out": str(out_path), "n_games": n_games, "half1": w1, "half2": w2}))


if __name__ == "__main__":
    main()
