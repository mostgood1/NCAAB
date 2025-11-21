#!/usr/bin/env python
"""Daily segment performance evaluation.

Segments games by tempo quartile and spread magnitude buckets to surface where
model performs better or worse. Writes outputs/segment_<date>.json.

Inputs:
  outputs/daily_results/results_<date>.csv (needs pred_total_model or pred_total, actual_total)
  features_* (optional for tempo ratings) -> uses features_curr.csv primarily.

Buckets:
  tempo_quartile: Q1 (slow) ... Q4 (fast) based on average(home_tempo_rating, away_tempo_rating)
  spread_bucket: abs(spread_home) grouped into [0-2, 2-5, 5-9, 9+]

Metrics per segment:
  n_games
  mean_residual_total
  mae_total
  std_total

Usage:
  python scripts/segment_performance.py --date YYYY-MM-DD
"""
from __future__ import annotations
import argparse, json, datetime as dt
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np

OUT = Path("outputs")
RES = OUT / "daily_results"

def load_results(d: str) -> pd.DataFrame:
    p = RES / f"results_{d}.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def load_features() -> pd.DataFrame:
    for name in ["features_curr.csv","features_last2.csv","features_all.csv"]:
        p = OUT / name
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default today)")
    args = ap.parse_args()
    date_str = args.date or dt.date.today().strftime("%Y-%m-%d")

    res = load_results(date_str)
    feats = load_features()

    if res.empty or 'home_team' not in res.columns:
        payload = {"date": date_str, "status": "no_data"}
        (OUT / f"segment_{date_str}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Segment performance: no data")
        return

    # Choose prediction column
    pred_col = 'pred_total_model' if 'pred_total_model' in res.columns else ('pred_total' if 'pred_total' in res.columns else None)
    if pred_col is None or 'actual_total' not in res.columns:
        payload = {"date": date_str, "status": "missing_columns"}
        (OUT / f"segment_{date_str}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Segment performance: missing columns")
        return

    df = res.copy()
    if not feats.empty and 'game_id' in feats.columns and 'game_id' in df.columns:
        try:
            feats['game_id'] = feats['game_id'].astype(str)
            df['game_id'] = df['game_id'].astype(str)
            df = df.merge(feats[['game_id','home_tempo_rating','away_tempo_rating']].drop_duplicates(), on='game_id', how='left')
        except Exception:
            pass

    # Tempo average
    if {'home_tempo_rating','away_tempo_rating'}.issubset(df.columns):
        ht = pd.to_numeric(df['home_tempo_rating'], errors='coerce')
        at = pd.to_numeric(df['away_tempo_rating'], errors='coerce')
        df['tempo_avg'] = (ht + at) / 2.0
    else:
        df['tempo_avg'] = np.nan

    pred = pd.to_numeric(df[pred_col], errors='coerce')
    actual = pd.to_numeric(df['actual_total'], errors='coerce')
    df['residual_total'] = pred - actual

    # Tempo quartiles
    tempo_series = df['tempo_avg']
    valid_tempo = tempo_series.notna()
    if valid_tempo.sum() >= 10:
        try:
            q = tempo_series[valid_tempo].quantile([0.25,0.5,0.75])
            def tempo_bucket(v):
                if pd.isna(v):
                    return 'unknown'
                if v <= q[0.25]: return 'Q1_slowest'
                if v <= q[0.5]: return 'Q2'
                if v <= q[0.75]: return 'Q3'
                return 'Q4_fastest'
            df['tempo_quartile'] = df['tempo_avg'].map(tempo_bucket)
        except Exception:
            df['tempo_quartile'] = 'unknown'
    else:
        df['tempo_quartile'] = 'unknown'

    # Spread magnitude bucket
    spread_col = 'spread_home' if 'spread_home' in df.columns else None
    if spread_col:
        sp = pd.to_numeric(df[spread_col], errors='coerce').abs()
        def spread_bucket(v):
            if pd.isna(v): return 'unknown'
            if v < 2: return '0_2'
            if v < 5: return '2_5'
            if v < 9: return '5_9'
            return '9_plus'
        df['spread_bucket'] = sp.map(spread_bucket)
    else:
        df['spread_bucket'] = 'unknown'

    segments: Dict[str, Dict[str, Any]] = {}
    def agg_segment(name: str, sub: pd.DataFrame):
        resid = pd.to_numeric(sub['residual_total'], errors='coerce').dropna()
        if resid.empty:
            return {"n_games": 0}
        return {
            "n_games": int(len(resid)),
            "mean_residual_total": float(resid.mean()),
            "mae_residual_total": float(resid.abs().mean()),
            "std_residual_total": float(resid.std()) if resid.std() > 0 else 0.0,
        }

    for bucket in sorted(df['tempo_quartile'].unique()):
        segments[f'tempo::{bucket}'] = agg_segment(f'tempo::{bucket}', df[df['tempo_quartile'] == bucket])
    for bucket in sorted(df['spread_bucket'].unique()):
        segments[f'spread::{bucket}'] = agg_segment(f'spread::{bucket}', df[df['spread_bucket'] == bucket])

    payload = {
        "date": date_str,
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "status": "ok",
        "segments": segments,
    }
    out_path = OUT / f"segment_{date_str}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Segment performance written -> {out_path}")

if __name__ == "__main__":
    main()
