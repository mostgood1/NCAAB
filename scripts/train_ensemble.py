"""Train (construct) ensemble predictions for a given date.

Usage:
  python scripts/train_ensemble.py -Date 2025-11-20
Outputs:
  outputs/predictions_model_ensemble_<date>.csv
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import argparse
import pathlib
from datetime import date as ddate

from src.models.ensemble_stack import build_ensemble, EnsembleConfig

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

def load_source(date_str: str) -> pd.DataFrame:
    base = pd.DataFrame()
    # Prefer unified predictions for richest data
    uni = OUT / f'predictions_unified_{date_str}.csv'
    if uni.exists():
        try:
            base = pd.read_csv(uni)
        except Exception:
            base = pd.DataFrame()
    if base.empty:
        model_path = OUT / f'predictions_model_{date_str}.csv'
        if model_path.exists():
            try:
                base = pd.read_csv(model_path)
            except Exception:
                base = pd.DataFrame()
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-Date', required=False, help='Target date YYYY-MM-DD (default: today)')
    ap.add_argument('-Out', required=False, help='Override output file path')
    ap.add_argument('-Cfg', required=False, help='JSON string for weights override')
    args = ap.parse_args()
    date_str = args.Date or ddate.today().strftime('%Y-%m-%d')
    df = load_source(date_str)
    if df.empty:
        raise SystemExit(f'No source predictions for {date_str}')
    cfg = EnsembleConfig()
    if args.Cfg:
        import json
        try:
            data = json.loads(args.Cfg)
            for k,v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, float(v))
        except Exception:
            print('Warning: could not parse Cfg JSON; using defaults')
    enriched = build_ensemble(df, cfg)
    keep = [c for c in ['game_id','date','home_team','away_team','pred_total_ensemble','pred_margin_ensemble'] if c in enriched.columns]
    out_df = enriched[keep].copy()
    out_path = pathlib.Path(args.Out) if args.Out else (OUT / f'predictions_model_ensemble_{date_str}.csv')
    out_df.to_csv(out_path, index=False)
    print(f'Wrote ensemble predictions to {out_path} (rows={len(out_df)})')

if __name__ == '__main__':
    main()
