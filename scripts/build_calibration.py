"""Utility script to build calibration artifact from historical distributional predictions.

Usage (PowerShell):
  .\.venv\Scripts\python.exe scripts/build_calibration.py \
      --preds outputs/predictions_distributional_hist.csv \
      --features outputs/features_hist.csv \
      --out outputs/models_dist/calibration_totals.json

Requires columns: pred_total_mu, pred_total_sigma and target_total or score columns.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from ncaab_model.train.calibration import build_z_recenter_artifact, save_artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True, help="CSV with pred_total_mu & pred_total_sigma")
    ap.add_argument("--features", type=Path, required=False, help="Features CSV with target_total/score columns (optional if preds already includes target_total)")
    ap.add_argument("--out", type=Path, required=True, help="Output artifact path")
    ap.add_argument("--min-rows", type=int, default=500, help="Minimum rows required for calibration")
    args = ap.parse_args()
    if not args.preds.exists():
        raise SystemExit(f"preds file not found: {args.preds}")
    df_preds = pd.read_csv(args.preds)
    if args.features and args.features.exists():
        df_feats = pd.read_csv(args.features)
        # Merge actual totals into preds if needed
        if "target_total" in df_feats.columns and "target_total" not in df_preds.columns:
            if "game_id" in df_preds.columns and "game_id" in df_feats.columns:
                df_feats["game_id"] = df_feats["game_id"].astype(str)
                df_preds["game_id"] = df_preds["game_id"].astype(str)
                df_preds = df_preds.merge(df_feats[["game_id","target_total"]], on="game_id", how="left")
    art = build_z_recenter_artifact(df_preds, min_rows=args.min_rows)
    save_artifact(art, args.out)
    print(f"Saved calibration artifact -> {args.out} (n={art.n_samples}, center={art.z_center:.4f}, scale={art.z_scale:.4f})")


if __name__ == "__main__":
    main()
