from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .model_totals import TotalsModel, _safe_read_csv, _select_features, OUT


def score_date(date_str: str, model_path: Path) -> dict:
    model = TotalsModel.load(model_path)
    # Load features for date
    candidates = [
        OUT / f"features_{date_str}_augmented.csv",
        OUT / f"features_{date_str}.csv",
        OUT / "features_curr_augmented.csv",
        OUT / "features_curr.csv",
    ]
    src = None
    df = pd.DataFrame()
    for p in candidates:
        if p.exists():
            src = p
            df = _safe_read_csv(p)
            break
    # If date-specific features lack augmented columns, try merging current augmented snapshot
    try:
        aug_p = OUT / "features_curr_augmented.csv"
        aug = _safe_read_csv(aug_p)
        if src is not None and df is not None and not df.empty and not aug.empty:
            # normalize keys
            if "game_id" in df.columns:
                df["game_id"] = df["game_id"].astype(str)
            if "game_id" in aug.columns:
                aug["game_id"] = aug["game_id"].astype(str)
            join_keys = [k for k in ("game_id", "date") if k in df.columns and k in aug.columns]
            if not join_keys:
                join_keys = ["game_id"]
            merged = df.merge(aug, on=join_keys, how="left", suffixes=("", "_aug"))
            # Promote augmented columns to base names when base is missing
            df = merged.copy()
            for c in model.feature_cols:
                aug_c = f"{c}_aug"
                if c not in df.columns and aug_c in df.columns:
                    df[c] = df[aug_c]
    except Exception:
        pass
    if df.empty:
        return {"error": "No features found for date", "date": date_str}
    # Normalize keys
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    X = _select_features(df)
    preds = model.predict(X)
    mean = preds.get("mean")
    q10 = preds.get("q0.1")
    q50 = preds.get("q0.5")
    q90 = preds.get("q0.9")
    out_df = pd.DataFrame({
        "game_id": df.get("game_id"),
        "date": df.get("date", pd.Series([date_str] * len(df))),
        "pred_total_model": mean,
        "pred_total_q10": q10,
        "pred_total_q50": q50,
        "pred_total_q90": q90,
        "pred_total_basis": pd.Series(["model_raw"] * len(df)),
    })
    out_path = OUT / f"predictions_model_totals_{date_str}.csv"
    out_df.to_csv(out_path, index=False)
    payload = {"date": date_str, "rows": int(len(out_df)), "source_features": str(src), "predictions_path": str(out_path)}
    (OUT / f"score_totals_{date_str}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    ap = argparse.ArgumentParser(description="Score totals for a date using trained model")
    ap.add_argument("--date", type=str, required=True, help="Target date YYYY-MM-DD")
    ap.add_argument("--model", type=str, default=str(OUT / "models" / "totals_v1.joblib"), help="Model path")
    args = ap.parse_args()
    payload = score_date(args.date, Path(args.model))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
