from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from .model_totals import OUT, _safe_read_csv


def integrate(date_str: str) -> dict:
    # Load enriched/unified predictions file for date
    candidates = [
        OUT / f"predictions_unified_enriched_{date_str}.csv",
        OUT / f"predictions_unified_{date_str}.csv",
        OUT / f"predictions_enriched_{date_str}.csv",
    ]
    base = pd.DataFrame()
    base_path = None
    for p in candidates:
        if p.exists():
            base = _safe_read_csv(p)
            base_path = p
            break
    if base.empty:
        return {"error": "No base predictions file found", "date": date_str}
    # Load model predictions
    mpath = OUT / f"predictions_model_totals_{date_str}.csv"
    mod = _safe_read_csv(mpath)
    if mod.empty:
        return {"error": "No model totals predictions found", "date": date_str}
    # Normalize keys
    for d in (base, mod):
        if "game_id" in d.columns:
            d["game_id"] = d["game_id"].astype(str)
    # Merge by game_id (and date if present)
    join_keys = [k for k in ("game_id", "date") if k in base.columns and k in mod.columns]
    if not join_keys:
        join_keys = ["game_id"]
    merged = base.merge(mod[[*join_keys, "pred_total_model", "pred_total_q10", "pred_total_q50", "pred_total_q90"]], on=join_keys, how="left")
    # Resolve duplicate suffixes: prefer model's quantiles where both exist
    for col in ("pred_total_q10", "pred_total_q50", "pred_total_q90"):
        cx, cy = f"{col}_x", f"{col}_y"
        if cx in merged.columns and cy in merged.columns:
            merged[col] = merged[cy]
            merged.drop(columns=[cx, cy], inplace=True)
    # Basis hint column and normalization: mark rows using model totals
    if "pred_total_model" in merged.columns:
        merged["pred_total_model_basis"] = merged["pred_total_model"].apply(lambda x: "model_raw" if pd.notna(x) else None)
        try:
            # If model totals are present for a row, set pred_total_basis to 'model'
            if "pred_total_basis" in merged.columns:
                merged["pred_total_basis"] = merged.apply(
                    lambda r: ("model" if pd.notna(r.get("pred_total_model")) else r.get("pred_total_basis")), axis=1
                )
        except Exception:
            pass
    # Write back enriched
    merged.to_csv(base_path, index=False)
    payload = {"date": date_str, "updated_file": str(base_path), "rows": int(len(merged)), "integrated_cols": [c for c in merged.columns if c.startswith("pred_total_q") or c in ("pred_total_model","pred_total_model_basis","pred_total_basis")]}
    (OUT / f"integrate_model_totals_{date_str}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    ap = argparse.ArgumentParser(description="Integrate model totals into unified/enriched predictions for a date")
    ap.add_argument("--date", type=str, required=True)
    args = ap.parse_args()
    payload = integrate(args.date)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
