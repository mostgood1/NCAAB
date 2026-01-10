from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from .model_totals import OUT, _safe_read_csv
import os

def _read_quantile_calibration() -> dict:
    try:
        p = OUT / "calibration_quantiles.json"
        if p.exists():
            import json as _json
            return _json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return {}

def _read_quantile_segmented() -> dict:
    try:
        p = OUT / "calibration_quantiles_segmented.json"
        if p.exists():
            import json as _json
            return _json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return {}

def _read_quantile_crps() -> dict:
    try:
        p = OUT / "calibration_quantiles_crps.json"
        if p.exists():
            import json as _json
            return _json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return {}

def _read_quantile_crps_segmented() -> dict:
    try:
        p = OUT / "calibration_quantiles_crps_segmented.json"
        if p.exists():
            import json as _json
            return _json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return {}


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
    # Apply CRPS quantile calibration first if available: shift q50 and scale spreads
    try:
        has_quantiles = {"pred_total_q10","pred_total_q50","pred_total_q90"}.issubset(merged.columns)
        crps_seg = _read_quantile_crps_segmented()
        if has_quantiles and crps_seg:
            edges = crps_seg.get("edges") or []
            bins = crps_seg.get("bins") or {}
            # Build tempo value per row
            def tempo_val(row):
                for key in ("closing_total","market_total","pred_total_q50"):
                    if key in row and pd.notna(row[key]):
                        return float(row[key])
                return None
            def tempo_bin(val):
                if val is None or len(edges) < 2:
                    return None
                e1, e2 = edges[0], edges[1]
                if val <= e1:
                    return "low"
                elif val <= e2:
                    return "mid"
                else:
                    return "high"
            # Apply per-row
            def apply_row(r):
                q10 = pd.to_numeric(r.get("pred_total_q10"), errors="coerce")
                q50 = pd.to_numeric(r.get("pred_total_q50"), errors="coerce")
                q90 = pd.to_numeric(r.get("pred_total_q90"), errors="coerce")
                if pd.isna(q50):
                    return r
                tv = tempo_val(r)
                bl = tempo_bin(tv)
                params = bins.get(bl) or {}
                shift = float(params.get("shift", 0.0))
                scale = float(params.get("scale", 1.0))
                r["pred_total_calibrated"] = q50 + shift
                if pd.notna(q10) and pd.notna(q90):
                    r["pred_total_q10"] = (q50 + shift) - scale * (q50 - q10)
                    r["pred_total_q90"] = (q50 + shift) + scale * (q90 - q50)
                return r
            merged = merged.apply(apply_row, axis=1)
            if "pred_total_basis" in merged.columns:
                merged["pred_total_basis"] = merged.apply(
                    lambda r: ("cal" if pd.notna(r.get("pred_total_calibrated")) else r.get("pred_total_basis")), axis=1
                )
        else:
            crps = _read_quantile_crps()
            if crps and has_quantiles:
                shift = float(crps.get("shift", 0.0))
                scale = float(crps.get("scale", 1.0))
                q10 = pd.to_numeric(merged["pred_total_q10"], errors="coerce")
                q50 = pd.to_numeric(merged["pred_total_q50"], errors="coerce")
                q90 = pd.to_numeric(merged["pred_total_q90"], errors="coerce")
                merged["pred_total_calibrated"] = (q50 + shift)
                merged["pred_total_q10"] = (q50 + shift) - scale * (q50 - q10)
                merged["pred_total_q90"] = (q50 + shift) + scale * (q90 - q50)
                if "pred_total_basis" in merged.columns:
                    merged["pred_total_basis"] = merged.apply(
                        lambda r: ("cal" if pd.notna(r.get("pred_total_calibrated")) else r.get("pred_total_basis")), axis=1
                    )
    except Exception:
        pass
    # If CRPS not available, apply simple q50 offset calibration
    try:
        cal = _read_quantile_calibration()
        if cal and ("pred_total_q50" in merged.columns) and ("pred_total_calibrated" not in merged.columns):
            q50 = pd.to_numeric(merged["pred_total_q50"], errors="coerce")
            offset = float(cal.get("q50_offset", 0.0))
            if q50.notna().any():
                merged["pred_total_calibrated"] = (q50 - offset)
                if "pred_total_basis" in merged.columns:
                    merged["pred_total_basis"] = merged.apply(
                        lambda r: ("cal" if pd.notna(r.get("pred_total_calibrated")) else r.get("pred_total_basis")), axis=1
                    )
    except Exception:
        pass
    # Optional: adjust q10/q90 spread using alpha to target empirical coverage
    try:
        seg = _read_quantile_segmented()
        alpha = float(seg.get("spread_alpha", 1.0))
        if alpha != 1.0 and {"pred_total_q10","pred_total_q50","pred_total_q90"}.issubset(merged.columns):
            q10 = pd.to_numeric(merged["pred_total_q10"], errors="coerce")
            q50 = pd.to_numeric(merged["pred_total_q50"], errors="coerce")
            q90 = pd.to_numeric(merged["pred_total_q90"], errors="coerce")
            # If pred_total_calibrated exists, anchor to it; else use q50
            anchor = pd.to_numeric(merged.get("pred_total_calibrated", q50), errors="coerce")
            d10 = (q50 - q10)
            d90 = (q90 - q50)
            merged["pred_total_q10"] = (anchor - alpha * d10)
            merged["pred_total_q90"] = (anchor + alpha * d90)
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
