from __future__ import annotations

import argparse
import json

import pandas as pd

from .model_totals import OUT, _safe_read_csv


def _read_margin_quantile_crps() -> dict:
    try:
        p = OUT / "calibration_margin_quantiles_crps.json"
        if p.exists():
            import json as _json

            return _json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return {}


def integrate(date_str: str) -> dict:
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
    if base.empty or base_path is None:
        return {"error": "No base predictions file found", "date": date_str}

    mpath = OUT / f"predictions_model_margins_{date_str}.csv"
    mod = _safe_read_csv(mpath)
    if mod.empty:
        return {"error": "No model margins predictions found", "date": date_str}

    # Drop merge suffix columns if this was integrated before.
    for base_col in (
        "pred_margin_model",
        "pred_margin_q10",
        "pred_margin_q50",
        "pred_margin_q90",
        "pred_margin_basis",
        "pred_margin_model_basis",
    ):
        for suf in ("_x", "_y"):
            c = f"{base_col}{suf}"
            if c in base.columns:
                base.drop(columns=[c], inplace=True)

    for d in (base, mod):
        if "game_id" in d.columns:
            d["game_id"] = d["game_id"].astype(str)

    join_keys = [k for k in ("game_id", "date") if k in base.columns and k in mod.columns]
    if not join_keys:
        join_keys = ["game_id"]

    merged = base.merge(
        mod[[*join_keys, "pred_margin_model", "pred_margin_q10", "pred_margin_q50", "pred_margin_q90"]],
        on=join_keys,
        how="left",
    )

    # Resolve duplicate suffixes for quantiles
    for col in ("pred_margin_q10", "pred_margin_q50", "pred_margin_q90"):
        cx, cy = f"{col}_x", f"{col}_y"
        if cx in merged.columns and cy in merged.columns:
            merged[col] = merged[cy]
            merged.drop(columns=[cx, cy], inplace=True)

    col = "pred_margin_model"
    cx, cy = f"{col}_x", f"{col}_y"
    if cx in merged.columns and cy in merged.columns:
        try:
            merged[col] = merged[cy].where(pd.notna(merged[cy]), merged[cx])
        except Exception:
            merged[col] = merged[cy]
        merged.drop(columns=[cx, cy], inplace=True)

    if "pred_margin_model" in merged.columns:
        merged["pred_margin_model_basis"] = merged["pred_margin_model"].apply(lambda x: "model_raw" if pd.notna(x) else None)
        try:
            if "pred_margin_basis" in merged.columns:
                merged["pred_margin_basis"] = merged.apply(
                    lambda r: ("model" if pd.notna(r.get("pred_margin_model")) else r.get("pred_margin_basis")), axis=1
                )
        except Exception:
            pass

    # Apply CRPS-style quantile calibration if available: shift q50 and scale spreads.
    try:
        has_quantiles = {"pred_margin_q10", "pred_margin_q50", "pred_margin_q90"}.issubset(merged.columns)
        crps = _read_margin_quantile_crps()
        if has_quantiles and crps:
            shift = float(crps.get("shift", 0.0))
            scale = float(crps.get("scale", 1.0))
            q10 = pd.to_numeric(merged["pred_margin_q10"], errors="coerce")
            q50 = pd.to_numeric(merged["pred_margin_q50"], errors="coerce")
            q90 = pd.to_numeric(merged["pred_margin_q90"], errors="coerce")
            merged["pred_margin_calibrated"] = (q50 + shift)
            merged["pred_margin_q10"] = (q50 + shift) - scale * (q50 - q10)
            merged["pred_margin_q90"] = (q50 + shift) + scale * (q90 - q50)
            if "pred_margin_basis" in merged.columns:
                merged["pred_margin_basis"] = merged.apply(
                    lambda r: ("cal" if pd.notna(r.get("pred_margin_calibrated")) else r.get("pred_margin_basis")), axis=1
                )
    except Exception:
        pass

    merged.to_csv(base_path, index=False)
    payload = {
        "date": date_str,
        "updated_file": str(base_path),
        "rows": int(len(merged)),
        "integrated_cols": [
            c
            for c in merged.columns
            if c.startswith("pred_margin_q") or c in ("pred_margin_model", "pred_margin_model_basis", "pred_margin_basis")
        ],
    }
    (OUT / f"integrate_model_margins_{date_str}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Integrate model margins into unified/enriched predictions for a date")
    ap.add_argument("--date", type=str, required=True)
    args = ap.parse_args()
    payload = integrate(args.date)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
