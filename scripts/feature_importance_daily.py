#!/usr/bin/env python
"""Daily feature importance extraction.

Loads LightGBM meta ensemble models (if available) and computes normalized gain
feature importances. Writes outputs/importance_<date>.json including top features
for totals and margin along with cumulative coverage.

If models missing or lightgbm unavailable, writes stub payload with status reason.

Usage:
  python scripts/feature_importance_daily.py --date YYYY-MM-DD
"""
from __future__ import annotations
import argparse, json, datetime as dt
from pathlib import Path
from typing import Any, Dict

OUT = Path("outputs")

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # graceful degrade
    lgb = None


def load_model(name: str):
    p = OUT / name
    if not p.exists() or lgb is None:
        return None
    try:
        booster = lgb.Booster(model_file=str(p))
        return booster
    except Exception:
        return None


def importance_payload(booster) -> Dict[str, Any]:
    try:
        feats = booster.feature_name()
        gains = booster.feature_importance(importance_type="gain")
        total_gain = float(sum(gains)) or 1.0
        items = []
        cum = 0.0
        order = sorted(zip(feats, gains), key=lambda x: x[1], reverse=True)
        for f, g in order:
            frac = float(g) / total_gain if total_gain else 0.0
            cum += frac
            items.append({"feature": f, "gain": float(g), "gain_frac": frac, "cumulative": cum})
        return {"features": items, "total_gain": total_gain}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default today)")
    args = ap.parse_args()
    date_str = args.date or dt.date.today().strftime("%Y-%m-%d")

    booster_totals = load_model("meta_ensemble_totals.txt")
    booster_margin = load_model("meta_ensemble_margin.txt")

    payload: Dict[str, Any] = {
        "date": date_str,
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "status": "ok" if (booster_totals or booster_margin) else "missing_models",
    }
    if booster_totals:
        payload["totals"] = importance_payload(booster_totals)
    if booster_margin:
        payload["margin"] = importance_payload(booster_margin)

    out_path = OUT / f"importance_{date_str}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote feature importance -> {out_path}")

if __name__ == "__main__":
    main()
