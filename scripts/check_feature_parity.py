#!/usr/bin/env python
"""Feature Parity Checker for Meta Probability Inference

Checks that today's enriched predictions include all features required by
the trained LightGBM meta probability models (cover/over). Writes a metrics
JSON under outputs/metrics/feature_parity_<date>.json and prints a concise
summary to stdout. Missing features are listed and counts provided.

Usage:
  python scripts/check_feature_parity.py --date YYYY-MM-DD
"""
from __future__ import annotations
import argparse, json, datetime as dt
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def load_feature_schemas() -> dict[str, list[str]]:
    schemas: dict[str, list[str]] = {}
    # New sidecars
    cover_new = OUT / 'meta_features_cover.json'
    total_new = OUT / 'meta_features_total.json'
    cj = _read_json(cover_new)
    tj = _read_json(total_new)
    if cj.get('features'):
        schemas['cover'] = list(map(str, cj['features']))
    if tj.get('features'):
        schemas['over'] = list(map(str, tj['features']))
    # Legacy sidecars from train_meta_probs_lgbm.py
    cover_legacy = OUT / 'meta_cover_lgbm_features.json'
    total_legacy = OUT / 'meta_over_lgbm_features.json'
    cj2 = _read_json(cover_legacy)
    tj2 = _read_json(total_legacy)
    if not schemas.get('cover') and cj2.get('features'):
        schemas['cover'] = list(map(str, cj2['features']))
    if not schemas.get('over') and tj2.get('features'):
        schemas['over'] = list(map(str, tj2['features']))
    return schemas


def check_parity(df: pd.DataFrame, required: list[str]) -> dict:
    cols = set(map(str, df.columns))
    req = list(map(str, required))
    # Alias mapping to tolerate schema evolution
    aliases = {
        'pred_total_cal': ['pred_total_calibrated'],
        'pred_margin_cal': ['pred_margin_calibrated'],
        'pred_total_model_new': ['pred_total_model_unified', 'pred_total_model'],
        'pred_margin_model_new': ['pred_margin_model'],
        'pred_total_model_weight': ['blend_market_w_total'],
        'pred_margin_sigma': ['sigma_margin_emp', 'sigma_margin_adj'],
        'pred_total_sigma': ['sigma_total_emp', 'sigma_total_adj'],
        'pred_total_pick': ['pick_total', 'pred_total'],
    }
    present = 0
    missing = []
    for r in req:
        if r in cols:
            present += 1
            continue
        alts = aliases.get(r, [])
        if any(a in cols for a in alts):
            present += 1
            continue
        missing.append(r)
    extra = sorted([c for c in cols if c not in req and c.startswith(('p_', 'pred_', 'edge_', 'sigma_', 'blend_'))])
    return {
        'required_count': len(req),
        'present_count': present,
        'missing_count': len(missing),
        'missing': sorted(missing),
        'extra_candidates': extra,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='YYYY-MM-DD (default today)')
    args = ap.parse_args()
    date_str = args.date or dt.date.today().strftime('%Y-%m-%d')

    enriched_path = OUT / f'predictions_unified_enriched_{date_str}.csv'
    if not enriched_path.exists():
        print(f"[warn] Enriched predictions not found: {enriched_path}")
        return 0
    try:
        df = pd.read_csv(enriched_path)
    except Exception as e:
        print(f"[error] Failed reading enriched: {e}")
        return 2

    schemas = load_feature_schemas()
    payload = {
        'date': date_str,
        'path': str(enriched_path),
        'rows': int(len(df)),
        'status': 'ok' if schemas else 'missing_schemas',
        'checks': {},
    }
    for kind, feats in schemas.items():
        payload['checks'][kind] = check_parity(df, feats)

    metrics_dir = OUT / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f'feature_parity_{date_str}.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    # Console summary
    for kind, res in payload['checks'].items():
        print(f"[parity] {kind}: required={res['required_count']} present={res['present_count']} missing={res['missing_count']}")
    if not payload['checks']:
        print("[parity] No schemas discovered; ensure meta trainer wrote sidecars.")
    else:
        miss_tot = sum(r.get('missing_count', 0) for r in payload['checks'].values())
        print(f"[parity] Missing total: {miss_tot} -> {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
