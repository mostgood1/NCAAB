from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply market-aware blend and guardrails to enriched predictions")
    p.add_argument("--date", help="Slate date YYYY-MM-DD; used to locate enriched CSV", required=True)
    p.add_argument("--w-market-total", type=float, default=0.65, dest="w_market_total", help="Weight for market in total blend [0,1]")
    p.add_argument("--w-market-margin", type=float, default=0.55, dest="w_market_margin", help="Weight for market in margin blend [0,1]")
    p.add_argument("--thr-total", type=float, default=20.0, dest="thr_total", help="Mismatch threshold absolute points for totals")
    p.add_argument("--thr-margin", type=float, default=8.0, dest="thr_margin", help="Mismatch threshold absolute points for spreads/margins")
    return p.parse_args()


def load_enriched(date: str) -> pd.DataFrame:
    path = Path(f"outputs/predictions_unified_enriched_{date}.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing enriched CSV: {path}")
    return pd.read_csv(path)


def to_float(s):
    try:
        return float(s)
    except Exception:
        return float('nan')


def apply_blend(df: pd.DataFrame, w_total: float, w_margin: float, thr_total: float, thr_margin: float) -> tuple[pd.DataFrame, dict]:
    # Choose model columns
    model_total_col = 'pred_total_calibrated' if 'pred_total_calibrated' in df.columns else 'pred_total'
    model_margin_col = 'pred_margin_calibrated' if 'pred_margin_calibrated' in df.columns else 'pred_margin'

    # Market columns
    mt_candidates = ['market_total', '_market_total_from_odds', '_market_total_from_odds_fallback', '_market_total_pair_med', 'closing_total']
    if 'market_total' not in df.columns:
        for c in mt_candidates:
            if c in df.columns:
                df['market_total'] = df[c]
                break
    if 'market_total' not in df.columns:
        df['market_total'] = pd.NA

    # market spread home -> market-implied margin (home minus away)
    spread_col = None
    for c in ['spread_home', 'closing_spread_home']:
        if c in df.columns:
            spread_col = c
            break
    market_margin = None
    if spread_col is not None:
        # If home is -x, market margin (home-away) is +x
        market_margin = -pd.to_numeric(df[spread_col], errors='coerce')

    m_total = pd.to_numeric(df['market_total'], errors='coerce')
    p_total = pd.to_numeric(df.get(model_total_col), errors='coerce')
    p_margin = pd.to_numeric(df.get(model_margin_col), errors='coerce')

    pred_total_blend = (1 - w_total) * p_total + w_total * m_total
    df['pred_total_market_blend'] = pred_total_blend
    df['blend_market_w_total'] = w_total

    if market_margin is not None:
        pred_margin_blend = (1 - w_margin) * p_margin + w_margin * market_margin
        df['pred_margin_market_blend'] = pred_margin_blend
        df['blend_market_w_margin'] = w_margin

    # Mismatch flags
    df['flag_market_total_mismatch'] = (p_total.sub(m_total).abs() > thr_total)
    if market_margin is not None:
        df['flag_market_margin_mismatch'] = (p_margin.sub(market_margin).abs() > thr_margin)
    else:
        df['flag_market_margin_mismatch'] = False

    # Guardrails: zero Kelly when mismatched
    for col in ['kelly_fraction_total', 'kelly_frac_spread']:
        if col in df.columns:
            guard_col = f'{col}_guarded'
            mask = df['flag_market_total_mismatch'] | df['flag_market_margin_mismatch']
            df[guard_col] = df[col]
            df.loc[mask, guard_col] = 0.0

    # Summary
    n = len(df)
    n_mismatch_total = int(df['flag_market_total_mismatch'].sum())
    n_mismatch_margin = int(df['flag_market_margin_mismatch'].sum())
    return df, {
        'rows': n,
        'w_market_total': w_total,
        'w_market_margin': w_margin,
        'thr_total': thr_total,
        'thr_margin': thr_margin,
        'mismatch_total_rows': n_mismatch_total,
        'mismatch_margin_rows': n_mismatch_margin,
    }


def main():
    args = parse_args()
    date = args.date
    df = load_enriched(date)
    df2, info = apply_blend(df, args.w_market_total, args.w_market_margin, args.thr_total, args.thr_margin)

    # Write back enriched CSV in-place
    out_path = Path(f"outputs/predictions_unified_enriched_{date}.csv")
    df2.to_csv(out_path, index=False)

    # Diagnostics JSON
    diag_dir = Path('outputs/diagnostics')
    diag_dir.mkdir(parents=True, exist_ok=True)
    info.update({
        'date': date,
        'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
        'path': str(out_path),
    })
    with open(diag_dir / f'market_blend_{date}.json', 'w') as f:
        json.dump(info, f, indent=2)
    print(json.dumps({'status': 'ok', **info}))


if __name__ == '__main__':
    main()
