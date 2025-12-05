import argparse
from pathlib import Path
import pandas as pd
import json

def load_df(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def compute_drift(limit_days: int, outputs: Path) -> dict:
    # Inputs
    preds_hist = load_df(outputs / 'predictions_history_enriched.csv')
    closing = load_df(outputs / 'games_with_closing.csv')
    last = load_df(outputs / 'games_with_last.csv')
    today_odds = load_df(outputs / 'games_with_odds_today.csv')
    if preds_hist.empty:
        return {"error": "missing predictions_history_enriched.csv"}
    # Normalize keys
    # Normalize game_id types across frames
    for df in (preds_hist, closing, last):
        if not df.empty and 'game_id' in df.columns:
            # Coerce to string then back to int where possible to align types
            df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
            try:
                df['game_id'] = df['game_id'].astype(int)
            except Exception:
                pass
    # Join closing lines
    # games_with_closing uses 'date_line' not 'date'; predictions_history_enriched has 'date'
    # We'll join on game_id and align dates loosely by available fields
    join_cols = ['game_id']
    have_cols_close = [c for c in ['closing_total','closing_spread_home'] if c in closing.columns]
    have_cols_last = [c for c in ['last_total','last_spread_home'] if c in last.columns]
    have_cols_today = [c for c in ['market_total','spread_home'] if c in today_odds.columns]
    merged = preds_hist.merge(closing[join_cols + have_cols_close], on=join_cols, how='left')
    if not last.empty and have_cols_last:
        merged = merged.merge(last[join_cols + have_cols_last], on=join_cols, how='left')
    if not today_odds.empty and have_cols_today:
        merged = merged.merge(today_odds[join_cols + have_cols_today], on=join_cols, how='left')
    # Filter recent window
    # Build a reference date for windowing
    ref_date = None
    if 'date' in preds_hist.columns:
        preds_hist['_date'] = pd.to_datetime(preds_hist['date'], errors='coerce')
        ref_date = preds_hist['_date'].max()
    elif 'date_line' in closing.columns:
        closing['_date_line'] = pd.to_datetime(closing['date_line'], errors='coerce')
        ref_date = closing['_date_line'].max()
    if ref_date is not None:
        cutoff = ref_date - pd.Timedelta(days=limit_days)
        # Filter using whichever date field exists
        if 'date' in merged.columns:
            merged['_date'] = pd.to_datetime(merged['date'], errors='coerce')
            merged = merged[merged['_date'] >= cutoff]
        elif 'date_line' in merged.columns:
            merged['_date_line'] = pd.to_datetime(merged['date_line'], errors='coerce')
            merged = merged[merged['_date_line'] >= cutoff]
    # Compute deltas
    deltas = {}
    if {'pred_total_calibrated','closing_total'}.issubset(merged.columns):
        dtot = merged['pred_total_calibrated'] - merged['closing_total']
        deltas['total'] = {
            'n': int(dtot.dropna().shape[0]),
            'mean': float(dtot.dropna().mean()) if dtot.dropna().shape[0] else 0.0,
            'median': float(dtot.dropna().median()) if dtot.dropna().shape[0] else 0.0,
            'std': float(dtot.dropna().std()) if dtot.dropna().shape[0] else 0.0,
            'p95_abs': float(dtot.dropna().abs().quantile(0.95)) if dtot.dropna().shape[0] else 0.0,
        }
        merged['delta_total'] = dtot
    elif {'pred_total_calibrated','last_total'}.issubset(merged.columns):
        dtot = merged['pred_total_calibrated'] - merged['last_total']
        deltas['total_last'] = {
            'n': int(dtot.dropna().shape[0]),
            'mean': float(dtot.dropna().mean()) if dtot.dropna().shape[0] else 0.0,
            'median': float(dtot.dropna().median()) if dtot.dropna().shape[0] else 0.0,
            'std': float(dtot.dropna().std()) if dtot.dropna().shape[0] else 0.0,
            'p95_abs': float(dtot.dropna().abs().quantile(0.95)) if dtot.dropna().shape[0] else 0.0,
        }
        merged['delta_total_last'] = dtot
    elif {'pred_total_calibrated','market_total'}.issubset(merged.columns):
        dtot = merged['pred_total_calibrated'] - merged['market_total']
        deltas['total_market_today'] = {
            'n': int(dtot.dropna().shape[0]),
            'mean': float(dtot.dropna().mean()) if dtot.dropna().shape[0] else 0.0,
            'median': float(dtot.dropna().median()) if dtot.dropna().shape[0] else 0.0,
            'std': float(dtot.dropna().std()) if dtot.dropna().shape[0] else 0.0,
            'p95_abs': float(dtot.dropna().abs().quantile(0.95)) if dtot.dropna().shape[0] else 0.0,
        }
        merged['delta_total_market_today'] = dtot
    if {'pred_margin_calibrated','closing_spread_home'}.issubset(merged.columns):
        dmar = merged['pred_margin_calibrated'] - merged['closing_spread_home']
        deltas['margin'] = {
            'n': int(dmar.dropna().shape[0]),
            'mean': float(dmar.dropna().mean()) if dmar.dropna().shape[0] else 0.0,
            'median': float(dmar.dropna().median()) if dmar.dropna().shape[0] else 0.0,
            'std': float(dmar.dropna().std()) if dmar.dropna().shape[0] else 0.0,
            'p95_abs': float(dmar.dropna().abs().quantile(0.95)) if dmar.dropna().shape[0] else 0.0,
        }
        merged['delta_margin'] = dmar
    elif {'pred_margin_calibrated','last_spread_home'}.issubset(merged.columns):
        dmar = merged['pred_margin_calibrated'] - merged['last_spread_home']
        deltas['margin_last'] = {
            'n': int(dmar.dropna().shape[0]),
            'mean': float(dmar.dropna().mean()) if dmar.dropna().shape[0] else 0.0,
            'median': float(dmar.dropna().median()) if dmar.dropna().shape[0] else 0.0,
            'std': float(dmar.dropna().std()) if dmar.dropna().shape[0] else 0.0,
            'p95_abs': float(dmar.dropna().abs().quantile(0.95)) if dmar.dropna().shape[0] else 0.0,
        }
        merged['delta_margin_last'] = dmar
    elif {'pred_margin_calibrated','spread_home'}.issubset(merged.columns):
        dmar = merged['pred_margin_calibrated'] - merged['spread_home']
        deltas['margin_market_today'] = {
            'n': int(dmar.dropna().shape[0]),
            'mean': float(dmar.dropna().mean()) if dmar.dropna().shape[0] else 0.0,
            'median': float(dmar.dropna().median()) if dmar.dropna().shape[0] else 0.0,
            'std': float(dmar.dropna().std()) if dmar.dropna().shape[0] else 0.0,
            'p95_abs': float(dmar.dropna().abs().quantile(0.95)) if dmar.dropna().shape[0] else 0.0,
        }
        merged['delta_margin_market_today'] = dmar
    # Persist daily
    daily_cols = [c for c in ['date','date_line','game_id','home_team','away_team','pred_total_calibrated','closing_total','delta_total','last_total','delta_total_last','market_total','delta_total_market_today','pred_margin_calibrated','closing_spread_home','delta_margin','last_spread_home','delta_margin_last','spread_home','delta_margin_market_today'] if c in merged.columns]
    (outputs / 'drift').mkdir(parents=True, exist_ok=True)
    daily_path = outputs / 'drift' / 'drift_market_daily.csv'
    try:
        merged[daily_cols].to_csv(daily_path, index=False)
    except Exception:
        pass
    return {
        'window_days': limit_days,
        'summary': deltas,
        'daily_path': str(daily_path),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit-days', type=int, default=45)
    ap.add_argument('--outputs', type=str, default=str(Path.cwd() / 'outputs'))
    args = ap.parse_args()
    res = compute_drift(args.limit_days, Path(args.outputs))
    out = Path(args.outputs) / 'drift' / 'drift_market_summary.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()