import argparse
from pathlib import Path
import json
import pandas as pd

def load_df(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', type=str, required=True)
    ap.add_argument('--outputs', type=str, default=str(Path.cwd() / 'outputs'))
    args = ap.parse_args()
    out_dir = Path(args.outputs)
    # Load model
    model = {}
    try:
        model = json.loads((out_dir / 'quantile_model.json').read_text(encoding='utf-8'))
    except Exception:
        pass
    qt = (model.get('residual_quantiles', {}).get('total') or {})
    qm = (model.get('residual_quantiles', {}).get('margin') or {})
    # Load today's enriched unified
    p = out_dir / f'predictions_unified_enriched_{args.date}.csv'
    df = load_df(p)
    if df.empty:
        print('{"error":"missing predictions_unified_enriched"}')
        return
    # Apply quantiles if available; fallback to existing columns
    if qt and {'pred_total'}.issubset(df.columns):
        df['q10_total'] = df['pred_total'] + float(qt.get('q10', 0.0))
        df['q50_total'] = df['pred_total'] + float(qt.get('q50', 0.0))
        df['q90_total'] = df['pred_total'] + float(qt.get('q90', 0.0))
    if qm and {'pred_margin'}.issubset(df.columns):
        df['q10_margin'] = df['pred_margin'] + float(qm.get('q10', 0.0))
        df['q50_margin'] = df['pred_margin'] + float(qm.get('q50', 0.0))
        df['q90_margin'] = df['pred_margin'] + float(qm.get('q90', 0.0))
    # Persist updated file
    df.to_csv(p, index=False)
    print('{"status":"updated","path":"' + str(p) + '"}')

if __name__ == '__main__':
    main()