import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path('outputs')

# Simple absolute residual conformal intervals using historical residuals.
# For each target (total, margin) we compute quantile of |residual| from calibration window.

def load_history(enriched_files, window_days=None):
    rows = []
    for f in enriched_files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            # Derive date from filename if absent
            if 'date' not in df.columns:
                date_part = f.name.replace('predictions_unified_enriched_','').replace('.csv','')
                df['date'] = date_part
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    all_df = pd.concat(rows, ignore_index=True)
    if window_days and 'date' in all_df.columns:
        all_df['date_dt'] = pd.to_datetime(all_df['date'], errors='coerce')
        cutoff = all_df['date_dt'].max() - pd.Timedelta(days=window_days)
        all_df = all_df[all_df['date_dt'] >= cutoff]
    return all_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', type=str, help='Target date YYYY-MM-DD (default today)')
    ap.add_argument('--window-days', type=int, default=30, help='Calibration window length')
    ap.add_argument('--alpha', type=float, default=0.1, help='Miscoverage rate (e.g., 0.1 for 90% intervals)')
    ap.add_argument('--out-dir', type=str, default='outputs')
    ap.add_argument('--enriched-file', type=str, help='Optional enriched file for target date intervals embedding')
    args = ap.parse_args()

    from datetime import datetime
    date_str = args.date or datetime.now().strftime('%Y-%m-%d')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched_files = sorted(out_dir.glob('predictions_unified_enriched_*.csv'))
    hist = load_history(enriched_files, window_days=args.window_days)

    artifact = {
        'date': date_str,
        'window_days': args.window_days,
        'alpha': args.alpha,
        'count_hist_games': int(len(hist))
    }

    # Need actual scores to compute residuals
    if {'home_score','away_score'}.issubset(hist.columns):
        hs = pd.to_numeric(hist['home_score'], errors='coerce')
        as_ = pd.to_numeric(hist['away_score'], errors='coerce')
        actual_total = hs + as_
        actual_margin = hs - as_
        mu_total = pd.to_numeric(hist.get('pred_total_model', hist.get('pred_total')), errors='coerce')
        mu_margin = pd.to_numeric(hist.get('pred_margin_model', hist.get('pred_margin')), errors='coerce')
        resid_total = actual_total - mu_total
        resid_margin = actual_margin - mu_margin
        # Absolute residuals
        abs_total = resid_total.abs().dropna()
        abs_margin = resid_margin.abs().dropna()
        if len(abs_total):
            q_total = np.quantile(abs_total, 1 - args.alpha)
            artifact['total_abs_quantile'] = float(q_total)
        if len(abs_margin):
            q_margin = np.quantile(abs_margin, 1 - args.alpha)
            artifact['margin_abs_quantile'] = float(q_margin)
    else:
        artifact['residuals_missing'] = True

    # Optional interval embedding back into target enriched file
    if args.enriched_file:
        ef = Path(args.enriched_file)
        if ef.exists():
            try:
                df_t = pd.read_csv(ef)
                mt_mu = pd.to_numeric(df_t.get('pred_total_model', df_t.get('pred_total')), errors='coerce')
                mm_mu = pd.to_numeric(df_t.get('pred_margin_model', df_t.get('pred_margin')), errors='coerce')
                if 'total_abs_quantile' in artifact:
                    w = artifact['total_abs_quantile']
                    df_t['pred_total_lower_conf'] = mt_mu - w
                    df_t['pred_total_upper_conf'] = mt_mu + w
                if 'margin_abs_quantile' in artifact:
                    w2 = artifact['margin_abs_quantile']
                    df_t['pred_margin_lower_conf'] = mm_mu - w2
                    df_t['pred_margin_upper_conf'] = mm_mu + w2
                df_t.to_csv(ef, index=False)
                artifact['intervals_embedded'] = True
            except Exception as e:
                artifact['interval_embed_error'] = str(e)

    (out_dir / f'residuals_{date_str}.json').write_text(json.dumps(artifact, indent=2))
    print('Conformal residual artifact written:', out_dir / f'residuals_{date_str}.json')

if __name__ == '__main__':
    main()
