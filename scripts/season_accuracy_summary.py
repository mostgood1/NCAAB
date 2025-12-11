import os
import glob
import json
import pandas as pd

OUT_DIR = os.path.join(os.getcwd(), 'outputs')
RESULTS_DIR = os.path.join(OUT_DIR, 'daily_results')

def load_all_results():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'results_*.csv')))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype=str, low_memory=False)
            df['date'] = df.get('date') if 'date' in df.columns else os.path.basename(f).split('_', 1)[1].split('.csv')[0]
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    # Coerce needed columns
    for c in ['ats_result','ou_result','ats_home_cover','ou_over','home_score','away_score']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce') if c in ['home_score','away_score'] else df[c]
    return df

def compute_accuracy(df: pd.DataFrame):
    out = {}
    if df.empty:
        return {'status':'empty'}
    # Model-implied ATS accuracy: compare predicted margin vs spread to actual cover side
    # Define: home covers if actual_margin > -spread_home; predicted home cover if pred_margin > -spread_home
    for c in ['pred_margin','spread_home','actual_margin']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    mask_ats = df[['pred_margin','spread_home','actual_margin']].notna().all(axis=1)
    ats_df = df.loc[mask_ats].copy()
    ats_df['pred_home_cover'] = ats_df['pred_margin'] > (-ats_df['spread_home'])
    ats_df['actual_home_cover'] = ats_df['actual_margin'] > (-ats_df['spread_home'])
    out['ats_total'] = int(len(ats_df))
    out['ats_correct'] = int((ats_df['pred_home_cover'] == ats_df['actual_home_cover']).sum())
    out['ats_accuracy'] = (out['ats_correct'] / out['ats_total']) if out['ats_total'] else None

    # Model-implied totals accuracy: compare pred_total vs market_total to actual over/under
    for c in ['pred_total','market_total','actual_total']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    mask_tot = df[['pred_total','market_total','actual_total']].notna().all(axis=1)
    tot_df = df.loc[mask_tot].copy()
    tot_df['pred_over'] = tot_df['pred_total'] > tot_df['market_total']
    tot_df['actual_over'] = tot_df['actual_total'] > tot_df['market_total']
    out['totals_total'] = int(len(tot_df))
    out['totals_correct'] = int((tot_df['pred_over'] == tot_df['actual_over']).sum())
    out['totals_accuracy'] = (out['totals_correct'] / out['totals_total']) if out['totals_total'] else None

    # Derivatives: half outcomes if present
    # Derivatives: 1H totals/margins using market_total_1h, spread_home_1h and predicted 1h values if present
    for c in ['pred_total_1h','pred_margin_1h','market_total_1h','spread_home_1h','actual_total_1h','actual_margin_1h']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    m1 = df[['pred_total_1h','market_total_1h','actual_total_1h']].notna().all(axis=1) if {'pred_total_1h','market_total_1h','actual_total_1h'}.issubset(df.columns) else pd.Series(False, index=df.index)
    th_df = df.loc[m1].copy()
    if not th_df.empty:
        th_df['pred_over_1h'] = th_df['pred_total_1h'] > th_df['market_total_1h']
        th_df['actual_over_1h'] = th_df['actual_total_1h'] > th_df['market_total_1h']
        out['totals_1h_total'] = int(len(th_df))
        out['totals_1h_correct'] = int((th_df['pred_over_1h'] == th_df['actual_over_1h']).sum())
        out['totals_1h_accuracy'] = (out['totals_1h_correct'] / out['totals_1h_total']) if out['totals_1h_total'] else None
    m2 = df[['pred_margin_1h','spread_home_1h','actual_margin_1h']].notna().all(axis=1) if {'pred_margin_1h','spread_home_1h','actual_margin_1h'}.issubset(df.columns) else pd.Series(False, index=df.index)
    mh_df = df.loc[m2].copy()
    if not mh_df.empty:
        mh_df['pred_home_cover_1h'] = mh_df['pred_margin_1h'] > (-mh_df['spread_home_1h'])
        mh_df['actual_home_cover_1h'] = mh_df['actual_margin_1h'] > (-mh_df['spread_home_1h'])
        out['ats_1h_total'] = int(len(mh_df))
        out['ats_1h_correct'] = int((mh_df['pred_home_cover_1h'] == mh_df['actual_home_cover_1h']).sum())
        out['ats_1h_accuracy'] = (out['ats_1h_correct'] / out['ats_1h_total']) if out['ats_1h_total'] else None

    # By month breakdown
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['date'].dt.to_period('M').astype(str)
        monthly = {}
        for m, g in df.groupby('month'):
            rec = {}
            if 'ats_home_cover' in g.columns:
                s = pd.to_numeric(g['ats_home_cover'], errors='coerce').dropna()
                rec['ats_total'] = int(len(s)); rec['ats_wins'] = int((s==1).sum())
                rec['ats_win_rate'] = (rec['ats_wins']/rec['ats_total']) if rec['ats_total'] else None
            if 'ou_over' in g.columns:
                s = pd.to_numeric(g['ou_over'], errors='coerce').dropna()
                rec['totals_total'] = int(len(s)); rec['totals_over_hits'] = int((s==1).sum())
                rec['totals_over_rate'] = (rec['totals_over_hits']/rec['totals_total']) if rec['totals_total'] else None
            monthly[m] = rec
        out['monthly'] = monthly
    return out

def main():
    df = load_all_results()
    summary = compute_accuracy(df)
    print(json.dumps(summary, indent=2))
    # Save to outputs for UI/API consumption
    os.makedirs(os.path.join(OUT_DIR, 'metrics'), exist_ok=True)
    with open(os.path.join(OUT_DIR, 'metrics', 'season_accuracy_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()