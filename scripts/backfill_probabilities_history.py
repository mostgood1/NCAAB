import math
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUTS = Path("outputs")

def _norm_cdf(z: pd.Series) -> pd.Series:
    return 0.5 * (1.0 + z.apply(lambda x: float(math.erf(x / math.sqrt(2.0))) if pd.notna(x) else np.nan))

def load_history() -> pd.DataFrame:
    p = OUTPUTS / 'predictions_history.csv'
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if 'game_id' in df.columns:
        df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
    return df

def load_lines() -> pd.DataFrame:
    # Aggregate both historical closing and last lines when available to maximize coverage
    close_files = sorted(OUTPUTS.glob('games_with_closing_*.csv'))
    last_files = sorted(OUTPUTS.glob('games_with_last_*.csv'))
    # Also include per-date align_period snapshots which carry book + market lines
    align_files = sorted(OUTPUTS.glob('align_period_*.csv'))
    frames = []
    if close_files:
        frames.append(pd.concat([pd.read_csv(f) for f in close_files], ignore_index=True))
    if last_files:
        frames.append(pd.concat([pd.read_csv(f) for f in last_files], ignore_index=True))
    if align_files:
        frames.append(pd.concat([pd.read_csv(f) for f in align_files], ignore_index=True))
    if frames:
        raw = pd.concat(frames, ignore_index=True)
    else:
        # Fallback to consolidated snapshots
        p_close = OUTPUTS / 'games_with_closing.csv'
        p_last = OUTPUTS / 'games_with_last.csv'
        src = p_close if p_close.exists() else p_last
        if not src.exists():
            raise FileNotFoundError(src)
        raw = pd.read_csv(src)
    if 'game_id' in raw.columns:
        raw['game_id'] = raw['game_id'].astype(str).str.replace(r'\.0$','', regex=True)
    # Derive helpful unified columns
    if 'market_total' not in raw.columns and 'total' in raw.columns:
        raw['market_total'] = pd.to_numeric(raw['total'], errors='coerce')
    if 'spread_home' not in raw.columns and 'home_spread' in raw.columns:
        raw['spread_home'] = pd.to_numeric(raw['home_spread'], errors='coerce')
    # Restrict to full_game where period available
    if 'period' in raw.columns:
        raw = raw[raw['period'].astype(str).str.lower().isin(['full_game','full','fg'])].copy()
    # Latest per game_id per market using last_update timestamp
    def _sel_latest(df: pd.DataFrame, market: str, col_map: dict[str,str]) -> pd.DataFrame:
        sub = df[df.get('market','').astype(str).str.lower() == market].copy()
        if sub.empty:
            return sub.iloc[0:0]
        if 'last_update' in sub.columns:
            sub['_lu'] = pd.to_datetime(sub['last_update'], errors='coerce')
            sub = sub.sort_values(['game_id','_lu']).drop_duplicates(subset=['game_id'], keep='last').drop(columns=['_lu'])
        else:
            sub = sub.sort_values(['game_id']).drop_duplicates(subset=['game_id'], keep='last')
        out = sub[['game_id']].copy()
        for k,v in col_map.items():
            out[k] = sub[v] if v in sub.columns else np.nan
        return out
    # Use explicit closing column names where available in historical files
    totals = _sel_latest(raw, 'totals', {'market_total':'market_total', 'closing_total':'close_total'})
    spreads = _sel_latest(raw, 'spreads', {'spread_home':'spread_home', 'closing_spread_home':'close_home_spread'})
    merged = totals.merge(spreads, on='game_id', how='outer')
    # closing aliases
    if 'closing_total' not in merged.columns or merged['closing_total'].isna().all():
        merged['closing_total'] = merged.get('market_total')
    if 'closing_spread_home' not in merged.columns or merged['closing_spread_home'].isna().all():
        merged['closing_spread_home'] = merged.get('spread_home')
    return merged

def enrich_probabilities(hist: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    work = hist.copy()
    if 'game_id' not in work.columns:
        print('[backfill] game_id missing in history; cannot merge lines.')
        return work
    # Merge lines
    for col in ['market_total','closing_total','spread_home','closing_spread_home']:
        if col in work.columns:
            work = work.drop(columns=[col])
    work = work.merge(lines, on='game_id', how='left')
    # Totals probabilities
    # Prefer clearly numeric prediction columns; avoid boolean flags like 'pred_total_adjusted'
    mu_candidates_ordered = [
        'pred_total',
        'pred_total_blend',
        'pred_total_seg',
        'pred_total_base',
        'pred_total_mu',
        'pred_total_raw'
    ]
    mu_cols = [c for c in mu_candidates_ordered if c in work.columns]
    if mu_cols:
        mu_series = None
        for c in mu_cols:
            s = pd.to_numeric(work.get(c), errors='coerce')
            if s.notna().any():
                mu_series = s
                break
        # Fallback: if none had numeric values, keep as NaN series
        mu = mu_series if mu_series is not None else pd.Series(np.nan, index=work.index)
        if 'pred_total_sigma' in work.columns:
            sigma = pd.to_numeric(work.get('pred_total_sigma'), errors='coerce')
        else:
            sigma = pd.Series(12.0, index=work.index)
        valid = sigma > 0
        # Use closing_total preferentially; fallback to market_total
        line_series = pd.to_numeric(work.get('closing_total'), errors='coerce') if 'closing_total' in work.columns else pd.Series(np.nan, index=work.index)
        if line_series.isna().all() and 'market_total' in work.columns:
            line_series = pd.to_numeric(work.get('market_total'), errors='coerce')
        p_over = work.get('p_over', pd.Series(np.nan, index=work.index))
        need_over = p_over.isna()
        if not line_series.isna().all():
            z = (line_series - mu) / sigma
            z = z.where(valid)
            p_over_calc = (1.0 - _norm_cdf(z)).where(valid).clip(lower=1e-4, upper=1-1e-4)
            p_under_calc = _norm_cdf(z).where(valid).clip(lower=1e-4, upper=1-1e-4)
            work.loc[need_over, 'p_over'] = p_over_calc[need_over]
            if 'p_under' not in work.columns or work['p_under'].isna().all():
                work['p_under'] = p_under_calc
        # Intervals if missing
        if 'pred_total_low_90' not in work.columns or work['pred_total_low_90'].isna().all():
            z90, z95 = 1.645, 1.96
            work['pred_total_low_90'] = (mu - z90 * sigma).where(valid)
            work['pred_total_high_90'] = (mu + z90 * sigma).where(valid)
            work['pred_total_low_95'] = (mu - z95 * sigma).where(valid)
            work['pred_total_high_95'] = (mu + z95 * sigma).where(valid)
    # Margin cover probability
    if {'pred_margin'}.issubset(work.columns):
        spread_series = pd.to_numeric(work.get('closing_spread_home') if 'closing_spread_home' in work.columns else work.get('spread_home'), errors='coerce')
        pm = pd.to_numeric(work.get('pred_margin'), errors='coerce')
        if 'pred_margin_sigma' in work.columns:
            sig_m = pd.to_numeric(work.get('pred_margin_sigma'), errors='coerce')
        else:
            sig_m = pd.Series(11.0, index=work.index)
            work['pred_margin_sigma'] = sig_m
        valid_m = sig_m > 0
        if valid_m.any():
            z_m = (spread_series - pm) / sig_m
            p_home_cover = (1.0 - _norm_cdf(z_m)).where(valid_m).clip(lower=1e-4, upper=1-1e-4)
            if 'p_home_cover_dist' not in work.columns or work['p_home_cover_dist'].isna().all():
                work['p_home_cover_dist'] = p_home_cover
    return work

def main():
    hist = load_history()
    lines = load_lines()
    # Debug: show lines availability
    mt_count = int(pd.to_numeric(lines.get('market_total', pd.Series(dtype=float)), errors='coerce').notna().sum())
    ct_count = int(pd.to_numeric(lines.get('closing_total', pd.Series(dtype=float)), errors='coerce').notna().sum())
    print(f"[backfill] Lines coverage: market_total_non_null={mt_count} closing_total_non_null={ct_count} rows={len(lines)}")
    enriched = enrich_probabilities(hist, lines)
    out = OUTPUTS / 'predictions_history_enriched.csv'
    enriched.to_csv(out, index=False)
    print(f"[backfill] Wrote enriched history -> {out} rows={len(enriched)}")
    # Simple stats
    stats = {
        'rows': len(enriched),
        'p_over_populated': int(enriched.get('p_over', pd.Series(dtype=float)).notna().sum()),
        'p_home_cover_dist_populated': int(enriched.get('p_home_cover_dist', pd.Series(dtype=float)).notna().sum()),
        'missing_market_total_rows': int(enriched.get('market_total', pd.Series(dtype=float)).isna().sum()) if 'market_total' in enriched.columns else -1,
    }
    print(f"[backfill] Stats: {stats}")

if __name__ == '__main__':
    main()
