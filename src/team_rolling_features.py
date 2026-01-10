from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()

def compute_rolling(bs: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    if bs.empty:
        return pd.DataFrame()
    df = bs.copy()
    # Normalize identifiers
    for c in ('date', 'home_team', 'away_team'):
        if c in df.columns:
            df[c] = df[c].astype(str)
    # Build team-date rows from augmented per-game metrics if present
    # Expect columns: home_pace, home_ts, home_3p_rate, home_to_rate, home_drb_rate, away_* equivalents
    rows = []
    for _, r in df.iterrows():
        d = str(r.get('date'))
        ht = str(r.get('home_team'))
        at = str(r.get('away_team'))
        if ht:
            rows.append({'team': ht, 'date': d,
                         'pace': float(r.get('home_pace') or np.nan),
                         'ts': float(r.get('home_ts') or np.nan),
                         'rate3p': float(r.get('home_3p_rate') or np.nan),
                         'to_rate': float(r.get('home_to_rate') or np.nan),
                         'drb_rate': float(r.get('home_drb_rate') or np.nan)})
        if at:
            rows.append({'team': at, 'date': d,
                         'pace': float(r.get('away_pace') or np.nan),
                         'ts': float(r.get('away_ts') or np.nan),
                         'rate3p': float(r.get('away_3p_rate') or np.nan),
                         'to_rate': float(r.get('away_to_rate') or np.nan),
                         'drb_rate': float(r.get('away_drb_rate') or np.nan)})
    td = pd.DataFrame(rows)
    if td.empty:
        return pd.DataFrame()
    # Ensure datetime sorting
    td['date'] = pd.to_datetime(td['date'], errors='coerce')
    td = td.dropna(subset=['team', 'date'])
    td = td.sort_values(['team', 'date'])
    # Rolling window per team
    def roll(col: str):
        return td.groupby('team')[col].transform(lambda s: s.rolling(window, min_periods=1).mean())
    td['pace5'] = roll('pace')
    td['ts5'] = roll('ts')
    td['rate3p5'] = roll('rate3p')
    td['to_rate5'] = roll('to_rate')
    td['drb_rate5'] = roll('drb_rate')
    # Output
    out = td[['team', 'date', 'pace5', 'ts5', 'rate3p5', 'to_rate5', 'drb_rate5']].copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    return out

def run(window: int = 5, recent: int = 120) -> dict:
    # Prefer per-date augmented feature files for richer coverage
    files = sorted([p for p in OUT.glob('features_*_augmented.csv')])
    sources = []
    for p in files:
        try:
            df = pd.read_csv(p)
            sources.append(df)
        except Exception:
            continue
    if not sources:
        # Fallback to consolidated recent augmented
        aug_recent = _safe_read_csv(OUT / 'features_augmented_recent.csv')
        if aug_recent.empty:
            return {'status': 'no_source'}
        sources = [aug_recent]
    src = pd.concat(sources, ignore_index=True)
    # Filter by recent days based on date if present
    if 'date' in src.columns:
        try:
            src['date'] = pd.to_datetime(src['date'], errors='coerce')
            cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=recent)
            src = src[src['date'] >= cutoff]
        except Exception:
            pass
    out = compute_rolling(src, window=window)
    if out.empty:
        return {'status': 'empty_output'}
    path = OUT / 'team_rolling_features_recent.csv'
    out.to_csv(path, index=False)
    return {'file': str(path), 'rows': int(len(out))}

def main():
    ap = argparse.ArgumentParser(description='Compute team rolling features from augmented metrics')
    ap.add_argument('--window', type=int, default=5, help='Rolling window length')
    ap.add_argument('--recent', type=int, default=120, help='Days back to include')
    args = ap.parse_args()
    res = run(window=args.window, recent=args.recent)
    print(res)

if __name__ == '__main__':
    main()
