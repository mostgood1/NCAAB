import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
SRC = OUT / 'games_with_last.csv'
DST = OUT / 'games_with_last_normalized.csv'

MAP = {
    'total': ['closing_total','total','odds_total','market_total'],
    'spread': ['closing_spread','spread','odds_spread','market_spread','home_spread_pred'],
    'home_ml_prob': ['home_ml_prob','market_moneyline_home_prob'],
    'home_ml_price': ['home_ml_price'],
}


def _read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def american_to_prob(price):
    try:
        price = float(price)
    except Exception:
        return np.nan
    if price > 0:
        return 100.0 / (price + 100.0)
    elif price < 0:
        return abs(price) / (abs(price) + 100.0)
    return np.nan


def main():
    df = _read_csv(SRC)
    if df.empty:
        raise SystemExit('Missing games_with_last.csv')
    # Normalize IDs
    if 'game_id' in df.columns:
        df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    # Build normalized columns
    for key, candidates in MAP.items():
        vals = np.nan
        for c in candidates:
            if c in df.columns:
                vals = df[c]
                break
        df[key] = vals if isinstance(vals, pd.Series) else np.nan
    # Derive home_ml_prob from price if missing
    if df['home_ml_prob'].isna().any() and 'home_ml_price' in df.columns:
        m = df['home_ml_prob'].copy()
        mask = m.isna()
        df.loc[mask, 'home_ml_prob'] = df.loc[mask, 'home_ml_price'].apply(american_to_prob)
    # Keep essential columns
    keep = [c for c in ['date','game_id','total','spread','home_ml_prob'] if c in df.columns]
    if 'home_ml_prob' not in keep:
        df['home_ml_prob'] = np.nan
        keep.append('home_ml_prob')
    if 'total' not in keep:
        df['total'] = np.nan
        keep.append('total')
    if 'spread' not in keep:
        df['spread'] = np.nan
        keep.append('spread')
    out = df[keep].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST, index=False)
    print('Wrote normalized odds to', str(DST))


if __name__ == '__main__':
    main()
