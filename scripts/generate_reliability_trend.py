import json, re, sys
from pathlib import Path
import pandas as pd

OUT = Path('outputs')
DATE_RE = re.compile(r'reliability_(\d{4}-\d{2}-\d{2})\.json$')

def load_daily():
    rows = []
    for p in sorted(OUT.glob('reliability_*.json')):
        m = DATE_RE.match(p.name)
        if not m:
            continue
        date = m.group(1)
        try:
            payload = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        # Flatten simple metrics (guard missing keys)
        row = {'date': date}
        for k in ('ece_total','ece_margin','brier_total','brier_margin','logloss_total','logloss_margin','sharpness_total','sharpness_margin'):
            v = payload.get(k)
            if isinstance(v,(int,float)):
                row[k] = v
        # Generic containers: reliability curves length etc.
        for k in ('reliability_bins_total','reliability_bins_margin'):
            bins = payload.get(k)
            if isinstance(bins,list):
                row[k+'_ct'] = len(bins)
        rows.append(row)
    return pd.DataFrame(rows)

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    for win in (3,7,14,30):
        for col in [c for c in df.columns if c not in ('date') and not c.endswith('_ct')]:
            df[f'{col}_roll{win}'] = df[col].rolling(win, min_periods=1).mean()
    return df

def main():
    df = load_daily()
    df = enrich(df)
    if df.empty:
        print('No reliability files found; nothing to aggregate.')
        return 0
    out_json = OUT / 'reliability_trend.json'
    out_csv = OUT / 'reliability_trend.csv'
    # Convert date back to string for JSON
    jdf = df.copy()
    jdf['date'] = jdf['date'].dt.strftime('%Y-%m-%d')
    out_json.write_text(json.dumps(jdf.to_dict(orient='records'), indent=2), encoding='utf-8')
    df.to_csv(out_csv, index=False)
    print(f'Wrote {out_json} and {out_csv} rows={len(df)}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
