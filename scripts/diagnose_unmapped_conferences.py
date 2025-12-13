import os
import glob
import pandas as pd
from pathlib import Path

ROOT = Path(os.getcwd())
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
DAILY = OUT / 'daily_results'
DIAG = OUT / 'diagnostics'
DIAG.mkdir(parents=True, exist_ok=True)

# Fallback normalize: lowercase, strip punctuation/spaces
import re

def simple_slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]", "-", str(name).lower())
    s = re.sub(r"-+", "-", s).strip('-')
    return s

# Load d1/conferences
conf_map: dict[str, str] = {}
for p in [DATA / 'd1_conferences.csv', DATA / 'conferences.csv']:
    if p.exists():
        try:
            df = pd.read_csv(p)
            cols = {c.lower(): c for c in df.columns}
            team_col = cols.get('team') or cols.get('name') or cols.get('school') or list(df.columns)[0]
            conf_col = cols.get('conference') or list(df.columns)[-1]
            for r in df[[team_col, conf_col]].dropna().itertuples(index=False):
                raw = str(getattr(r, team_col))
                conf = str(getattr(r, conf_col)).strip()
                conf_map[raw] = conf
                conf_map[simple_slug(raw)] = conf
        except Exception:
            pass

# Load provider aliases and team_map for additional slugging
alias_map: dict[str, str] = {}
for p in [DATA / 'provider_aliases.csv', DATA / 'team_map.csv']:
    if p.exists():
        try:
            df = pd.read_csv(p)
            cols = {c.lower(): c for c in df.columns}
            raw_col = cols.get('alias') or cols.get('raw') or list(df.columns)[0]
            canon_col = cols.get('canonical') or cols.get('canon') or list(df.columns)[-1]
            for r in df[[raw_col, canon_col]].dropna().itertuples(index=False):
                alias = str(getattr(r, raw_col))
                canon = str(getattr(r, canon_col))
                alias_map[alias] = canon
                alias_map[simple_slug(alias)] = canon
        except Exception:
            pass

# Aggregate daily results
files = sorted(glob.glob(str(DAILY / 'results_*.csv')))
if not files:
    print('No daily results found in outputs/daily_results')
    raise SystemExit(0)

df = pd.concat([pd.read_csv(f) for f in files if os.path.exists(f)], ignore_index=True)

teams_cols = [c for c in ['home_team','home','away_team','away'] if c in df.columns]
if not teams_cols:
    print('No team columns found in daily results')
    raise SystemExit(0)

rows = []
for col in teams_cols:
    for name in df[col].astype(str).dropna().unique():
        # Try raw
        conf = conf_map.get(name)
        if not conf:
            # Try alias to canonical
            canon = alias_map.get(name) or alias_map.get(simple_slug(name))
            key = canon or name
            conf = conf_map.get(key) or conf_map.get(simple_slug(key))
        if not conf:
            rows.append({'team': name, 'slug': simple_slug(name)})

# Aggregate
agg = {}
for r in rows:
    k = r['slug']
    agg[k] = agg.get(k, 0) + 1

out_df = pd.DataFrame(sorted(([k, v] for k, v in agg.items()), key=lambda x: x[1], reverse=True), columns=['slug','count'])
out_path = DIAG / 'unmapped_teams.csv'
out_df.to_csv(out_path, index=False)
print(f'Wrote {out_path} with {len(out_df)} unmapped slugs')

# Also emit a sample JSON for quick review
sample = {
    'total_unmapped': int(len(out_df)),
    'top_20': out_df.head(20).to_dict(orient='records')
}
(pd.Series(sample)).to_json(DIAG / 'unmapped_teams.json')
print('Wrote diagnostics JSON')
