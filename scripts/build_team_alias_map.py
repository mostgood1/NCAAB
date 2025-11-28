import pandas as pd
from pathlib import Path
import json

OUT = Path('outputs')
ALIAS_JSON = OUT / 'team_alias_map.json'

NORMALIZE_RULES = [
    (" university", ""), (" univ", ""), (" state", " st"), ("&", " and "), ("-", " "), ("  ", " ")
]

def norm(name: str) -> str:
    if not isinstance(name,str):
        return ''
    n = name.lower().strip()
    for a,b in NORMALIZE_RULES:
        n = n.replace(a,b)
    return ' '.join(n.split())

def main():
    enriched_files = list(OUT.glob('predictions_unified_enriched_*.csv'))
    records = []
    for fp in enriched_files[-120:]:  # limit to recent window
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if df.empty or not {'home_team','away_team'}.issubset(df.columns):
            continue
        for col in ['home_team','away_team']:
            for v in df[col].dropna().unique().tolist():
                records.append(v)
    ser = pd.Series(records)
    base_norm = ser.apply(norm)
    # Group by normalized form; pick most frequent original variant
    alias_map = {}
    for nform, grp in ser.groupby(base_norm):
        # frequency count of original variants
        top = grp.value_counts().index.tolist()
        if not top:
            continue
        canonical = top[0]
        variants = list(dict.fromkeys(top))
        alias_map[canonical] = {'normalized': nform, 'variants': variants, 'count': len(grp)}
    with open(ALIAS_JSON,'w',encoding='utf-8') as fh:
        json.dump(alias_map, fh, indent=2)
    print('Alias map written:', ALIAS_JSON, 'canonical_count:', len(alias_map))

if __name__ == '__main__':
    main()
