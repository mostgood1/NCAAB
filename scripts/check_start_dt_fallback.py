import pandas as pd
import datetime as dt
from typing import Any
import sys

TZ_ABBR_MAP = {
    'UTC': 0, 'Z': 0,
    'HST': -10, 'AKST': -9,
    'PST': -8, 'PDT': -7,
    'MST': -7, 'MDT': -6,
    'CST': -6, 'CDT': -5,
    'EST': -5, 'EDT': -4,
}

def derive_iso(row: dict[str, Any]) -> str | None:
    loc = (row.get('start_time_local') or '').strip()
    abbr = ((row.get('start_tz_abbr') or '')).strip().upper()
    if loc:
        parts = loc.split()
        if len(parts) >= 2 and abbr in TZ_ABBR_MAP:
            date, time = parts[0], parts[1]
            off = TZ_ABBR_MAP[abbr]
            iso_local = f"{date}T{time}:00" + ("Z" if off == 0 else ("+" if off > 0 else "-") + str(abs(off)).rjust(2, '0') + ":00")
            try:
                d = pd.to_datetime(iso_local, errors='coerce', utc=True)
                if pd.notna(d):
                    return d.strftime('%Y-%m-%dT%H:%M:%SZ')
            except Exception:
                pass
    iso = row.get('start_time_iso')
    if iso:
        try:
            d = pd.to_datetime(str(iso).replace('Z','+00:00'), errors='coerce', utc=True)
            if pd.notna(d):
                return d.strftime('%Y-%m-%dT%H:%M:%SZ')
        except Exception:
            pass
    st = row.get('start_time')
    if st:
        try:
            s = str(st)
            d = pd.to_datetime(s.replace('Z','+00:00'), errors='coerce', utc=True)
            if pd.notna(d):
                return d.strftime('%Y-%m-%dT%H:%M:%SZ')
        except Exception:
            pass
    return None

if __name__ == '__main__':
    date_q = sys.argv[1] if len(sys.argv) > 1 else '2025-12-02'
    path = f'outputs/predictions_unified_enriched_{date_q}.csv'
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print({'error': str(e)})
        sys.exit(1)
    if 'date' in df.columns:
        df = df[df['date'].astype(str).str.strip() == date_q]
    iso_vals = []
    for r in df.to_dict(orient='records'):
        iso_vals.append(derive_iso(r))
    sd = pd.to_datetime(pd.Series(iso_vals).astype(str).str.replace('Z','+00:00', regex=False), errors='coerce', utc=True)
    nan_idx = list(sd[sd.isna()].index)
    summary = {
        'rows_considered': int(len(df)),
        'nan_count': int(len(nan_idx)),
    }
    ids = []
    if 'game_id' in df.columns:
        gid = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        ids = list(gid.iloc[nan_idx])
    print({'summary': summary, 'nan_game_ids': ids[:50]})
