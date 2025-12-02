import argparse
import datetime as dt
from typing import Any
import pandas as pd
from zoneinfo import ZoneInfo

TZ_ABBR_MAP = {
    'UTC': 0, 'Z': 0,
    'HST': -10, 'AKST': -9,
    'PST': -8, 'PDT': -7,
    'MST': -7, 'MDT': -6,
    'CST': -6, 'CDT': -5,
    'EST': -5, 'EDT': -4,
}


def _derive_iso(row: dict[str, Any]) -> str | None:
    """Canonical ISO derivation with preference for local+abbr, else ISO, else start_time.
    Returns a UTC Z string like 'YYYY-MM-DDTHH:MM:SSZ'.
    """
    try:
        loc = (row.get('start_time_local') or '').strip()
        abbr = ((row.get('start_tz_abbr') or '')).strip().upper()
        if loc:
            parts = loc.split()
            if len(parts) >= 2 and abbr in TZ_ABBR_MAP:
                date, time = parts[0], parts[1]
                off = TZ_ABBR_MAP[abbr]
                iso_local = f"{date}T{time}:00" + ("Z" if off == 0 else ("+" if off > 0 else "-") + str(abs(off)).rjust(2, '0') + ":00")
                d = pd.to_datetime(iso_local, errors='coerce', utc=True)
                if pd.notna(d):
                    return d.strftime('%Y-%m-%dT%H:%M:%SZ')
        # fallback: start_time_iso
        iso = row.get('start_time_iso')
        if iso:
            d = pd.to_datetime(str(iso).replace('Z','+00:00'), errors='coerce', utc=True)
            if pd.notna(d):
                return d.strftime('%Y-%m-%dT%H:%M:%SZ')
        # fallback: start_time (assume UTC or explicit offset)
        st = row.get('start_time')
        if st:
            s = str(st)
            d = pd.to_datetime(s.replace('Z','+00:00'), errors='coerce', utc=True)
            if pd.notna(d):
                return d.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception:
        pass
    return None


def _apply_defaults(row: dict[str, Any]) -> dict[str, Any]:
    """Apply defaults and persist stable UTC start datetime for downstream usage."""
    # Default missing tz abbr for display to CST
    abbr = row.get('start_tz_abbr')
    try:
        if abbr is None or (isinstance(abbr, float) and pd.isna(abbr)) or str(abbr).strip() == '' or str(abbr).strip().lower() == 'nan':
            row['start_tz_abbr'] = 'CST'
    except Exception:
        row['start_tz_abbr'] = 'CST'
    # Derive ISO if missing
    if not row.get('start_time_iso'):
        iso = _derive_iso(row)
        if iso:
            row['start_time_iso'] = iso
    # Persist _start_dt when missing or NaN
    _sdt = row.get('_start_dt')
    _sdt_missing = False
    try:
        _sdt_missing = (_sdt is None) or (isinstance(_sdt, float) and pd.isna(_sdt)) or (str(_sdt).strip() == '' or str(_sdt).strip().lower() == 'nan')
    except Exception:
        _sdt_missing = True
    if _sdt_missing:
        iso2 = _derive_iso(row)
        if iso2:
            try:
                row['_start_dt'] = pd.to_datetime(str(iso2).replace('Z','+00:00'), errors='coerce', utc=True)
            except Exception:
                row['_start_dt'] = iso2
    # Ensure display_date exists (prefer local date, else Central from ISO)
    if not row.get('display_date'):
        loc = (row.get('start_time_local') or '').strip()
        if loc and len(loc.split()) >= 2:
            row['display_date'] = loc.split()[0]
        else:
            iso = row.get('start_time_iso')
            if iso:
                try:
                    ts = pd.to_datetime(str(iso).replace('Z','+00:00'), errors='coerce', utc=True)
                    if pd.notna(ts):
                        ts_c = ts.tz_convert(ZoneInfo('America/Chicago'))
                        row['display_date'] = ts_c.strftime('%Y-%m-%d')
                except Exception:
                    pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('date', nargs='?', default=dt.date.today().strftime('%Y-%m-%d'))
    ap.add_argument('--inplace', action='store_true', help='Write back to original file')
    args = ap.parse_args()

    src = f'outputs/predictions_unified_enriched_{args.date}.csv'
    df = pd.read_csv(src)
    # Normalize game_id type
    if 'game_id' in df.columns:
        df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    # Apply only for the selected slate date if date column exists
    if 'date' in df.columns:
        df_sel = df[df['date'].astype(str).str.strip() == args.date].copy()
    else:
        df_sel = df.copy()

    recs = []
    for r in df_sel.to_dict(orient='records'):
        recs.append(_apply_defaults(r))
    df_norm = pd.DataFrame(recs)

    # Merge normalized columns back into full frame
    cols_norm = ['start_tz_abbr','start_time_iso','display_date','_start_dt']
    for c in cols_norm:
        if c in df_norm.columns:
            df.loc[df_sel.index, c] = df_norm[c].values
    # Serialize _start_dt to ISO strings
    if '_start_dt' in df.columns:
        try:
            dtv = pd.to_datetime(df['_start_dt'], errors='coerce')
            mask = dtv.notna()
            df.loc[mask, '_start_dt'] = dtv[mask].dt.tz_convert('UTC').dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        except Exception:
            # Best-effort stringification
            df['_start_dt'] = df['_start_dt'].astype(str)

    if args.inplace:
        out = src
    else:
        out = f'outputs/predictions_unified_enriched_{args.date}_normalized.csv'
    df.to_csv(out, index=False)

    # Report summary
    sd = pd.to_datetime(df.loc[df_sel.index, '_start_dt'], errors='coerce', utc=True)
    nan_count = int(sd.isna().sum())
    print({'date': args.date, 'out': out, 'normalized_rows': int(len(df_sel)), 'nan__start_dt': nan_count})


if __name__ == '__main__':
    main()
