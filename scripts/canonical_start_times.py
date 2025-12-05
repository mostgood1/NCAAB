import argparse
import json
from pathlib import Path
import pandas as pd
from zoneinfo import ZoneInfo

def to_utc_iso_from_central(date_str: str, time_str: str) -> str | None:
    try:
        # Accept '6:00 PM' or '18:00'
        t = time_str.strip().upper().replace(' CT','').replace(' CST','').replace(' CDT','')
        for fmt in ('%I:%M %p','%H:%M'):
            try:
                import datetime as dt
                naive = dt.datetime.strptime(f"{date_str} {t}", f"%Y-%m-%d {fmt}")
                local_dt = naive.replace(tzinfo=ZoneInfo('America/Chicago'))
                utc_dt = local_dt.astimezone(ZoneInfo('UTC'))
                return utc_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            except Exception:
                pass
    except Exception:
        pass
    return None


def load_espn_subset(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        # Expect list of items with id and start dates
        lookup = {}
        for r in data if isinstance(data, list) else []:
            gid = str(r.get('id') or r.get('game_id') or '').strip()
            if not gid:
                continue
            lookup[gid] = r
        return lookup
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('date', help='YYYY-MM-DD slate date')
    args = ap.parse_args()

    outdir = Path('outputs')
    games_path = outdir / f'games_{args.date}.csv'
    espn_subset_path = outdir / f'schedule_espn_subset_{args.date}.json'
    out_path = outdir / f'canonical_start_times_{args.date}.csv'

    df = pd.read_csv(games_path) if games_path.exists() else pd.DataFrame()
    if 'game_id' in df.columns:
        df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    espn_lookup = load_espn_subset(espn_subset_path)

    rows = []
    for r in df.to_dict(orient='records'):
        gid = str(r.get('game_id') or '').strip()
        # Prefer ISO fields if present
        iso = None
        for key in ('start_time_iso','commence_time','start_time'):
            val = r.get(key)
            if val:
                d = pd.to_datetime(str(val).replace('Z','+00:00'), errors='coerce', utc=True)
                if pd.notna(d):
                    iso = d.strftime('%Y-%m-%dT%H:%M:%SZ')
                    break
        # If missing, try ESPN subset
        if not iso and gid in espn_lookup:
            e = espn_lookup[gid]
            # ESPN may have date/time split or an ISO
            e_iso = e.get('start_time_iso') or e.get('dateTime') or e.get('start')
            if e_iso:
                d = pd.to_datetime(str(e_iso).replace('Z','+00:00'), errors='coerce', utc=True)
                if pd.notna(d):
                    iso = d.strftime('%Y-%m-%dT%H:%M:%SZ')
            if not iso:
                disp_date = str(e.get('display_date') or r.get('display_date') or '').strip()
                disp_time = str(e.get('start_time_display') or r.get('start_time_display') or '').strip()
                if disp_date and disp_time:
                    iso = to_utc_iso_from_central(disp_date, disp_time)
        # As last resort, display/local from games
        if not iso:
            disp_date = str(r.get('display_date') or '').strip()
            disp_time = str(r.get('start_time_display') or '').strip()
            if disp_date and disp_time:
                iso = to_utc_iso_from_central(disp_date, disp_time)
        # Build Central display fields
        display_date = None
        start_time_display = None
        start_time_local = None
        start_tz_abbr = 'CST'
        if iso:
            ts = pd.to_datetime(iso.replace('Z','+00:00'), errors='coerce', utc=True)
            if pd.notna(ts):
                ts_c = ts.tz_convert(ZoneInfo('America/Chicago'))
                display_date = ts_c.strftime('%Y-%m-%d')
                # Windows-safe 12h format without leading zero
                start_time_display = ts_c.strftime('%I:%M %p').lstrip('0') if hasattr(ts_c, 'strftime') else None
                start_time_local = ts_c.strftime('%Y-%m-%d %H:%M')
        rows.append({
            'game_id': gid,
            'date': args.date,
            'start_time_iso': iso,
            '_start_dt': iso,
            'display_date': display_date,
            'start_time_display': start_time_display,
            'start_time_local': start_time_local,
            'start_tz_abbr': start_tz_abbr,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    nonnull = int(out_df['_start_dt'].notna().sum()) if '_start_dt' in out_df.columns else 0
    print({'path': str(out_path), 'rows': len(out_df), 'nonnull__start_dt': nonnull})

if __name__ == '__main__':
    main()
