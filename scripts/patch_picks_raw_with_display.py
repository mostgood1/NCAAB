import os
import sys
import pandas as pd

def main(date_str: str) -> None:
    out_dir = os.path.join(os.getcwd(), 'outputs')
    picks_path = os.path.join(out_dir, 'picks_raw.csv')
    disp_path = os.path.join(out_dir, f'predictions_display_{date_str}.csv')
    enr_path = os.path.join(out_dir, f'predictions_unified_enriched_{date_str}.csv')
    if not os.path.exists(picks_path):
        print(f"[patch] missing picks_raw: {picks_path}")
        return
    if not os.path.exists(disp_path) and not os.path.exists(enr_path):
        print(f"[patch] missing display and enriched for date: {date_str}")
        return
    df_p = pd.read_csv(picks_path, dtype=str, low_memory=False)
    df_d = pd.read_csv(disp_path, dtype=str, low_memory=False) if os.path.exists(disp_path) else pd.DataFrame()
    df_e = pd.read_csv(enr_path, dtype=str, low_memory=False) if os.path.exists(enr_path) else pd.DataFrame()
    # Normalize keys
    for df in (df_p, df_d, df_e):
        if 'game_id' in df.columns:
            df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    # Build line map from display (prefer market/closing; fallback to model pred_total)
    line_map = {}
    team_map = {}
    for r in df_d.to_dict(orient='records'):
        gid = str(r.get('game_id') or '').strip()
        if gid:
            ln = None
            for c in ('market_total', 'closing_total', 'total', 'line_total'):
                v = r.get(c)
                if v is not None and str(v).strip() != '':
                    try:
                        ln = float(v)
                        break
                    except Exception:
                        continue
            if ln is None:
                pv = r.get('pred_total')
                if pv is not None and str(pv).strip() != '':
                    try:
                        ln = float(pv)
                    except Exception:
                        ln = None
            line_map[gid] = ln
            team_map[gid] = {
                'home_team': r.get('home_team') or r.get('home_team_name'),
                'away_team': r.get('away_team') or r.get('away_team_name'),
            }
    # Enriched fallback for line and pred_total
    pred_map = {}
    for r in df_e.to_dict(orient='records'):
        gid = str(r.get('game_id') or '').strip()
        if gid:
            ln = None
            for c in ('market_total', 'closing_total'):
                v = r.get(c)
                if v is not None and str(v).strip() != '':
                    try:
                        ln = float(v)
                        break
                    except Exception:
                        continue
            if ln is None:
                pv = r.get('pred_total')
                if pv is not None and str(pv).strip() != '':
                    try:
                        ln = float(pv)
                    except Exception:
                        ln = None
            if gid not in line_map or line_map[gid] is None:
                line_map[gid] = ln
            # Keep pred_total mapping too
            try:
                pt = r.get('pred_total')
                pred_map[gid] = float(pt) if pt is not None and str(pt).strip() != '' else pred_map.get(gid)
            except Exception:
                pass
    # Patch OU rows for the date: fill line and missing teams
    def _patch_row(r: dict) -> dict:
        d = str(r.get('date') or '')
        rc = str(r.get('rec_code') or '').upper()
        mkt = str(r.get('market') or '').lower()
        gid = str(r.get('game_id') or '').strip()
        if d == date_str and (rc == 'OU' or 'total' in mkt):
            # line: prefer display/enriched; fallback to model pred_total
            if not r.get('line') or str(r.get('line')).strip() == '':
                if gid and gid in line_map and line_map[gid] is not None:
                    r['line'] = line_map[gid]
                elif gid in pred_map and pred_map[gid] is not None:
                    r['line'] = pred_map[gid]
            # teams
            for side in ('home_team', 'away_team'):
                val = r.get(side)
                if val is None or str(val).strip() == '':
                    if gid and gid in team_map:
                        t = team_map[gid].get(side)
                        if t:
                            r[side] = t
            # pred_total fill
            if (not r.get('pred_total')) or str(r.get('pred_total')).strip() == '':
                if gid in pred_map and pred_map[gid] is not None:
                    r['pred_total'] = pred_map[gid]
        return r
    patched = [ _patch_row(dict(x)) for x in df_p.to_dict(orient='records') ]
    df_out = pd.DataFrame(patched)
    df_out.to_csv(picks_path, index=False)
    print(f"[patch] wrote patched picks_raw with {len(df_out)} rows")

if __name__ == '__main__':
    date_arg = sys.argv[1] if len(sys.argv) > 1 else pd.Timestamp.now(tz='US/Eastern').date().isoformat()
    main(date_arg)
