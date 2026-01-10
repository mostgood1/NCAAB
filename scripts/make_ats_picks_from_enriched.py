import os
import sys
import pandas as pd

def main(date_str: str) -> None:
    out_dir = os.path.join(os.getcwd(), 'outputs')
    enr_path = os.path.join(out_dir, f'predictions_unified_enriched_{date_str}.csv')
    picks_path = os.path.join(out_dir, 'picks_raw.csv')
    if not os.path.exists(enr_path):
        print(f"[ats] missing enriched: {enr_path}")
        return
    df = pd.read_csv(enr_path, dtype=str, low_memory=False)
    if df.empty:
        print("[ats] enriched empty")
        return
    # Normalize types
    for c in ('pred_margin','closing_spread_home','home_spread','away_spread'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'game_id' in df.columns:
        df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    # Filter rows with at least one spread column
    spread_cols = [c for c in ['closing_spread_home','home_spread','away_spread'] if c in df.columns]
    if not spread_cols:
        print("[ats] enriched missing spread columns")
        return
    has_spread = df[spread_cols].notna().any(axis=1)
    df = df[has_spread].copy()
    if df.empty:
        print("[ats] no spread info in enriched")
        return
    # Select bet side from pred_margin
    def _bet_side(row: pd.Series) -> str:
        pm = row.get('pred_margin')
        try:
            return 'home' if float(pm) >= 0 else 'away'
        except Exception:
            return 'home'
    # Compute signed line: prefer closing_spread_home -> home_spread -> away_spread
    def _signed_line(row: pd.Series) -> float | None:
        sel = _bet_side(row)
        for base_col in ('closing_spread_home','home_spread'):
            v = row.get(base_col)
            try:
                if v is not None and pd.notna(v):
                    base = float(v)
                    return base if sel == 'home' else (0 - base)
            except Exception:
                continue
        # Fallback to away_spread
        v = row.get('away_spread')
        try:
            if v is not None and pd.notna(v):
                base = float(v)
                return base if sel == 'away' else (0 - base)
        except Exception:
            pass
        return None
    # Edge: abs(pred_margin - signed_line)
    def _edge_val(row: pd.Series) -> float | None:
        try:
            pm = float(row.get('pred_margin'))
            ln = _signed_line(row)
            if ln is None:
                return None
            return abs(pm - ln)
        except Exception:
            return None
    out = pd.DataFrame()
    out['game_id'] = df.get('game_id')
    out['date'] = date_str
    out['home_team'] = df.get('home_team') if 'home_team' in df.columns else df.get('home_team_name')
    out['away_team'] = df.get('away_team') if 'away_team' in df.columns else df.get('away_team_name')
    out['market'] = 'spreads'
    out['period'] = 'full_game'
    out['bet'] = df.apply(_bet_side, axis=1)
    out['line'] = df.apply(_signed_line, axis=1)
    out['price'] = None
    out['edge'] = df.apply(_edge_val, axis=1)
    out['pred_margin'] = df.get('pred_margin')
    out['pred_total'] = None
    out['rec_type'] = 'Spread'
    out['rec_code'] = 'ATS'
    # Drop rows without line
    out = out[out['line'].notna()]
    if out.empty:
        print("[ats] no valid ATS rows produced")
        return
    # Append into picks_raw.csv
    if os.path.exists(picks_path):
        base = pd.read_csv(picks_path, dtype=str, low_memory=False)
        # Remove any existing ATS rows for date to avoid duplicates
        try:
            base['date'] = pd.to_datetime(base['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            keep = ~( (base['date'] == date_str) & (base.get('rec_code','').astype(str).str.upper() == 'ATS') )
            base = base[keep]
        except Exception:
            pass
        merged = pd.concat([base, out], ignore_index=True)
        merged.to_csv(picks_path, index=False)
        print(f"[ats] appended {len(out)} ATS rows into picks_raw.csv (total {len(merged)})")
    else:
        out.to_csv(picks_path, index=False)
        print(f"[ats] wrote picks_raw.csv with {len(out)} ATS rows")

if __name__ == '__main__':
    date_arg = sys.argv[1] if len(sys.argv) > 1 else pd.Timestamp.now(tz='US/Eastern').date().isoformat()
    main(date_arg)
