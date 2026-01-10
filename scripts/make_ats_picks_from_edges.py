import os
import sys
import pandas as pd

def main(date_str: str) -> None:
    out_dir = os.path.join(os.getcwd(), 'outputs')
    edges_path = os.path.join(out_dir, f'align_period_{date_str}_edges.csv')
    picks_path = os.path.join(out_dir, 'picks_raw.csv')
    if not os.path.exists(edges_path):
        print(f"[ats-edges] missing edges: {edges_path}")
        return
    df = pd.read_csv(edges_path, dtype=str, low_memory=False)
    if df.empty:
        print("[ats-edges] edges empty")
        return
    for c in ('edge_margin','pred_margin','home_spread','away_spread','closing_spread_home'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'game_id' in df.columns:
        df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    # Restrict to full_game period when available
    if 'period' in df.columns:
        df = df[df['period'].astype(str).str.lower() == 'full_game']
    # Restrict to spreads market when present
    if 'market' in df.columns:
        df = df[df['market'].astype(str).str.lower() == 'spreads']
    if df.empty:
        print("[ats-edges] no spreads rows in edges")
        return
    def _bet_side(row: pd.Series) -> str:
        # Prefer edge_margin sign; fallback to pred_margin
        try:
            em = float(row.get('edge_margin')) if row.get('edge_margin') is not None else None
        except Exception:
            em = None
        if em is not None:
            return 'home' if em >= 0 else 'away'
        try:
            pm = float(row.get('pred_margin')) if row.get('pred_margin') is not None else None
        except Exception:
            pm = None
        if pm is not None:
            return 'home' if pm >= 0 else 'away'
        return 'home'
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
        v = row.get('away_spread')
        try:
            if v is not None and pd.notna(v):
                base = float(v)
                return base if sel == 'away' else (0 - base)
        except Exception:
            pass
        return None
    def _edge_val(row: pd.Series) -> float | None:
        try:
            # use abs edge_margin when present, else compute from pred_margin and signed line
            em = row.get('edge_margin')
            if em is not None and pd.notna(em):
                return abs(float(em))
        except Exception:
            pass
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
    out['home_team'] = df.get('home_team_name') if 'home_team_name' in df.columns else df.get('home_team')
    out['away_team'] = df.get('away_team_name') if 'away_team_name' in df.columns else df.get('away_team')
    out['market'] = 'spreads'
    out['period'] = 'full_game'
    out['bet'] = df.apply(_bet_side, axis=1)
    out['line'] = df.apply(_signed_line, axis=1)
    out['price'] = df.apply(lambda r: (r['home_spread_price'] if str(_bet_side(r)).lower() == 'home' else r['away_spread_price']) if {'home_spread_price','away_spread_price'}.issubset(df.columns) else None, axis=1)
    out['edge'] = df.apply(_edge_val, axis=1)
    out['pred_margin'] = df.get('pred_margin')
    out['pred_total'] = None
    out['rec_type'] = 'Spread'
    out['rec_code'] = 'ATS'
    # Drop rows without line
    out = out[out['line'].notna()]
    if out.empty:
        print("[ats-edges] no valid ATS rows produced")
        return
    # Append into picks_raw.csv
    if os.path.exists(picks_path):
        base = pd.read_csv(picks_path, dtype=str, low_memory=False)
        try:
            base['date'] = pd.to_datetime(base['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            keep = ~( (base['date'] == date_str) & (base.get('rec_code','').astype(str).str.upper() == 'ATS') )
            base = base[keep]
        except Exception:
            pass
        merged = pd.concat([base, out], ignore_index=True)
        merged.to_csv(picks_path, index=False)
        print(f"[ats-edges] appended {len(out)} ATS rows into picks_raw.csv (total {len(merged)})")
    else:
        out.to_csv(picks_path, index=False)
        print(f"[ats-edges] wrote picks_raw.csv with {len(out)} ATS rows")

if __name__ == '__main__':
    date_arg = sys.argv[1] if len(sys.argv) > 1 else pd.Timestamp.now(tz='US/Eastern').date().isoformat()
    main(date_arg)
