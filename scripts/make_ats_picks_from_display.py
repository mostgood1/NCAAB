import os
import sys
import pandas as pd

def main(date_str: str) -> None:
    out_dir = os.path.join(os.getcwd(), 'outputs')
    disp_path = os.path.join(out_dir, f'predictions_display_{date_str}.csv')
    picks_path = os.path.join(out_dir, 'picks_raw.csv')
    if not os.path.exists(disp_path):
        print(f"[ats-disp] missing display: {disp_path}")
        return
    df = pd.read_csv(disp_path, dtype=str, low_memory=False)
    if df.empty:
        print("[ats-disp] display empty")
        return
    # Normalize types
    for c in ('pred_margin',):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'game_id' in df.columns:
        df['game_id'] = df['game_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    # Use pred_margin to pick side; line unknown
    def _bet_side(row: pd.Series) -> str:
        pm = row.get('pred_margin')
        try:
            return 'home' if float(pm) >= 0 else 'away'
        except Exception:
            return 'home'
    out = pd.DataFrame()
    out['game_id'] = df.get('game_id')
    out['date'] = date_str
    out['home_team'] = df.get('home_team')
    out['away_team'] = df.get('away_team')
    out['market'] = 'spreads'
    out['period'] = 'full_game'
    out['bet'] = df.apply(_bet_side, axis=1)
    out['line'] = None
    out['price'] = None
    out['edge'] = df.get('pred_margin').abs() if 'pred_margin' in df.columns else None
    out['pred_margin'] = df.get('pred_margin')
    out['pred_total'] = None
    out['rec_type'] = 'Spread'
    out['rec_code'] = 'ATS'
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
        print(f"[ats-disp] appended {len(out)} ATS rows into picks_raw.csv (total {len(merged)})")
    else:
        out.to_csv(picks_path, index=False)
        print(f"[ats-disp] wrote picks_raw.csv with {len(out)} ATS rows")

if __name__ == '__main__':
    date_arg = sys.argv[1] if len(sys.argv) > 1 else pd.Timestamp.now(tz='US/Eastern').date().isoformat()
    main(date_arg)
