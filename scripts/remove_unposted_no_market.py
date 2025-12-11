import pandas as pd
from pathlib import Path
import sys

def main(date: str, out_dir: str):
    outp = Path(out_dir)
    enr = outp / f"predictions_unified_enriched_{date}.csv"
    dis = outp / f"predictions_display_{date}.csv"
    games_curr = outp / f"games_{date}.csv"
    if not enr.exists():
        print("[safeguard] skipped (missing enriched)")
        return 0
    df = pd.read_csv(enr)
    gid_today = set()
    if games_curr.exists():
        gc = pd.read_csv(games_curr)
        if 'game_id' in gc.columns:
            gid_today = set(gc['game_id'].astype(str))
    def should_drop(r):
        market_status = str(r.get('market_status','')).strip().lower()
        mt = r.get('market_total')
        sh = r.get('spread_home')
        gid = str(r.get('game_id',''))
        no_market = (market_status == 'no market (unposted)') or (pd.isna(mt) and pd.isna(sh))
        not_in_provider = (gid_today and (gid not in gid_today))
        return no_market and not_in_provider
    before = len(df)
    df2 = df[~df.apply(should_drop, axis=1)].copy()
    removed = before - len(df2)
    if removed > 0:
        df2.to_csv(enr, index=False)
        keep = [
            'game_id','date','home_team','away_team','pred_total','pred_margin',
            'display_date','start_time_display','display_time_str','start_tz_abbr',
            'market_total','spread_home','start_time_iso','_start_dt'
        ]
        cols = [c for c in keep if c in df2.columns]
        df2[cols].to_csv(dis, index=False)
        print(f"[safeguard] removed={removed} unposted/no-market rows not in provider slate")
    else:
        print("[safeguard] no unposted/no-market rows to remove")
    return 0

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: remove_unposted_no_market.py <date> <outputs_dir>")
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
