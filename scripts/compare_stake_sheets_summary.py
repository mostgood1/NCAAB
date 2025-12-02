import pandas as pd
from pathlib import Path


def safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def summarize(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"label": label, "rows": 0}
    amt_col = next((c for c in df.columns if c.lower().startswith('stake') and 'amount' in c.lower()), None)
    ev_col = next((c for c in df.columns if 'ev' in c.lower()), None)
    roi_col = next((c for c in df.columns if 'roi' in c.lower()), None)
    out = {"label": label, "rows": len(df)}
    if amt_col:
        out["total_stake"] = float(pd.to_numeric(df[amt_col], errors='coerce').sum())
    if ev_col:
        out["avg_ev"] = float(pd.to_numeric(df[ev_col], errors='coerce').mean())
    if roi_col:
        out["avg_roi"] = float(pd.to_numeric(df[roi_col], errors='coerce').mean())
    return out


def main():
    out_dir = Path('outputs')
    base = safe_read(out_dir / 'stake_sheet_today.csv')
    cal = safe_read(out_dir / 'stake_sheet_today_cal.csv')
    if base.empty and cal.empty:
        print('[stake-compare] No stake sheets found to summarize.')
        return
    s_base = summarize(base, 'baseline')
    s_cal = summarize(cal, 'calibrated')
    # Join by game_id if present to see common vs unique picks
    common = 0
    if {'game_id'}.issubset(base.columns) and {'game_id'}.issubset(cal.columns):
        common = len(set(base['game_id']).intersection(set(cal['game_id'])))
    res = pd.DataFrame([s_base, s_cal])
    res['common_picks'] = common
    out_path = out_dir / 'stake_sheet_today_summary.csv'
    res.to_csv(out_path, index=False)
    print(f"[stake-compare] Wrote {out_path}")


if __name__ == '__main__':
    main()
