import pandas as pd
from pathlib import Path


def safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def main():
    out_dir = Path('outputs')
    base = safe_read(out_dir / 'stake_sheet_today.csv')
    cal = safe_read(out_dir / 'stake_sheet_today_cal.csv')
    if base.empty and cal.empty:
        print('[stake-risk] No stake sheets found.')
        return

    def top_exposures(df: pd.DataFrame, n: int = 5) -> dict:
        if df.empty:
            return {"rows": 0}
        amt_col = next((c for c in df.columns if c.lower().startswith('stake') and 'amount' in c.lower()), None)
        market_col = next((c for c in df.columns if 'market' in c.lower()), None)
        team_cols = [c for c in df.columns if 'team' in c.lower()]
        out = {"rows": len(df)}
        if amt_col:
            df2 = df[[amt_col] + ([market_col] if market_col else []) + team_cols].copy()
            df2[amt_col] = pd.to_numeric(df2[amt_col], errors='coerce').fillna(0.0)
            df2 = df2.sort_values(amt_col, ascending=False)
            top = df2.head(n)
            out["top_total_stake"] = float(top[amt_col].sum())
            if market_col and market_col in top.columns:
                out["market_cluster_counts"] = dict(top[market_col].value_counts())
        return out

    base_top = top_exposures(base)
    cal_top = top_exposures(cal)
    res = pd.DataFrame([
        {"label": "baseline", **base_top},
        {"label": "calibrated", **cal_top},
    ])
    out_path = out_dir / 'stake_risk_summary.csv'
    res.to_csv(out_path, index=False)
    print(f"[stake-risk] Wrote {out_path}")


if __name__ == '__main__':
    main()
