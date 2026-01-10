import sys
import os
import pandas as pd
import numpy as np

OUT = os.path.join(os.getcwd(), 'outputs')

def first_present(df: pd.DataFrame, cols: list[str]) -> str | None:
    for c in cols:
        if c in df.columns:
            return c
    return None

def coerce_float(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    try:
        return pd.to_numeric(series, errors='coerce')
    except Exception:
        return series

def load_source(date_str: str) -> pd.DataFrame:
    paths = [
        os.path.join(OUT, f"predictions_unified_enriched_{date_str}.csv"),
        os.path.join(OUT, f"predictions_unified_{date_str}.csv"),
        os.path.join(OUT, f"predictions_display_{date_str}.csv"),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, dtype=str, low_memory=False)
                return df
            except Exception:
                continue
    return pd.DataFrame()

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/make_ats_picks_file.py YYYY-MM-DD")
        sys.exit(1)
    date_str = sys.argv[1].strip()
    src = load_source(date_str)
    if src.empty:
        print(f"No source snapshot found for {date_str}; nothing to write.")
        sys.exit(0)

    # Normalize key columns
    if 'game_id' in src.columns:
        src['game_id'] = src['game_id'].astype(str)

    # Determine spreads and margin columns
    c_spread_home = first_present(src, ['closing_spread_home','home_spread','spread_home'])
    c_spread_away = first_present(src, ['away_spread'])
    c_pred_margin = first_present(src, [
        'pred_margin_market_blend',
        'pred_margin_blend',
        'pred_margin',
    ])

    # Coerce numeric columns
    sh = coerce_float(src.get(c_spread_home))
    sa = coerce_float(src.get(c_spread_away))
    pm = coerce_float(src.get(c_pred_margin))

    # Build output rows
    out = pd.DataFrame()
    out['game_id'] = src.get('game_id')
    out['date'] = date_str
    out['home_team'] = src.get('home_team') if 'home_team' in src.columns else src.get('home_team_name')
    out['away_team'] = src.get('away_team') if 'away_team' in src.columns else src.get('away_team_name')

    # selection side based on margin vs market margin (0 - home spread)
    def pick_side(idx: int) -> str:
        try:
            pmv = float(pm.iloc[idx]) if (pm is not None and idx < len(pm) and pd.notna(pm.iloc[idx])) else None
        except Exception:
            pmv = None
        try:
            shv = float(sh.iloc[idx]) if (sh is not None and idx < len(sh) and pd.notna(sh.iloc[idx])) else None
        except Exception:
            shv = None
        # Fallback: infer from away spread if home spread missing
        if shv is None and sa is not None and idx < len(sa) and pd.notna(sa.iloc[idx]):
            try:
                av = float(sa.iloc[idx])
                # if away spread present, home spread is negative of away for symmetry
                shv = -av
            except Exception:
                shv = None
        if (pmv is None) or (shv is None):
            # Default to home if insufficient info
            return 'home'
        mkt_margin = (0.0 - shv)
        delta = pmv - mkt_margin
        return 'home' if delta >= 0 else 'away'

    out['ats_side'] = [pick_side(i) for i in range(len(out))]

    # Include spreads to help the API compute signed line
    out['closing_spread_home'] = sh
    # common alias for some readers
    out['spread_home'] = sh

    # Diagnostic columns
    def delta_val(idx: int) -> float | None:
        try:
            pmv = float(pm.iloc[idx]) if (pm is not None and idx < len(pm) and pd.notna(pm.iloc[idx])) else None
        except Exception:
            pmv = None
        try:
            shv = float(sh.iloc[idx]) if (sh is not None and idx < len(sh) and pd.notna(sh.iloc[idx])) else None
        except Exception:
            shv = None
        if (pmv is None) or (shv is None):
            return None
        return pmv - (0.0 - shv)

    out['_delta'] = [delta_val(i) for i in range(len(out))]
    out['_pred_margin_blend'] = pm

    # Create output directory and write CSV
    out_dir = os.path.join(OUT, 'picks')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ats_picks_{date_str}.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} ATS picks to {out_path}")

if __name__ == '__main__':
    main()
