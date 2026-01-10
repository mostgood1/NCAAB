import sys
import os
import pandas as pd
import numpy as np

OUT = os.path.join(os.getcwd(), 'outputs')

def to_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/make_picks_raw_for_date.py YYYY-MM-DD")
        sys.exit(1)
    date_str = sys.argv[1].strip()
    edges_path = os.path.join(OUT, f"align_period_{date_str}_edges.csv")
    enr_path = os.path.join(OUT, f"predictions_unified_enriched_{date_str}.csv")
    base_path = os.path.join(OUT, f"predictions_unified_{date_str}.csv")
    df = pd.DataFrame()
    df_enr = pd.DataFrame()
    # Try edges first (for lines/prices); fall back to enriched/base for totals and spreads
    if os.path.exists(edges_path):
        try:
            df = pd.read_csv(edges_path, dtype=str, low_memory=False)
        except Exception:
            df = pd.DataFrame()
    if os.path.exists(enr_path):
        try:
            df_enr = pd.read_csv(enr_path, dtype=str, low_memory=False)
        except Exception:
            df_enr = pd.DataFrame()
    if df_enr.empty and os.path.exists(base_path):
        try:
            df_enr = pd.read_csv(base_path, dtype=str, low_memory=False)
        except Exception:
            df_enr = pd.DataFrame()
    if df.empty and df_enr.empty:
        print("No sources found; nothing to write.")
        sys.exit(0)
    # Coerce key numerics on both frames
    for _df in (df, df_enr):
        for c in [
            'edge_total','edge_margin','total','over_price','under_price',
            'home_spread','home_spread_price','away_spread','away_spread_price',
            'moneyline_home','moneyline_away','pred_total','pred_margin',
            'market_total','closing_total','closing_spread_home'
        ]:
            if c in _df.columns:
                _df[c] = pd.to_numeric(_df[c], errors='coerce')
    # Filter to full game period where applicable
    try:
        if 'period' in df.columns:
            df = df[df['period'].astype(str).str.lower() == 'full_game']
    except Exception:
        pass

    picks_rows = []

    # Totals
    # Prefer edges for totals if present; else build from enriched snapshot
    try:
        m_tot = df['market'].astype(str).str.lower() == 'totals'
    except Exception:
        m_tot = pd.Series([False]*len(df))
    tots = df[m_tot].copy()
    # If edges has no totals, synthesize from enriched/base
    if tots.empty and not df_enr.empty:
        # Construct a minimal frame with game_id, teams, pred_total, market_total
        cols = df_enr.columns
        def pick_col(*names):
            for nm in names:
                if nm in cols:
                    return nm
            return None
        gid_c = pick_col('game_id')
        ht_c = pick_col('home_team','home_team_name')
        at_c = pick_col('away_team','away_team_name')
        pt_c = pick_col('pred_total','pred_total_model')
        mt_c = pick_col('market_total','closing_total','total')
        if gid_c and ht_c and at_c and pt_c:
            tots = pd.DataFrame({
                'game_id': df_enr[gid_c].astype(str),
                'home_team': df_enr[ht_c],
                'away_team': df_enr[at_c],
                'pred_total': pd.to_numeric(df_enr[pt_c], errors='coerce') if pt_c else None,
                'total': pd.to_numeric(df_enr[mt_c], errors='coerce') if mt_c else None,
                'over_price': None,
                'under_price': None,
            })
    for r in tots.to_dict(orient='records'):
        gid = str(r.get('game_id') or '')
        home = r.get('home_team') or r.get('home_team_name')
        away = r.get('away_team') or r.get('away_team_name')
        pt = to_float(r.get('pred_total'))
        ln = to_float(r.get('total'))
        side = None
        if (pt is not None) and (ln is not None):
            side = 'Over' if pt > ln else 'Under'
        elif r.get('edge_total') is not None:
            try:
                et = float(r.get('edge_total'))
                if np.isfinite(et):
                    side = 'Over' if et >= 0 else 'Under'
            except Exception:
                side = None
        line = ln
        price = r.get('over_price') if side == 'Over' else r.get('under_price')
        edge_val = abs(float(r.get('edge_total'))) if r.get('edge_total') is not None else None
        picks_rows.append({
            'game_id': gid,
            'date': date_str,
            'home_team': home,
            'away_team': away,
            'market': 'totals',
            'period': 'full_game',
            'bet': side or '',
            'line': line,
            'price': price,
            'edge': edge_val,
            'pred_total': pt,
            'pred_margin': to_float(r.get('pred_margin')),
            'edge_total': r.get('edge_total'),
            'edge_margin': r.get('edge_margin'),
            'total': r.get('total'),
            'over_price': r.get('over_price'),
            'under_price': r.get('under_price'),
        })

    # Spreads
    # Prefer edges for spreads; else synthesize from enriched/base
    try:
        m_spr = df['market'].astype(str).str.lower() == 'spreads'
    except Exception:
        m_spr = pd.Series([False]*len(df))
    sprs = df[m_spr].copy()
    if sprs.empty and not df_enr.empty:
        cols = df_enr.columns
        def pick_col2(*names):
            for nm in names:
                if nm in cols:
                    return nm
            return None
        gid_c = pick_col2('game_id')
        ht_c = pick_col2('home_team','home_team_name')
        at_c = pick_col2('away_team','away_team_name')
        pm_c = pick_col2('pred_margin_market_blend','pred_margin_blend','pred_margin')
        hs_c = pick_col2('closing_spread_home','home_spread','spread_home')
        as_c = pick_col2('away_spread')
        if gid_c and ht_c and at_c and pm_c:
            sprs = pd.DataFrame({
                'game_id': df_enr[gid_c].astype(str),
                'home_team': df_enr[ht_c],
                'away_team': df_enr[at_c],
                'pred_margin': pd.to_numeric(df_enr[pm_c], errors='coerce') if pm_c else None,
                'home_spread': pd.to_numeric(df_enr[hs_c], errors='coerce') if hs_c else None,
                'closing_spread_home': pd.to_numeric(df_enr[hs_c], errors='coerce') if hs_c else None,
                'away_spread': pd.to_numeric(df_enr[as_c], errors='coerce') if as_c else None,
                'edge_margin': None,
                'home_spread_price': None,
                'away_spread_price': None,
            })
    for r in sprs.to_dict(orient='records'):
        gid = str(r.get('game_id') or '')
        home = r.get('home_team') or r.get('home_team_name')
        away = r.get('away_team') or r.get('away_team_name')
        em = to_float(r.get('edge_margin'))
        pm = to_float(r.get('pred_margin'))
        # side: home if edge_margin >= 0, else away (fallback pred_margin sign)
        side_home = True
        if em is not None:
            side_home = (em >= 0)
        elif pm is not None:
            side_home = (pm >= 0)
        hs = to_float(r.get('home_spread'))
        chs = to_float(r.get('closing_spread_home'))
        aw = to_float(r.get('away_spread'))
        base = chs if chs is not None else hs
        line = None
        if base is not None:
            line = base if side_home else (0.0 - base)
        elif aw is not None:
            line = aw if (not side_home) else (0.0 - aw)
        price = r.get('home_spread_price') if side_home else r.get('away_spread_price')
        edge_val = abs(float(r.get('edge_margin'))) if r.get('edge_margin') is not None else None
        picks_rows.append({
            'game_id': gid,
            'date': date_str,
            'home_team': home,
            'away_team': away,
            'market': 'spreads',
            'period': 'full_game',
            'bet': ('home' if side_home else 'away'),
            'line': line,
            'price': price,
            'edge': edge_val,
            'pred_total': to_float(r.get('pred_total')),
            'pred_margin': pm,
            'edge_total': r.get('edge_total'),
            'edge_margin': r.get('edge_margin'),
            'home_spread': r.get('home_spread'),
            'home_spread_price': r.get('home_spread_price'),
            'away_spread': r.get('away_spread'),
            'away_spread_price': r.get('away_spread_price'),
        })

    picks_df = pd.DataFrame(picks_rows)
    # Write date-scoped picks_raw and refresh global picks_raw by appending
    out_date = os.path.join(OUT, f"picks_raw_{date_str}.csv")
    os.makedirs(OUT, exist_ok=True)
    picks_df.to_csv(out_date, index=False)
    # Update picks_raw.csv: append rows but keep unique by game_id+market
    base_path = os.path.join(OUT, 'picks_raw.csv')
    try:
        if os.path.exists(base_path):
            base_df = pd.read_csv(base_path, dtype=str, low_memory=False)
        else:
            base_df = pd.DataFrame()
    except Exception:
        base_df = pd.DataFrame()
    merged = pd.concat([base_df, picks_df], ignore_index=True)
    try:
        if {'game_id','market'}.issubset(merged.columns):
            merged['game_id'] = merged['game_id'].astype(str)
            merged = merged.drop_duplicates(subset=['game_id','market'])
    except Exception:
        pass
    merged.to_csv(base_path, index=False)
    print(f"Wrote {len(picks_df)} rows to {out_date} and updated picks_raw.csv ({len(merged)} rows)")

if __name__ == '__main__':
    main()
