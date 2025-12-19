from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import json
import os

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'


def pick_latest_date_from_history() -> str | None:
    p = OUT / 'predictions_history_enriched.csv'
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, dtype=str)
        if 'date' not in df.columns:
            return None
        dates = (
            pd.to_datetime(df['date'], errors='coerce')
            .dropna()
            .dt.strftime('%Y-%m-%d')
            .unique()
        )
        if len(dates) == 0:
            return None
        return sorted(dates)[-1]
    except Exception:
        return None


def load_enriched_for_date(date: str) -> pd.DataFrame:
    # Prefer unified_enriched_<date>.csv if present, else filter history
    p1 = OUT / f'predictions_unified_enriched_{date}.csv'
    if p1.exists():
        try:
            df = pd.read_csv(p1)
            # If file is empty or has no columns, treat as missing
            if df is None or (getattr(df, 'empty', True) and len(getattr(df, 'columns', [])) == 0):
                raise ValueError('empty_enriched_file')
            return df
        except Exception:
            # Fallback to history
            pass
    p2 = OUT / 'predictions_history_enriched.csv'
    if p2.exists():
        df = pd.read_csv(p2)
        if 'date' in df.columns:
            try:
                return df[df['date'].astype(str) == date].copy()
            except Exception:
                return df.iloc[0:0].copy()
    return pd.DataFrame()


def pick_latest_date() -> str | None:
    # Prefer per-day unified_enriched files if available
    try:
        cands = sorted(OUT.glob('predictions_unified_enriched_*.csv'))
        if cands:
            # Extract dates from filenames
            dates: list[str] = []
            for p in cands:
                s = p.stem.replace('predictions_unified_enriched_', '')
                try:
                    d = pd.to_datetime(s, errors='coerce')
                    if pd.notna(d):
                        dates.append(d.strftime('%Y-%m-%d'))
                except Exception:
                    pass
            if dates:
                return sorted(set(dates))[-1]
    except Exception:
        pass
    # Fallback to history file
    return pick_latest_date_from_history()


def infer_basis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Initialize basis columns if missing
    for col in ['pred_total_basis','pred_margin_basis']:
        if col not in df.columns:
            df[col] = pd.NA
    # Total basis
    try:
        pt = pd.to_numeric(df.get('pred_total'), errors='coerce')
        pt_cal = pd.to_numeric(df.get('pred_total_calibrated'), errors='coerce') if 'pred_total_calibrated' in df.columns else None
        pt_model = pd.to_numeric(df.get('pred_total_model'), errors='coerce') if 'pred_total_model' in df.columns else None
        mask = df['pred_total_basis'].isna()
        if pt_cal is not None:
            m = mask & (pt.notna() & pt_cal.notna() & (pt == pt_cal))
            df.loc[m, 'pred_total_basis'] = 'cal'
            mask = df['pred_total_basis'].isna()
        if pt_model is not None:
            m = mask & (pt.notna() & pt_model.notna() & (pt == pt_model))
            df.loc[m, 'pred_total_basis'] = 'model'
            mask = df['pred_total_basis'].isna()
        df.loc[df['pred_total_basis'].isna(), 'pred_total_basis'] = 'unknown'
    except Exception:
        pass
    # Margin basis
    try:
        pm = pd.to_numeric(df.get('pred_margin'), errors='coerce')
        pm_cal = pd.to_numeric(df.get('pred_margin_calibrated'), errors='coerce') if 'pred_margin_calibrated' in df.columns else None
        pm_model = pd.to_numeric(df.get('pred_margin_model'), errors='coerce') if 'pred_margin_model' in df.columns else None
        mask = df['pred_margin_basis'].isna()
        if pm_cal is not None:
            m = mask & (pm.notna() & pm_cal.notna() & (pm == pm_cal))
            df.loc[m, 'pred_margin_basis'] = 'cal'
            mask = df['pred_margin_basis'].isna()
        if pm_model is not None:
            m = mask & (pm.notna() & pm_model.notna() & (pm == pm_model))
            df.loc[m, 'pred_margin_basis'] = 'model'
            mask = df['pred_margin_basis'].isna()
        df.loc[df['pred_margin_basis'].isna(), 'pred_margin_basis'] = 'unknown'
    except Exception:
        pass
    return df


def derive_display_fields(df: pd.DataFrame, date: str) -> pd.DataFrame:
    df = df.copy()
    # display_date
    if 'display_date' not in df.columns:
        df['display_date'] = date
    # display_time_str: prefer display_time_ampm, else start_time_display, else empty
    if 'display_time_str' not in df.columns:
        if 'display_time_ampm' in df.columns:
            df['display_time_str'] = df['display_time_ampm']
        elif 'start_time_display' in df.columns:
            df['display_time_str'] = df['start_time_display']
        else:
            df['display_time_str'] = ''
    # start_time_display passthrough if present
    if 'start_time_display' not in df.columns:
        df['start_time_display'] = df['display_time_str']
    # ensure presence of optional fields
    if 'start_time' not in df.columns:
        df['start_time'] = pd.NA
    return df


def make_display(date: str) -> dict:
    df = load_enriched_for_date(date)
    if df.empty:
        return {'status': 'error', 'reason': 'no_enriched', 'date': date}
    # Keep only games rows (drop dup markets)
    if 'market' in df.columns:
        # Prefer full-game unique rows by game_id/home_team/away_team
        keep_cols = ['game_id','home_team','away_team']
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df.drop_duplicates(subset=keep_cols) if keep_cols else df
    # Minimal columns expected by index snapshot block
    cols_keep = [
        'game_id','home_team','away_team','pred_total','pred_margin',
        'pred_total_basis','pred_margin_basis','market_total','spread_home',
        'display_date','display_time_str','start_time','start_time_display'
    ]
    # Enrich basis + display fields
    df = infer_basis(df)
    df = derive_display_fields(df, date)
    # Coerce numeric where appropriate
    for c in ['pred_total','pred_margin','market_total','spread_home']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    disp = df[[c for c in cols_keep if c in df.columns]].copy()
    out_path = OUT / f'predictions_display_{date}.csv'
    disp.to_csv(out_path, index=False)
    # Also archive
    arch = OUT / 'archive' / date
    arch.mkdir(parents=True, exist_ok=True)
    (arch / f'predictions_display_{date}.csv').write_text(out_path.read_text(encoding='utf-8'), encoding='utf-8')
    return {'status': 'ok', 'rows': int(len(disp)), 'path': str(out_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='YYYY-MM-DD')
    ap.add_argument('--latest', action='store_true', help='Use latest date from enriched files or history')
    args = ap.parse_args()
    date = args.date
    if not date and args.latest:
        date = pick_latest_date()
    if not date:
        print(json.dumps({'status': 'error', 'reason': 'no_date'}))
        return 2
    res = make_display(date)
    print(json.dumps(res))
    return 0 if res.get('status') == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
