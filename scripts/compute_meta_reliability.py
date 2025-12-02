import argparse
import glob
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1] / 'outputs'


def _find_recent_enriched(limit_days: int) -> list[str]:
    paths = sorted(glob.glob(str(OUT / 'predictions_unified_enriched_*.csv')))
    if limit_days <= 0:
        return paths[-180:]
    cutoff = datetime.utcnow() - timedelta(days=limit_days)
    sel = []
    for p in paths:
        base = Path(p).name
        try:
            date_part = base.replace('predictions_unified_enriched_', '').replace('.csv', '')
            dt = datetime.strptime(date_part, '%Y-%m-%d')
        except Exception:
            dt = None
        if dt is None or dt >= cutoff:
            sel.append(p)
    return sel[-180:]


def _detect_prob_col(df: pd.DataFrame, patterns: list[str]) -> str | None:
    cols = [c for c in df.columns if any(c.lower().startswith(p) for p in patterns)]
    # Prefer explicitly calibrated/meta columns when present
    pref_order = ['_cal', '_meta', '_lgbm']
    cols = sorted(cols, key=lambda c: (min([i for i,p in enumerate(pref_order) if p in c] or [len(pref_order)]), c))
    for c in cols:
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().sum() >= 100 and s.between(0,1).mean() > 0.9:
            return c
    return None


def _derive_targets(df: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    # Cover target
    y_cover = None
    for c in ['ats_home_win','cover_home']:
        if c in df.columns:
            y = pd.to_numeric(df[c], errors='coerce')
            y_cover = y if y_cover is None else y_cover.fillna(y)
    if 'ats_result' in df.columns:
        s = df['ats_result'].astype(str).str.lower()
        y = s.map(lambda v: 1.0 if ('home' in v and 'cover' in v) else (0.0 if ('away' in v and 'cover' in v) else np.nan))
        y_cover = y if y_cover is None else y_cover.fillna(y)

    # Over target
    y_over = None
    for c in ['ou_over_win','went_over']:
        if c in df.columns:
            y = pd.to_numeric(df[c], errors='coerce')
            y_over = y if y_over is None else y_over.fillna(y)
    if 'ou_result' in df.columns:
        s = df['ou_result'].astype(str).str.lower()
        y = s.map(lambda v: 1.0 if 'over' in v else (0.0 if 'under' in v else np.nan))
        y_over = y if y_over is None else y_over.fillna(y)

    # Fallback for over: actual vs market
    if (y_over is None or y_over.isna().all()) and {'actual_total','market_total'}.issubset(df.columns):
        at = pd.to_numeric(df['actual_total'], errors='coerce')
        mt = pd.to_numeric(df['market_total'], errors='coerce')
        y_over = (at > mt).astype(float)

    return y_cover, y_over


def _reliability(y: pd.Series, p: pd.Series, bins: int = 10) -> tuple[pd.DataFrame, float, float, int]:
    yt = pd.to_numeric(y, errors='coerce')
    pt = pd.to_numeric(p, errors='coerce')
    mask = yt.isin([0.0,1.0]) & pt.between(0,1)
    yt = yt[mask]
    pt = pt[mask]
    n = int(len(yt))
    if n < 50:
        return pd.DataFrame(columns=['bin_low','bin_high','n','confidence','accuracy']), float('nan'), float('nan'), 0
    edges = np.linspace(0,1,bins+1)
    idx = np.digitize(pt, edges, right=True)
    rows = []
    ece = 0.0
    for b in range(1, bins+1):
        sel = (idx == b)
        if sel.any():
            conf = float(pt[sel].mean())
            acc = float(yt[sel].mean())
            cnt = int(sel.sum())
            ece += (cnt / n) * abs(acc - conf)
            rows.append({'bin_low': float(edges[b-1]), 'bin_high': float(edges[b]), 'n': cnt, 'confidence': conf, 'accuracy': acc})
        else:
            rows.append({'bin_low': float(edges[b-1]), 'bin_high': float(edges[b]), 'n': 0, 'confidence': np.nan, 'accuracy': np.nan})
    brier = float(np.mean((pt - yt) ** 2)) if n > 0 else float('nan')
    return pd.DataFrame(rows), float(ece), brier, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit-days', type=int, default=45)
    ap.add_argument('--out-dir', type=str, default=str(OUT))
    args = ap.parse_args()

    files = _find_recent_enriched(args.limit_days)
    if not files:
        print('[meta] No enriched files found for reliability computation')
        return 0
    frames = []
    for p in files:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            continue
    if not frames:
        print('[meta] No readable enriched files')
        return 0
    df = pd.concat(frames, ignore_index=True)

    # Detect prediction columns (raw + calibrated variants when present)
    # Cover
    p_cover_raw = None
    for c in ['p_home_cover_meta', 'p_home_cover', 'p_home_cover_dist', 'p_home_cover_cdf', 'p_home_cover_mix']:
        if c in df.columns:
            p_cover_raw = c; break
    p_cover_cal = 'p_home_cover_meta_cal' if 'p_home_cover_meta_cal' in df.columns else None
    # Over
    p_over_raw = None
    for c in ['p_over_meta', 'p_over', 'p_over_dist', 'p_over_cdf', 'p_over_mix']:
        if c in df.columns:
            p_over_raw = c; break
    p_over_cal = 'p_over_meta_cal' if 'p_over_meta_cal' in df.columns else None
    y_cover, y_over = _derive_targets(df)

    rel_rows = []
    summary = {}

    # Cover raw
    if p_cover_raw and y_cover is not None:
        rel_c_raw, ece_c_raw, brier_c_raw, n_c_raw = _reliability(y_cover, df[p_cover_raw])
        rel_c_raw['type'] = 'cover'; rel_c_raw['variant'] = 'raw'; rel_c_raw['p_col'] = p_cover_raw
        rel_rows.append(rel_c_raw)
        summary.update({'ece_cover_raw': ece_c_raw, 'brier_cover_raw': brier_c_raw, 'rows_cover_raw': n_c_raw, 'p_col_cover_raw': p_cover_raw})
    # Cover calibrated
    if p_cover_cal and y_cover is not None:
        rel_c_cal, ece_c_cal, brier_c_cal, n_c_cal = _reliability(y_cover, df[p_cover_cal])
        rel_c_cal['type'] = 'cover'; rel_c_cal['variant'] = 'cal'; rel_c_cal['p_col'] = p_cover_cal
        rel_rows.append(rel_c_cal)
        summary.update({'ece_cover_cal': ece_c_cal, 'brier_cover_cal': brier_c_cal, 'rows_cover_cal': n_c_cal, 'p_col_cover_cal': p_cover_cal})
    # Over raw
    if p_over_raw and y_over is not None:
        rel_o_raw, ece_o_raw, brier_o_raw, n_o_raw = _reliability(y_over, df[p_over_raw])
        rel_o_raw['type'] = 'over'; rel_o_raw['variant'] = 'raw'; rel_o_raw['p_col'] = p_over_raw
        rel_rows.append(rel_o_raw)
        summary.update({'ece_over_raw': ece_o_raw, 'brier_over_raw': brier_o_raw, 'rows_over_raw': n_o_raw, 'p_col_over_raw': p_over_raw})
    # Over calibrated
    if p_over_cal and y_over is not None:
        rel_o_cal, ece_o_cal, brier_o_cal, n_o_cal = _reliability(y_over, df[p_over_cal])
        rel_o_cal['type'] = 'over'; rel_o_cal['variant'] = 'cal'; rel_o_cal['p_col'] = p_over_cal
        rel_rows.append(rel_o_cal)
        summary.update({'ece_over_cal': ece_o_cal, 'brier_over_cal': brier_o_cal, 'rows_over_cal': n_o_cal, 'p_col_over_cal': p_over_cal})

    if rel_rows:
        rel_df = pd.concat(rel_rows, ignore_index=True)
        # Preserve variant and source column when present
        keep_cols = ['type','variant','p_col','bin_low','bin_high','n','confidence','accuracy']
        rel_df = rel_df[[c for c in keep_cols if c in rel_df.columns]]
        rel_df.to_csv(OUT / 'meta_reliability.csv', index=False)
        summary['status'] = 'ok'
    else:
        summary['status'] = 'missing'
    summary['timestamp_utc'] = datetime.utcnow().isoformat()
    (OUT / 'meta_ece.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('[meta] Wrote meta_reliability.csv and meta_ece.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
