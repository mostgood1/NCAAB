import sys
import json
import datetime as dt
from pathlib import Path
import re
import pandas as pd
import numpy as np

# Inline copy of the single-date accuracy computation to avoid import path issues
def compute_daily(date_str: str, outputs_dir: str = 'outputs'):
    dr_path = Path(outputs_dir) / 'daily_results' / f'results_{date_str}.csv'
    disp_path = Path(outputs_dir) / f'predictions_display_{date_str}.csv'
    if not dr_path.exists() or not disp_path.exists():
        return {
            'date': date_str,
            'error': f'missing files for date: results={dr_path.exists()} display={disp_path.exists()}'
        }
    dr = pd.read_csv(dr_path, dtype={'game_id': str})
    disp = pd.read_csv(disp_path, dtype={'game_id': str})
    cols = ['game_id','pred_total','pred_margin','closing_total','closing_spread_home']
    disp_cols = [c for c in cols if c in disp.columns]
    df = dr.merge(disp[disp_cols], on='game_id', how='left', suffixes=('', '_disp'))

    def coalesce(col: str) -> pd.Series:
        a = pd.to_numeric(df.get(col), errors='coerce') if col in df.columns else pd.Series(np.nan, index=df.index)
        b = pd.to_numeric(df.get(f"{col}_disp"), errors='coerce') if f"{col}_disp" in df.columns else pd.Series(np.nan, index=df.index)
        out = a.copy()
        m = out.isna() & b.notna()
        out[m] = b[m]
        return out

    pm = coalesce('pred_margin')
    am = pd.to_numeric(df.get('actual_margin'), errors='coerce')
    mw = pm.notna() & am.notna()
    winners = {
        'n': int(mw.sum()),
        'acc': float(((pm[mw] > 0).astype(int) == (am[mw] > 0).astype(int)).mean()) if mw.sum() > 0 else None,
    }

    pt = coalesce('pred_total')
    # Ensure line is a Series; prefer market_total then closing_total
    if 'market_total' in df.columns:
        line = pd.to_numeric(df['market_total'], errors='coerce')
    elif 'closing_total' in df.columns:
        line = pd.to_numeric(df['closing_total'], errors='coerce')
    else:
        line = pd.Series(np.nan, index=df.index)
    ou = df['ou_result_full'] if 'ou_result_full' in df.columns else pd.Series(np.nan, index=df.index)
    mask_t = pt.notna() & line.notna() & ou.notna() & ou.isin(['Over','Under'])
    totals = {
        'n': int(mask_t.sum()),
        'acc': float((((pt[mask_t] > line[mask_t]).astype(int)) == (ou[mask_t] == 'Over').astype(int)).mean()) if mask_t.sum() > 0 else None,
    }

    sp = pd.to_numeric(df['spread_home'], errors='coerce') if 'spread_home' in df.columns else pd.Series(np.nan, index=df.index)
    ats_res = df.get('ats_result')
    mask_ats = pm.notna() & sp.notna() & ats_res.notna() & ats_res.isin(['Home Cover','Away Cover'])
    ats = {
        'n': int(mask_ats.sum()),
        'acc': float((((pm[mask_ats] > -sp[mask_ats]).astype(int)) == (ats_res[mask_ats] == 'Home Cover').astype(int)).mean()) if mask_ats.sum() > 0 else None,
    }

    return {'date': date_str, 'winners': winners, 'totals': totals, 'ats': ats}

# Finalize dependency (optional, used when daily_results missing)
try:
    from ncaab_model.cli import finalize_day as finalize_day_cli
except Exception:
    finalize_day_cli = None

OUTPUTS = Path('outputs')
METRICS_DIR = OUTPUTS / 'metrics'
DAILY_RESULTS_DIR = OUTPUTS / 'daily_results'

DATE_RE = re.compile(r"predictions_display_(\d{4}-\d{2}-\d{2})\.csv$")


def find_available_dates(end_date: str | None) -> list[str]:
    """Find dates by scanning outputs/predictions_display_YYYY-MM-DD.csv.

    If end_date provided, filter to <= end_date.
    """
    if not OUTPUTS.exists():
        return []
    dates: list[str] = []
    for p in OUTPUTS.glob('predictions_display_*.csv'):
        m = DATE_RE.search(p.name)
        if not m:
            continue
        dates.append(m.group(1))
    dates = sorted(set(dates))
    if end_date:
        try:
            end = dt.date.fromisoformat(end_date)
            dates = [d for d in dates if dt.date.fromisoformat(d) <= end]
        except Exception:
            pass
    return dates


def ensure_daily_results(date: str, finalize_missing: bool = True) -> bool:
    """Ensure outputs/daily_results/results_<date>.csv exists.

    Returns True if results exist after this call, False otherwise.
    """
    DAILY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DAILY_RESULTS_DIR / f'results_{date}.csv'
    if out_path.exists():
        return True
    if not finalize_missing:
        return False
    if finalize_day_cli is None:
        print(json.dumps({'warn': 'finalize_day_cli unavailable; skip finalize', 'date': date}))
        return False
    try:
        # Explicitly pass plain values to avoid Typer OptionInfo defaults leaking
        finalize_day_cli(
            date=date,
            provider='espn',
            games_csv=OUTPUTS / 'games_all.csv',
            predictions_csv=OUTPUTS / 'predictions_week.csv',
            odds_csv=OUTPUTS / 'games_with_last.csv',
            boxscores_csv=OUTPUTS / 'boxscores_prev.csv',
            out_dir=DAILY_RESULTS_DIR,
            overwrite=True,
            include_halves=True,
            halftime_cutoff_min=45,
            secondary_provider='ncaa',
            use_cache=True,
            overrides_csv=None,
        )
    except SystemExit:
        # Typer Exit codes map to SystemExit; file may still be written
        pass
    except Exception as e:
        print(json.dumps({'date': date, 'error': f'finalize failed: {e}'}))
        return out_path.exists()
    return out_path.exists()


def aggregate_metrics(daily: list[dict]) -> dict:
    """Aggregate winners/totals/ats accuracy across days, weighted by counts."""
    agg = {
        'winners': {'n': 0, 'acc': None},
        'totals': {'n': 0, 'acc': None},
        'ats': {'n': 0, 'acc': None},
    }
    # Weighted accuracy = correct / total. Each daily dict has n and acc.
    for key in ['winners', 'totals', 'ats']:
        total_n = 0
        total_correct = 0.0
        for d in daily:
            m = d.get(key) or {}
            n = int(m.get('n') or 0)
            acc = m.get('acc')
            if n > 0 and acc is not None:
                total_n += n
                total_correct += acc * n
        agg[key]['n'] = total_n
        agg[key]['acc'] = (total_correct / total_n) if total_n > 0 else None
    return agg


def run_season(end_date: str | None = None, finalize_missing: bool = True, write_daily_json: bool = True) -> dict:
    """Compute season accuracy up to end_date (inclusive). Optionally finalize missing daily_results."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    end = end_date or yesterday
    dates = find_available_dates(end)
    results: list[dict] = []
    for date in dates:
        ok = ensure_daily_results(date, finalize_missing=finalize_missing)
        if not ok:
            results.append({'date': date, 'error': 'daily_results missing'})
            continue
        res = compute_daily(date)
        results.append(res)
        if write_daily_json:
            (OUTPUTS / f'daily_accuracy_{date}.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    agg = aggregate_metrics(results)
    payload = {
        'end_date': end,
        'dates': dates,
        'aggregate': agg,
    }
    # Write outputs
    (METRICS_DIR / 'season_accuracy_summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    # Also write CSV with per-day metrics
    rows = []
    for r in results:
        date = r.get('date')
        w = r.get('winners') or {}
        t = r.get('totals') or {}
        a = r.get('ats') or {}
        rows.append({
            'date': date,
            'winners_n': w.get('n'), 'winners_acc': w.get('acc'),
            'totals_n': t.get('n'), 'totals_acc': t.get('acc'),
            'ats_n': a.get('n'), 'ats_acc': a.get('acc'),
            'error': r.get('error'),
        })
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / 'season_accuracy_daily.csv', index=False)
    return payload


if __name__ == '__main__':
    # Usage: python scripts/compute_season_accuracy.py [YYYY-MM-DD]
    end_date = sys.argv[1] if len(sys.argv) > 1 else None
    payload = run_season(end_date=end_date, finalize_missing=True, write_daily_json=True)
    print(json.dumps(payload, indent=2))
