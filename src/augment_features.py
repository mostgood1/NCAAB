import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

STAT_SUFFIXES = {
    'fga': ['fga','field_goals_attempted'],
    'fgm': ['fgm','field_goals_made'],
    'fta': ['fta','free_throws_attempted'],
    'ftm': ['ftm','free_throws_made'],
    'tpa': ['tpa','3pa','three_pointers_attempted'],
    'tpm': ['tpm','3pm','three_pointers_made'],
    'orb': ['orb','offensive_rebounds'],
    'drb': ['drb','defensive_rebounds'],
    'to':  ['to','turnovers'],
    'pts': ['pts','points']
}

TEAM_KEYS = ['home_team','away_team']
DATE_KEY = 'date'
GAME_ID = 'game_id'

# Basic timezone abbreviation to offset (hours) mapping for US timezones
TZ_ABBR_OFFSETS = {
    'ET': -5, 'EST': -5, 'EDT': -4,
    'CT': -6, 'CST': -6, 'CDT': -5,
    'MT': -7, 'MST': -7, 'MDT': -6,
    'PT': -8, 'PST': -8, 'PDT': -7,
    'AKT': -9, 'AKST': -9, 'AKDT': -8,
    'HST': -10,
}


def _find_col(df: pd.DataFrame, prefix: str, suffixes: List[str]) -> str | None:
    """Find a column with given prefix and any of suffixes; returns column name or None."""
    for s in suffixes:
        cand = f"{prefix}_{s}"
        if cand in df.columns:
            return cand
    # fallback: exact suffix without prefix (rare)
    for s in suffixes:
        if s in df.columns:
            return s
    return None


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce').fillna(0.0)


def _compute_team_features(row: pd.Series, side: str) -> Dict[str, float]:
    # side: 'home' or 'away'
    opp = 'away' if side == 'home' else 'home'
    # Prefer precomputed metrics if present
    poss_col = f"{side}_possessions"
    to_rate_col = f"{side}_tov_rate"
    orb_rate_col = f"{side}_orb_rate"
    if poss_col in row.index or to_rate_col in row.index or orb_rate_col in row.index:
        poss = float(row.get(poss_col) or 0.0)
        to_rate = float(row.get(to_rate_col) or np.nan)
        # Approximate team DRB% as 1 - opponent ORB%
        opp_orb_rate = float(row.get(f"{opp}_orb_rate") or np.nan)
        drb_rate = (1.0 - opp_orb_rate) if not np.isnan(opp_orb_rate) else np.nan
        return {
            f'{side}_poss': poss,
            f'{side}_pace': poss,
            f'{side}_ts': np.nan,  # not derivable without FGA/FTA
            f'{side}_3p_rate': np.nan,
            f'{side}_to_rate': to_rate,
            f'{side}_drb_rate': drb_rate,
        }
    def gv(prefix: str, key: str) -> float:
        col = _find_col(row.to_frame().T, f"{prefix}", STAT_SUFFIXES[key])
        return float(row.get(col, 0.0) or 0.0) if col else 0.0
    fga = gv(side, 'fga'); fgm = gv(side, 'fgm')
    fta = gv(side, 'fta'); ftm = gv(side, 'ftm')
    tpa = gv(side, 'tpa'); tpm = gv(side, 'tpm')
    orb = gv(side, 'orb'); drb = gv(side, 'drb')
    tov = gv(side, 'to')
    pts = gv(side, 'pts')
    opp_fga = gv(opp, 'fga'); opp_orb = gv(opp, 'orb'); opp_to = gv(opp, 'to'); opp_fta = gv(opp, 'fta')
    # Possessions (estimate) via Hollinger formula
    team_poss = fga + 0.475 * fta - orb + tov
    opp_poss = opp_fga + 0.475 * opp_fta - opp_orb + opp_to
    poss = 0.5 * (team_poss + opp_poss)
    poss = max(poss, 1.0)
    # Metrics
    ts_denom = 2.0 * (fga + 0.475 * fta)
    ts = (pts / ts_denom) if ts_denom > 0 else np.nan
    three_rate = (tpa / fga) if fga > 0 else np.nan
    to_rate = (tov / poss) if poss > 0 else np.nan
    drb_rate = drb / (drb + opp_orb) if (drb + opp_orb) > 0 else np.nan
    return {
        f'{side}_poss': poss,
        f'{side}_pace': poss,  # pace proxy (per-game possessions)
        f'{side}_ts': ts,
        f'{side}_3p_rate': three_rate,
        f'{side}_to_rate': to_rate,
        f'{side}_drb_rate': drb_rate,
    }


def _compute_b2b_and_rest(df: pd.DataFrame) -> pd.DataFrame:
    # Requires DATE_KEY and TEAM_KEYS present
    df = df.copy()
    df[DATE_KEY] = pd.to_datetime(df.get(DATE_KEY), errors='coerce')
    for tk in TEAM_KEYS:
        # Build team date series
        ser = df[[tk, DATE_KEY]].dropna()
        ser = ser.sort_values([tk, DATE_KEY])
        prev = ser.groupby(tk)[DATE_KEY].shift(1)
        days_since = (ser[DATE_KEY] - prev).dt.days
        # Map back to original rows by index alignment
        df[f'{tk}_days_since'] = days_since.reindex(ser.index).reindex(df.index).values
        df[f'{tk}_b2b'] = (df[f'{tk}_days_since'] == 1).astype(float)
    return df


def augment_boxscores(boxscores: pd.DataFrame) -> pd.DataFrame:
    if boxscores.empty:
        return pd.DataFrame()
    df = boxscores.copy()
    # Ensure required identifiers
    for key in [GAME_ID, DATE_KEY] + TEAM_KEYS:
        if key not in df.columns:
            df[key] = df.get(key)  # may be None
    rows: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        base = {
            GAME_ID: row.get(GAME_ID),
            DATE_KEY: row.get(DATE_KEY),
            'home_team': row.get('home_team'),
            'away_team': row.get('away_team'),
        }
        feats_h = _compute_team_features(row, 'home')
        feats_a = _compute_team_features(row, 'away')
        out = {**base, **feats_h, **feats_a}
        rows.append(out)
    aug = pd.DataFrame(rows)
    # Rest/B2B flags
    aug = _compute_b2b_and_rest(aug)
    # Timezone offset hours from start_tz_abbr if present
    if 'start_tz_abbr' in boxscores.columns:
        try:
            tz_series = boxscores['start_tz_abbr'].astype(str).str.upper().str.strip()
            aug['tz_offset_hours'] = tz_series.apply(lambda ab: float(TZ_ABBR_OFFSETS.get(ab, np.nan)))
        except Exception:
            aug['tz_offset_hours'] = np.nan
    else:
        aug['tz_offset_hours'] = np.nan
    # Home advantage: 1 when not neutral site, else 0; fallback NaN when unavailable
    if 'neutral_site' in boxscores.columns:
        try:
            ns = boxscores['neutral_site'].astype(str).str.lower().str.strip()
            ns_flag = ns.isin(['1','true','yes','y','t'])
            aug['home_adv'] = (~ns_flag).astype(float)
        except Exception:
            aug['home_adv'] = np.nan
    else:
        aug['home_adv'] = np.nan
    # Skip rolling aggregates to keep alignment simple when dates are missing
    # Keep a compact set
    keep = [GAME_ID, DATE_KEY, 'home_team','away_team',
            'home_pace','home_ts','home_3p_rate','home_to_rate','home_drb_rate',
            'away_pace','away_ts','away_3p_rate','away_to_rate','away_drb_rate',
            'home_days_since','home_b2b','away_days_since','away_b2b',
            'tz_offset_hours','home_adv']
    for k in list(keep):
        if k not in aug.columns:
            keep.remove(k)
    return aug[keep].copy()


def _load_boxscores() -> pd.DataFrame:
    # Prefer consolidated boxscores; fallback to last2
    candidates = [OUT / 'boxscores.csv', OUT / 'boxscores_last2.csv']
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                return df
            except Exception:
                continue
    return pd.DataFrame()


def save_augmented(aug: pd.DataFrame, out_path: Path) -> None:
    if aug.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aug.to_csv(out_path, index=False)


def run(recent: int | None = None, date: str | None = None) -> Dict[str, str]:
    bs = _load_boxscores()
    if bs.empty:
        return {'status': 'empty_boxscores'}
    bs[DATE_KEY] = pd.to_datetime(bs.get(DATE_KEY), errors='coerce')
    results: Dict[str, str] = {}
    if date:
        d = pd.to_datetime(date, errors='coerce')
        sub = bs[bs[DATE_KEY] == d]
        aug = augment_boxscores(sub)
        out_path = OUT / f'features_augmented_{date}.csv'
        save_augmented(aug, out_path)
        results['file'] = str(out_path)
    else:
        if recent is None:
            recent = 60
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=recent)
        # If dates are missing entirely, process full dataset
        if bs[DATE_KEY].notna().any():
            sub = bs[bs[DATE_KEY] >= cutoff]
        else:
            sub = bs
        aug = augment_boxscores(sub)
        out_path = OUT / 'features_augmented_recent.csv'
        save_augmented(aug, out_path)
        results['file'] = str(out_path)
    results['rows'] = str(int(len(aug))) if isinstance(aug, pd.DataFrame) else '0'
    return results


def main():
    ap = argparse.ArgumentParser(description='Augment features from boxscores to fill missing data (pace, TS%, 3P rate, TO rate, DRB, rest/b2b).')
    ap.add_argument('--recent', type=int, default=60, help='Days back to include')
    ap.add_argument('--date', type=str, default='', help='Specific YYYY-MM-DD date to process')
    args = ap.parse_args()
    date = args.date.strip() or None
    res = run(recent=args.recent, date=date)
    print(res)

if __name__ == '__main__':
    main()
