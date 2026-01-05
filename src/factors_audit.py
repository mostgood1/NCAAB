from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

DEFAULT_EXPECTED: dict[str, list[str]] = {
    # Regex-like substrings to look for in feature column names
    "tempo_pace": ["pace", "tempo", "possessions"],
    "efficiency": ["off_eff", "def_eff", "off_rating", "def_rating", "adjusted"],
    "shooting": ["fg", "3p", "3pa", "ft", "ts%", "efg"],
    "turnovers": ["to_rate", "turnover"],
    "rebounding": ["orb", "drb", "reb", "rebound"],
    "fouls": ["pf", "foul"],
    "rest_schedule": ["rest", "days_since", "back_to_back"],
    "travel_tz": ["travel", "distance", "timezone", "tz", "altitude"],
    "venue": ["neutral", "home_adv", "venue"],
}


def _parse_date(s: str | None) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return dt.datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        return None


def _resolve_dates(start: Optional[str], end: Optional[str], recent: Optional[int]) -> list[str]:
    if recent:
        cand = sorted([p for p in OUT.glob("features_*.csv")])
        tokens = []
        for p in cand:
            try:
                tokens.append(p.stem.split("_")[-1])
            except Exception:
                continue
        dates = [t for t in tokens if len(t) == 10 and t[4] == '-' and t[7] == '-']
        return dates[-recent:]
    if start and end:
        dt_start = dt.datetime.strptime(start, "%Y-%m-%d")
        dt_end = dt.datetime.strptime(end, "%Y-%m-%d")
        step = (dt_end - dt_start).days
        if step < 0:
            dt_start, dt_end = dt_end, dt_start
            step = -step
        return [(dt_start + dt.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(step + 1)]
    if start and not end:
        return [start]
    today = dt.datetime.now().strftime("%Y-%m-%d")
    return [today]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _load_expected_config() -> dict[str, list[str]]:
    cfg_path = OUT / "factors_expected.json"
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): [str(x) for x in v] for k, v in data.items() if isinstance(v, list)}
        except Exception:
            pass
    return DEFAULT_EXPECTED


def _coerce_actual_total(df: pd.DataFrame) -> pd.Series:
    for cand in ("actual_total", "final_total", "total_final"):
        if cand in df.columns:
            try:
                return pd.to_numeric(df[cand], errors="coerce")
            except Exception:
                pass
    # Try score columns
    score_cols = [c for c in df.columns if str(c).lower() in ("score_home", "home_score", "home_points")]
    score_cols_away = [c for c in df.columns if str(c).lower() in ("score_away", "away_score", "away_points")]
    if score_cols and score_cols_away:
        try:
            sh = pd.to_numeric(df[score_cols[0]], errors="coerce")
            sa = pd.to_numeric(df[score_cols_away[0]], errors="coerce")
            return sh + sa
        except Exception:
            pass
    return pd.Series([np.nan] * len(df))


def audit_factors(start: Optional[str], end: Optional[str], recent: Optional[int]) -> dict[str, Any]:
    dates = _resolve_dates(start, end, recent)
    # Load features for dates; try date-specific files first, fallback to aggregate
    feats_list = []
    for d in dates:
        for pat in [OUT / f"features_{d}.csv", OUT / "features_curr.csv", OUT / "features_all.csv"]:
            df = _safe_read_csv(pat)
            if not df.empty:
                df["_src"] = str(pat)
                feats_list.append(df)
                break
    feats = pd.concat(feats_list, ignore_index=True) if feats_list else pd.DataFrame()
    # Load results for actual totals
    results_list = []
    for d in dates:
        p = OUT / "daily_results" / f"results_{d}.csv"
        df = _safe_read_csv(p)
        if not df.empty:
            df["_src"] = str(p)
            results_list.append(df)
    results = pd.concat(results_list, ignore_index=True) if results_list else pd.DataFrame()
    payload: dict[str, Any] = {"dates": dates, "features_rows": int(len(feats)), "results_rows": int(len(results))}
    if feats.empty or results.empty:
        payload["error"] = "Missing features or results"
        return payload
    # Normalize keys
    for d in (feats, results):
        if "game_id" in d.columns:
            d["game_id"] = d["game_id"].astype(str)
        if "date" in d.columns:
            d["date"] = d["date"].astype(str)
    join_keys = [k for k in ("game_id", "date") if k in feats.columns and k in results.columns]
    if not join_keys:
        join_keys = ["game_id"]
    df = feats.merge(results, on=join_keys, how="inner", suffixes=("", "_res"))
    if df.empty:
        payload["error"] = "Join produced no rows"
        return payload
    df["actual_total"] = _coerce_actual_total(df)
    # Expected factor categories
    expected = _load_expected_config()
    present: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    cols = [str(c) for c in df.columns]
    for cat, needles in expected.items():
        found = []
        for n in needles:
            for c in cols:
                if n.lower() in c.lower():
                    found.append(c)
        present[cat] = sorted(list(set(found)))
        missing_needles = [n for n in needles if not any(n.lower() in c.lower() for c in cols)]
        missing[cat] = missing_needles
    # Simple correlation with actual total (numeric columns only)
    num_df = df.select_dtypes(include=[np.number]).copy()
    # Drop obvious non-feature columns
    for drop in ("actual_total",):
        if drop in num_df.columns:
            pass
    corrs = {}
    try:
        y = pd.to_numeric(df["actual_total"], errors="coerce")
        for c in num_df.columns:
            if c == "actual_total":
                continue
            try:
                x = pd.to_numeric(num_df[c], errors="coerce")
                if x.notna().sum() > 20 and y.notna().sum() > 20:
                    corr = float(pd.concat([x, y], axis=1).corr().iloc[0, 1])
                    if np.isfinite(corr):
                        corrs[c] = corr
            except Exception:
                continue
    except Exception:
        corrs = {}
    # Top correlated features
    top_pos = sorted([(k, v) for k, v in corrs.items() if v > 0], key=lambda x: -x[1])[:20]
    top_neg = sorted([(k, v) for k, v in corrs.items() if v < 0], key=lambda x: x[1])[:20]
    payload.update({
        "present": present,
        "missing": missing,
        "top_corr_pos": top_pos,
        "top_corr_neg": top_neg,
    })
    out_path = OUT / f"factors_audit_{dates[0]}_{dates[-1]}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    ap = argparse.ArgumentParser(description="Audit features coverage and correlation for totals modeling")
    ap.add_argument("--start", type=str, help="Start date YYYY-MM-DD", default=None)
    ap.add_argument("--end", type=str, help="End date YYYY-MM-DD", default=None)
    ap.add_argument("--recent", type=int, help="Use N most recent feature files", default=None)
    args = ap.parse_args()
    payload = audit_factors(_parse_date(args.start), _parse_date(args.end), args.recent)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
