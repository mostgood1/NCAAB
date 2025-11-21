"""Compute adaptive team-level rolling residual variance.

Uses residuals_history.csv (created by residuals_daily.py) to compute per-team rolling
standard deviation for totals and margin residuals.

Output: outputs/team_variance_<date>.json with structure:
{
  "date": "YYYY-MM-DD",
  "generated_at": ISO_TS,
  "teams": {
     "Marshall": {"total_std": 11.2, "margin_std": 6.3, "n": 28},
     ...
  },
  "global": {"total_std_median": X, "margin_std_median": Y}
}

Logic:
 - Load residuals_history.csv (must contain columns: game_id, actual_total, residual_total, residual_margin, date plus team identification columns if available).
 - If team columns absent, attempt to join games_<date>.csv for team names; else skip.
 - Rolling window: last 30 resolved games per team (or all if <30).
 - Minimum sample size threshold: n>=5 else fallback to global median.
"""
from __future__ import annotations
import json, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

OUT = Path("outputs")

def _safe(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="Target date YYYY-MM-DD (default: today)")
    ap.add_argument("--window", type=int, default=30, help="Rolling window size per team")
    args = ap.parse_args()
    today = dt.datetime.now().strftime("%Y-%m-%d")
    date_str = args.date or today

    hist = _safe(OUT / "residuals_history.csv")
    if hist.empty:
        print("No residuals_history.csv present; skipping team variance generation")
        return
    # Attempt to recover team columns from predictions if missing
    if not {"home_team","away_team"}.issubset(hist.columns):
        # Try merge with games_<date>.csv and games_curr for mapping
        candidates = []
        for gname in [f"games_{date_str}.csv", "games_curr.csv"]:
            candidates.append(_safe(OUT / gname))
        gfull = pd.concat([g for g in candidates if not g.empty], ignore_index=True)
        if not gfull.empty and "game_id" in gfull.columns:
            gfull["game_id"] = gfull["game_id"].astype(str)
            hist["game_id"] = hist["game_id"].astype(str)
            # Only keep necessary mapping columns
            keep = [c for c in ["game_id","home_team","away_team"] if c in gfull.columns]
            if keep:
                hist = hist.merge(gfull[keep].drop_duplicates(), on="game_id", how="left")
    # Build per-row team residual assignment: each game contributes residual_total & residual_margin to both teams (signed for margin?)
    # For variance we only care about magnitude dispersion; use absolute margin residual magnitude.
    rows = []
    if {"home_team","away_team","residual_total","residual_margin"}.issubset(hist.columns):
        for _, r in hist.iterrows():
            rt = r.get("residual_total")
            rm = r.get("residual_margin")
            # Skip unresolved
            if pd.isna(rt) or pd.isna(rm):
                continue
            rows.append({"team": r.get("home_team"), "residual_total": rt, "residual_margin": abs(rm), "date": r.get("date")})
            rows.append({"team": r.get("away_team"), "residual_total": rt, "residual_margin": abs(rm), "date": r.get("date")})
    tdf = pd.DataFrame(rows)
    if tdf.empty:
        print("No team residual rows constructed; exiting")
        return
    # Sort by date for rolling window selection
    tdf_sorted = tdf.sort_values("date")
    out_map = {}
    for team, grp in tdf_sorted.groupby("team"):
        # Keep only last window entries
        last_grp = grp.tail(args.window)
        if last_grp.empty:
            continue
        total_vals = pd.to_numeric(last_grp["residual_total"], errors="coerce").dropna()
        margin_vals = pd.to_numeric(last_grp["residual_margin"], errors="coerce").dropna()
        n = int(min(len(total_vals), len(margin_vals)))
        if n < 2:
            continue
        out_map[team] = {
            "total_std": float(np.std(total_vals, ddof=0)) if len(total_vals) else None,
            "margin_std": float(np.std(margin_vals, ddof=0)) if len(margin_vals) else None,
            "n": n,
        }
    # Global medians for fallback
    all_total_std = [v["total_std"] for v in out_map.values() if v.get("total_std") is not None]
    all_margin_std = [v["margin_std"] for v in out_map.values() if v.get("margin_std") is not None]
    payload = {
        "date": date_str,
        "generated_at": dt.datetime.now().isoformat(),
        "teams": out_map,
        "global": {
            "total_std_median": float(np.median(all_total_std)) if all_total_std else None,
            "margin_std_median": float(np.median(all_margin_std)) if all_margin_std else None,
            "teams_with_stats": len(out_map),
        }
    }
    out_path = OUT / f"team_variance_{date_str}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote team variance JSON to {out_path}")

if __name__ == "__main__":
    main()
