"""Feature leakage scan.
Identifies potentially leaky feature columns that might embed post-game information (final scores, closing lines, etc.).

Heuristics:
  1. Name-based: column name matches /(final|score|actual|closing|result|winner|margin_actual)/i
  2. Value-based: near-perfect correlation (>|0.985|) or equality with actual outcome for completed games:
     - actual_total, actual_margin, home_score, away_score.
  3. Temporal mismatch: feature row date equals or exceeds game date plus a buffer (suggests after-the-fact enrichment) when a timestamp column present.

Output JSON: outputs/leakage_<date>.json
Structure:
{
  "date": ..., "generated_at": ..., "n_games": int,
  "suspicious_columns": [ {"column": str, "reasons": [..], "corr_actual_total": float|null, ...} ],
  "summary": {"n_suspicious": int, "name_flagged": int, "corr_flagged": int, "temporal_flagged": int}
}

Usage:
  python scripts/leakage_scan.py --date YYYY-MM-DD
"""
from __future__ import annotations
import argparse, json, datetime as dt, re
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("outputs")
NAME_PATTERN = re.compile(r"(final|score|actual|closing|result|winner|margin_actual)", re.IGNORECASE)


def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def _load_features(date_str: str) -> pd.DataFrame:
    # Prefer dated features_<date>.csv then features_curr.csv then features.csv
    for name in [f"features_{date_str}.csv", "features_curr.csv", "features.csv"]:
        df = _safe_read_csv(OUT / name)
        if not df.empty:
            return df
    # Historical fallback (with priors)
    df = _safe_read_csv(OUT / f"features_{date_str}_with_priors.csv")
    return df


def _load_daily_results(date_str: str) -> pd.DataFrame:
    p = OUT / "daily_results" / f"results_{date_str}.csv"
    return _safe_read_csv(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="Target date YYYY-MM-DD (default today)")
    args = ap.parse_args()
    date_str = args.date or dt.date.today().strftime("%Y-%m-%d")

    feats = _load_features(date_str)
    results = _load_daily_results(date_str)
    suspicious = []
    reasons_map = {}

    if not feats.empty and "game_id" in feats.columns:
        feats["game_id"] = feats["game_id"].astype(str)
    if not results.empty and "game_id" in results.columns:
        results["game_id"] = results["game_id"].astype(str)

    # Merge outcomes for correlation tests
    merged = pd.DataFrame()
    if not feats.empty and not results.empty and "game_id" in feats.columns and "game_id" in results.columns:
        merged = feats.merge(results[[c for c in ["game_id","home_score","away_score","actual_total","pred_total"] if c in results.columns]], on="game_id", how="left")
        # Derive actual_margin when scores present
        if {"home_score","away_score"}.issubset(merged.columns):
            try:
                hs = pd.to_numeric(merged["home_score"], errors="coerce")
                as_ = pd.to_numeric(merged["away_score"], errors="coerce")
                merged["actual_margin"] = hs - as_
                merged["actual_total"] = merged.get("actual_total", hs + as_)
            except Exception:
                pass

    outcome_cols = [c for c in ["actual_total","actual_margin","home_score","away_score"] if c in merged.columns]

    # Identify candidate feature columns (exclude obvious identifiers / meta)
    exclude_prefixes = {"game_id","home_team","away_team","date","start_time"}
    feature_cols = [c for c in feats.columns if c not in exclude_prefixes and not c.startswith("_")]

    for col in feature_cols:
        col_ser = pd.to_numeric(feats[col], errors="coerce") if feats[col].dtype.kind in "biufc" else feats[col].astype(str)
        rlist = []
        # Name-based flag
        if NAME_PATTERN.search(col):
            rlist.append("name_pattern")
        # Correlation-based flags
        corr_vals = {}
        if outcome_cols and merged.notna().any(axis=1):
            try:
                if col in merged.columns:
                    fnum = pd.to_numeric(merged[col], errors="coerce")
                    for outc in outcome_cols:
                        onum = pd.to_numeric(merged[outc], errors="coerce")
                        pair = pd.DataFrame({"f": fnum, "o": onum}).dropna()
                        if len(pair) >= 8 and pair["f"].nunique() > 1 and pair["o"].nunique() > 1:
                            corr = pair.corr().iloc[0,1]
                            corr_vals[outc] = float(corr)
                            if abs(corr) > 0.985:
                                rlist.append(f"corr_{outc}")
                        else:
                            corr_vals[outc] = None
            except Exception:
                pass
        # Equality check: if values equal outcome exactly for >60% of completed games
        try:
            if col in merged.columns and "actual_total" in merged.columns:
                fnum = pd.to_numeric(merged[col], errors="coerce")
                at = pd.to_numeric(merged["actual_total"], errors="coerce")
                valid = fnum.notna() & at.notna()
                if valid.sum() >= 8:
                    eq_rate = (np.isclose(fnum[valid], at[valid], atol=0.0)).mean()
                    if eq_rate > 0.60:
                        rlist.append("direct_match_actual_total")
                        corr_vals["eq_rate_actual_total"] = float(eq_rate)
        except Exception:
            pass
        # Temporal mismatch: feature has timestamp after game date (columns ending _updated/_ts/_time)
        try:
            date_cols = [c for c in feats.columns if re.search(r"(updated|timestamp|ts|time)$", c, re.IGNORECASE)]
            if date_cols and "date" in feats.columns:
                fdate = pd.to_datetime(feats.get("date"), errors="coerce")
                for dc in date_cols:
                    tser = pd.to_datetime(feats.get(dc), errors="coerce")
                    if tser.notna().any() and fdate.notna().any():
                        # Flag if any timestamp > (game date + 4 hours)
                        mask_future = (tser - fdate) > pd.Timedelta(hours=4)
                        if mask_future.sum() > 0 and dc in col:
                            rlist.append("temporal_future")
        except Exception:
            pass
        if rlist:
            suspicious.append({
                "column": col,
                "reasons": sorted(set(rlist)),
                **{f"corr_{k}": v for k,v in corr_vals.items()}  # embed correlation metrics
            })

    summary = {
        "n_suspicious": len(suspicious),
        "name_flagged": sum(1 for s in suspicious if any(r.startswith("name") for r in s["reasons"])),
        "corr_flagged": sum(1 for s in suspicious if any(r.startswith("corr") for r in s["reasons"])),
        "temporal_flagged": sum(1 for s in suspicious if any(r == "temporal_future" for r in s["reasons"]))
    }

    payload = {
        "date": date_str,
        "generated_at": dt.datetime.utcnow().isoformat(),
        "n_games": int(results.shape[0]) if not results.empty else 0,
        "suspicious_columns": suspicious,
        "summary": summary,
    }
    OUT.mkdir(exist_ok=True, parents=True)
    out_path = OUT / f"leakage_{date_str}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Leakage scan complete -> {out_path} (suspicious={summary['n_suspicious']})")

if __name__ == "__main__":
    main()
