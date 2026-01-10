from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

OUT = Path("outputs")

def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()

def _load_calibration_edges() -> list[float] | None:
    p = OUT / "calibration_quantiles_crps_segmented.json"
    if p.exists():
        try:
            obj = json.loads(p.read_text())
            edges = obj.get("edges")
            if isinstance(edges, list) and len(edges) == 2:
                return [float(edges[0]), float(edges[1])]
        except Exception:
            return None
    return None

def _tempo_bin(mt: float, edges: list[float] | None) -> str:
    if not np.isfinite(mt):
        return "unknown"
    if not edges:
        # Fallback: rough tertiles
        if mt < 145:
            return "low"
        elif mt < 153:
            return "mid"
        return "high"
    e1, e2 = edges
    if mt < e1:
        return "low"
    if mt < e2:
        return "mid"
    return "high"

def build_for_date(date_str: str) -> Path | None:
    enrich_path = OUT / f"predictions_unified_enriched_{date_str}.csv"
    df = _safe_read_csv(enrich_path)
    if df.empty:
        return None
    # Normalize key columns
    for c in ("game_id","date"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    # Ensure numeric
    def num(c):
        return pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
    market_total = num("market_total") if "market_total" in df.columns else num("closing_total")
    closing_total = num("closing_total")
    q10 = num("q10")
    q50 = num("q50")
    q90 = num("q90")
    pred_cal = num("pred_total_calibrated")
    # Derived deltas and spreads
    df_out = pd.DataFrame(index=df.index)
    df_out["game_id"] = df.get("game_id", df.index.astype(str))
    df_out["date"] = df.get("date", date_str)
    # Team columns for robust fallback joins with augmented features
    if "home_team" in df.columns:
        df_out["home_team"] = df["home_team"].astype(str)
    if "away_team" in df.columns:
        df_out["away_team"] = df["away_team"].astype(str)
    df_out["market_total"] = market_total
    df_out["closing_total"] = closing_total
    df_out["q10"] = q10
    df_out["q50"] = q50
    df_out["q90"] = q90
    df_out["pred_total_calibrated"] = pred_cal
    df_out["delta_q50_market"] = q50 - market_total
    df_out["delta_cal_market"] = pred_cal - market_total
    df_out["spread_q"] = q90 - q10
    # Tempo bins
    edges = _load_calibration_edges()
    df_out["tempo_bin"] = [ _tempo_bin(mt, edges) for mt in df_out["market_total"].tolist() ]
    df_out["tempo_bin_low"] = (df_out["tempo_bin"] == "low").astype(int)
    df_out["tempo_bin_mid"] = (df_out["tempo_bin"] == "mid").astype(int)
    df_out["tempo_bin_high"] = (df_out["tempo_bin"] == "high").astype(int)
    # Proxies for tempo ratings
    df_out["tempo_rating_sum"] = df_out["market_total"]
    df_out["away_tempo_rating"] = df_out["market_total"] / 2.0
    df_out["home_tempo_rating"] = df_out["market_total"] / 2.0
    # Neutral site flag passthrough or default
    df_out["neutral_site"] = df["neutral_site"] if "neutral_site" in df.columns else 0
    # Start timezone passthrough if present
    if "start_tz_abbr" in df.columns:
        df_out["start_tz_abbr"] = df["start_tz_abbr"]
    # Rest flags: default 0 (can be populated by schedule integration later)
    df_out["rest_home"] = 0
    df_out["rest_away"] = 0
    out_path = OUT / f"features_{date_str}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="Date YYYY-MM-DD; if omitted, build for all enriched files", default=None)
    args = ap.parse_args()
    if args.date:
        p = build_for_date(args.date)
        if p:
            print(json.dumps({"built": str(p)}))
        else:
            print(json.dumps({"error": "No enriched file for date"}))
    else:
        files = sorted(glob.glob(str(OUT / "predictions_unified_enriched_*.csv")))
        built = []
        for f in files:
            d = Path(f).stem.replace("predictions_unified_enriched_", "")
            p = build_for_date(d)
            if p:
                built.append(str(p))
        print(json.dumps({"built_count": len(built)}))

if __name__ == "__main__":
    main()