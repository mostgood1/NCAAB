from __future__ import annotations
import glob
import json
from pathlib import Path
import pandas as pd

OUT = Path("outputs")

def _collect_dates(start: str | None = None, end: str | None = None, recent: int | None = None) -> list[str]:
    files = sorted(glob.glob(str(OUT / "daily_results" / "results_*.csv")))
    dates = [Path(f).stem.replace("results_", "") for f in files]
    if start or end:
        def in_range(d):
            return (start is None or d >= start) and (end is None or d <= end)
        dates = [d for d in dates if in_range(d)]
    if recent:
        dates = dates[-recent:]
    return dates

def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()

def _join_preds_results(date: str) -> pd.DataFrame:
    preds = _safe_read_csv(OUT / f"predictions_unified_enriched_{date}.csv")
    res = _safe_read_csv(OUT / f"daily_results/results_{date}.csv")
    if preds.empty or res.empty:
        return pd.DataFrame()
    for d in (preds, res):
        if "game_id" in d.columns:
            d["game_id"] = d["game_id"].astype(str)
    keys = [c for c in ["game_id","home_team","away_team","date"] if c in preds.columns and c in res.columns]
    if not keys:
        keys = [c for c in ["home_team","away_team"] if c in preds.columns and c in res.columns]
    # Select minimal result columns to avoid suffix collisions
    keep = list(set(keys + [c for c in ["market_total","actual_total","final_total"] if c in res.columns]))
    res_min = res[keep].copy()
    # Normalize and disambiguate result-side columns to avoid suffix collisions
    if "actual_total" not in res_min.columns and "final_total" in res_min.columns:
        res_min = res_min.rename(columns={"final_total": "actual_total"})
    if "market_total" in res_min.columns:
        res_min = res_min.rename(columns={"market_total": "market_total_res"})
    if "actual_total" in res_min.columns:
        res_min = res_min.rename(columns={"actual_total": "actual_total_res"})
    merged = pd.merge(preds, res_min, on=keys, how="inner")
    return merged

def _compute_ou_accuracy(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"count": 0, "correct": 0, "pct": None}
    # Prefer result-side totals when available
    mt = df.get("market_total_res") if "market_total_res" in df.columns else df.get("market_total")
    at = df.get("actual_total_res") if "actual_total_res" in df.columns else df.get("actual_total")
    mt = pd.to_numeric(mt, errors="coerce") if mt is not None else pd.Series(dtype=float)
    at = pd.to_numeric(at, errors="coerce") if at is not None else pd.Series(dtype=float)
    res_ok = mt.notna() & at.notna()
    df = df[res_ok].copy()
    if df.empty:
        return {"count": 0, "correct": 0, "pct": None}
    out = {}
    # Model raw
    if "pred_total_model" in df.columns:
        pm = pd.to_numeric(df["pred_total_model"], errors="coerce")
        mask = (pm.notna() & mt.notna() & at.notna()).reindex(df.index, fill_value=False)
        idx = df.index[mask]
        diff_pred = (pm.loc[idx] - mt.loc[idx])
        diff_act = (at.loc[idx] - mt.loc[idx])
        correct = ((diff_pred > 0) == (diff_act > 0)).sum()
        cnt = int(len(idx))
        out["model"] = {"count": cnt, "correct": int(correct), "pct": (float(correct)/cnt if cnt else None)}
    # Calibrated q50
    if "pred_total_calibrated" in df.columns:
        pc = pd.to_numeric(df["pred_total_calibrated"], errors="coerce")
        mask = (pc.notna() & mt.notna() & at.notna()).reindex(df.index, fill_value=False)
        idx = df.index[mask]
        diff_pred = (pc.loc[idx] - mt.loc[idx])
        diff_act = (at.loc[idx] - mt.loc[idx])
        correct = ((diff_pred > 0) == (diff_act > 0)).sum()
        cnt = int(len(idx))
        out["cal"] = {"count": cnt, "correct": int(correct), "pct": (float(correct)/cnt if cnt else None)}
    # Blend basis rows: use whichever value was applied in display via basis
    if "pred_total_basis" in df.columns:
        mask_basis = (df["pred_total_basis"].isin(["blend"]) & mt.notna() & at.notna()).reindex(df.index, fill_value=False)
        idx = df.index[mask_basis]
        # Prefer calibrated value, fallback to model
        pv = pd.to_numeric(df.get("pred_total_calibrated"), errors="coerce")
        pv = pv.where(pv.notna(), pd.to_numeric(df.get("pred_total_model"), errors="coerce"))
        pv = pv.loc[idx]
        pv = pv[pv.notna()]
        idx = pv.index
        if len(idx):
            diff_pred = (pv - mt.loc[idx])
            diff_act = (at.loc[idx] - mt.loc[idx])
            correct = ((diff_pred > 0) == (diff_act > 0)).sum()
            cnt = int(len(idx))
            out["blend"] = {"count": cnt, "correct": int(correct), "pct": (float(correct)/cnt if cnt else None)}
    # Threshold-based accuracy for higher-confidence picks
    thresholds = [3, 5, 7, 10]
    out_thresh = {}
    # Prepare common aligned series
    pm = pd.to_numeric(df.get("pred_total_model"), errors="coerce") if "pred_total_model" in df.columns else pd.Series(index=df.index, dtype=float)
    pc = pd.to_numeric(df.get("pred_total_calibrated"), errors="coerce") if "pred_total_calibrated" in df.columns else pd.Series(index=df.index, dtype=float)
    for t in thresholds:
        # Model
        mask_m = (pm.notna() & mt.notna() & at.notna() & ((pm - mt).abs() >= t)).reindex(df.index, fill_value=False)
        idx_m = df.index[mask_m]
        corr_m = (((pm.loc[idx_m] - mt.loc[idx_m]) > 0) == ((at.loc[idx_m] - mt.loc[idx_m]) > 0)).sum()
        cnt_m = int(len(idx_m))
        # Calibrated
        mask_c = (pc.notna() & mt.notna() & at.notna() & ((pc - mt).abs() >= t)).reindex(df.index, fill_value=False)
        idx_c = df.index[mask_c]
        corr_c = (((pc.loc[idx_c] - mt.loc[idx_c]) > 0) == ((at.loc[idx_c] - mt.loc[idx_c]) > 0)).sum()
        cnt_c = int(len(idx_c))
        out_thresh[str(t)] = {
            "model": {"count": cnt_m, "correct": int(corr_m), "pct": (float(corr_m)/cnt_m if cnt_m else None)},
            "cal": {"count": cnt_c, "correct": int(corr_c), "pct": (float(corr_c)/cnt_c if cnt_c else None)},
        }
    out["by_threshold"] = out_thresh
    # Tails strategy: pick over if market_total <= q10c; pick under if market_total >= q90c
    tails = {}
    if "pred_total_q10" in df.columns and "pred_total_q90" in df.columns:
        q10c = pd.to_numeric(df["pred_total_q10"], errors="coerce")
        q90c = pd.to_numeric(df["pred_total_q90"], errors="coerce")
        # Align indices before comparisons
        mt_a = mt.reindex(df.index)
        at_a = at.reindex(df.index)
        q10_a = q10c.reindex(df.index)
        q90_a = q90c.reindex(df.index)
        mask_over = (mt_a.notna() & at_a.notna() & q10_a.notna() & (mt_a <= q10_a)).reindex(df.index, fill_value=False)
        idx_o = df.index[mask_over]
        corr_o = ((at_a.loc[idx_o] > mt_a.loc[idx_o])).sum()
        cnt_o = int(len(idx_o))
        mask_under = (mt_a.notna() & at_a.notna() & q90_a.notna() & (mt_a >= q90_a)).reindex(df.index, fill_value=False)
        idx_u = df.index[mask_under]
        corr_u = ((at_a.loc[idx_u] < mt_a.loc[idx_u])).sum()
        cnt_u = int(len(idx_u))
        tails = {
            "over": {"count": cnt_o, "correct": int(corr_o), "pct": (float(corr_o)/cnt_o if cnt_o else None)},
            "under": {"count": cnt_u, "correct": int(corr_u), "pct": (float(corr_u)/cnt_u if cnt_u else None)},
            "total": {"count": (cnt_o + cnt_u), "correct": int(corr_o + corr_u), "pct": (float(corr_o + corr_u)/(cnt_o + cnt_u) if (cnt_o + cnt_u) else None)}
        }
    out["tails"] = tails
    # Probability picks from quantiles: estimate CDF via piecewise-linear between q10/q50/q90
    prob_picks = {}
    if "pred_total_q10" in df.columns and "pred_total_q50" in df.columns and "pred_total_q90" in df.columns:
        q10c = pd.to_numeric(df["pred_total_q10"], errors="coerce").reindex(df.index)
        q50c = pd.to_numeric(df["pred_total_q50"], errors="coerce").reindex(df.index)
        q90c = pd.to_numeric(df["pred_total_q90"], errors="coerce").reindex(df.index)
        mt_a = mt.reindex(df.index)
        at_a = at.reindex(df.index)
        thresholds = [0.6, 0.65, 0.7]
        for th in thresholds:
            # Compute F(market)
            F = pd.Series(index=df.index, dtype=float)
            # Below q10
            mask_b = mt_a.notna() & q10c.notna() & (mt_a <= q10c)
            F.loc[mask_b] = 0.1 * (mt_a.loc[mask_b] - (mt_a.loc[mask_b])) / (q10c.loc[mask_b] - (mt_a.loc[mask_b])) if False else 0.1
            # Between q10 and q50
            mask_m1 = mt_a.notna() & q10c.notna() & q50c.notna() & (mt_a > q10c) & (mt_a <= q50c)
            F.loc[mask_m1] = 0.1 + 0.4 * ((mt_a.loc[mask_m1] - q10c.loc[mask_m1]) / (q50c.loc[mask_m1] - q10c.loc[mask_m1])).clip(lower=0, upper=1)
            # Between q50 and q90
            mask_m2 = mt_a.notna() & q50c.notna() & q90c.notna() & (mt_a > q50c) & (mt_a <= q90c)
            F.loc[mask_m2] = 0.5 + 0.4 * ((mt_a.loc[mask_m2] - q50c.loc[mask_m2]) / (q90c.loc[mask_m2] - q50c.loc[mask_m2])).clip(lower=0, upper=1)
            # Above q90
            mask_a = mt_a.notna() & q90c.notna() & (mt_a > q90c)
            F.loc[mask_a] = 0.9
            P_over = 1.0 - F
            P_under = F
            # Picks
            idx_over = df.index[(P_over >= th) & mt_a.notna() & at_a.notna()]
            idx_under = df.index[(P_under >= th) & mt_a.notna() & at_a.notna()]
            corr_over = (at_a.loc[idx_over] > mt_a.loc[idx_over]).sum()
            corr_under = (at_a.loc[idx_under] < mt_a.loc[idx_under]).sum()
            cnt_over = int(len(idx_over))
            cnt_under = int(len(idx_under))
            prob_picks[str(th)] = {
                "over": {"count": cnt_over, "correct": int(corr_over), "pct": (float(corr_over)/cnt_over if cnt_over else None)},
                "under": {"count": cnt_under, "correct": int(corr_under), "pct": (float(corr_under)/cnt_under if cnt_under else None)},
                "total": {"count": (cnt_over + cnt_under), "correct": int(corr_over + corr_under), "pct": (float(corr_over + corr_under)/(cnt_over + cnt_under) if (cnt_over + cnt_under) else None)}
            }
    out["prob_picks"] = prob_picks
    return out

def evaluate(start: str | None = None, end: str | None = None, recent: int | None = None) -> dict:
    dates = _collect_dates(start, end, recent)
    frames = []
    for d in dates:
        joined = _join_preds_results(d)
        if not joined.empty:
            frames.append(joined)
    if not frames:
        return {"dates": dates, "by_basis": {}, "error": "No joined predictions/results"}
    df = pd.concat(frames, ignore_index=True)
    by_basis = _compute_ou_accuracy(df)
    payload = {"dates": dates, "by_basis": by_basis}
    (OUT / "eval_ou_accuracy.json").write_text(json.dumps(payload, indent=2))
    return payload

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate over/under percent correct for predictions vs actuals")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--recent", type=int, default=14)
    args = ap.parse_args()
    payload = evaluate(args.start, args.end, args.recent)
    print(json.dumps(payload))

if __name__ == "__main__":
    main()