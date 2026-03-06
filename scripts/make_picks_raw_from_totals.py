import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

DEF_OUT = Path(__file__).resolve().parents[1] / "outputs"


def _dedupe_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "game_id" not in df.columns:
        return df
    work = df.copy()
    try:
        work["game_id"] = work["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        work = work[work["game_id"].ne("")].copy()
    except Exception:
        pass
    if work.empty:
        return work
    score = pd.Series(0, index=work.index, dtype=float)
    for col in ["market_total", "closing_total", "total", "pred_total", "p_over_blend", "p_over", "p_over_final"]:
        if col in work.columns:
            try:
                score = score + pd.to_numeric(work[col], errors="coerce").notna().astype(float)
            except Exception:
                pass
    work["_score"] = score
    work = work.sort_values(["game_id", "_score"], ascending=[True, False], na_position="last")
    work = work.drop_duplicates(subset=["game_id"], keep="first")
    return work.drop(columns=["_score"], errors="ignore").reset_index(drop=True)


def _find_prob_col(df: pd.DataFrame) -> str | None:
    for c in [
        "p_over_blend",  # preferred when sim blend available
        "p_over_sim",
        "p_over_quantile",
        "p_over_final",
        "p_over_meta_cal",
        "p_over_display",
        "p_over",
        "p_over_emp",
        "p_over_dist",
        "p_over_ensemble",
    ]:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return None


def _find_market_total_col(df: pd.DataFrame, prefer_closing: bool = True) -> str | None:
    cols = ["closing_total", "market_total", "total"] if prefer_closing else ["market_total", "closing_total", "total"]
    for c in cols:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return None


def _find_pred_total_col(df: pd.DataFrame) -> str | None:
    # Prefer calibrated model totals first
    for c in ["pred_total_calibrated", "pred_total", "pred_total_market_blend", "pred_total_blend"]:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return None

def _find_sigma_col(df: pd.DataFrame) -> str | None:
    for c in ["sigma_total_quantile", "sigma_total_emp", "sigma_total_adj", "pred_total_sigma"]:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return None


def _load_enriched_or_display(outputs: Path, date: str) -> pd.DataFrame:
    # Prefer enriched first (it contains probability columns), then fallback to display
    enr = outputs / f"predictions_unified_enriched_{date}.csv"
    disp = outputs / f"predictions_display_{date}.csv"
    df = None
    if enr.exists():
        try:
            df = pd.read_csv(enr)
        except Exception:
            df = None
    if df is None and disp.exists():
        try:
            df = pd.read_csv(disp)
        except Exception:
            df = None
    return df if df is not None else pd.DataFrame()

def _try_merge_sim_blend(df: pd.DataFrame, outputs: Path, date: str) -> pd.DataFrame:
    """If sim_blend_<date>.csv exists, merge its probabilities and market_total into df."""
    simb = outputs / f"sim_blend_{date}.csv"
    if not simb.exists() or df.empty:
        return df
    try:
        sb = pd.read_csv(simb)
        # normalize keys
        if "game_id" in df.columns and "game_id" in sb.columns:
            df["game_id"] = df["game_id"].astype(str)
            sb["game_id"] = sb["game_id"].astype(str)
            df = df.merge(sb[["game_id","p_over_blend","p_over_sim","market_total","mu_total_sim","q50_total_sim"]], on="game_id", how="left")
        elif all(c in df.columns for c in ["home_team","away_team"]) and all(c in sb.columns for c in ["home_team","away_team"]):
            df = df.merge(sb[["home_team","away_team","p_over_blend","p_over_sim","market_total","mu_total_sim","q50_total_sim"]], on=["home_team","away_team"], how="left")
    except Exception:
        pass
    return df

def _merge_quantiles_for_date(df: pd.DataFrame, outputs: Path, date: str) -> pd.DataFrame:
    qhist = outputs / "quantiles_history.csv"
    if not qhist.exists() or df.empty:
        return df
    try:
        qh = pd.read_csv(qhist)
        qh["game_id"] = qh["game_id"].astype(str)
        qh["date"] = pd.to_datetime(qh["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        qh = qh[qh["date"] == date]
        df = df.merge(qh, on=["game_id"], how="left")
        q10 = pd.to_numeric(df.get("q10_total"), errors="coerce")
        q50 = pd.to_numeric(df.get("q50_total"), errors="coerce")
        q90 = pd.to_numeric(df.get("q90_total"), errors="coerce")
        line = pd.to_numeric(df.get("closing_total"), errors="coerce")
        if line.isna().all():
            line = pd.to_numeric(df.get("market_total"), errors="coerce")
        if line.isna().all() and "total" in df.columns:
            line = pd.to_numeric(df.get("total"), errors="coerce")
        cdf = pd.Series(np.nan, index=df.index)
        mid1 = line.notna() & q10.notna() & q50.notna() & (line >= q10) & (line <= q50)
        cdf.loc[mid1] = 0.1 + 0.4 * ((line[mid1] - q10[mid1]) / (q50[mid1] - q10[mid1]).replace(0, np.nan))
        mid2 = line.notna() & q50.notna() & q90.notna() & (line > q50) & (line <= q90)
        cdf.loc[mid2] = 0.5 + 0.4 * ((line[mid2] - q50[mid2]) / (q90[mid2] - q50[mid2]).replace(0, np.nan))
        left = line.notna() & q10.notna() & (line < q10)
        cdf.loc[left] = 0.1 * (line[left] / q10[left]).replace(0, np.nan)
        right = line.notna() & q90.notna() & (line > q90)
        cdf.loc[right] = 0.9 + 0.1 * ((line[right] - q90[right]) / q90[right]).replace(0, np.nan)
        df["p_over_quantile"] = 1.0 - cdf
        df["sigma_total_quantile"] = (q90 - q10) / 2.563103131089201
        return df
    except Exception:
        return df


def build_totals_picks(date: str, outputs: Path, p_hi: float, p_lo: float) -> pd.DataFrame:
    df = _load_enriched_or_display(outputs, date)
    if df.empty:
        return pd.DataFrame()
    # Normalize IDs
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)
        df = _dedupe_games(df)
    # Merge sim blend if available
    df = _try_merge_sim_blend(df, outputs, date)
    # Merge quantiles for the date to ensure p_over_quantile and sigma_total_quantile are available
    df = _merge_quantiles_for_date(df, outputs, date)
    prob_col = _find_prob_col(df)
    market_col = _find_market_total_col(df, prefer_closing=True)
    pred_col = _find_pred_total_col(df)
    sigma_col = _find_sigma_col(df)
    po = pd.to_numeric(df[prob_col], errors="coerce") if prob_col else pd.Series(np.nan, index=df.index)
    mkt = pd.to_numeric(df[market_col], errors="coerce") if market_col else pd.Series(np.nan, index=df.index)
    pred = pd.to_numeric(df[pred_col], errors="coerce") if pred_col else pd.Series(np.nan, index=df.index)
    sigma = pd.to_numeric(df[sigma_col], errors="coerce") if sigma_col else pd.Series(np.nan, index=df.index)

    # --- Segmented gating prototype ---
    def _segment_for_line(x: float) -> str:
        try:
            v = float(x)
        except Exception:
            return "unknown"
        # Buckets by market/closing total bands
        if v <= 135:
            return "very_low"
        if v <= 145:
            return "low"
        if v <= 155:
            return "mid"
        if v <= 165:
            return "high"
        return "very_high"

    def _load_segment_thresholds(outputs: Path, default_hi: float, default_lo: float) -> dict:
        # Optional external mapping at outputs/metrics/ou_segment_thresholds.json
        mp = outputs / "metrics" / "ou_segment_thresholds.json"
        if mp.exists():
            try:
                m = pd.read_json(mp, typ='series').to_dict()  # {segment: {hi:.., lo:..}}
                # Normalize to nested dict
                segs = {}
                for k, v in m.items():
                    if isinstance(v, dict):
                        hi = float(v.get('hi', default_hi))
                        lo = float(v.get('lo', default_lo))
                        segs[str(k)] = {"hi": hi, "lo": lo}
                return segs
            except Exception:
                pass
        # Defaults: global hi/lo applied across segments
        base = {"hi": float(os.environ.get("NCAAB_P_OVER_THRESHOLD_HIGH", default_hi)),
                "lo": float(os.environ.get("NCAAB_P_OVER_THRESHOLD_LOW", default_lo))}
        return {s: base for s in ("very_low","low","mid","high","very_high","unknown")}

    # Load tuned OU policy if available (delta-based), used as fallback when segmented gating yields too few)
    tau = None
    pmin = 0.0
    smax = 0.0
    policy_path = outputs / "metrics" / "ou_selection_policy.json"
    if policy_path.exists():
        try:
            policy = pd.read_json(policy_path)
            sel = policy.get("selected")
            if isinstance(sel, dict):
                tau = float(sel.get("tau") or 0.0)
                pmin = float(sel.get("pmin") or 0.0)
                smax = float(sel.get("sigma_max") or 0.0)
        except Exception:
            pass
    # Fallback tau if not present
    if tau is None:
        tau = float(os.environ.get("NCAAB_OU_TAU", "12"))

    # 1) Primary: segmented probability gating using market/closing total bands (hold 0.60/0.40 by default)
    seg_map = _load_segment_thresholds(outputs, default_hi=p_hi, default_lo=p_lo)
    mt_for_seg = mkt.copy()
    # fallback to market_total column if closing_total missing in mkt
    if mt_for_seg.isna().all() and "market_total" in df.columns:
        mt_for_seg = pd.to_numeric(df.get("market_total"), errors="coerce")
    segs = mt_for_seg.apply(_segment_for_line)
    # decide side from probability
    po_use_full = pd.to_numeric(df.get("p_over_blend"), errors="coerce") if "p_over_blend" in df.columns else po
    side_prob = pd.Series(np.where(po_use_full.ge(0.5), "over", "under"), index=df.index)
    # row-wise thresholds
    thr_hi = segs.map(lambda s: seg_map.get(s, {"hi": p_hi, "lo": p_lo}).get("hi", p_hi))
    thr_lo = segs.map(lambda s: seg_map.get(s, {"hi": p_hi, "lo": p_lo}).get("lo", p_lo))
    gate_prob = (side_prob.eq("over") & po_use_full.ge(thr_hi)) | (side_prob.eq("under") & po_use_full.le(thr_lo))
    picks = df[gate_prob & mt_for_seg.notna()].copy()

    # 2) Fallback: delta-based tuner policy if segmented gating yields too few
    try:
        min_recs = int(os.environ.get("NCAAB_MIN_OU_RECS", "5"))
    except Exception:
        min_recs = 5
    if len(picks) < max(1, min_recs):
        delta = (pred - mkt).abs()
        sel = delta.ge(tau)
        if smax > 0 and sigma.notna().any():
            sel = sel & sigma.le(smax)
        if pmin > 0 and po.notna().any():
            sel = sel & po.ge(pmin)
        picks = df[sel & mkt.notna()].copy()

    # 3) Final fallback: select top-|delta| to reach min picks if still below threshold
    if len(picks) < max(1, min_recs):
        delta = (pred - mkt).abs()
        base = df[mkt.notna()].copy()
        base = base.assign(_abs_delta=delta)
        base = base.sort_values("_abs_delta", ascending=False)
        picks = base.head(max(1, min_recs)).copy()

    # Build output schema aligned with ATS picks_raw generator
    out = pd.DataFrame()
    # Populate IDs and team names from the source df using picks' index
    try:
        src = df.loc[picks.index]
    except Exception:
        src = picks
    out["game_id"] = src.get("game_id")
    out["date"] = date
    out["home_team"] = src.get("home_team")
    out["away_team"] = src.get("away_team")
    out["market"] = "totals"
    out["period"] = "full_game"
    # Determine bet
    po_b_all = pd.to_numeric(df.get("p_over_blend"), errors="coerce") if "p_over_blend" in df.columns else pd.Series(np.nan, index=df.index)
    po_m_all = pd.to_numeric(df.get("p_over"), errors="coerce") if "p_over" in df.columns else pd.Series(np.nan, index=df.index)
    po_use_all = po_b_all.fillna(po_m_all)
    po_use = po_use_all.reindex(picks.index)
    if po_use.notna().any():
        out["bet"] = np.where(po_use.ge(0.5), "over", "under")
    else:
        delta_signed = (pred - mkt).reindex(picks.index)
        out["bet"] = np.where(delta_signed.gt(0), "over", "under")
    # Line: prefer merged market_total, fallback to mkt
    mt_pick = pd.to_numeric(picks.get("market_total"), errors="coerce") if "market_total" in picks.columns else pd.Series(np.nan, index=picks.index)
    if mt_pick.isna().all():
        mt_pick = mkt.reindex(picks.index)
    out["line"] = mt_pick
    out["price"] = None
    # Edge: absolute difference between pred and market
    pr = pred.reindex(picks.index)
    edge = (pr - mt_pick).abs() if pr.notna().any() else pd.Series(np.nan, index=picks.index)
    # If no model pred_total set, fallback to sim mean/median
    if edge.isna().all() and ("mu_total_sim" in picks.columns or "q50_total_sim" in picks.columns):
        mu_sim = pd.to_numeric(picks.get("mu_total_sim"), errors="coerce") if "mu_total_sim" in picks.columns else pd.Series(np.nan, index=picks.index)
        q50_sim = pd.to_numeric(picks.get("q50_total_sim"), errors="coerce") if "q50_total_sim" in picks.columns else pd.Series(np.nan, index=picks.index)
        pr_alt = mu_sim.fillna(q50_sim)
        edge = (pr_alt - mt_pick).abs()
        pr = pr_alt
    out["edge"] = edge
    out["pred_margin"] = None
    out["pred_total"] = pr
    out["rec_type"] = "Total"
    out["rec_code"] = "OU"
    # Backfill team names if missing using the day's games file
    try:
        need_home = out.get("home_team") is None or pd.isna(out.get("home_team")).any() or (out.get("home_team").astype(str).eq("")).any()
        need_away = out.get("away_team") is None or pd.isna(out.get("away_team")).any() or (out.get("away_team").astype(str).eq("")).any()
    except Exception:
        need_home = True
        need_away = True
    if need_home or need_away:
        try:
            games_path = outputs / f"games_{date}.csv"
            if games_path.exists():
                gf = pd.read_csv(games_path)
                # Normalize key type
                if "game_id" in gf.columns:
                    gf["game_id"] = gf["game_id"].astype(str)
                if "game_id" in out.columns:
                    out["game_id"] = out["game_id"].astype(str)
                # Only keep needed columns to avoid collisions
                cols = [c for c in ["game_id","home_team","away_team"] if c in gf.columns]
                if cols:
                    m = out.merge(gf[cols], on="game_id", how="left", suffixes=("", "_g"))
                    # Prefer existing values; fill missing from _g
                    if "home_team" in m.columns and "home_team_g" in m.columns:
                        m["home_team"] = m["home_team"].where(m["home_team"].notna() & (m["home_team"].astype(str).ne("")), m["home_team_g"])
                        m = m.drop(columns=["home_team_g"])
                    if "away_team" in m.columns and "away_team_g" in m.columns:
                        m["away_team"] = m["away_team"].where(m["away_team"].notna() & (m["away_team"].astype(str).ne("")), m["away_team_g"])
                        m = m.drop(columns=["away_team_g"])
                    out = m
            else:
                # Fallback to display snapshot if games file missing
                disp_path = outputs / f"predictions_display_{date}.csv"
                if disp_path.exists():
                    df_disp = pd.read_csv(disp_path)
                    if "game_id" in df_disp.columns:
                        df_disp["game_id"] = df_disp["game_id"].astype(str)
                    if "game_id" in out.columns:
                        out["game_id"] = out["game_id"].astype(str)
                    cols = [c for c in ["game_id","home_team","away_team"] if c in df_disp.columns]
                    if cols:
                        m = out.merge(df_disp[cols], on="game_id", how="left", suffixes=("", "_d"))
                        if "home_team" in m.columns and "home_team_d" in m.columns:
                            m["home_team"] = m["home_team"].where(m["home_team"].notna() & (m["home_team"].astype(str).ne("")), m["home_team_d"])
                            m = m.drop(columns=["home_team_d"])
                        if "away_team" in m.columns and "away_team_d" in m.columns:
                            m["away_team"] = m["away_team"].where(m["away_team"].notna() & (m["away_team"].astype(str).ne("")), m["away_team_d"])
                            m = m.drop(columns=["away_team_d"])
                        out = m
        except Exception:
            pass
    return out


def merge_into_picks_raw(out_df: pd.DataFrame, outputs: Path) -> Path:
    raw_path = outputs / "picks_raw.csv"
    if raw_path.exists():
        try:
            base = pd.read_csv(raw_path)
            # Deduplicate columns if needed
            base = base.loc[:, ~base.columns.duplicated()]
        except Exception:
            base = pd.DataFrame()
    else:
        base = pd.DataFrame()
    merged = pd.concat([base, out_df], ignore_index=True) if not out_df.empty else base
    try:
        dedupe_subset = [c for c in ["date", "game_id", "market", "period"] if c in merged.columns]
        if dedupe_subset:
            merged["_edge_abs"] = pd.to_numeric(merged.get("edge"), errors="coerce").abs()
            merged = (
                merged.sort_values(dedupe_subset + ["_edge_abs"], ascending=[True] * len(dedupe_subset) + [False], na_position="last")
                .drop_duplicates(subset=dedupe_subset, keep="first")
                .drop(columns=["_edge_abs"], errors="ignore")
                .reset_index(drop=True)
            )
    except Exception:
        pass
    merged.to_csv(raw_path, index=False)
    return raw_path


def main():
    ap = argparse.ArgumentParser(description="Create totals picks and merge into picks_raw.csv using p_over thresholds")
    ap.add_argument("--date", required=True, help="Date YYYY-MM-DD")
    ap.add_argument("--outputs", default=str(DEF_OUT))
    ap.add_argument("--p-high", type=float, default=float(os.environ.get("NCAAB_P_OVER_THRESHOLD_HIGH", "0.58")))
    ap.add_argument("--p-low", type=float, default=float(os.environ.get("NCAAB_P_OVER_THRESHOLD_LOW", "0.42")))
    args = ap.parse_args()
    out_dir = Path(args.outputs)
    out_df = build_totals_picks(args.date, out_dir, args.p_high, args.p_low)
    if out_df.empty:
        print(f"[skip] No totals picks selected for {args.date} (check probabilities and thresholds)")
        sys.exit(0)
    raw_path = merge_into_picks_raw(out_df, out_dir)
    print(f"[ok] Wrote totals picks into {raw_path} rows_appended={len(out_df)}")


if __name__ == "__main__":
    main()
