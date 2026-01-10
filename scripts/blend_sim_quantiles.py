import sys
import json
from pathlib import Path
import pandas as pd
import math

DEFAULT_W_SIM = 0.4


def robust_p_over_model(row: pd.Series) -> float:
    # Prefer model-side probabilities only; avoid using simulation/market columns
    candidates = [
        # From enriched/base predictions (prefer most finalized/calibrated variants)
        "p_over_final",
        "p_over_final_base",
        "p_over_display",
        "p_over_display_base",
        "p_over_meta_cal",
        "p_over_meta_cal_base",
        "p_over_meta",
        "p_over_meta_base",
        "p_over",               # raw model-side probability
        "p_over_base",
        # Occasionally named as calibrated
        "p_over_calibrated",
        "p_over_calibrated_base",
    ]
    for col in candidates:
        try:
            if col in row and pd.notna(row[col]):
                return float(row[col])
        except Exception:
            continue
    return float(float("nan"))


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: blend_sim_quantiles.py <date> [outputs_dir] [weight_sim]"}))
        return 1
    date = sys.argv[1]
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs")
    w_sim = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_W_SIM

    sim_path = out_dir / f"sim_quantiles_{date}.csv"
    base_cal = out_dir / f"predictions_model_calibrated_{date}.csv"
    base_int = out_dir / f"predictions_model_interval_{date}.csv"

    if not sim_path.exists():
        print(json.dumps({"date": date, "error": f"sim file missing: {sim_path}"}))
        return 2

    # Robust read: if file has only header (no rows), treat as empty
    try:
        sim = pd.read_csv(sim_path)
    except Exception:
        sim = pd.DataFrame()
    if sim is None or getattr(sim, 'shape', (0,0))[0] == 0:
        print(json.dumps({"date": date, "warning": "sim_quantiles appears empty; skipping blend"}))
        out_path = out_dir / f"sim_blend_{date}.csv"
        # Write header-only blend CSV to preserve downstream expectations
        pd.DataFrame(columns=[
            "date","game_id","home_team","away_team","market_total",
            "mu_total_sim","q50_total_sim","p_over_sim","p_over_model","p_over_blend","weight_sim"
        ]).to_csv(out_path, index=False)
        return 0

    base = None
    # Prefer interval file (has CI bands needed for sigma); fallback to calibrated means only
    if base_int.exists():
        base = pd.read_csv(base_int)
    elif base_cal.exists():
        base = pd.read_csv(base_cal)

    # Optional: bring in enriched predictions for market/closing totals and model probabilities
    enriched = None
    enr_path = out_dir / f".tmp_modelfirst_predictions_unified_enriched_{date}.csv"
    if enr_path.exists():
        try:
            enriched = pd.read_csv(enr_path)
        except Exception:
            enriched = None

    # Optional: extract market totals from games_with_last.csv
    market_map = None
    gwl_path = out_dir / "games_with_last.csv"
    if gwl_path.exists():
        try:
            gwl = pd.read_csv(gwl_path)
            # Filter to date and totals market
            if "date_game" in gwl.columns:
                gwl = gwl[gwl["date_game"].astype(str) == str(date)]
            if "market" in gwl.columns:
                gwl = gwl[gwl["market"].astype(str).str.lower() == "totals"]
            if "period" in gwl.columns:
                gwl = gwl[gwl["period"].astype(str).str.lower() == "full_game"]
            if "game_id" in gwl.columns and "total" in gwl.columns:
                tmp = gwl.copy()
                tmp["game_id"] = tmp["game_id"].astype(str)
                tmp["total"] = pd.to_numeric(tmp["total"], errors="coerce")
                market_map = tmp.groupby("game_id")["total"].last()
        except Exception:
            market_map = None

    if base is not None:
        # Try joining on game_id if exists; else synthesize from teams
        id_col = "game_id" if "game_id" in sim.columns and "game_id" in base.columns else None
        if id_col is None:
            if all(c in sim.columns for c in ["home_team", "away_team"]) and \
               all(c in base.columns for c in ["home_team", "away_team"]):
                sim["_gid"] = sim["home_team"].astype(str).str.upper()+"_vs_"+sim["away_team"].astype(str).str.upper()
                base["_gid"] = base["home_team"].astype(str).str.upper()+"_vs_"+base["away_team"].astype(str).str.upper()
                id_col = "_gid"
        if id_col:
            try:
                sim[id_col] = sim[id_col].astype(str)
                base[id_col] = base[id_col].astype(str)
            except Exception:
                pass
            merged = sim.merge(base, on=id_col, how="left", suffixes=("_sim","_base"))
        else:
            merged = sim.copy()
    else:
        merged = sim.copy()

    # Derive model-side p_over if not explicitly present, using calibrated totals and intervals
    def _derive_p_over_model(m: pd.DataFrame) -> pd.Series:
        # Get mu from calibrated total
        mu = None
        if "pred_total_calibrated" in m.columns:
            mu = pd.to_numeric(m["pred_total_calibrated"], errors="coerce")
        elif "pred_total_calibrated_base" in m.columns:
            mu = pd.to_numeric(m["pred_total_calibrated_base"], errors="coerce")

        # Estimate sigma from CI90 or CI75
        sigma = None
        if "pred_total_ci90_low" in m.columns and "pred_total_ci90_high" in m.columns:
            low = pd.to_numeric(m["pred_total_ci90_low"], errors="coerce")
            high = pd.to_numeric(m["pred_total_ci90_high"], errors="coerce")
            sigma = (high - low) / (2.0 * 1.6448536269514722)
        elif "pred_total_ci90_low_base" in m.columns and "pred_total_ci90_high_base" in m.columns:
            low = pd.to_numeric(m["pred_total_ci90_low_base"], errors="coerce")
            high = pd.to_numeric(m["pred_total_ci90_high_base"], errors="coerce")
            sigma = (high - low) / (2.0 * 1.6448536269514722)
        elif "pred_total_ci75_low" in m.columns and "pred_total_ci75_high" in m.columns:
            low = pd.to_numeric(m["pred_total_ci75_low"], errors="coerce")
            high = pd.to_numeric(m["pred_total_ci75_high"], errors="coerce")
            sigma = (high - low) / (2.0 * 1.1503493803760082)
        elif "pred_total_ci75_low_base" in m.columns and "pred_total_ci75_high_base" in m.columns:
            low = pd.to_numeric(m["pred_total_ci75_low_base"], errors="coerce")
            high = pd.to_numeric(m["pred_total_ci75_high_base"], errors="coerce")
            sigma = (high - low) / (2.0 * 1.1503493803760082)

        mt = None
        if "market_total" in m.columns:
            mt = pd.to_numeric(m["market_total"], errors="coerce")
        elif "market_total_sim" in m.columns:
            mt = pd.to_numeric(m["market_total_sim"], errors="coerce")
        elif "closing_total" in m.columns:
            mt = pd.to_numeric(m["closing_total"], errors="coerce")
        elif "closing_total_sim" in m.columns:
            mt = pd.to_numeric(m["closing_total_sim"], errors="coerce")

        if mu is None or sigma is None or mt is None:
            return pd.Series([float("nan")] * len(m))
        z = (mt - mu) / sigma
        # CDF of standard normal via erf
        cdf = 0.5 * (1.0 + (z / abs(z)).fillna(0))  # placeholder for shape
        try:
            cdf = 0.5 * (1.0 + (z.apply(lambda t: math.erf(t / math.sqrt(2.0)))))
        except Exception:
            pass
        return 1.0 - cdf

    # Blend p_over: if both available, weighted average; else use whichever exists
    # If market/closing totals missing, try join from enriched and odds last
    if enriched is not None:
        try:
            id_col = "game_id" if "game_id" in merged.columns and "game_id" in enriched.columns else None
            if (
                id_col is None
                and all(c in merged.columns for c in ["home_team","away_team"])
                and all(c in enriched.columns for c in ["home_team","away_team"])
            ):
                merged["_gid"] = merged["home_team"].astype(str).str.upper()+"_vs_"+merged["away_team"].astype(str).str.upper()
                enriched["_gid"] = enriched["home_team"].astype(str).str.upper()+"_vs_"+enriched["away_team"].astype(str).str.upper()
                id_col = "_gid"
            if id_col:
                # Ensure join keys align in dtype/format
                try:
                    merged[id_col] = merged[id_col].astype(str)
                    enriched[id_col] = enriched[id_col].astype(str)
                except Exception:
                    pass
                # Select only available columns to avoid KeyErrors
                desired_cols = [
                    id_col,
                    "market_total","closing_total",
                    # model probability columns from enriched
                    "p_over_final","p_over_display","p_over_meta_cal","p_over_meta","p_over",
                    # calibrated totals and intervals for derivation fallback
                    "pred_total_calibrated","pred_total_ci90_low","pred_total_ci90_high",
                    "pred_total_ci75_low","pred_total_ci75_high",
                ]
                use_cols = [c for c in desired_cols if c in enriched.columns]
                merged = merged.merge(enriched[use_cols], on=id_col, how="left")
                if "_gid" in merged.columns:
                    merged = merged.drop(columns=["_gid"])
        except Exception:
            pass
    if market_map is not None and "game_id" in merged.columns:
        try:
            merged["game_id"] = merged["game_id"].astype(str)
            if "market_total" in merged.columns:
                mt_from_map = merged["game_id"].map(market_map)
                merged["market_total"] = pd.to_numeric(merged["market_total"], errors="coerce").fillna(mt_from_map)
            else:
                merged = merged.merge(market_map.rename("market_total").reset_index(), on="game_id", how="left")
        except Exception:
            pass

    # Simulation-side p_over from quantiles if not present; handle suffixed merge variants first
    if "p_over_market" in merged.columns:
        p_over_sim = pd.to_numeric(merged["p_over_market"], errors="coerce")
    elif "p_over_market_sim" in merged.columns:
        p_over_sim = pd.to_numeric(merged["p_over_market_sim"], errors="coerce")
    else:
        mu_sim = None
        if "mu_total" in merged.columns:
            mu_sim = pd.to_numeric(merged["mu_total"], errors="coerce")
        elif "mu_total_sim" in merged.columns:
            mu_sim = pd.to_numeric(merged["mu_total_sim"], errors="coerce")
        q10 = None
        q90 = None
        if "q10_total" in merged.columns and "q90_total" in merged.columns:
            q10 = pd.to_numeric(merged["q10_total"], errors="coerce")
            q90 = pd.to_numeric(merged["q90_total"], errors="coerce")
        elif "q10_total_sim" in merged.columns and "q90_total_sim" in merged.columns:
            q10 = pd.to_numeric(merged["q10_total_sim"], errors="coerce")
            q90 = pd.to_numeric(merged["q90_total_sim"], errors="coerce")
        mt_series = None
        if "market_total" in merged.columns:
            mt_series = pd.to_numeric(merged["market_total"], errors="coerce")
        elif "closing_total" in merged.columns:
            mt_series = pd.to_numeric(merged["closing_total"], errors="coerce")
        if mu_sim is not None and q10 is not None and q90 is not None and mt_series is not None:
            # Approximate sigma from central 80% width
            sigma_sim = (q90 - q10) / 2.563103131089195
            z_sim = (mt_series - mu_sim) / sigma_sim
            try:
                cdf_sim = 0.5 * (1.0 + z_sim.apply(lambda t: math.erf(t / math.sqrt(2.0))))
            except Exception:
                cdf_sim = pd.Series([float("nan")] * len(merged))
            p_over_sim = 1.0 - cdf_sim
        else:
            p_over_sim = pd.Series([float("nan")] * len(merged))
    # If numeric conversion yielded all NaNs, try derivation fallback
    if isinstance(p_over_sim, pd.Series) and p_over_sim.isna().all():
        mu_sim = None
        if "mu_total" in merged.columns:
            mu_sim = pd.to_numeric(merged["mu_total"], errors="coerce")
        elif "mu_total_sim" in merged.columns:
            mu_sim = pd.to_numeric(merged["mu_total_sim"], errors="coerce")
        q10 = None
        q90 = None
        if "q10_total" in merged.columns and "q90_total" in merged.columns:
            q10 = pd.to_numeric(merged["q10_total"], errors="coerce")
            q90 = pd.to_numeric(merged["q90_total"], errors="coerce")
        elif "q10_total_sim" in merged.columns and "q90_total_sim" in merged.columns:
            q10 = pd.to_numeric(merged["q10_total_sim"], errors="coerce")
            q90 = pd.to_numeric(merged["q90_total_sim"], errors="coerce")
        mt_series = None
        if "market_total" in merged.columns:
            mt_series = pd.to_numeric(merged["market_total"], errors="coerce")
        elif "closing_total" in merged.columns:
            mt_series = pd.to_numeric(merged["closing_total"], errors="coerce")
        if mu_sim is not None and q10 is not None and q90 is not None and mt_series is not None:
            sigma_sim = (q90 - q10) / 2.563103131089195
            z_sim = (mt_series - mu_sim) / sigma_sim
            try:
                cdf_sim = 0.5 * (1.0 + z_sim.apply(lambda t: math.erf(t / math.sqrt(2.0))))
            except Exception:
                cdf_sim = pd.Series([float("nan")] * len(merged))
            p_over_sim = 1.0 - cdf_sim
    # Vectorized selection of model-side p_over from enriched/base columns
    p_over_model = None
    model_prob_cols = [
        "p_over_final","p_over_display","p_over_meta_cal","p_over_meta",
        "p_over","p_over_base","p_over_calibrated","p_over_calibrated_base",
    ]
    for col in model_prob_cols:
        if col in merged.columns:
            s = pd.to_numeric(merged[col], errors="coerce")
            if p_over_model is None:
                p_over_model = s
            else:
                p_over_model = p_over_model.fillna(s)
    if p_over_model is None:
        p_over_model = pd.Series([float("nan")] * len(merged))
    # If model prob missing for some rows, derive from intervals and fill per-row
    derived_model = _derive_p_over_model(merged)
    try:
        p_over_model = pd.to_numeric(p_over_model, errors="coerce")
    except Exception:
        pass
    # Attempt vectorized derivation with CI bands when available
    try:
        import numpy as np
        mu = None
        if "pred_total_calibrated" in merged.columns:
            mu = pd.to_numeric(merged["pred_total_calibrated"], errors="coerce")
        elif "pred_total_calibrated_base" in merged.columns:
            mu = pd.to_numeric(merged["pred_total_calibrated_base"], errors="coerce")
        elif "pred_total_model" in merged.columns:
            mu = pd.to_numeric(merged["pred_total_model"], errors="coerce")
        elif "pred_total_model_base" in merged.columns:
            mu = pd.to_numeric(merged["pred_total_model_base"], errors="coerce")

        low = None; high = None
        if "pred_total_ci90_low" in merged.columns and "pred_total_ci90_high" in merged.columns:
            low = pd.to_numeric(merged["pred_total_ci90_low"], errors="coerce")
            high = pd.to_numeric(merged["pred_total_ci90_high"], errors="coerce")
            sigma = (high - low) / (2.0 * 1.6448536269514722)
        elif "pred_total_ci75_low" in merged.columns and "pred_total_ci75_high" in merged.columns:
            low = pd.to_numeric(merged["pred_total_ci75_low"], errors="coerce")
            high = pd.to_numeric(merged["pred_total_ci75_high"], errors="coerce")
            sigma = (high - low) / (2.0 * 1.1503493803760082)
        else:
            sigma = pd.to_numeric(merged.get("pred_total_sigma"), errors="coerce")

        mt = None
        if "market_total" in merged.columns:
            mt = pd.to_numeric(merged["market_total"], errors="coerce")
        elif "closing_total" in merged.columns:
            mt = pd.to_numeric(merged["closing_total"], errors="coerce")

        if mu is not None and sigma is not None and mt is not None:
            z = (mt.values - mu.values) / sigma.values
            cdf = 0.5 * (1.0 + np.vectorize(lambda t: math.erf(t / math.sqrt(2.0)))(z))
            derived_vec = 1.0 - cdf
            derived_vec = pd.Series(derived_vec, index=merged.index)
            # Fill missing entries and fully replace if nothing existed
            if isinstance(p_over_model, pd.Series) and not p_over_model.notna().any():
                p_over_model = derived_vec
            else:
                p_over_model = p_over_model.fillna(derived_vec)
    except Exception:
        # Fallback to previously computed derived_model if vectorized path fails
        if isinstance(p_over_model, pd.Series) and not p_over_model.notna().any():
            p_over_model = derived_model
        else:
            p_over_model = p_over_model.fillna(derived_model)
    p_over_blend = p_over_sim.copy()
    has_model = p_over_model.notna()
    has_sim = p_over_sim.notna()

    p_over_blend[has_model & has_sim] = (w_sim * p_over_sim[has_model & has_sim] + (1.0 - w_sim) * p_over_model[has_model & has_sim])
    p_over_blend[~has_model & has_sim] = p_over_sim[~has_model & has_sim]
    p_over_blend[has_model & ~has_sim] = p_over_model[has_model & ~has_sim]

    # Prefer market_total from sim; fallback to closing_total (handle suffixed)
    if "market_total" not in merged.columns:
        if "market_total_sim" in merged.columns:
            merged["market_total"] = pd.to_numeric(merged["market_total_sim"], errors="coerce")
        elif "closing_total" in merged.columns:
            merged["market_total"] = pd.to_numeric(merged["closing_total"], errors="coerce")
        elif "closing_total_sim" in merged.columns:
            merged["market_total"] = pd.to_numeric(merged["closing_total_sim"], errors="coerce")

    mu_total_series = None
    if "mu_total" in merged.columns:
        mu_total_series = merged["mu_total"]
    elif "mu_total_sim" in merged.columns:
        mu_total_series = merged["mu_total_sim"]

    q50_total_series = None
    if "q50_total" in merged.columns:
        q50_total_series = merged["q50_total"]
    elif "q50_total_sim" in merged.columns:
        q50_total_series = merged["q50_total_sim"]

    # Construct output frame using raw array values to prevent index alignment issues
    out = pd.DataFrame({
        "date": [date] * len(merged),
        "game_id": merged.get("game_id").values if "game_id" in merged.columns else [None] * len(merged),
        "home_team": merged.get("home_team").values if "home_team" in merged.columns else [None] * len(merged),
        "away_team": merged.get("away_team").values if "away_team" in merged.columns else [None] * len(merged),
        "market_total": pd.to_numeric(merged.get("market_total"), errors="coerce").values if "market_total" in merged.columns else [float("nan")] * len(merged),
        "mu_total_sim": pd.to_numeric(mu_total_series, errors="coerce").values if mu_total_series is not None else [float("nan")] * len(merged),
        "q50_total_sim": pd.to_numeric(q50_total_series, errors="coerce").values if q50_total_series is not None else [float("nan")] * len(merged),
        "p_over_sim": pd.to_numeric(p_over_sim, errors="coerce").values if isinstance(p_over_sim, pd.Series) else p_over_sim,
        "p_over_model": pd.to_numeric(p_over_model, errors="coerce").values if isinstance(p_over_model, pd.Series) else p_over_model,
        "p_over_blend": pd.to_numeric(p_over_blend, errors="coerce").values if isinstance(p_over_blend, pd.Series) else p_over_blend,
        "weight_sim": [w_sim] * len(merged),
    })

    # Deduplicate to one row per game: prefer game_id, else normalized team pair
    try:
        if "game_id" in out.columns and out["game_id"].notna().any():
            out["game_id"] = out["game_id"].astype(str)
            out = out.drop_duplicates(subset=["game_id"]).reset_index(drop=True)
        else:
            def _pair_key(r: pd.Series) -> str:
                ht = str(r.get("home_team") or "").strip().upper()
                at = str(r.get("away_team") or "").strip().upper()
                return ht + "__VS__" + at
            out["_pair_key"] = out.apply(_pair_key, axis=1)
            out = out.drop_duplicates(subset=["_pair_key"]).drop(columns=["_pair_key"]).reset_index(drop=True)
    except Exception:
        pass

    out_path = out_dir / f"sim_blend_{date}.csv"
    out.to_csv(out_path, index=False)
    try:
        cnt_model = pd.to_numeric(out["p_over_model"], errors="coerce").notna().sum()
        cnt_sim = pd.to_numeric(out["p_over_sim"], errors="coerce").notna().sum()
    except Exception:
        cnt_model = None
        cnt_sim = None
    print(json.dumps({
        "date": str(date),
        "wrote": str(out_path),
        "rows": int(len(out)),
        "count_p_over_model": int(cnt_model) if cnt_model is not None else None,
        "count_p_over_sim": int(cnt_sim) if cnt_sim is not None else None
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
