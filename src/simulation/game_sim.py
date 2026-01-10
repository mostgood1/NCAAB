import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

# Lightweight Monte Carlo simulator using baseline totals/margins.
# Assumes per-game means for total and margin and estimates per-team variance
# with a shared correlation parameter.

DEFAULT_RHO = 0.3  # positive correlation between team scores
DEFAULT_TOTAL_SIGMA = 14.0  # fallback spread of total points
DEFAULT_SAMPLES = 4000


def _resolve_mean_total_margin(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    for tot_col in [
        "pred_total_calibrated",
        "pred_total_interval_mean",
        "pred_total",
        "total_pred",
    ]:
        if tot_col in row and pd.notna(row[tot_col]):
            total = float(row[tot_col])
            break
    else:
        total = None

    for mar_col in [
        "pred_margin_calibrated",
        "pred_margin_interval_mean",
        "pred_margin",
        "margin_pred",
    ]:
        if mar_col in row and pd.notna(row[mar_col]):
            margin = float(row[mar_col])
            break
    else:
        margin = None

    return total, margin


def _resolve_total_sigma(row: pd.Series) -> float:
    for sig_col in [
        "interval_total_std",
        "interval_total_sigma",
        "tot_sigma",
        "sigma_total",
    ]:
        if sig_col in row and pd.notna(row[sig_col]):
            try:
                return float(row[sig_col])
            except Exception:
                continue
    return DEFAULT_TOTAL_SIGMA


def _resolve_market_total(row: pd.Series) -> Optional[float]:
    for mcol in [
        "market_total",
        "closing_total",
        "total",
        "ou_line",
    ]:
        if mcol in row and pd.notna(row[mcol]):
            try:
                return float(row[mcol])
            except Exception:
                continue
    return None


def _resolve_keys(df: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str]]:
    # Returns (id_col, home_col, away_col)
    id_candidates = ["game_id", "id"]
    home_candidates = ["home_team", "home"]
    away_candidates = ["away_team", "away"]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    home_col = next((c for c in home_candidates if c in df.columns), None)
    away_col = next((c for c in away_candidates if c in df.columns), None)
    if id_col is None:
        # Create synthetic id if teams exist
        if home_col and away_col:
            df["_gid"] = df[home_col].astype(str).str.upper() + "_vs_" + df[away_col].astype(str).str.upper()
            id_col = "_gid"
        else:
            id_col = "index"
            df["index"] = np.arange(len(df))
    return id_col, home_col, away_col


def simulate_game_row(row: pd.Series, rho: float = DEFAULT_RHO, samples: int = DEFAULT_SAMPLES) -> dict:
    total_mean, margin_mean = _resolve_mean_total_margin(row)
    if total_mean is None or margin_mean is None:
        return {
            "sim_ok": False,
            "mu_total": total_mean,
            "mu_margin": margin_mean,
        }
    sigma_total = _resolve_total_sigma(row)
    # Infer equal per-team sigma from total variance and correlation
    var_total = sigma_total ** 2
    # var(total) = 2*s^2*(1+rho) with s = per-team sigma (assuming equal)
    s2 = var_total / (2.0 * (1.0 + rho))
    sigma_team = float(np.sqrt(max(s2, 1e-6)))

    mu_home = (total_mean + margin_mean) / 2.0
    mu_away = (total_mean - margin_mean) / 2.0

    cov = rho * (sigma_team ** 2)
    cov_mat = np.array([[sigma_team ** 2, cov], [cov, sigma_team ** 2]], dtype=float)
    means = np.array([mu_home, mu_away], dtype=float)

    try:
        samples_arr = np.random.multivariate_normal(means, cov_mat, size=samples)
    except np.linalg.LinAlgError:
        # Fallback: independent normals
        samples_arr = np.column_stack([
            np.random.normal(mu_home, sigma_team, size=samples),
            np.random.normal(mu_away, sigma_team, size=samples),
        ])

    home_pts = np.clip(samples_arr[:, 0], 0.0, None)
    away_pts = np.clip(samples_arr[:, 1], 0.0, None)
    totals = home_pts + away_pts
    margins = home_pts - away_pts

    q10_t = float(np.quantile(totals, 0.10))
    q50_t = float(np.quantile(totals, 0.50))
    q90_t = float(np.quantile(totals, 0.90))

    q10_m = float(np.quantile(margins, 0.10))
    q50_m = float(np.quantile(margins, 0.50))
    q90_m = float(np.quantile(margins, 0.90))

    market_total = _resolve_market_total(row)
    p_over_market = None
    if market_total is not None:
        p_over_market = float(np.mean(totals > market_total))

    return {
        "sim_ok": True,
        "mu_total": float(np.mean(totals)),
        "mu_margin": float(np.mean(margins)),
        "q10_total": q10_t,
        "q50_total": q50_t,
        "q90_total": q90_t,
        "q10_margin": q10_m,
        "q50_margin": q50_m,
        "q90_margin": q90_m,
        "p_over_market": p_over_market,
        "market_total": market_total,
    }


def run_simulations_for_date(out_dir: Path, date: str,
                             preds_path: Optional[Path] = None,
                             lines_path: Optional[Path] = None,
                             samples: int = DEFAULT_SAMPLES,
                             rho: float = DEFAULT_RHO) -> Path:
    out_dir = Path(out_dir)
    if preds_path is None:
        # Prefer enriched unified predictions (contains IDs/teams); fallback to calibrated/model
        enr_path = out_dir / f"predictions_unified_enriched_{date}.csv"
        cal_path = out_dir / f"predictions_model_calibrated_{date}.csv"
        base_path = out_dir / f"predictions_model_{date}.csv"
        preds_path = enr_path if enr_path.exists() else (cal_path if cal_path.exists() else base_path)
    if lines_path is None:
        lines_path = out_dir / "games_with_last.csv"

    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")

    preds = pd.read_csv(preds_path)
    if "date" in preds.columns:
        preds = preds[preds["date"].astype(str) == str(date)]

    # If no predictions for the date, write a header-only CSV to avoid empty-file parse errors downstream
    if preds.shape[0] == 0:
        out_path = out_dir / f"sim_quantiles_{date}.csv"
        header_cols = [
            "date","game_id","home_team","away_team",
            "sim_ok","mu_total","mu_margin",
            "q10_total","q50_total","q90_total",
            "q10_margin","q50_margin","q90_margin",
            "p_over_market","market_total",
        ]
        pd.DataFrame(columns=header_cols).to_csv(out_path, index=False)
        return out_path

    # Try to enrich with market totals via games_with_last.csv (optional)
    market = None
    if lines_path.exists():
        try:
            market_df = pd.read_csv(lines_path)
            if "date" in market_df.columns:
                market_df = market_df[market_df["date"].astype(str) == str(date)]
            # Prefer joining on game_id and carry team names + market total
            if "game_id" in preds.columns and "game_id" in market_df.columns:
                if "game_id" in market_df.columns:
                    market_df = market_df.drop_duplicates(subset=["game_id"])  # prevent explosion on merge
                cols = ["game_id"]
                if "home_team" in market_df.columns: cols.append("home_team")
                if "away_team" in market_df.columns: cols.append("away_team")
                # Normalize market total column
                if "market_total" in market_df.columns:
                    cols.append("market_total")
                elif "closing_total" in market_df.columns:
                    market_df["market_total"] = market_df["closing_total"]
                    cols.append("market_total")
                market = market_df[cols].copy()
                preds = preds.merge(market, on="game_id", how="left")
            else:
                # Fallback: join on team names if both sources have them
                id_p, home_p, away_p = _resolve_keys(preds)
                id_m, home_m, away_m = _resolve_keys(market_df)
                if home_p and away_p and home_m and away_m:
                    market_df = market_df.drop_duplicates(subset=[home_m, away_m])
                    preds["_h"] = preds[home_p].astype(str).str.upper()
                    preds["_a"] = preds[away_p].astype(str).str.upper()
                    market_df["_h"] = market_df[home_m].astype(str).str.upper()
                    market_df["_a"] = market_df[away_m].astype(str).str.upper()
                    if "market_total" not in market_df.columns and "closing_total" in market_df.columns:
                        market_df["market_total"] = market_df["closing_total"]
                    join_cols = ["_h", "_a"]
                    cols_to_keep = ["market_total", "home_team", "away_team"]
                    preds = preds.merge(market_df[join_cols + [c for c in cols_to_keep if c in market_df.columns]], on=join_cols, how="left")
        except Exception:
            pass

    # Simulate per row
    results = []
    id_col, home_col, away_col = _resolve_keys(preds)
    for _, r in preds.iterrows():
        sim_res = simulate_game_row(r, rho=rho, samples=samples)
        out = {
            "date": date,
            "game_id": r.get(id_col),
            "home_team": r.get("home_team") if "home_team" in preds.columns else r.get(home_col),
            "away_team": r.get("away_team") if "away_team" in preds.columns else r.get(away_col),
        }
        out.update(sim_res)
        results.append(out)

    sim_df = pd.DataFrame(results)
    out_path = out_dir / f"sim_quantiles_{date}.csv"
    sim_df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("date", type=str)
    ap.add_argument("--outputs", type=str, default=str(Path("outputs")))
    ap.add_argument("--preds-file", type=str, default="")
    ap.add_argument("--lines-file", type=str, default="")
    ap.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    ap.add_argument("--rho", type=float, default=DEFAULT_RHO)
    args = ap.parse_args()

    out_dir = Path(args.outputs)
    preds_path = Path(args.preds_file) if args.preds_file else None
    lines_path = Path(args.lines_file) if args.lines_file else None

    path = run_simulations_for_date(out_dir, args.date, preds_path, lines_path, args.samples, args.rho)
    print({"wrote": str(path)})
