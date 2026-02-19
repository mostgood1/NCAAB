from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.simulation.game_sim import run_simulations_for_date
from src.simulation.sim_backtest import (
    _join_sim_results,
    _load_results_for_date,
    _load_sim_for_date,
    _resolve_dates_from_results,
)


def _parse_date(s: str | None) -> Optional[str]:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(str(s).strip()).isoformat()
    except Exception:
        return None


@dataclasses.dataclass
class SimAccuracyBacktestConfig:
    out_dir: Path
    start: Optional[str] = None
    end: Optional[str] = None
    recent: Optional[int] = None
    engine: str = "events"
    samples: int = 2000
    rho: float = 0.25
    recompute: bool = False
    out_prefix: str = "sim_accuracy"
    sim_quantiles_prefix: str = "sim_quantiles_"
    sim_meta_prefix: str = "sim_meta_"
    # Optional: use 5-min interval actuals (from ESPN PBP) to compute regulation totals (@40)
    # and OT diagnostics, without changing the existing final-score-based metrics.
    interval_actuals_prefix: str = "interval_actuals_5min_"
    include_ot_diagnostics: bool = False
    calibration_json: Optional[Path] = None
    strip_spread_bins: bool = False


def _load_interval_actuals_5min(out_dir: Path, date_iso: str, prefix: str) -> pd.DataFrame:
    try:
        path = Path(out_dir) / f"{prefix}{date_iso}.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame()
        if "game_id" not in df.columns or "end_min" not in df.columns:
            return pd.DataFrame()
        df = df.copy()
        df["game_id"] = df["game_id"].astype(str)
        df["end_min"] = pd.to_numeric(df["end_min"], errors="coerce")
        # normalize common naming
        if "actual_total_score_end" in df.columns:
            df["actual_total_score_end"] = pd.to_numeric(df["actual_total_score_end"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def _ou_side(total: float | None, line: float | None) -> str | None:
    try:
        if total is None or line is None:
            return None
        if not (np.isfinite(float(total)) and np.isfinite(float(line))):
            return None
        if float(total) > float(line):
            return "O"
        if float(total) < float(line):
            return "U"
        return "P"
    except Exception:
        return None


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    # No SciPy in this project; use math.erf with a Python loop.
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    return 0.5 * (1.0 + np.asarray([math.erf(float(v) * inv_sqrt2) for v in z], dtype=float))


def _crps_normal(mu: pd.Series, sigma: pd.Series, y: pd.Series) -> pd.Series:
    """CRPS for Normal(mu, sigma) against observation y.

    Formula: CRPS = sigma * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ], z=(y-mu)/sigma
    """
    mu2 = pd.to_numeric(mu, errors="coerce")
    sig2 = pd.to_numeric(sigma, errors="coerce")
    y2 = pd.to_numeric(y, errors="coerce")
    out = pd.Series(np.nan, index=mu2.index, dtype=float)
    m = mu2.notna() & sig2.notna() & y2.notna() & (sig2 > 1e-9)
    if not m.any():
        return out
    z = ((y2[m] - mu2[m]) / sig2[m]).to_numpy(dtype=float)
    phi = _norm_pdf(z)
    Phi = _norm_cdf(z)
    crps = sig2[m].to_numpy(dtype=float) * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    out.loc[m] = crps
    return out


def _nll_normal(mu: pd.Series, sigma: pd.Series, y: pd.Series) -> pd.Series:
    """Negative log-likelihood for Normal(mu, sigma) against observation y (up to exact constants)."""
    mu2 = pd.to_numeric(mu, errors="coerce")
    sig2 = pd.to_numeric(sigma, errors="coerce")
    y2 = pd.to_numeric(y, errors="coerce")
    out = pd.Series(np.nan, index=mu2.index, dtype=float)
    m = mu2.notna() & sig2.notna() & y2.notna() & (sig2 > 1e-9)
    if not m.any():
        return out
    z2 = (((y2[m] - mu2[m]) / sig2[m]) ** 2).astype(float)
    out.loc[m] = 0.5 * np.log(2.0 * math.pi * (sig2[m].astype(float) ** 2)) + 0.5 * z2
    return out


def run_sim_accuracy_backtest(cfg: SimAccuracyBacktestConfig) -> dict[str, Any]:
    cfg.out_dir = Path(cfg.out_dir)
    start = _parse_date(cfg.start)
    end = _parse_date(cfg.end)

    dates = _resolve_dates_from_results(cfg.out_dir, start=start, end=end, recent=cfg.recent)
    if not dates:
        return {"error": "No results_*.csv dates found", "out_dir": str(cfg.out_dir)}

    bt_dir = cfg.out_dir / "backtests"
    bt_dir.mkdir(parents=True, exist_ok=True)

    per_game_rows: list[pd.DataFrame] = []
    per_date_rows: list[dict[str, Any]] = []

    def _bucket_edges(s: pd.Series, bins: list[float]) -> pd.Series:
        try:
            return pd.cut(s.astype(float), bins=bins, right=False, include_lowest=True)
        except Exception:
            return pd.Series([pd.NA] * len(s), index=s.index)

    def _acc_from_mask(correct: pd.Series, mask: pd.Series) -> tuple[int, Optional[float]]:
        if correct is None or mask is None:
            return (0, None)
        cc = pd.to_numeric(correct.where(mask), errors="coerce")
        n = int(cc.notna().sum())
        if n <= 0:
            return (0, None)
        return (n, float(cc.mean()))

    for d in dates:
        sim_path = cfg.out_dir / f"{cfg.sim_quantiles_prefix}{d}.csv"
        if cfg.recompute or (not sim_path.exists()):
            run_simulations_for_date(
                cfg.out_dir,
                d,
                samples=int(cfg.samples),
                rho=float(cfg.rho),
                engine=str(cfg.engine),
                quantiles_out_prefix=str(cfg.sim_quantiles_prefix),
                meta_out_prefix=str(cfg.sim_meta_prefix),
                calibration_json=cfg.calibration_json,
                strip_spread_bins=bool(cfg.strip_spread_bins),
            )

        sim = _load_sim_for_date(cfg.out_dir, d, sim_quantiles_prefix=str(cfg.sim_quantiles_prefix))
        res = _load_results_for_date(cfg.out_dir, d)
        merged = _join_sim_results(sim, res)
        if merged.empty:
            per_date_rows.append({"date": d, "skipped": True, "reason": "empty_join", "n": 0})
            continue

        merged = merged.copy()
        merged["date"] = d

        def _s(col: str) -> pd.Series:
            if col in merged.columns:
                return pd.to_numeric(merged[col], errors="coerce")
            return pd.Series(np.nan, index=merged.index, dtype=float)

        # Coerce numeric in-place for known cols when they exist (keeps downstream merges tidy)
        for c in [
            "actual_total",
            "actual_margin",
            "market_total",
            "spread_home",
            "q50_total",
            "q50_margin",
            "mu_total",
            "mu_margin",
            "actual_total_1h",
            "actual_margin_1h",
            "home_score_1h",
            "away_score_1h",
            "market_total_1h",
            "spread_home_1h",
            "q50_total_1h",
            "q50_margin_1h",
            "mu_total_1h",
            "mu_margin_1h",
        ]:
            if c in merged.columns:
                merged[c] = pd.to_numeric(merged[c], errors="coerce")

        # Choose q50 when available, else mu
        if "q50_total" in merged.columns and merged["q50_total"].notna().any():
            pred_total = merged["q50_total"]
            total_source = "q50_total"
        else:
            pred_total = _s("mu_total")
            total_source = "mu_total"

        if "q50_margin" in merged.columns and merged["q50_margin"].notna().any():
            pred_margin = merged["q50_margin"]
            margin_source = "q50_margin"
        else:
            pred_margin = _s("mu_margin")
            margin_source = "mu_margin"

        # 1H predictions
        if "q50_total_1h" in merged.columns and merged["q50_total_1h"].notna().any():
            pred_total_1h = merged["q50_total_1h"]
            total_1h_source = "q50_total_1h"
        else:
            pred_total_1h = _s("mu_total_1h")
            total_1h_source = "mu_total_1h"

        if "q50_margin_1h" in merged.columns and merged["q50_margin_1h"].notna().any():
            pred_margin_1h = merged["q50_margin_1h"]
            margin_1h_source = "q50_margin_1h"
        else:
            pred_margin_1h = _s("mu_margin_1h")
            margin_1h_source = "mu_margin_1h"

        actual_total = _s("actual_total")
        actual_margin = _s("actual_margin")
        market_total = _s("market_total")
        spread_home = _s("spread_home")

        # Distribution params (Normal) when available
        mu_total = _s("mu_total")
        sigma_total = _s("sigma_total")
        mu_margin = _s("mu_margin")
        sigma_margin = _s("sigma_margin")

        mu_total_1h = _s("mu_total_1h")
        sigma_total_1h = _s("sigma_total_1h")
        mu_margin_1h = _s("mu_margin_1h")
        sigma_margin_1h = _s("sigma_margin_1h")

        # Optional OT diagnostics via interval actuals (regulation @40 from PBP)
        interval_df = pd.DataFrame()
        if bool(cfg.include_ot_diagnostics):
            interval_df = _load_interval_actuals_5min(cfg.out_dir, d, str(cfg.interval_actuals_prefix))

        actual_total_reg40 = pd.Series(np.nan, index=merged.index, dtype=float)
        actual_margin_reg40 = pd.Series(np.nan, index=merged.index, dtype=float)
        actual_total_final_from_intervals = pd.Series(np.nan, index=merged.index, dtype=float)
        actual_margin_final_from_intervals = pd.Series(np.nan, index=merged.index, dtype=float)
        is_ot_game = pd.Series(0.0, index=merged.index, dtype=float)
        ot_points = pd.Series(np.nan, index=merged.index, dtype=float)
        ou_flipped_by_ot = pd.Series(np.nan, index=merged.index, dtype=float)

        if bool(cfg.include_ot_diagnostics) and (not interval_df.empty):
            try:
                # Derive per-game regulation total (@40) and max endpoint (final incl OT when present)
                if "actual_total_score_end" in interval_df.columns:
                    base = interval_df.dropna(subset=["end_min", "actual_total_score_end"]).copy()
                    base["end_min"] = base["end_min"].astype(int)
                    reg = (
                        base[base["end_min"] == 40]
                        .sort_values(["game_id"])
                        .drop_duplicates(subset=["game_id"], keep="last")
                        .rename(columns={"actual_total_score_end": "actual_total_reg40"})
                    )
                    fin = (
                        base.sort_values(["game_id", "end_min"])
                        .groupby("game_id", as_index=False)
                        .last()
                        .rename(columns={"actual_total_score_end": "actual_total_final_from_intervals", "end_min": "max_end_min"})
                    )

                    # If home/away end scores are present, compute margins as well.
                    if "actual_home_score_end" in base.columns and "actual_away_score_end" in base.columns:
                        base["actual_home_score_end"] = pd.to_numeric(base["actual_home_score_end"], errors="coerce")
                        base["actual_away_score_end"] = pd.to_numeric(base["actual_away_score_end"], errors="coerce")
                        reg_m = (
                            base[base["end_min"] == 40]
                            .sort_values(["game_id"])
                            .drop_duplicates(subset=["game_id"], keep="last")
                            .assign(actual_margin_reg40=lambda d: d["actual_home_score_end"] - d["actual_away_score_end"])
                            [["game_id", "actual_margin_reg40"]]
                        )
                        fin_m = (
                            base.sort_values(["game_id", "end_min"])
                            .groupby("game_id", as_index=False)
                            .last()
                            .assign(actual_margin_final_from_intervals=lambda d: d["actual_home_score_end"] - d["actual_away_score_end"])
                            [["game_id", "actual_margin_final_from_intervals"]]
                        )
                    else:
                        reg_m = None
                        fin_m = None

                    # Optional metadata columns (if present)
                    meta_cols = [c for c in ["is_ot_game", "ot_periods"] if c in base.columns]
                    meta = None
                    if meta_cols:
                        meta = (
                            base.sort_values(["game_id", "end_min"])
                            .groupby("game_id", as_index=False)
                            .last()[["game_id"] + meta_cols]
                        )

                    g = reg[["game_id", "actual_total_reg40"]].merge(
                        fin[["game_id", "actual_total_final_from_intervals", "max_end_min"]], on="game_id", how="outer"
                    )
                    if reg_m is not None:
                        g = g.merge(reg_m, on="game_id", how="left")
                    if fin_m is not None:
                        g = g.merge(fin_m, on="game_id", how="left")
                    if meta is not None:
                        g = g.merge(meta, on="game_id", how="left")

                    # Map into merged rows by game_id
                    g["game_id"] = g["game_id"].astype(str)
                    merged_gid = merged.get("game_id").astype(str) if "game_id" in merged.columns else pd.Series("", index=merged.index)

                    g_map_reg = dict(zip(g["game_id"], pd.to_numeric(g.get("actual_total_reg40"), errors="coerce")))
                    g_map_fin = dict(zip(g["game_id"], pd.to_numeric(g.get("actual_total_final_from_intervals"), errors="coerce")))
                    g_map_max = dict(zip(g["game_id"], pd.to_numeric(g.get("max_end_min"), errors="coerce")))

                    actual_total_reg40 = merged_gid.map(g_map_reg).astype(float)
                    actual_total_final_from_intervals = merged_gid.map(g_map_fin).astype(float)
                    max_end = merged_gid.map(g_map_max).astype(float)

                    if "actual_margin_reg40" in g.columns:
                        g_map_m40 = dict(zip(g["game_id"], pd.to_numeric(g.get("actual_margin_reg40"), errors="coerce")))
                        actual_margin_reg40 = merged_gid.map(g_map_m40).astype(float)
                    if "actual_margin_final_from_intervals" in g.columns:
                        g_map_mfin = dict(zip(g["game_id"], pd.to_numeric(g.get("actual_margin_final_from_intervals"), errors="coerce")))
                        actual_margin_final_from_intervals = merged_gid.map(g_map_mfin).astype(float)

                    # is_ot_game: prefer explicit flag, else infer from max endpoint
                    if "is_ot_game" in g.columns:
                        g_map_isot = dict(zip(g["game_id"], pd.to_numeric(g.get("is_ot_game"), errors="coerce")))
                        is_ot_game = merged_gid.map(g_map_isot).fillna(0.0)
                    else:
                        is_ot_game = (max_end > 40).astype(float).fillna(0.0)

                    # OT points computed from intervals (final - reg40)
                    ot_points = (actual_total_final_from_intervals - actual_total_reg40).where(
                        actual_total_final_from_intervals.notna() & actual_total_reg40.notna()
                    )

                    # OU flipped by OT relative to market_total
                    try:
                        line = pd.to_numeric(market_total, errors="coerce")
                        s_reg = actual_total_reg40.combine(line, lambda t, l: _ou_side(t, l))
                        s_fin = actual_total_final_from_intervals.combine(line, lambda t, l: _ou_side(t, l))
                        flipped = (s_reg.notna() & s_fin.notna() & (s_reg != "P") & (s_fin != "P") & (s_reg != s_fin))
                        ou_flipped_by_ot = flipped.astype(float)
                    except Exception:
                        pass
            except Exception:
                # Leave diagnostics as NaN/0 when parsing fails
                pass

        # 1H actuals/markets when present
        actual_total_1h = _s("actual_total_1h")
        market_total_1h = _s("market_total_1h")
        spread_home_1h = _s("spread_home_1h")

        hs1 = _s("home_score_1h")
        as1 = _s("away_score_1h")
        if actual_total_1h.isna().all() and hs1.notna().any() and as1.notna().any():
            actual_total_1h = hs1 + as1

        if "actual_margin_1h" in merged.columns and merged["actual_margin_1h"].notna().any():
            actual_margin_1h = _s("actual_margin_1h")
        elif hs1.notna().any() and as1.notna().any():
            actual_margin_1h = hs1 - as1
        else:
            actual_margin_1h = pd.Series(np.nan, index=merged.index)

        # Winners accuracy: exclude ties
        mw = pred_margin.notna() & actual_margin.notna() & (actual_margin != 0)
        win_correct = ((pred_margin[mw] > 0) == (actual_margin[mw] > 0))

        # Totals accuracy vs market_total: exclude pushes
        mt = pred_total.notna() & actual_total.notna() & market_total.notna() & (actual_total != market_total)
        tot_correct = ((pred_total[mt] > market_total[mt]) == (actual_total[mt] > market_total[mt]))

        # Optional: regulation totals accuracy vs market_total (using PBP @40). Exclude pushes.
        mt_reg40 = (
            bool(cfg.include_ot_diagnostics)
            and pred_total.notna().any()
            and market_total.notna().any()
        )
        if mt_reg40:
            mt40 = pred_total.notna() & actual_total_reg40.notna() & market_total.notna() & (actual_total_reg40 != market_total)
            tot_correct_reg40 = ((pred_total[mt40] > market_total[mt40]) == (actual_total_reg40[mt40] > market_total[mt40]))
        else:
            mt40 = pd.Series(False, index=merged.index)
            tot_correct_reg40 = pd.Series(dtype=bool)

        # Distribution scoring: CRPS / NLL (final and optional reg40)
        crps_total_final = _crps_normal(mu_total, sigma_total, actual_total)
        nll_total_final = _nll_normal(mu_total, sigma_total, actual_total)
        crps_margin_final = _crps_normal(mu_margin, sigma_margin, actual_margin)
        nll_margin_final = _nll_normal(mu_margin, sigma_margin, actual_margin)

        crps_total_reg40 = _crps_normal(mu_total, sigma_total, actual_total_reg40) if bool(cfg.include_ot_diagnostics) else pd.Series(np.nan, index=merged.index)
        nll_total_reg40 = _nll_normal(mu_total, sigma_total, actual_total_reg40) if bool(cfg.include_ot_diagnostics) else pd.Series(np.nan, index=merged.index)
        crps_margin_reg40 = _crps_normal(mu_margin, sigma_margin, actual_margin_reg40) if bool(cfg.include_ot_diagnostics) else pd.Series(np.nan, index=merged.index)
        nll_margin_reg40 = _nll_normal(mu_margin, sigma_margin, actual_margin_reg40) if bool(cfg.include_ot_diagnostics) else pd.Series(np.nan, index=merged.index)

        crps_total_1h = _crps_normal(mu_total_1h, sigma_total_1h, actual_total_1h)
        nll_total_1h = _nll_normal(mu_total_1h, sigma_total_1h, actual_total_1h)
        crps_margin_1h = _crps_normal(mu_margin_1h, sigma_margin_1h, actual_margin_1h)
        nll_margin_1h = _nll_normal(mu_margin_1h, sigma_margin_1h, actual_margin_1h)

        # ATS accuracy vs spread_home: exclude pushes
        ma = pred_margin.notna() & actual_margin.notna() & spread_home.notna()
        push = (actual_margin[ma] + spread_home[ma]).abs() < 1e-9
        ma2 = ma.copy()
        ma2.loc[ma] = ~push
        ats_correct = ((pred_margin[ma2] + spread_home[ma2] > 0) == (actual_margin[ma2] + spread_home[ma2] > 0))

        # 1H Winners/Totals/ATS (exclude ties/pushes)
        mw1 = pred_margin_1h.notna() & actual_margin_1h.notna() & (actual_margin_1h != 0)
        win_correct_1h = ((pred_margin_1h[mw1] > 0) == (actual_margin_1h[mw1] > 0))

        mt1 = (
            pred_total_1h.notna()
            & actual_total_1h.notna()
            & market_total_1h.notna()
            & (actual_total_1h != market_total_1h)
        )
        tot_correct_1h = (
            (pred_total_1h[mt1] > market_total_1h[mt1])
            == (actual_total_1h[mt1] > market_total_1h[mt1])
        )

        ma1 = pred_margin_1h.notna() & actual_margin_1h.notna() & spread_home_1h.notna()
        push1 = (actual_margin_1h[ma1] + spread_home_1h[ma1]).abs() < 1e-9
        ma1b = ma1.copy()
        ma1b.loc[ma1] = ~push1
        ats_correct_1h = (
            (pred_margin_1h[ma1b] + spread_home_1h[ma1b] > 0)
            == (actual_margin_1h[ma1b] + spread_home_1h[ma1b] > 0)
        )

        per_date_rows.append(
            {
                "date": d,
                "skipped": False,
                "n_games": int(len(merged)),
                "winners_n": int(mw.sum()),
                "winners_acc": float(win_correct.mean()) if mw.any() else None,
                "totals_n": int(mt.sum()),
                "totals_acc": float(tot_correct.mean()) if mt.any() else None,
                "totals_reg40_n": int(mt40.sum()) if bool(cfg.include_ot_diagnostics) else 0,
                "totals_reg40_acc": float(tot_correct_reg40.mean()) if (bool(cfg.include_ot_diagnostics) and mt40.any()) else None,
                "crps_total_final": float(pd.to_numeric(crps_total_final, errors="coerce").dropna().mean())
                if pd.to_numeric(crps_total_final, errors="coerce").notna().any()
                else None,
                "crps_total_reg40": float(pd.to_numeric(crps_total_reg40, errors="coerce").dropna().mean())
                if (bool(cfg.include_ot_diagnostics) and pd.to_numeric(crps_total_reg40, errors="coerce").notna().any())
                else None,
                "crps_margin_final": float(pd.to_numeric(crps_margin_final, errors="coerce").dropna().mean())
                if pd.to_numeric(crps_margin_final, errors="coerce").notna().any()
                else None,
                "crps_margin_reg40": float(pd.to_numeric(crps_margin_reg40, errors="coerce").dropna().mean())
                if (bool(cfg.include_ot_diagnostics) and pd.to_numeric(crps_margin_reg40, errors="coerce").notna().any())
                else None,
                "crps_total_1h": float(pd.to_numeric(crps_total_1h, errors="coerce").dropna().mean())
                if pd.to_numeric(crps_total_1h, errors="coerce").notna().any()
                else None,
                "crps_margin_1h": float(pd.to_numeric(crps_margin_1h, errors="coerce").dropna().mean())
                if pd.to_numeric(crps_margin_1h, errors="coerce").notna().any()
                else None,
                "ats_n": int(ma2.sum()),
                "ats_acc": float(ats_correct.mean()) if ma2.any() else None,
                "winners_1h_n": int(mw1.sum()),
                "winners_1h_acc": float(win_correct_1h.mean()) if mw1.any() else None,
                "totals_1h_n": int(mt1.sum()),
                "totals_1h_acc": float(tot_correct_1h.mean()) if mt1.any() else None,
                "ats_1h_n": int(ma1b.sum()),
                "ats_1h_acc": float(ats_correct_1h.mean()) if ma1b.any() else None,
                "ot_games_n": int(pd.to_numeric(is_ot_game, errors="coerce").fillna(0.0).gt(0).sum()) if bool(cfg.include_ot_diagnostics) else 0,
                "ou_flipped_by_ot_n": int(pd.to_numeric(ou_flipped_by_ot, errors="coerce").fillna(0.0).gt(0).sum()) if bool(cfg.include_ot_diagnostics) else 0,
                "pred_total_source": total_source,
                "pred_margin_source": margin_source,
                "pred_total_1h_source": total_1h_source,
                "pred_margin_1h_source": margin_1h_source,
            }
        )

        df_out = pd.DataFrame(
            {
                "date": d,
                "game_id": merged.get("game_id"),
                "home_team": merged.get("home_team"),
                "away_team": merged.get("away_team"),
                "pred_total": pred_total,
                "pred_margin": pred_margin,
                "pred_total_1h": pred_total_1h,
                "pred_margin_1h": pred_margin_1h,
                "actual_total": actual_total,
                "actual_total_reg40": actual_total_reg40 if bool(cfg.include_ot_diagnostics) else np.nan,
                "actual_total_final_from_intervals": actual_total_final_from_intervals if bool(cfg.include_ot_diagnostics) else np.nan,
                "actual_margin_reg40": actual_margin_reg40 if bool(cfg.include_ot_diagnostics) else np.nan,
                "actual_margin_final_from_intervals": actual_margin_final_from_intervals if bool(cfg.include_ot_diagnostics) else np.nan,
                "is_ot_game": is_ot_game if bool(cfg.include_ot_diagnostics) else np.nan,
                "ot_points": ot_points if bool(cfg.include_ot_diagnostics) else np.nan,
                "ou_flipped_by_ot": ou_flipped_by_ot if bool(cfg.include_ot_diagnostics) else np.nan,
                "actual_margin": actual_margin,
                "actual_total_1h": actual_total_1h,
                "actual_margin_1h": actual_margin_1h,
                "market_total": market_total,
                "spread_home": spread_home,
                "market_total_1h": market_total_1h,
                "spread_home_1h": spread_home_1h,
                "mu_total": mu_total,
                "sigma_total": sigma_total,
                "mu_margin": mu_margin,
                "sigma_margin": sigma_margin,
                "mu_total_1h": mu_total_1h,
                "sigma_total_1h": sigma_total_1h,
                "mu_margin_1h": mu_margin_1h,
                "sigma_margin_1h": sigma_margin_1h,
                "crps_total_final": crps_total_final,
                "nll_total_final": nll_total_final,
                "crps_margin_final": crps_margin_final,
                "nll_margin_final": nll_margin_final,
                "crps_total_reg40": crps_total_reg40 if bool(cfg.include_ot_diagnostics) else np.nan,
                "nll_total_reg40": nll_total_reg40 if bool(cfg.include_ot_diagnostics) else np.nan,
                "crps_margin_reg40": crps_margin_reg40 if bool(cfg.include_ot_diagnostics) else np.nan,
                "nll_margin_reg40": nll_margin_reg40 if bool(cfg.include_ot_diagnostics) else np.nan,
                "crps_total_1h": crps_total_1h,
                "nll_total_1h": nll_total_1h,
                "crps_margin_1h": crps_margin_1h,
                "nll_margin_1h": nll_margin_1h,
            }
        )
        if len(df_out):
            df_out["winner_correct"] = np.nan
            df_out.loc[mw, "winner_correct"] = win_correct.astype(float).to_numpy()
            df_out["total_correct"] = np.nan
            df_out.loc[mt, "total_correct"] = tot_correct.astype(float).to_numpy()

            if bool(cfg.include_ot_diagnostics):
                df_out["total_correct_reg40"] = np.nan
                if isinstance(mt40, pd.Series) and mt40.any():
                    df_out.loc[mt40, "total_correct_reg40"] = tot_correct_reg40.astype(float).to_numpy()
            df_out["ats_correct"] = np.nan
            df_out.loc[ma2, "ats_correct"] = ats_correct.astype(float).to_numpy()

            df_out["winner_correct_1h"] = np.nan
            df_out.loc[mw1, "winner_correct_1h"] = win_correct_1h.astype(float).to_numpy()
            df_out["total_correct_1h"] = np.nan
            df_out.loc[mt1, "total_correct_1h"] = tot_correct_1h.astype(float).to_numpy()
            df_out["ats_correct_1h"] = np.nan
            df_out.loc[ma1b, "ats_correct_1h"] = ats_correct_1h.astype(float).to_numpy()

        per_game_rows.append(df_out)

    per_date = pd.DataFrame(per_date_rows).sort_values("date")
    per_game = pd.concat(per_game_rows, ignore_index=True) if per_game_rows else pd.DataFrame()

    def _agg_acc(n_col: str, acc_col: str) -> dict[str, Any]:
        ok = (per_date.get("skipped") == False) if "skipped" in per_date.columns else pd.Series(True, index=per_date.index)
        n = pd.to_numeric(per_date.loc[ok, n_col], errors="coerce")
        acc = pd.to_numeric(per_date.loc[ok, acc_col], errors="coerce")
        mask = n.notna() & acc.notna() & (n > 0)
        if not mask.any():
            return {"n": 0, "acc": None}
        correct = float(np.sum((n[mask] * acc[mask]).astype(float)))
        total = float(np.sum(n[mask].astype(float)))
        return {"n": int(total), "acc": float(correct / total) if total > 0 else None}

    def _agg_mean(col: str) -> dict[str, Any]:
        try:
            ok = (per_date.get("skipped") == False) if "skipped" in per_date.columns else pd.Series(True, index=per_date.index)
            v = pd.to_numeric(per_date.loc[ok, col], errors="coerce")
            n = int(v.notna().sum())
            return {"n": n, "mean": float(v.dropna().mean()) if n > 0 else None}
        except Exception:
            return {"n": 0, "mean": None}

    summary = {
        "range": {"start": dates[0], "end": dates[-1], "n_dates": int(len(dates))},
        "engine": str(cfg.engine),
        "samples": int(cfg.samples),
        "rho": float(cfg.rho),
        "sim_quantiles_prefix": str(cfg.sim_quantiles_prefix),
        "sim_meta_prefix": str(cfg.sim_meta_prefix),
        "include_ot_diagnostics": bool(cfg.include_ot_diagnostics),
        "interval_actuals_prefix": str(cfg.interval_actuals_prefix),
        "dates_scored": int(per_date.loc[per_date.get("skipped") == False].shape[0]) if "skipped" in per_date.columns else int(len(per_date)),
        "dates_skipped": int(per_date.get("skipped").sum()) if "skipped" in per_date.columns else 0,
        "winners": _agg_acc("winners_n", "winners_acc"),
        "totals": _agg_acc("totals_n", "totals_acc"),
        "totals_reg40": _agg_acc("totals_reg40_n", "totals_reg40_acc") if bool(cfg.include_ot_diagnostics) else {"n": 0, "acc": None},
        "ats": _agg_acc("ats_n", "ats_acc"),
        "winners_1h": _agg_acc("winners_1h_n", "winners_1h_acc"),
        "totals_1h": _agg_acc("totals_1h_n", "totals_1h_acc"),
        "ats_1h": _agg_acc("ats_1h_n", "ats_1h_acc"),
        "scoring": {
            "crps_total_final": _agg_mean("crps_total_final"),
            "crps_total_reg40": _agg_mean("crps_total_reg40") if bool(cfg.include_ot_diagnostics) else {"n": 0, "mean": None},
            "crps_margin_final": _agg_mean("crps_margin_final"),
            "crps_margin_reg40": _agg_mean("crps_margin_reg40") if bool(cfg.include_ot_diagnostics) else {"n": 0, "mean": None},
            "crps_total_1h": _agg_mean("crps_total_1h"),
            "crps_margin_1h": _agg_mean("crps_margin_1h"),
        },
    }

    if bool(cfg.include_ot_diagnostics) and (not per_date.empty):
        try:
            ok = (per_date.get("skipped") == False) if "skipped" in per_date.columns else pd.Series(True, index=per_date.index)
            ot_games = pd.to_numeric(per_date.loc[ok, "ot_games_n"], errors="coerce").fillna(0.0)
            flips = pd.to_numeric(per_date.loc[ok, "ou_flipped_by_ot_n"], errors="coerce").fillna(0.0)
            summary["ot"] = {
                "ot_games": int(ot_games.sum()),
                "ou_flipped_by_ot": int(flips.sum()),
            }
        except Exception:
            pass

    out_stem = f"{cfg.out_prefix}_{dates[0]}_{dates[-1]}"
    out_game = bt_dir / f"{out_stem}.csv"
    out_per_date = bt_dir / f"{out_stem}_per_date.csv"
    out_summary = bt_dir / f"{out_stem}_summary.json"

    # Breakdown outputs (computed from per_game for robustness)
    out_by_month = bt_dir / f"{out_stem}_by_month.csv"
    out_by_edge = bt_dir / f"{out_stem}_by_edge_bucket.csv"

    if not per_game.empty:
        per_game.to_csv(out_game, index=False, na_rep="")
    per_date.to_csv(out_per_date, index=False, na_rep="")

    # Build breakdowns
    breakdown_month_rows: list[dict[str, Any]] = []
    breakdown_edge_rows: list[dict[str, Any]] = []

    if not per_game.empty:
        pg = per_game.copy()
        pg["month"] = pg["date"].astype(str).str.slice(0, 7)

        metrics = [
            ("winners", "winner_correct", None),
            ("totals", "total_correct", "total"),
            ("ats", "ats_correct", "ats"),
            ("winners_1h", "winner_correct_1h", None),
            ("totals_1h", "total_correct_1h", "total_1h"),
            ("ats_1h", "ats_correct_1h", "ats_1h"),
        ]

        for metric_name, correct_col, edge_kind in metrics:
            # Month breakdown
            if correct_col in pg.columns:
                for m, g in pg.groupby("month", dropna=False):
                    n, acc = _acc_from_mask(g[correct_col], g[correct_col].notna())
                    breakdown_month_rows.append({"metric": metric_name, "month": str(m), "n": n, "acc": acc})

            # Edge buckets
            if edge_kind and correct_col in pg.columns:
                if edge_kind == "total":
                    edge = (pd.to_numeric(pg.get("pred_total"), errors="coerce") - pd.to_numeric(pg.get("market_total"), errors="coerce")).abs()
                elif edge_kind == "total_1h":
                    edge = (pd.to_numeric(pg.get("pred_total_1h"), errors="coerce") - pd.to_numeric(pg.get("market_total_1h"), errors="coerce")).abs()
                elif edge_kind == "ats":
                    edge = (pd.to_numeric(pg.get("pred_margin"), errors="coerce") + pd.to_numeric(pg.get("spread_home"), errors="coerce")).abs()
                elif edge_kind == "ats_1h":
                    edge = (pd.to_numeric(pg.get("pred_margin_1h"), errors="coerce") + pd.to_numeric(pg.get("spread_home_1h"), errors="coerce")).abs()
                else:
                    edge = pd.Series(np.nan, index=pg.index)

                bins = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, float("inf")]
                bucket = _bucket_edges(edge, bins=bins)
                tmp = pg[[correct_col]].copy()
                tmp["bucket"] = bucket
                for b, g in tmp.groupby("bucket", dropna=False, observed=False):
                    n, acc = _acc_from_mask(g[correct_col], g[correct_col].notna())
                    breakdown_edge_rows.append({"metric": metric_name, "bucket": str(b), "n": n, "acc": acc})

    by_month = pd.DataFrame(breakdown_month_rows)
    by_edge = pd.DataFrame(breakdown_edge_rows)

    if not by_month.empty:
        by_month.to_csv(out_by_month, index=False, na_rep="")
    if not by_edge.empty:
        by_edge.to_csv(out_by_edge, index=False, na_rep="")

    summary["breakdowns"] = {
        "by_month_csv": str(out_by_month) if (not by_month.empty) else None,
        "by_edge_bucket_csv": str(out_by_edge) if (not by_edge.empty) else None,
    }

    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "wrote": {
            "per_game": str(out_game),
            "per_date": str(out_per_date),
            "summary": str(out_summary),
            "by_month": str(out_by_month) if (out_by_month.exists()) else None,
            "by_edge_bucket": str(out_by_edge) if (out_by_edge.exists()) else None,
        },
        "n_dates": int(len(dates)),
        "n_games": int(len(per_game)) if not per_game.empty else 0,
        "dates_skipped": int(summary.get("dates_skipped") or 0),
    }
