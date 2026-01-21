from __future__ import annotations

import dataclasses
import datetime as dt
import json
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
            )

        sim = _load_sim_for_date(cfg.out_dir, d)
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
                "ats_n": int(ma2.sum()),
                "ats_acc": float(ats_correct.mean()) if ma2.any() else None,
                "winners_1h_n": int(mw1.sum()),
                "winners_1h_acc": float(win_correct_1h.mean()) if mw1.any() else None,
                "totals_1h_n": int(mt1.sum()),
                "totals_1h_acc": float(tot_correct_1h.mean()) if mt1.any() else None,
                "ats_1h_n": int(ma1b.sum()),
                "ats_1h_acc": float(ats_correct_1h.mean()) if ma1b.any() else None,
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
                "actual_margin": actual_margin,
                "actual_total_1h": actual_total_1h,
                "actual_margin_1h": actual_margin_1h,
                "market_total": market_total,
                "spread_home": spread_home,
                "market_total_1h": market_total_1h,
                "spread_home_1h": spread_home_1h,
            }
        )
        if len(df_out):
            df_out["winner_correct"] = np.nan
            df_out.loc[mw, "winner_correct"] = win_correct.astype(float).to_numpy()
            df_out["total_correct"] = np.nan
            df_out.loc[mt, "total_correct"] = tot_correct.astype(float).to_numpy()
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

    summary = {
        "range": {"start": dates[0], "end": dates[-1], "n_dates": int(len(dates))},
        "engine": str(cfg.engine),
        "samples": int(cfg.samples),
        "rho": float(cfg.rho),
        "dates_scored": int(per_date.loc[per_date.get("skipped") == False].shape[0]) if "skipped" in per_date.columns else int(len(per_date)),
        "dates_skipped": int(per_date.get("skipped").sum()) if "skipped" in per_date.columns else 0,
        "winners": _agg_acc("winners_n", "winners_acc"),
        "totals": _agg_acc("totals_n", "totals_acc"),
        "ats": _agg_acc("ats_n", "ats_acc"),
        "winners_1h": _agg_acc("winners_1h_n", "winners_1h_acc"),
        "totals_1h": _agg_acc("totals_1h_n", "totals_1h_acc"),
        "ats_1h": _agg_acc("ats_1h_n", "ats_1h_acc"),
    }

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
