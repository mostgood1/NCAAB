from __future__ import annotations

import datetime as dt
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ncaab_model.config import settings
from ncaab_model.data.adapters.espn_playbyplay import fetch_playbyplay, extract_cum_totals_5min, infer_ot_periods


@dataclass(frozen=True)
class Segments5MinBacktestConfig:
    out_dir: Path
    start: str
    end: str
    engine: str = "events"
    samples: int = 2000
    rho: float = 0.25
    recompute_sims: bool = False
    use_cache: bool = True
    sleep_seconds: float = 0.15
    max_games: int = 0
    out_prefix: str = "segments_5min"
    sim_segments_prefix: str = "sim_segments_"
    sim_quantiles_prefix: str = "sim_quantiles_"
    sim_meta_prefix: str = "sim_meta_"
    emit_ot_rows: bool = True
    max_ot_periods: int = 4


def _date_range(start_iso: str, end_iso: str) -> Iterable[dt.date]:
    s = dt.date.fromisoformat(start_iso)
    e = dt.date.fromisoformat(end_iso)
    cur = s
    one = dt.timedelta(days=1)
    while cur <= e:
        yield cur
        cur += one


def _pinball(q: float, y: float, tau: float) -> float:
    # quantile loss (pinball)
    # tau in (0,1)
    e = y - q
    return float(max(tau * e, (tau - 1.0) * e))


def _ensure_sim_segments_for_date(cfg: Segments5MinBacktestConfig, date_iso: str) -> Optional[Path]:
    out_dir = Path(cfg.out_dir)
    seg_path = out_dir / f"{cfg.sim_segments_prefix}{date_iso}.csv"

    if seg_path.exists() and not cfg.recompute_sims:
        return seg_path

    try:
        from src.simulation.game_sim import run_simulations_for_date

        run_simulations_for_date(
            out_dir=out_dir,
            date=date_iso,
            samples=int(cfg.samples),
            rho=float(cfg.rho),
            engine=str(cfg.engine),
            quantiles_out_prefix=str(cfg.sim_quantiles_prefix),
            segments_out_prefix=str(cfg.sim_segments_prefix),
            meta_out_prefix=str(cfg.sim_meta_prefix),
        )
    except Exception:
        return None

    return seg_path if seg_path.exists() else None


def run_segments_5min_backtest(cfg: Segments5MinBacktestConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_bt = out_dir / "backtests"
    out_bt.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n_dates = 0
    n_games = 0
    n_missing_pbp = 0
    n_missing_segments = 0
    n_ot_games = 0
    n_ot_rows = 0
    ot_points: list[float] = []

    for d in _date_range(cfg.start, cfg.end):
        date_iso = d.isoformat()
        n_dates += 1

        seg_path = _ensure_sim_segments_for_date(cfg, date_iso)
        if seg_path is None or not seg_path.exists():
            n_missing_segments += 1
            continue

        try:
            seg = pd.read_csv(seg_path)
        except Exception:
            n_missing_segments += 1
            continue

        if seg.empty or "game_id" not in seg.columns:
            continue

        seg["game_id"] = seg["game_id"].astype(str)
        seg["end_min"] = pd.to_numeric(seg.get("end_min"), errors="coerce")
        seg = seg.dropna(subset=["end_min"])

        # We evaluate cumulative totals at each 5-minute endpoint.
        # Prefer cumulative columns (score_end) since those are the meaningful rollups.
        needed = [
            "q10_total_score_end",
            "q50_total_score_end",
            "q90_total_score_end",
            "mu_total_score_end",
        ]
        for c in needed:
            if c not in seg.columns:
                seg[c] = np.nan

        seg = seg[["game_id", "end_min"] + needed].copy()

        for gid, gdf in seg.groupby("game_id"):
            if cfg.max_games and int(cfg.max_games) > 0 and n_games >= int(cfg.max_games):
                break

            payload = fetch_playbyplay(gid, use_cache=cfg.use_cache)
            fetched_from = payload.get("_fetched_from") if isinstance(payload, dict) else None
            did_network = isinstance(fetched_from, str) and fetched_from.startswith("network")
            plays = payload.get("plays") if isinstance(payload, dict) else None
            if payload is None or not isinstance(plays, list) or len(plays) == 0:
                n_missing_pbp += 1
                if did_network and cfg.sleep_seconds and float(cfg.sleep_seconds) > 0:
                    time.sleep(float(cfg.sleep_seconds))
                continue

            base_endpoints = [5, 10, 15, 20, 25, 30, 35, 40]
            ot_periods = infer_ot_periods(payload) if bool(cfg.emit_ot_rows) else 0
            endpoints = list(base_endpoints)
            if ot_periods > 0:
                n_ot_games += 1
                extra = [40 + 5 * i for i in range(1, min(int(cfg.max_ot_periods), int(ot_periods)) + 1)]
                endpoints = sorted(set(endpoints + extra))

            actual = extract_cum_totals_5min(payload, endpoints=endpoints)
            actual_map = {int(x.end_min): int(x.total_score) for x in actual}

            # OT points summary (final OT endpoint vs regulation endpoint)
            if ot_periods > 0:
                try:
                    reg = float(actual_map.get(40))
                    fin = float(actual_map.get(40 + 5 * int(min(int(cfg.max_ot_periods), int(ot_periods)))))
                    if math.isfinite(reg) and math.isfinite(fin):
                        ot_points.append(float(fin - reg))
                except Exception:
                    pass

            # Preds for this game
            pred_map = {}
            for _, r in gdf.iterrows():
                try:
                    end_min = int(r["end_min"])
                except Exception:
                    continue
                pred_map[end_min] = {
                    "mu": float(r.get("mu_total_score_end")) if pd.notna(r.get("mu_total_score_end")) else None,
                    "q10": float(r.get("q10_total_score_end")) if pd.notna(r.get("q10_total_score_end")) else None,
                    "q50": float(r.get("q50_total_score_end")) if pd.notna(r.get("q50_total_score_end")) else None,
                    "q90": float(r.get("q90_total_score_end")) if pd.notna(r.get("q90_total_score_end")) else None,
                }

            # Emit rows for all extracted actual endpoints.
            for end_min, y in actual_map.items():
                end_min_i = int(end_min)
                p = pred_map.get(end_min_i)

                # Regulation endpoints (<=40) should always have preds; OT endpoints generally won't.
                if p and p.get("q50") is not None and math.isfinite(float(p.get("q50"))):
                    q50 = float(p.get("q50"))
                    err = float(q50) - float(y)
                    rows.append(
                        {
                            "date": date_iso,
                            "game_id": str(gid),
                            "end_min": int(end_min_i),
                            "actual_total": float(y),
                            "pred_mu": p.get("mu"),
                            "pred_q10": p.get("q10"),
                            "pred_q50": p.get("q50"),
                            "pred_q90": p.get("q90"),
                            "err_q50": err,
                            "abs_err_q50": abs(err),
                            "pinball_q10": _pinball(float(p.get("q10")) if p.get("q10") is not None else float("nan"), float(y), 0.10)
                            if p.get("q10") is not None and math.isfinite(float(p.get("q10")))
                            else None,
                            "pinball_q50": _pinball(float(p.get("q50")), float(y), 0.50),
                            "pinball_q90": _pinball(float(p.get("q90")) if p.get("q90") is not None else float("nan"), float(y), 0.90)
                            if p.get("q90") is not None and math.isfinite(float(p.get("q90")))
                            else None,
                            "is_ot_game": 1 if ot_periods > 0 else 0,
                            "is_ot_endpoint": 1 if end_min_i > 40 else 0,
                            "ot_periods": int(ot_periods),
                        }
                    )
                elif bool(cfg.emit_ot_rows) and ot_periods > 0 and end_min_i > 40:
                    # OT-only row: keep for reconciliation, exclude from metrics.
                    rows.append(
                        {
                            "date": date_iso,
                            "game_id": str(gid),
                            "end_min": int(end_min_i),
                            "actual_total": float(y),
                            "pred_mu": None,
                            "pred_q10": None,
                            "pred_q50": None,
                            "pred_q90": None,
                            "err_q50": None,
                            "abs_err_q50": None,
                            "pinball_q10": None,
                            "pinball_q50": None,
                            "pinball_q90": None,
                            "is_ot_game": 1,
                            "is_ot_endpoint": 1,
                            "ot_periods": int(ot_periods),
                        }
                    )
                    n_ot_rows += 1

            n_games += 1

            if did_network and cfg.sleep_seconds and float(cfg.sleep_seconds) > 0:
                time.sleep(float(cfg.sleep_seconds))

        if cfg.max_games and int(cfg.max_games) > 0 and n_games >= int(cfg.max_games):
            break

    df = pd.DataFrame(rows)

    # Summary metrics
    summary: dict = {
        "start": cfg.start,
        "end": cfg.end,
        "engine": cfg.engine,
        "samples": int(cfg.samples),
        "rho": float(cfg.rho),
        "recompute_sims": bool(cfg.recompute_sims),
        "use_cache": bool(cfg.use_cache),
        "dates_considered": int(n_dates),
        "games_processed": int(n_games),
        "missing_sim_segments_dates": int(n_missing_segments),
        "missing_pbp_games": int(n_missing_pbp),
        "emit_ot_rows": bool(cfg.emit_ot_rows),
        "max_ot_periods": int(cfg.max_ot_periods),
        "ot_games": int(n_ot_games),
        "ot_rows_emitted": int(n_ot_rows),
    }

    if df.empty:
        summary.update(
            {
                "rows": 0,
                "mae_q50": None,
                "bias_q50": None,
                "by_end_min": [],
            }
        )
    else:
        summary["rows"] = int(len(df))
        # Metrics are evaluated on regulation endpoints with valid predictions.
        df_metrics = df.copy()
        if "end_min" in df_metrics.columns:
            df_metrics["end_min"] = pd.to_numeric(df_metrics["end_min"], errors="coerce")
        df_metrics["pred_q50"] = pd.to_numeric(df_metrics.get("pred_q50"), errors="coerce")
        df_metrics["abs_err_q50"] = pd.to_numeric(df_metrics.get("abs_err_q50"), errors="coerce")
        df_metrics["err_q50"] = pd.to_numeric(df_metrics.get("err_q50"), errors="coerce")
        df_metrics = df_metrics[(df_metrics["end_min"] <= 40) & df_metrics["pred_q50"].notna()]

        summary["mae_q50"] = float(df_metrics["abs_err_q50"].mean()) if not df_metrics.empty else None
        summary["bias_q50"] = float(df_metrics["err_q50"].mean()) if not df_metrics.empty else None

        by_end = []
        df_by_end = df_metrics.copy()
        for end_min, g in df_by_end.groupby("end_min"):
            by_end.append(
                {
                    "end_min": int(end_min),
                    "n": int(len(g)),
                    "mae_q50": float(g["abs_err_q50"].mean()),
                    "bias_q50": float(g["err_q50"].mean()),
                    "pinball_q10": float(pd.to_numeric(g["pinball_q10"], errors="coerce").dropna().mean())
                    if pd.to_numeric(g["pinball_q10"], errors="coerce").notna().any()
                    else None,
                    "pinball_q50": float(pd.to_numeric(g["pinball_q50"], errors="coerce").dropna().mean())
                    if pd.to_numeric(g["pinball_q50"], errors="coerce").notna().any()
                    else None,
                    "pinball_q90": float(pd.to_numeric(g["pinball_q90"], errors="coerce").dropna().mean())
                    if pd.to_numeric(g["pinball_q90"], errors="coerce").notna().any()
                    else None,
                }
            )
        by_end.sort(key=lambda x: x["end_min"])
        summary["by_end_min"] = by_end

        if ot_points:
            try:
                arr = np.asarray(ot_points, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    summary["ot_points"] = {
                        "n": int(arr.size),
                        "mean": float(np.mean(arr)),
                        "median": float(np.median(arr)),
                        "max": float(np.max(arr)),
                    }
            except Exception:
                pass

    # Write artifacts
    tag = f"{cfg.start}_to_{cfg.end}".replace(":", "-")
    out_csv = out_bt / f"{cfg.out_prefix}_{tag}.csv"
    out_json = out_bt / f"{cfg.out_prefix}_{tag}.json"

    try:
        df.to_csv(out_csv, index=False)
    except Exception:
        pass

    try:
        out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass

    summary["out_csv"] = str(out_csv)
    summary["out_json"] = str(out_json)
    return summary
