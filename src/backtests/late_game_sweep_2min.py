from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ncaab_model.config import settings


@dataclass(frozen=True)
class LateGame2MinSweepConfig:
    empirical_profile_json: Path
    out_dir: Path
    out_prefix: str = "late_game_2min_sweep"
    sims_per_bucket: int = 20000
    seed: int = 1337


def default_config(
    empirical_profile_json: Optional[Path] = None,
    out_prefix: str = "late_game_2min_sweep",
    sims_per_bucket: int = 20000,
    seed: int = 1337,
) -> LateGame2MinSweepConfig:
    if empirical_profile_json is None:
        empirical_profile_json = (
            settings.outputs_dir
            / "backtests"
            / "late_game_2min_profile_holdout_2026-01-11_to_2026-01-19.json"
        )
    return LateGame2MinSweepConfig(
        empirical_profile_json=Path(empirical_profile_json),
        out_dir=settings.outputs_dir,
        out_prefix=str(out_prefix),
        sims_per_bucket=int(sims_per_bucket),
        seed=int(seed),
    )


def _safe_float(v: object) -> Optional[float]:
    try:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _load_empirical_targets(profile_json: Path) -> dict:
    payload = json.loads(profile_json.read_text(encoding="utf-8"))
    by_bucket = payload.get("by_bucket") or []

    # Collapse home/away into symmetric abs-margin buckets.
    # Targets keyed by: tie, lead_1_2, lead_3_5, lead_6_9, lead_10p
    agg: dict[str, dict[str, float]] = {}
    for row in by_bucket:
        b = str(row.get("bucket") or "")
        mean = _safe_float(row.get("mean_last2_total_points"))
        n = int(row.get("n") or 0)
        if not b or mean is None or n <= 0:
            continue
        if b == "tie":
            key = "tie"
        elif b.endswith("_lead_1_2"):
            key = "lead_1_2"
        elif b.endswith("_lead_3_5"):
            key = "lead_3_5"
        elif b.endswith("_lead_6_9"):
            key = "lead_6_9"
        elif b.endswith("_lead_10p"):
            key = "lead_10p"
        else:
            continue
        if key not in agg:
            agg[key] = {"sum": 0.0, "n": 0.0}
        agg[key]["sum"] += float(mean) * float(n)
        agg[key]["n"] += float(n)

    targets: dict[str, dict[str, float]] = {}
    for key, v in agg.items():
        n = float(v.get("n") or 0.0)
        if n <= 0:
            continue
        targets[key] = {"mean": float(v["sum"] / n), "n": float(n)}

    if not targets:
        raise ValueError(f"No usable targets in {profile_json}")

    return targets


def _simulate_single_possession_points(
    to_rate: float,
    ft_trip: float,
    three_rate: float,
    ft_pct: float,
    p2: float,
    p3: float,
    rng: np.random.Generator,
) -> int:
    u = float(rng.random())
    if u < float(to_rate):
        return 0
    u2 = float(rng.random())
    if u2 < float(ft_trip):
        return int(rng.binomial(2, float(np.clip(ft_pct, 0.0, 1.0))))
    u3 = float(rng.random())
    if u3 < float(three_rate):
        return 3 if float(rng.random()) < float(np.clip(p3, 0.0, 1.0)) else 0
    return 2 if float(rng.random()) < float(np.clip(p2, 0.0, 1.0)) else 0


def _calibrate_shooting_to_ppp(
    ppp_target: float,
    to_rate: float,
    ft_trip: float,
    three_rate: float,
) -> tuple[float, float, float]:
    """Lightweight copy of the simulator's calibration to keep the sweep realistic."""
    ppp_t = float(np.clip(float(ppp_target), 0.75, 1.35))
    ft_pct = float(np.clip(0.72 + (ppp_t - 1.0) * 0.06, 0.62, 0.82))
    p2 = 0.50
    p3 = 0.34

    base_e = (1.0 - float(to_rate)) * (
        float(ft_trip) * (2.0 * ft_pct)
        + (1.0 - float(ft_trip))
        * (
            float(three_rate) * (3.0 * p3)
            + (1.0 - float(three_rate)) * (2.0 * p2)
        )
    )
    if base_e <= 1e-9:
        return ft_pct, float(p2), float(p3)

    k = float(np.clip(ppp_t / base_e, 0.70, 1.35))
    p2 = float(np.clip(p2 * k, 0.35, 0.72))
    p3 = float(np.clip(p3 * k, 0.25, 0.52))
    ft_pct = float(np.clip(ft_pct * (0.5 + 0.5 * k), 0.60, 0.86))
    return ft_pct, p2, p3


def _sim_last2_points_for_margin(
    margin: int,
    params: dict[str, float],
    rng: np.random.Generator,
) -> int:
    # This is an intentionally small, self-contained replica of the late-foul heuristic
    # used by the possession-timeline segment generator.

    # Baseline rates (roughly consistent with defaults)
    to_h = 0.175
    ft_h = 0.115
    three_h = 0.36
    to_a = 0.175
    ft_a = 0.115
    three_a = 0.36

    # Baseline shooting calibrated to a realistic PPP.
    # This matters a lot for matching last-2-min empirical point means.
    ppp_target = float(params.get("ppp_target", 1.05))
    ft_pct_h, p2_h, p3_h = _calibrate_shooting_to_ppp(ppp_target, to_h, ft_h, three_h)
    ft_pct_a, p2_a, p3_a = _calibrate_shooting_to_ppp(ppp_target, to_a, ft_a, three_a)

    # Clock model: gamma around a base mean possession duration.
    # 18s gives ~6-7 combined possessions in 2:00 before adjustments; fouls will shorten.
    base_mean_s = float(params.get("base_mean_s", 18.0))
    shape = 2.0
    scale = float(max(1.0, base_mean_s / shape))

    close_dt_mult = float(params["close_dt_mult"])
    trail_dt_mult = float(params["trail_dt_mult"])
    lead_dt_mult = float(params["lead_dt_mult"])
    trail_three_delta = float(params["trail_three_delta"])
    lead_ft_delta = float(params["lead_ft_delta"])
    lead_to_delta = float(params["lead_to_delta"])
    lead_three_delta = float(params["lead_three_delta"])

    margin_thresh = int(params.get("margin_thresh", 3))
    close_margin = int(params.get("close_margin", 2))

    # Start of last 2:00: random possession
    home = 0
    away = 0
    t = 0.0
    home_ball = bool(rng.random() < 0.5)

    while t < 120.0:
        time_remaining = 120.0 - t
        # In the segment generator, "margin" is based on cumulative home-away points.
        # Here, we start with the provided margin at 2:00.
        cur_margin = int(margin + home - away)

        adj_home = (to_h, ft_h, three_h)
        adj_away = (to_a, ft_a, three_a)
        dt_mult = 1.0

        if abs(cur_margin) <= close_margin:
            dt_mult = close_dt_mult

        if home_ball:
            if cur_margin <= -margin_thresh:
                # Home trailing on offense
                adj_home = (to_h, ft_h, min(0.48, three_h + trail_three_delta))
                dt_mult = min(dt_mult, trail_dt_mult)
            elif cur_margin >= margin_thresh:
                # Home leading on offense (away fouls)
                adj_home = (
                    float(np.clip(to_h + lead_to_delta, 0.11, 0.25)),
                    float(np.clip(ft_h + lead_ft_delta, 0.06, 0.18)),
                    float(np.clip(three_h + lead_three_delta, 0.25, 0.50)),
                )
                dt_mult = min(dt_mult, lead_dt_mult)
        else:
            if cur_margin >= margin_thresh:
                # Away trailing on offense
                adj_away = (to_a, ft_a, min(0.48, three_a + trail_three_delta))
                dt_mult = min(dt_mult, trail_dt_mult)
            elif cur_margin <= -margin_thresh:
                # Away leading on offense (home fouls)
                adj_away = (
                    float(np.clip(to_a + lead_to_delta, 0.11, 0.25)),
                    float(np.clip(ft_a + lead_ft_delta, 0.06, 0.18)),
                    float(np.clip(three_a + lead_three_delta, 0.25, 0.50)),
                )
                dt_mult = min(dt_mult, lead_dt_mult)

        dt = float(rng.gamma(shape, scale))
        dt = float(np.clip(dt * dt_mult, 4.0, 40.0))
        t += dt

        if home_ball:
            th, fh, trh = adj_home
            home += _simulate_single_possession_points(th, fh, trh, ft_pct_h, p2_h, p3_h, rng)
        else:
            ta, fa, tra = adj_away
            away += _simulate_single_possession_points(ta, fa, tra, ft_pct_a, p2_a, p3_a, rng)
        home_ball = not home_ball

        # If we're basically out of time, stop.
        if time_remaining <= 2.0:
            break

    return int(home + away)


def _evaluate_params_against_targets(
    targets: dict,
    params: dict[str, float],
    sims_per_bucket: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(int(seed))

    # Representative margins for each bucket (symmetric; we average +m and -m)
    reps = {
        "tie": 0,
        "lead_1_2": 2,
        "lead_3_5": 4,
        "lead_6_9": 7,
        "lead_10p": 15,
    }

    rows = []
    loss = 0.0
    wsum = 0.0

    for key, meta in targets.items():
        m = int(reps.get(key, 0))
        n_target = float(meta.get("n") or 1.0)
        target_mean = float(meta.get("mean"))

        # Simulate both sides leading and average to remove sign bias.
        sims = []
        half = int(max(1, sims_per_bucket // 2))
        for _ in range(half):
            sims.append(_sim_last2_points_for_margin(+m, params, rng))
        for _ in range(sims_per_bucket - half):
            sims.append(_sim_last2_points_for_margin(-m, params, rng))

        sims_arr = np.asarray(sims, dtype=float)
        pred_mean = float(np.mean(sims_arr))
        err = pred_mean - target_mean

        rows.append(
            {
                "bucket": key,
                "target_mean": target_mean,
                "pred_mean": pred_mean,
                "err": err,
                "weight_n": n_target,
            }
        )

        # Weighted squared error
        loss += float(n_target) * float(err * err)
        wsum += float(n_target)

    loss = float(loss / max(1e-9, wsum))
    return {
        "loss": loss,
        "by_bucket": rows,
    }


def run_late_game_2min_sweep(cfg: LateGame2MinSweepConfig) -> dict:
    if not cfg.empirical_profile_json.exists():
        raise FileNotFoundError(f"Missing empirical profile json: {cfg.empirical_profile_json}")

    targets = _load_empirical_targets(cfg.empirical_profile_json)

    # Parameter grid (small on purpose)
    grid = []
    for close_dt_mult in [0.88, 0.92, 0.96]:
        for trail_dt_mult in [0.80, 0.85, 0.90]:
            for lead_dt_mult in [0.65, 0.75, 0.85]:
                for trail_three_delta in [0.03, 0.04, 0.06]:
                    for lead_ft_delta in [0.04, 0.06, 0.08, 0.10]:
                        for lead_to_delta in [-0.02, -0.01, 0.0]:
                            for lead_three_delta in [-0.05, -0.03, -0.01]:
                                grid.append(
                                    {
                                        "close_dt_mult": float(close_dt_mult),
                                        "trail_dt_mult": float(trail_dt_mult),
                                        "lead_dt_mult": float(lead_dt_mult),
                                        "trail_three_delta": float(trail_three_delta),
                                        "lead_ft_delta": float(lead_ft_delta),
                                        "lead_to_delta": float(lead_to_delta),
                                        "lead_three_delta": float(lead_three_delta),
                                        "margin_thresh": 3,
                                        "close_margin": 2,
                                        "base_mean_s": 18.0,
                                    }
                                )

    results = []
    best = None

    for idx, params in enumerate(grid):
        # Deterministic-ish per-row seed to keep comparisons stable
        seed = int(cfg.seed) + int(idx * 17)
        ev = _evaluate_params_against_targets(targets, params, int(cfg.sims_per_bucket), seed)
        row = {**params, "loss": float(ev["loss"])}
        results.append(row)

        if best is None or float(ev["loss"]) < float(best["loss"]):
            best = {
                "loss": float(ev["loss"]),
                "params": dict(params),
                "detail": ev,
            }

    df = pd.DataFrame(results).sort_values("loss", ascending=True).reset_index(drop=True)

    out_bt = Path(cfg.out_dir) / "backtests" / "sweep"
    out_bt.mkdir(parents=True, exist_ok=True)

    out_csv = out_bt / f"{cfg.out_prefix}.csv"
    out_json = out_bt / f"{cfg.out_prefix}.json"

    df.to_csv(out_csv, index=False)

    best_params = best["params"] if best else {}
    env_block = {
        "NCAAB_LATE_CLOSE_DT_MULT": best_params.get("close_dt_mult"),
        "NCAAB_LATE_TRAIL_DT_MULT": best_params.get("trail_dt_mult"),
        "NCAAB_LATE_LEAD_DT_MULT": best_params.get("lead_dt_mult"),
        "NCAAB_LATE_TRAIL_3PA_DELTA": best_params.get("trail_three_delta"),
        "NCAAB_LATE_LEAD_FT_DELTA": best_params.get("lead_ft_delta"),
        "NCAAB_LATE_LEAD_TO_DELTA": best_params.get("lead_to_delta"),
        "NCAAB_LATE_LEAD_3PA_DELTA": best_params.get("lead_three_delta"),
        "NCAAB_LATE_MARGIN_THRESH": best_params.get("margin_thresh"),
        "NCAAB_LATE_CLOSE_MARGIN": best_params.get("close_margin"),
    }

    summary = {
        "empirical_profile_json": str(cfg.empirical_profile_json),
        "grid_rows": int(len(df)),
        "sims_per_bucket": int(cfg.sims_per_bucket),
        "best": {
            "loss": float(best["loss"]) if best else None,
            "params": best_params,
            "by_bucket": (best["detail"].get("by_bucket") if best else None),
            "env": env_block,
        },
        "out_csv": str(out_csv),
        "out_json": str(out_json),
    }

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
