import json
import os

import numpy as np
import pandas as pd

from ncaab_model.data.cache import read_json

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

# Lightweight Monte Carlo simulator using baseline totals/margins.
# Assumes per-game means for total and margin and estimates per-team variance
# with a shared correlation parameter.

DEFAULT_RHO = 0.3  # positive correlation between team scores
DEFAULT_TOTAL_SIGMA = 14.0  # fallback spread of total points
DEFAULT_SAMPLES = 4000

# Event-driven simulation defaults (possession -> event -> points)
DEFAULT_EVENT_TO_RATE = 0.175
DEFAULT_EVENT_FT_TRIP_RATE = 0.115  # P(trip to line | non-turnover)
DEFAULT_EVENT_3PA_RATE = 0.36       # P(3PA | non-TO, non-FT)

# Reasonable college shooting baselines
BASE_FT_PCT = 0.72
BASE_2P_PCT = 0.50
BASE_3P_PCT = 0.34

# Pace/possessions modeling (used when tempo/pace inputs are present)
DEFAULT_PACE = 69.0
DEFAULT_PACE_SIGMA = 3.5
PACE_MIN = 55.0
PACE_MAX = 85.0

HALF_FRAC_DEFAULT = 0.5


def _hash_prob_vec_short(v: object) -> Optional[str]:
    """Return a short, stable hash for a segment probability vector."""
    try:
        if v is None:
            return None
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size <= 0:
            return None
        a = np.where(np.isfinite(a), a, 0.0)
        # Stable text serialization (avoid platform-dependent float repr).
        s = ",".join(f"{float(x):.10g}" for x in a.tolist())
        return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:12]
    except Exception:
        return None


def _prob_vec_len(v: object) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(np.asarray(v, dtype=float).reshape(-1).size)
    except Exception:
        return None


def _resolve_half_frac(row: pd.Series) -> float:
    # Prefer explicit half fraction, then derive from projections/predictions/markets.
    try:
        v = row.get("half_frac")
        if v is not None and pd.notna(v):
            vf = float(v)
            if 0 < vf < 1:
                return float(np.clip(vf, 0.35, 0.65))
    except Exception:
        pass

    # Projection-based ratio (most stable when available)
    try:
        ph = row.get("proj_home")
        pa = row.get("proj_away")
        ph1 = row.get("proj_home_1h")
        pa1 = row.get("proj_away_1h")
        if all(pd.notna(x) for x in [ph, pa, ph1, pa1]):
            den = float(ph) + float(pa)
            num = float(ph1) + float(pa1)
            if den > 1e-6 and 0 < num < den * 0.95:
                return float(np.clip(num / den, 0.35, 0.65))
    except Exception:
        pass

    # Model prediction-based ratio
    try:
        t = row.get("pred_total")
        t1 = row.get("pred_total_1h")
        if pd.notna(t) and pd.notna(t1):
            den = float(t)
            num = float(t1)
            if den > 1e-6 and 0 < num < den * 0.95:
                return float(np.clip(num / den, 0.35, 0.65))
    except Exception:
        pass

    # Market-based ratio
    try:
        mt = _resolve_market_total(row)
        mt1 = _resolve_market_total_1h(row)
        if mt is not None and mt1 is not None:
            den = float(mt)
            num = float(mt1)
            if den > 1e-6 and 0 < num < den * 0.95:
                return float(np.clip(num / den, 0.35, 0.65))
    except Exception:
        pass

    return float(HALF_FRAC_DEFAULT)


def _norm_team_key(v: object) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip().lower()
    return s if s else None


def _to_game_id_str(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return s if s else None


def _resolve_sim_engine(engine: str, mean_source_used: str) -> str:
    e = (engine or "auto").strip().lower() or "auto"
    if e in {"auto", "default"}:
        # When we're explicitly feature-driven, default to the ground-up engine.
        return "events" if mean_source_used == "features" else "normal"
    if e in {"events", "event", "pbp", "play_by_play", "play-by-play"}:
        return "events"
    return "normal"


def _safe_bool(v: object) -> bool:
    try:
        if v is None:
            return False
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        s = str(v).strip().lower()
        return s in {"1", "true", "yes", "y", "t"}
    except Exception:
        return False


def _safe_float(v: object) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if isinstance(v, (int, float, np.floating, np.integer)):
            return float(v)
        s = str(v).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _derive_event_rates(row: pd.Series, side: str) -> tuple[float, float, float]:
    """Return (to_rate, ft_trip_rate, three_rate) for a team-side."""
    to_rate = float(DEFAULT_EVENT_TO_RATE)
    ft_trip = float(DEFAULT_EVENT_FT_TRIP_RATE)
    three_rate = float(DEFAULT_EVENT_3PA_RATE)

    # Prefer learned rates from boxscore-derived rolling features when present.
    # These are merged in `run_simulations_for_date` as:
    #   <side>_team_event_to_rate, <side>_team_event_3p_rate, <side>_team_event_fta_rate
    try:
        lr_to = _safe_float(row.get(f"{side}_team_event_to_rate"))
        if lr_to is not None:
            to_rate = float(lr_to)
    except Exception:
        pass
    try:
        lr_3p = _safe_float(row.get(f"{side}_team_event_3p_rate"))
        if lr_3p is not None:
            three_rate = float(lr_3p)
    except Exception:
        pass
    try:
        lr_fta = _safe_float(row.get(f"{side}_team_event_fta_rate"))
        if lr_fta is not None and lr_fta > 0:
            # Approx trips/poss ≈ (FTA/2)/poss = 0.5*fta_rate; convert to conditional rate.
            trips_per_poss = 0.5 * float(lr_fta)
            denom = max(1e-6, 1.0 - float(to_rate))
            ft_trip = float(trips_per_poss / denom)
    except Exception:
        pass

    # Light feature hooks when available.
    # These columns exist in some pipelines; when absent, we stick to defaults.
    b2b = _safe_bool(row.get(f"{side}_team_back_to_back")) or _safe_bool(row.get(f"b2b_{side}"))
    rest = _safe_float(row.get(f"{side}_team_rest_days"))
    if rest is None:
        rest = _safe_float(row.get(f"rest_{side}"))

    if b2b:
        to_rate += 0.010
        ft_trip -= 0.005
    if rest is not None:
        if rest >= 5:
            to_rate -= 0.010
        elif rest <= 1:
            to_rate += 0.005

    # Keep within sensible bounds
    to_rate = float(np.clip(to_rate, 0.11, 0.25))
    ft_trip = float(np.clip(ft_trip, 0.06, 0.18))
    three_rate = float(np.clip(three_rate, 0.25, 0.50))
    return to_rate, ft_trip, three_rate


def _calibrate_shooting_to_ppp(
    ppp_target: float,
    to_rate: float,
    ft_trip: float,
    three_rate: float,
) -> tuple[float, float, float]:
    """Choose (ft_pct, p2, p3) so the implied expected PPP is close to ppp_target."""
    ppp_t = float(np.clip(float(ppp_target), 0.75, 1.35))
    ft_pct = float(np.clip(BASE_FT_PCT + (ppp_t - 1.0) * 0.06, 0.62, 0.82))
    p2 = float(BASE_2P_PCT)
    p3 = float(BASE_3P_PCT)

    # Expected points per possession under the simple event model.
    # Events are conditioned on not being a turnover.
    base_e = (1.0 - to_rate) * (
        ft_trip * (2.0 * ft_pct)
        + (1.0 - ft_trip)
        * (
            three_rate * (3.0 * p3)
            + (1.0 - three_rate) * (2.0 * p2)
        )
    )
    if base_e <= 1e-9:
        return ft_pct, p2, p3

    k = float(np.clip(ppp_t / base_e, 0.70, 1.35))
    # Scale makes modestly; keep FT less volatile.
    p2 = float(np.clip(p2 * k, 0.35, 0.72))
    p3 = float(np.clip(p3 * k, 0.25, 0.52))
    ft_pct = float(np.clip(ft_pct * (0.5 + 0.5 * k), 0.60, 0.86))
    return ft_pct, p2, p3


def _simulate_team_points(
    poss: int,
    to_rate: float,
    ft_trip: float,
    three_rate: float,
    ft_pct: float,
    p2: float,
    p3: float,
    rng: np.random.Generator,
) -> tuple[int, dict]:
    """Simulate one team's points over `poss` possessions and return (points, stats)."""
    if poss <= 0:
        return 0, {"poss": 0, "tov": 0, "fta": 0, "fga2": 0, "fga3": 0}

    u = rng.random(poss)
    is_to = u < to_rate

    u2 = rng.random(poss)
    is_ft = (~is_to) & (u2 < ft_trip)
    is_shot = (~is_to) & (~is_ft)

    u3 = rng.random(poss)
    is_3 = is_shot & (u3 < three_rate)
    is_2 = is_shot & (~is_3)

    # FT trips: 2 shots (simplified)
    n_ft = int(is_ft.sum())
    ft_made = rng.binomial(2, _clip01(ft_pct), size=n_ft).sum() if n_ft else 0

    n2 = int(is_2.sum())
    n3 = int(is_3.sum())
    two_made = int((rng.random(n2) < _clip01(p2)).sum()) if n2 else 0
    three_made = int((rng.random(n3) < _clip01(p3)).sum()) if n3 else 0

    pts = int(2 * two_made + 3 * three_made + ft_made)
    stats = {
        "poss": int(poss),
        "tov": int(is_to.sum()),
        "fta": int(2 * n_ft),
        "fga2": int(n2),
        "fga3": int(n3),
    }
    return pts, stats


def _simulate_events_samples(
    row: pd.Series,
    samples: int,
    pace_mu: float,
    pace_sigma: float,
    half_frac: float,
    mu_home: float,
    mu_away: float,
    rng: np.random.Generator,
) -> dict:
    """Ground-up simulation via possessions and shot/FT/TO events."""
    poss_mu = float(np.clip(float(pace_mu), PACE_MIN, PACE_MAX))
    poss_sigma = float(max(0.25, float(pace_sigma)))

    # Shared possessions per game creates realistic total correlation.
    poss_game = rng.normal(loc=poss_mu, scale=poss_sigma, size=int(samples))
    poss_game = np.clip(np.round(poss_game), PACE_MIN, PACE_MAX).astype(int)
    poss_1h = rng.binomial(poss_game, float(np.clip(half_frac, 0.35, 0.65)))
    poss_2h = poss_game - poss_1h

    ppp_home = float(np.clip(float(mu_home) / max(1.0, poss_mu), 0.75, 1.35))
    ppp_away = float(np.clip(float(mu_away) / max(1.0, poss_mu), 0.75, 1.35))

    to_h, ft_h, three_h = _derive_event_rates(row, "home")
    to_a, ft_a, three_a = _derive_event_rates(row, "away")
    ft_pct_h, p2_h, p3_h = _calibrate_shooting_to_ppp(ppp_home, to_h, ft_h, three_h)
    ft_pct_a, p2_a, p3_a = _calibrate_shooting_to_ppp(ppp_away, to_a, ft_a, three_a)

    home = np.zeros(int(samples), dtype=float)
    away = np.zeros(int(samples), dtype=float)
    home_1h = np.zeros(int(samples), dtype=float)
    away_1h = np.zeros(int(samples), dtype=float)

    # Per-sample boxscore-like aggregates (game + 1H). These are integer counts.
    home_tov = np.zeros(int(samples), dtype=int)
    away_tov = np.zeros(int(samples), dtype=int)
    home_fta = np.zeros(int(samples), dtype=int)
    away_fta = np.zeros(int(samples), dtype=int)
    home_fga2 = np.zeros(int(samples), dtype=int)
    away_fga2 = np.zeros(int(samples), dtype=int)
    home_fga3 = np.zeros(int(samples), dtype=int)
    away_fga3 = np.zeros(int(samples), dtype=int)
    home_poss = np.zeros(int(samples), dtype=int)
    away_poss = np.zeros(int(samples), dtype=int)

    home_tov_1h = np.zeros(int(samples), dtype=int)
    away_tov_1h = np.zeros(int(samples), dtype=int)
    home_fta_1h = np.zeros(int(samples), dtype=int)
    away_fta_1h = np.zeros(int(samples), dtype=int)
    home_fga2_1h = np.zeros(int(samples), dtype=int)
    away_fga2_1h = np.zeros(int(samples), dtype=int)
    home_fga3_1h = np.zeros(int(samples), dtype=int)
    away_fga3_1h = np.zeros(int(samples), dtype=int)
    home_poss_1h = np.zeros(int(samples), dtype=int)
    away_poss_1h = np.zeros(int(samples), dtype=int)

    # Light aggregation of event stats (means over samples)
    agg = {
        "poss_game_mean": float(np.mean(poss_game)),
        "home_to_rate": float(to_h),
        "away_to_rate": float(to_a),
        "home_ft_trip_rate": float(ft_h),
        "away_ft_trip_rate": float(ft_a),
        "home_3pa_rate": float(three_h),
        "away_3pa_rate": float(three_a),
        "home_ft_pct": float(ft_pct_h),
        "away_ft_pct": float(ft_pct_a),
        "home_p2": float(p2_h),
        "away_p2": float(p2_a),
        "home_p3": float(p3_h),
        "away_p3": float(p3_a),
    }

    for i in range(int(samples)):
        h1, hs1 = _simulate_team_points(int(poss_1h[i]), to_h, ft_h, three_h, ft_pct_h, p2_h, p3_h, rng)
        a1, as1 = _simulate_team_points(int(poss_1h[i]), to_a, ft_a, three_a, ft_pct_a, p2_a, p3_a, rng)
        h2, hs2 = _simulate_team_points(int(poss_2h[i]), to_h, ft_h, three_h, ft_pct_h, p2_h, p3_h, rng)
        a2, as2 = _simulate_team_points(int(poss_2h[i]), to_a, ft_a, three_a, ft_pct_a, p2_a, p3_a, rng)
        home_1h[i] = float(h1)
        away_1h[i] = float(a1)
        home[i] = float(h1 + h2)
        away[i] = float(a1 + a2)

        # Store aggregates (game and 1H)
        home_tov_1h[i] = int(hs1.get("tov", 0))
        away_tov_1h[i] = int(as1.get("tov", 0))
        home_fta_1h[i] = int(hs1.get("fta", 0))
        away_fta_1h[i] = int(as1.get("fta", 0))
        home_fga2_1h[i] = int(hs1.get("fga2", 0))
        away_fga2_1h[i] = int(as1.get("fga2", 0))
        home_fga3_1h[i] = int(hs1.get("fga3", 0))
        away_fga3_1h[i] = int(as1.get("fga3", 0))
        home_poss_1h[i] = int(hs1.get("poss", 0))
        away_poss_1h[i] = int(as1.get("poss", 0))

        home_tov[i] = int(hs1.get("tov", 0)) + int(hs2.get("tov", 0))
        away_tov[i] = int(as1.get("tov", 0)) + int(as2.get("tov", 0))
        home_fta[i] = int(hs1.get("fta", 0)) + int(hs2.get("fta", 0))
        away_fta[i] = int(as1.get("fta", 0)) + int(as2.get("fta", 0))
        home_fga2[i] = int(hs1.get("fga2", 0)) + int(hs2.get("fga2", 0))
        away_fga2[i] = int(as1.get("fga2", 0)) + int(as2.get("fga2", 0))
        home_fga3[i] = int(hs1.get("fga3", 0)) + int(hs2.get("fga3", 0))
        away_fga3[i] = int(as1.get("fga3", 0)) + int(as2.get("fga3", 0))
        home_poss[i] = int(hs1.get("poss", 0)) + int(hs2.get("poss", 0))
        away_poss[i] = int(as1.get("poss", 0)) + int(as2.get("poss", 0))

    return {
        "home": home,
        "away": away,
        "home_1h": home_1h,
        "away_1h": away_1h,
        "home_tov": home_tov,
        "away_tov": away_tov,
        "home_fta": home_fta,
        "away_fta": away_fta,
        "home_fga2": home_fga2,
        "away_fga2": away_fga2,
        "home_fga3": home_fga3,
        "away_fga3": away_fga3,
        "home_poss": home_poss,
        "away_poss": away_poss,
        "home_tov_1h": home_tov_1h,
        "away_tov_1h": away_tov_1h,
        "home_fta_1h": home_fta_1h,
        "away_fta_1h": away_fta_1h,
        "home_fga2_1h": home_fga2_1h,
        "away_fga2_1h": away_fga2_1h,
        "home_fga3_1h": home_fga3_1h,
        "away_fga3_1h": away_fga3_1h,
        "home_poss_1h": home_poss_1h,
        "away_poss_1h": away_poss_1h,
        "agg": agg,
    }


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
        # 2-shot trip (simplified)
        return int(rng.binomial(2, _clip01(ft_pct)))
    u3 = float(rng.random())
    if u3 < float(three_rate):
        return 3 if float(rng.random()) < _clip01(p3) else 0
    return 2 if float(rng.random()) < _clip01(p2) else 0


def _segment_5min_quantiles_from_events_timeline(
    poss_1h: np.ndarray,
    poss_2h: np.ndarray,
    home_params: tuple[float, float, float, float, float, float],
    away_params: tuple[float, float, float, float, float, float],
    rng: np.random.Generator,
    enable_late_foul: bool = True,
) -> list[dict]:
    return _segment_quantiles_from_events_timeline(
        poss_1h=poss_1h,
        poss_2h=poss_2h,
        home_params=home_params,
        away_params=away_params,
        rng=rng,
        end_mins=[5, 10, 15, 20, 25, 30, 35, 40],
        enable_late_foul=enable_late_foul,
    )


def _segment_2min_quantiles_from_events_timeline(
    poss_1h: np.ndarray,
    poss_2h: np.ndarray,
    home_params: tuple[float, float, float, float, float, float],
    away_params: tuple[float, float, float, float, float, float],
    rng: np.random.Generator,
    enable_late_foul: bool = True,
) -> list[dict]:
    return _segment_quantiles_from_events_timeline(
        poss_1h=poss_1h,
        poss_2h=poss_2h,
        home_params=home_params,
        away_params=away_params,
        rng=rng,
        end_mins=list(range(2, 41, 2)),
        enable_late_foul=enable_late_foul,
    )


def _segments_grid_min_from_env() -> int:
    try:
        v = (os.environ.get("NCAAB_SEGMENTS_GRID_MIN") or "").strip()
        if not v:
            return 5
        g = int(float(v))
        if g <= 2:
            return 2
        return 5
    except Exception:
        return 5


def _segment_quantiles_from_events_timeline(
    poss_1h: np.ndarray,
    poss_2h: np.ndarray,
    home_params: tuple[float, float, float, float, float, float],
    away_params: tuple[float, float, float, float, float, float],
    rng: np.random.Generator,
    end_mins: list[int],
    enable_late_foul: bool = True,
) -> list[dict]:
    """Derive cumulative endpoints by simulating a possession timeline.

    Default behavior remains the 5-minute grid. A 2-minute grid is supported
    (2,4,...,40) and is used when end_mins is provided as such.
    """

    # Endpoints in minutes (regulation)
    end_mins = [int(x) for x in end_mins if x is not None]
    end_mins = [x for x in end_mins if 0 < int(x) <= 40]
    end_mins = sorted(set(end_mins))
    if not end_mins:
        end_mins = [5, 10, 15, 20, 25, 30, 35, 40]
    n1 = sum(1 for m in end_mins if int(m) <= 20)
    if n1 <= 0 or n1 >= len(end_mins):
        # Defensive fallback: assume standard halves split.
        n1 = int(len(end_mins) // 2)

    n = int(len(poss_1h))
    home_cum = np.zeros((n, len(end_mins)), dtype=float)
    away_cum = np.zeros((n, len(end_mins)), dtype=float)

    to_h, ft_h, three_h, ft_pct_h, p2_h, p3_h = home_params
    to_a, ft_a, three_a, ft_pct_a, p2_a, p3_a = away_params

    # Late-game heuristic parameters (tunable via env for sweeps)
    late_foul_time_s = float(np.clip(_safe_float(os.getenv("NCAAB_LATE_FOUL_TIME_SEC")) or 120.0, 30.0, 240.0))
    # Defaults are tuned against empirical last-2-minute scoring profile; can be overridden via env.
    close_dt_mult = float(np.clip(_safe_float(os.getenv("NCAAB_LATE_CLOSE_DT_MULT")) or 0.88, 0.60, 1.10))
    trail_dt_mult = float(np.clip(_safe_float(os.getenv("NCAAB_LATE_TRAIL_DT_MULT")) or 0.80, 0.50, 1.10))
    lead_dt_mult = float(np.clip(_safe_float(os.getenv("NCAAB_LATE_LEAD_DT_MULT")) or 0.65, 0.50, 1.10))
    trail_three_delta = float(np.clip(_safe_float(os.getenv("NCAAB_LATE_TRAIL_3PA_DELTA")) or 0.06, 0.00, 0.20))
    lead_ft_delta = float(np.clip(_safe_float(os.getenv("NCAAB_LATE_LEAD_FT_DELTA")) or 0.10, 0.00, 0.30))
    lead_to_delta = float(np.clip(_safe_float(os.getenv("NCAAB_LATE_LEAD_TO_DELTA")) or -0.02, -0.10, 0.10))
    lead_three_delta = float(np.clip(_safe_float(os.getenv("NCAAB_LATE_LEAD_3PA_DELTA")) or -0.05, -0.20, 0.20))

    margin_thresh = int(_safe_float(os.getenv("NCAAB_LATE_MARGIN_THRESH")) or 3)
    close_margin = int(_safe_float(os.getenv("NCAAB_LATE_CLOSE_MARGIN")) or 2)
    late_foul_max_abs_margin = int(np.clip(_safe_float(os.getenv("NCAAB_LATE_FOUL_MAX_ABS_MARGIN")) or 10, 2, 40))
    late_foul_max_abs_margin_early = int(
        np.clip(_safe_float(os.getenv("NCAAB_LATE_FOUL_MAX_ABS_MARGIN_EARLY")) or 6, 1, late_foul_max_abs_margin)
    )
    late_foul_ramp_end_s = float(np.clip(_safe_float(os.getenv("NCAAB_LATE_FOUL_RAMP_END_SEC")) or 60.0, 5.0, late_foul_time_s))

    for i in range(n):
        # Possessions per team per half
        p1 = int(max(0, int(poss_1h[i])))
        p2 = int(max(0, int(poss_2h[i])))

        # Combined possessions in a half are roughly 2x per-team possessions.
        npos_1h = int(2 * p1)
        npos_2h = int(2 * p2)

        # Simulate 1H timeline
        t = 0.0
        home = 0
        away = 0
        # mean seconds per combined possession
        mean_s_1h = 1200.0 / float(max(1, npos_1h))
        shape = 2.0
        scale_1h = float(max(1.0, mean_s_1h / shape))
        next_ep_idx = 0
        # Alternate possessions starting randomly
        home_ball = bool(rng.random() < 0.5)
        for _ in range(npos_1h):
            # Possession duration
            dt = float(rng.gamma(shape, scale_1h))
            dt = float(np.clip(dt, 4.0, 40.0))
            t += dt

            if home_ball:
                home += _simulate_single_possession_points(to_h, ft_h, three_h, ft_pct_h, p2_h, p3_h, rng)
            else:
                away += _simulate_single_possession_points(to_a, ft_a, three_a, ft_pct_a, p2_a, p3_a, rng)
            home_ball = not home_ball

            # Record any crossed endpoints within 1H
            while next_ep_idx < n1 and t >= end_mins[next_ep_idx] * 60.0:
                home_cum[i, next_ep_idx] = float(home)
                away_cum[i, next_ep_idx] = float(away)
                next_ep_idx += 1
            if t >= 1200.0:
                break

        # Fill any remaining 1H endpoints with final 1H totals
        while next_ep_idx < n1:
            home_cum[i, next_ep_idx] = float(home)
            away_cum[i, next_ep_idx] = float(away)
            next_ep_idx += 1

        # Simulate 2H timeline, continuing the cumulative totals
        t = 0.0
        mean_s_2h = 1200.0 / float(max(1, npos_2h))
        scale_2h = float(max(1.0, mean_s_2h / shape))
        home_ball = bool(rng.random() < 0.5)
        next_ep_idx = n1

        for _ in range(npos_2h):
            time_remaining = 1200.0 - t
            margin = home - away

            # Simple late-foul / clock effects in last 2 minutes of 2H.
            adj_home = (to_h, ft_h, three_h)
            adj_away = (to_a, ft_a, three_a)
            dt_mult = 1.0
            # Late-foul / clock effects: only plausible in close-ish games, and the maximum
            # margin where fouling is active should widen as the clock gets lower.
            if enable_late_foul and time_remaining <= late_foul_time_s:
                # Make adjustments conditional on BOTH margin and who has the ball.
                # Heuristic intent:
                # - Trailing offense plays faster + shoots more 3s.
                # - Trailing defense fouls more, which increases FT-trip for the LEADING offense.
                # - Leading offense tends to burn clock.

                # Ramp the max margin from an early (more conservative) value at ~2:00 remaining
                # to the full value by ~1:00 remaining.
                if late_foul_time_s > late_foul_ramp_end_s:
                    ramp = float((late_foul_time_s - time_remaining) / (late_foul_time_s - late_foul_ramp_end_s))
                else:
                    ramp = 1.0
                ramp = float(np.clip(ramp, 0.0, 1.0))
                eff_max_margin = int(round(late_foul_max_abs_margin_early + ramp * (late_foul_max_abs_margin - late_foul_max_abs_margin_early)))
                if abs(margin) > eff_max_margin:
                    pass
                else:

                    # Close games: modestly faster possessions.
                    if abs(margin) <= close_margin:
                        dt_mult = close_dt_mult

                    # Home has ball
                    if home_ball:
                        if margin <= -margin_thresh:
                            # Home trailing on offense: faster + more 3s.
                            adj_home = (
                                to_h,
                                ft_h,
                                min(0.48, three_h + trail_three_delta),
                            )
                            dt_mult = min(dt_mult, trail_dt_mult)
                        elif margin >= margin_thresh:
                            # Home leading on offense: away may foul -> more FTs for home; clock stoppages.
                            adj_home = (
                                float(np.clip(to_h + lead_to_delta, 0.11, 0.25)),
                                float(np.clip(ft_h + lead_ft_delta, 0.06, 0.18)),
                                float(np.clip(three_h + lead_three_delta, 0.25, 0.50)),
                            )
                            dt_mult = min(dt_mult, lead_dt_mult)

                    # Away has ball
                    else:
                        if margin >= margin_thresh:
                            # Away trailing on offense: faster + more 3s.
                            adj_away = (
                                to_a,
                                ft_a,
                                min(0.48, three_a + trail_three_delta),
                            )
                            dt_mult = min(dt_mult, trail_dt_mult)
                        elif margin <= -margin_thresh:
                            # Away leading on offense: home may foul -> more FTs for away; clock stoppages.
                            adj_away = (
                                float(np.clip(to_a + lead_to_delta, 0.11, 0.25)),
                                float(np.clip(ft_a + lead_ft_delta, 0.06, 0.18)),
                                float(np.clip(three_a + lead_three_delta, 0.25, 0.50)),
                            )
                            dt_mult = min(dt_mult, lead_dt_mult)

            dt = float(rng.gamma(shape, scale_2h))
            dt = float(np.clip(dt * dt_mult, 4.0, 40.0))
            t += dt

            if home_ball:
                th, fh, trh = adj_home
                home += _simulate_single_possession_points(th, fh, trh, ft_pct_h, p2_h, p3_h, rng)
            else:
                ta, fa, tra = adj_away
                away += _simulate_single_possession_points(ta, fa, tra, ft_pct_a, p2_a, p3_a, rng)
            home_ball = not home_ball

            # Record any crossed endpoints within game
            while next_ep_idx < len(end_mins) and t >= (end_mins[next_ep_idx] - 20) * 60.0:
                home_cum[i, next_ep_idx] = float(home)
                away_cum[i, next_ep_idx] = float(away)
                next_ep_idx += 1
            if t >= 1200.0:
                break

        # Fill any remaining endpoints with final game totals
        while next_ep_idx < len(end_mins):
            home_cum[i, next_ep_idx] = float(home)
            away_cum[i, next_ep_idx] = float(away)
            next_ep_idx += 1

    total_cum = home_cum + away_cum
    margin_cum = home_cum - away_cum

    def _q(v: np.ndarray) -> tuple[float, float, float]:
        return (
            float(np.quantile(v, 0.10)),
            float(np.quantile(v, 0.50)),
            float(np.quantile(v, 0.90)),
        )

    grid_min = int(min(np.diff(np.array(end_mins, dtype=int))) if len(end_mins) > 1 else 5)
    grid_min = int(2 if grid_min <= 2 else 5)

    rows: list[dict] = []
    for j, end_min in enumerate(end_mins):
        h_end = home_cum[:, j].astype(float)
        a_end = away_cum[:, j].astype(float)
        t_end = total_cum[:, j].astype(float)
        m_end = margin_cum[:, j].astype(float)
        q_h_end = _q(h_end)
        q_a_end = _q(a_end)
        q_t_end = _q(t_end)
        q_m_end = _q(m_end)
        rows.append(
            {
                "start_min": int(max(0, int(end_min) - int(grid_min))),
                "end_min": int(end_min),
                "half": 1 if int(end_min) <= 20 else 2,
                "mu_home_score_end": float(np.mean(h_end)),
                "mu_away_score_end": float(np.mean(a_end)),
                "mu_total_score_end": float(np.mean(t_end)),
                "mu_margin_score_end": float(np.mean(m_end)),
                "q10_home_score_end": q_h_end[0],
                "q50_home_score_end": q_h_end[1],
                "q90_home_score_end": q_h_end[2],
                "q10_away_score_end": q_a_end[0],
                "q50_away_score_end": q_a_end[1],
                "q90_away_score_end": q_a_end[2],
                "q10_total_score_end": q_t_end[0],
                "q50_total_score_end": q_t_end[1],
                "q90_total_score_end": q_t_end[2],
                "q10_margin_score_end": q_m_end[0],
                "q50_margin_score_end": q_m_end[1],
                "q90_margin_score_end": q_m_end[2],
            }
        )

    return rows


def _segment_5min_quantiles_from_points(
    home_pts: np.ndarray,
    away_pts: np.ndarray,
    home_1h: np.ndarray,
    away_1h: np.ndarray,
    rng: np.random.Generator,
    seg_probs_half1: Optional[np.ndarray] = None,
    seg_probs_half2: Optional[np.ndarray] = None,
    home_stats: Optional[dict[str, np.ndarray]] = None,
    away_stats: Optional[dict[str, np.ndarray]] = None,
) -> list[dict]:
    """Derive 5-minute segment scoring distributions that roll up to 1H/full.

    This is intentionally light-weight: we take the already-simulated team points
    for 1H and full game and split them into 4x 5-min segments per half using a
    multinomial allocation. This guarantees that:
      - segment sums equal 1H and full-game points per sample
      - segment cumulative at 20 min matches 1H distributions
    """

    def _to_nonneg_int(x: float) -> int:
        try:
            if not np.isfinite(x):
                return 0
            return int(max(0, int(np.rint(float(x)))))
        except Exception:
            return 0

    home_pts = np.asarray(home_pts, dtype=float)
    away_pts = np.asarray(away_pts, dtype=float)
    home_1h = np.asarray(home_1h, dtype=float)
    away_1h = np.asarray(away_1h, dtype=float)
    n = int(len(home_pts))
    if n <= 0 or int(len(away_pts)) != n or int(len(home_1h)) != n or int(len(away_1h)) != n:
        return []

    def _norm_probs(v: Optional[np.ndarray]) -> np.ndarray:
        if v is None:
            return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        try:
            a = np.asarray(v, dtype=float).reshape(-1)
        except Exception:
            return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        if a.size != 4:
            return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        a = np.where(np.isfinite(a), a, 0.0)
        a = np.clip(a, 0.0, None)
        s = float(a.sum())
        if s <= 0:
            return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        return (a / s).astype(float)

    # 40-minute game in 8 segments of 5 minutes each (4 per half)
    probs_1h = _norm_probs(seg_probs_half1)
    probs_2h = _norm_probs(seg_probs_half2)

    home_seg_pts = np.zeros((n, 8), dtype=int)
    away_seg_pts = np.zeros((n, 8), dtype=int)

    metrics = ["poss", "tov", "fta", "fga2", "fga3"]
    home_seg_metrics: dict[str, np.ndarray] = {}
    away_seg_metrics: dict[str, np.ndarray] = {}
    if home_stats is not None and away_stats is not None:
        for m in metrics:
            if (m in home_stats and f"{m}_1h" in home_stats) and (m in away_stats and f"{m}_1h" in away_stats):
                home_seg_metrics[m] = np.zeros((n, 8), dtype=int)
                away_seg_metrics[m] = np.zeros((n, 8), dtype=int)

    for i in range(n):
        h1_pts = _to_nonneg_int(home_1h[i])
        a1_pts = _to_nonneg_int(away_1h[i])
        h_full_pts = _to_nonneg_int(home_pts[i])
        a_full_pts = _to_nonneg_int(away_pts[i])

        # Robustness: if caller provides incoherent 1H/full samples (possible when
        # 1H and full-game are simulated independently), clamp 1H to full so that
        # segment sums match full-game totals and 2H residuals are non-negative.
        h1_pts = min(h1_pts, h_full_pts)
        a1_pts = min(a1_pts, a_full_pts)

        h2_pts = max(0, h_full_pts - h1_pts)
        a2_pts = max(0, a_full_pts - a1_pts)

        home_seg_pts[i, 0:4] = rng.multinomial(h1_pts, probs_1h)
        away_seg_pts[i, 0:4] = rng.multinomial(a1_pts, probs_1h)
        home_seg_pts[i, 4:8] = rng.multinomial(h2_pts, probs_2h)
        away_seg_pts[i, 4:8] = rng.multinomial(a2_pts, probs_2h)

        if home_seg_metrics:
            for m, hseg in home_seg_metrics.items():
                full_h = _to_nonneg_int(float(home_stats.get(m)[i]))
                one_h = _to_nonneg_int(float(home_stats.get(f"{m}_1h")[i]))
                full_a = _to_nonneg_int(float(away_stats.get(m)[i]))
                one_a = _to_nonneg_int(float(away_stats.get(f"{m}_1h")[i]))

                one_h = min(one_h, full_h)
                one_a = min(one_a, full_a)

                two_h = max(0, full_h - one_h)
                two_a = max(0, full_a - one_a)

                hseg[i, 0:4] = rng.multinomial(one_h, probs_1h)
                away_seg_metrics[m][i, 0:4] = rng.multinomial(one_a, probs_1h)
                hseg[i, 4:8] = rng.multinomial(two_h, probs_2h)
                away_seg_metrics[m][i, 4:8] = rng.multinomial(two_a, probs_2h)

    home_cum = np.cumsum(home_seg_pts, axis=1)
    away_cum = np.cumsum(away_seg_pts, axis=1)
    total_cum = home_cum + away_cum
    margin_cum = home_cum - away_cum

    rows: list[dict] = []
    for seg_idx in range(8):
        start_min = int(seg_idx * 5)
        end_min = int((seg_idx + 1) * 5)
        half = 1 if seg_idx < 4 else 2

        h_seg = home_seg_pts[:, seg_idx].astype(float)
        a_seg = away_seg_pts[:, seg_idx].astype(float)
        t_seg = h_seg + a_seg
        m_seg = h_seg - a_seg

        h_end = home_cum[:, seg_idx].astype(float)
        a_end = away_cum[:, seg_idx].astype(float)
        t_end = total_cum[:, seg_idx].astype(float)
        m_end = margin_cum[:, seg_idx].astype(float)

        def _q(arr: np.ndarray) -> tuple[float, float, float]:
            qq = np.quantile(arr, [0.10, 0.50, 0.90])
            return float(qq[0]), float(qq[1]), float(qq[2])

        q_h_seg = _q(h_seg)
        q_a_seg = _q(a_seg)
        q_t_seg = _q(t_seg)
        q_m_seg = _q(m_seg)
        q_h_end = _q(h_end)
        q_a_end = _q(a_end)
        q_t_end = _q(t_end)
        q_m_end = _q(m_end)

        extra: dict[str, float] = {}
        if home_seg_metrics:
            for m in metrics:
                if m not in home_seg_metrics:
                    continue
                h_m = home_seg_metrics[m][:, seg_idx].astype(float)
                a_m = away_seg_metrics[m][:, seg_idx].astype(float)
                t_m = h_m + a_m
                q_t_m = _q(t_m)
                extra.update(
                    {
                        f"mu_home_{m}_seg": float(np.mean(h_m)),
                        f"mu_away_{m}_seg": float(np.mean(a_m)),
                        f"mu_total_{m}_seg": float(np.mean(t_m)),
                        f"q10_total_{m}_seg": float(q_t_m[0]),
                        f"q50_total_{m}_seg": float(q_t_m[1]),
                        f"q90_total_{m}_seg": float(q_t_m[2]),
                    }
                )

        rows.append(
            {
                "segment": int(seg_idx + 1),
                "half": int(half),
                "start_min": int(start_min),
                "end_min": int(end_min),
                # Segment points
                "mu_home_pts_seg": float(np.mean(h_seg)),
                "mu_away_pts_seg": float(np.mean(a_seg)),
                "mu_total_pts_seg": float(np.mean(t_seg)),
                "mu_margin_pts_seg": float(np.mean(m_seg)),
                "q10_home_pts_seg": q_h_seg[0],
                "q50_home_pts_seg": q_h_seg[1],
                "q90_home_pts_seg": q_h_seg[2],
                "q10_away_pts_seg": q_a_seg[0],
                "q50_away_pts_seg": q_a_seg[1],
                "q90_away_pts_seg": q_a_seg[2],
                "q10_total_pts_seg": q_t_seg[0],
                "q50_total_pts_seg": q_t_seg[1],
                "q90_total_pts_seg": q_t_seg[2],
                "q10_margin_pts_seg": q_m_seg[0],
                "q50_margin_pts_seg": q_m_seg[1],
                "q90_margin_pts_seg": q_m_seg[2],
                # Cumulative score through segment end
                "mu_home_score_end": float(np.mean(h_end)),
                "mu_away_score_end": float(np.mean(a_end)),
                "mu_total_score_end": float(np.mean(t_end)),
                "mu_margin_score_end": float(np.mean(m_end)),
                "q10_home_score_end": q_h_end[0],
                "q50_home_score_end": q_h_end[1],
                "q90_home_score_end": q_h_end[2],
                "q10_away_score_end": q_a_end[0],
                "q50_away_score_end": q_a_end[1],
                "q90_away_score_end": q_a_end[2],
                "q10_total_score_end": q_t_end[0],
                "q50_total_score_end": q_t_end[1],
                "q90_total_score_end": q_t_end[2],
                "q10_margin_score_end": q_m_end[0],
                "q50_margin_score_end": q_m_end[1],
                "q90_margin_score_end": q_m_end[2],
                **extra,
            }
        )

    return rows


def _segment_grid_quantiles_from_points(
    home_pts: np.ndarray,
    away_pts: np.ndarray,
    home_1h: np.ndarray,
    away_1h: np.ndarray,
    rng: np.random.Generator,
    grid_min: int,
    abs_margin_proxy: Optional[float] = None,
    seg_probs_half1: Optional[np.ndarray] = None,
    seg_probs_half2: Optional[np.ndarray] = None,
    home_stats: Optional[dict[str, np.ndarray]] = None,
    away_stats: Optional[dict[str, np.ndarray]] = None,
) -> list[dict]:
    """Derive segment scoring distributions on an arbitrary time grid from point samples.

    - grid_min=5 => identical to the existing 5-min output.
    - grid_min=2 => emits 20 segments (10 per half) with endpoints 2..40 by 2.

    This preserves the key invariants of the original implementation:
      - segment sums equal 1H and full-game points per sample
      - cumulative at 20 min matches 1H distributions
    """

    try:
        gm = int(grid_min)
    except Exception:
        gm = 5
    if gm == 5:
        return _segment_5min_quantiles_from_points(
            home_pts=home_pts,
            away_pts=away_pts,
            home_1h=home_1h,
            away_1h=away_1h,
            seg_probs_half1=seg_probs_half1,
            seg_probs_half2=seg_probs_half2,
            home_stats=home_stats,
            away_stats=away_stats,
            rng=rng,
        )
    if gm != 2:
        # Only 2 or 5 are supported for now.
        return _segment_5min_quantiles_from_points(
            home_pts=home_pts,
            away_pts=away_pts,
            home_1h=home_1h,
            away_1h=away_1h,
            seg_probs_half1=seg_probs_half1,
            seg_probs_half2=seg_probs_half2,
            home_stats=home_stats,
            away_stats=away_stats,
            rng=rng,
        )


    # Optional late-game shaping for 2-min point-allocation segments.
    # This is a lightweight way to make the final 2-minute segment share depend on
    # the (simulated) game closeness, without switching to time-aware segments.
    #
    # Disabled by default to preserve existing behavior.
    try:
        enable_late_alloc_shape = _safe_bool(os.environ.get("NCAAB_LATE_ALLOC_SHAPE"))
    except Exception:
        enable_late_alloc_shape = False
    try:
        late_alloc_close_max = int(_safe_float(os.environ.get("NCAAB_LATE_ALLOC_CLOSE_MAX")) or 6)
    except Exception:
        late_alloc_close_max = 6
    try:
        late_alloc_blowout_min = int(_safe_float(os.environ.get("NCAAB_LATE_ALLOC_BLOWOUT_MIN")) or 14)
    except Exception:
        late_alloc_blowout_min = 14
    try:
        late_alloc_last_mult_close = float(_safe_float(os.environ.get("NCAAB_LATE_ALLOC_LAST_MULT_CLOSE")) or 1.12)
    except Exception:
        late_alloc_last_mult_close = 1.12
    try:
        late_alloc_last_mult_blowout = float(_safe_float(os.environ.get("NCAAB_LATE_ALLOC_LAST_MULT_BLOWOUT")) or 0.88)
    except Exception:
        late_alloc_last_mult_blowout = 0.88

    late_alloc_close_max = int(max(0, late_alloc_close_max))
    late_alloc_blowout_min = int(max(late_alloc_close_max + 1, late_alloc_blowout_min))
    late_alloc_last_mult_close = float(np.clip(late_alloc_last_mult_close, 0.50, 2.00))
    late_alloc_last_mult_blowout = float(np.clip(late_alloc_last_mult_blowout, 0.25, 1.50))

    def _shape_2h_probs_for_margin(abs_margin: int) -> np.ndarray:
        if not enable_late_alloc_shape:
            return probs_2h_2
        try:
            am = int(max(0, abs_margin))
        except Exception:
            am = 0
        if am <= late_alloc_close_max:
            mult = late_alloc_last_mult_close
        elif am >= late_alloc_blowout_min:
            mult = late_alloc_last_mult_blowout
        else:
            # Linear interpolation between close and blowout multipliers.
            denom = float(max(1, (late_alloc_blowout_min - late_alloc_close_max)))
            t = float((am - late_alloc_close_max) / denom)
            t = float(np.clip(t, 0.0, 1.0))
            mult = float((1.0 - t) * late_alloc_last_mult_close + t * late_alloc_last_mult_blowout)

        p = np.asarray(probs_2h_2, dtype=float).copy()
        p[-1] = float(max(0.0, p[-1] * mult))
        s = float(p.sum())
        if s <= 0:
            return probs_2h_2
        return (p / s).astype(float)

    def _to_nonneg_int(x: float) -> int:
        try:
            if not np.isfinite(x):
                return 0
            return int(max(0, int(np.rint(float(x)))))
        except Exception:
            return 0

    home_pts = np.asarray(home_pts, dtype=float)
    away_pts = np.asarray(away_pts, dtype=float)
    home_1h = np.asarray(home_1h, dtype=float)
    away_1h = np.asarray(away_1h, dtype=float)
    n = int(len(home_pts))
    if n <= 0 or int(len(away_pts)) != n or int(len(home_1h)) != n or int(len(away_1h)) != n:
        return []

    def _norm_probs_4(v: Optional[np.ndarray]) -> np.ndarray:
        if v is None:
            return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        try:
            a = np.asarray(v, dtype=float).reshape(-1)
        except Exception:
            return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        if a.size != 4:
            return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        a = np.where(np.isfinite(a), a, 0.0)
        a = np.clip(a, 0.0, None)
        s = float(a.sum())
        if s <= 0:
            return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        return (a / s).astype(float)

    def _norm_probs_10(v: Optional[np.ndarray]) -> np.ndarray:
        if v is None:
            return np.array([0.1] * 10, dtype=float)
        try:
            a = np.asarray(v, dtype=float).reshape(-1)
        except Exception:
            return np.array([0.1] * 10, dtype=float)
        if a.size != 10:
            return np.array([0.1] * 10, dtype=float)
        a = np.where(np.isfinite(a), a, 0.0)
        a = np.clip(a, 0.0, None)
        s = float(a.sum())
        if s <= 0:
            return np.array([0.1] * 10, dtype=float)
        return (a / s).astype(float)

    def _half_probs_2min(probs_5min_blocks: np.ndarray) -> np.ndarray:
        """Convert 4x5-minute block weights into 10x2-minute segment probabilities.

        We build per-minute weights (20 minutes per half), spreading each 5-minute block
        uniformly across its 5 constituent minutes. Each 2-minute segment then sums the
        weights of its two minutes.
        """
        p = np.asarray(probs_5min_blocks, dtype=float).reshape(-1)
        if p.size != 4:
            p = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        p = np.where(np.isfinite(p), p, 0.0)
        p = np.clip(p, 0.0, None)
        s = float(p.sum())
        if s <= 0:
            p = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
            s = float(p.sum())
        p = (p / s).astype(float)

        minute_w = np.zeros(20, dtype=float)
        for j in range(4):
            start = int(j * 5)
            end = int((j + 1) * 5)
            minute_w[start:end] = float(p[j]) / 5.0

        seg_w = np.zeros(10, dtype=float)
        for k in range(10):
            m0 = int(k * 2)
            seg_w[k] = float(minute_w[m0] + minute_w[m0 + 1])

        ss = float(seg_w.sum())
        if ss <= 0:
            return np.array([0.1] * 10, dtype=float)
        return (seg_w / ss).astype(float)

    # Allow callers to pass either 4x5-min block weights (legacy) or 10x2-min weights.
    use_direct_10 = False
    try:
        if seg_probs_half1 is not None and np.asarray(seg_probs_half1, dtype=float).reshape(-1).size == 10:
            use_direct_10 = True
    except Exception:
        use_direct_10 = False

    if use_direct_10:
        probs_1h_2 = _norm_probs_10(seg_probs_half1)
        probs_2h_2 = _norm_probs_10(seg_probs_half2)
    else:
        probs_1h_5 = _norm_probs_4(seg_probs_half1)
        probs_2h_5 = _norm_probs_4(seg_probs_half2)
        probs_1h_2 = _half_probs_2min(probs_1h_5)
        probs_2h_2 = _half_probs_2min(probs_2h_5)

    # IMPORTANT: shape based on a per-game proxy (spread / expected margin), not per-sample
    # realized margins. Per-sample shaping can suppress close games because MC tails include
    # blowout outcomes even when the median game is close.
    abs_margin_i = None
    try:
        if abs_margin_proxy is not None and np.isfinite(float(abs_margin_proxy)):
            abs_margin_i = int(max(0, int(np.rint(float(abs_margin_proxy)))))
    except Exception:
        abs_margin_i = None
    probs_2h_i = _shape_2h_probs_for_margin(abs_margin_i if abs_margin_i is not None else 0)

    # 40-minute game in 20 segments of 2 minutes each (10 per half)
    home_seg_pts = np.zeros((n, 20), dtype=int)
    away_seg_pts = np.zeros((n, 20), dtype=int)

    metrics = ["poss", "tov", "fta", "fga2", "fga3"]
    home_seg_metrics: dict[str, np.ndarray] = {}
    away_seg_metrics: dict[str, np.ndarray] = {}
    if home_stats is not None and away_stats is not None:
        for m in metrics:
            if (m in home_stats and f"{m}_1h" in home_stats) and (m in away_stats and f"{m}_1h" in away_stats):
                home_seg_metrics[m] = np.zeros((n, 20), dtype=int)
                away_seg_metrics[m] = np.zeros((n, 20), dtype=int)

    for i in range(n):
        h1_pts = _to_nonneg_int(home_1h[i])
        a1_pts = _to_nonneg_int(away_1h[i])
        h_full_pts = _to_nonneg_int(home_pts[i])
        a_full_pts = _to_nonneg_int(away_pts[i])

        h2_pts = max(0, h_full_pts - h1_pts)
        a2_pts = max(0, a_full_pts - a1_pts)

        home_seg_pts[i, 0:10] = rng.multinomial(h1_pts, probs_1h_2)
        away_seg_pts[i, 0:10] = rng.multinomial(a1_pts, probs_1h_2)

        home_seg_pts[i, 10:20] = rng.multinomial(h2_pts, probs_2h_i)
        away_seg_pts[i, 10:20] = rng.multinomial(a2_pts, probs_2h_i)

        if home_seg_metrics:
            for m, hseg in home_seg_metrics.items():
                full_h = _to_nonneg_int(float(home_stats.get(m)[i]))
                one_h = _to_nonneg_int(float(home_stats.get(f"{m}_1h")[i]))
                full_a = _to_nonneg_int(float(away_stats.get(m)[i]))
                one_a = _to_nonneg_int(float(away_stats.get(f"{m}_1h")[i]))

                two_h = max(0, full_h - one_h)
                two_a = max(0, full_a - one_a)

                hseg[i, 0:10] = rng.multinomial(one_h, probs_1h_2)
                away_seg_metrics[m][i, 0:10] = rng.multinomial(one_a, probs_1h_2)
                hseg[i, 10:20] = rng.multinomial(two_h, probs_2h_i)
                away_seg_metrics[m][i, 10:20] = rng.multinomial(two_a, probs_2h_i)

    home_cum = np.cumsum(home_seg_pts, axis=1)
    away_cum = np.cumsum(away_seg_pts, axis=1)
    total_cum = home_cum + away_cum
    margin_cum = home_cum - away_cum

    def _q(arr: np.ndarray) -> tuple[float, float, float]:
        qq = np.quantile(arr, [0.10, 0.50, 0.90])
        return float(qq[0]), float(qq[1]), float(qq[2])

    rows: list[dict] = []
    for seg_idx in range(20):
        start_min = int(seg_idx * 2)
        end_min = int((seg_idx + 1) * 2)
        half = 1 if int(end_min) <= 20 else 2

        h_seg = home_seg_pts[:, seg_idx].astype(float)
        a_seg = away_seg_pts[:, seg_idx].astype(float)
        t_seg = h_seg + a_seg
        m_seg = h_seg - a_seg

        h_end = home_cum[:, seg_idx].astype(float)
        a_end = away_cum[:, seg_idx].astype(float)
        t_end = total_cum[:, seg_idx].astype(float)
        m_end = margin_cum[:, seg_idx].astype(float)

        q_h_seg = _q(h_seg)
        q_a_seg = _q(a_seg)
        q_t_seg = _q(t_seg)
        q_m_seg = _q(m_seg)
        q_h_end = _q(h_end)
        q_a_end = _q(a_end)
        q_t_end = _q(t_end)
        q_m_end = _q(m_end)

        extra: dict[str, float] = {}
        if home_seg_metrics:
            for m in metrics:
                if m not in home_seg_metrics:
                    continue
                h_m = home_seg_metrics[m][:, seg_idx].astype(float)
                a_m = away_seg_metrics[m][:, seg_idx].astype(float)
                t_m = h_m + a_m
                q_t_m = _q(t_m)
                extra.update(
                    {
                        f"mu_home_{m}_seg": float(np.mean(h_m)),
                        f"mu_away_{m}_seg": float(np.mean(a_m)),
                        f"mu_total_{m}_seg": float(np.mean(t_m)),
                        f"q10_total_{m}_seg": float(q_t_m[0]),
                        f"q50_total_{m}_seg": float(q_t_m[1]),
                        f"q90_total_{m}_seg": float(q_t_m[2]),
                    }
                )

        rows.append(
            {
                "segment": int(seg_idx + 1),
                "half": int(half),
                "start_min": int(start_min),
                "end_min": int(end_min),
                "mu_home_pts_seg": float(np.mean(h_seg)),
                "mu_away_pts_seg": float(np.mean(a_seg)),
                "mu_total_pts_seg": float(np.mean(t_seg)),
                "mu_margin_pts_seg": float(np.mean(m_seg)),
                "q10_home_pts_seg": q_h_seg[0],
                "q50_home_pts_seg": q_h_seg[1],
                "q90_home_pts_seg": q_h_seg[2],
                "q10_away_pts_seg": q_a_seg[0],
                "q50_away_pts_seg": q_a_seg[1],
                "q90_away_pts_seg": q_a_seg[2],
                "q10_total_pts_seg": q_t_seg[0],
                "q50_total_pts_seg": q_t_seg[1],
                "q90_total_pts_seg": q_t_seg[2],
                "q10_margin_pts_seg": q_m_seg[0],
                "q50_margin_pts_seg": q_m_seg[1],
                "q90_margin_pts_seg": q_m_seg[2],
                "mu_home_score_end": float(np.mean(h_end)),
                "mu_away_score_end": float(np.mean(a_end)),
                "mu_total_score_end": float(np.mean(t_end)),
                "mu_margin_score_end": float(np.mean(m_end)),
                "q10_home_score_end": q_h_end[0],
                "q50_home_score_end": q_h_end[1],
                "q90_home_score_end": q_h_end[2],
                "q10_away_score_end": q_a_end[0],
                "q50_away_score_end": q_a_end[1],
                "q90_away_score_end": q_a_end[2],
                "q10_total_score_end": q_t_end[0],
                "q50_total_score_end": q_t_end[1],
                "q90_total_score_end": q_t_end[2],
                "q10_margin_score_end": q_m_end[0],
                "q50_margin_score_end": q_m_end[1],
                "q90_margin_score_end": q_m_end[2],
                **extra,
            }
        )

    return rows


def _resolve_mean_total_margin_from_features(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    # Feature-only mean construction (no model/blend required):
    # - possessions from pace/poss estimates
    # - per-team PPP from either rolling PPP columns or off/def ratings
    poss = _resolve_pace_mu(row)
    if poss is None:
        return None, None
    poss = float(np.clip(float(poss), PACE_MIN, PACE_MAX))

    # Prefer PPP-style means when present (points per possession)
    def _ppp(side: str) -> Optional[float]:
        own = row.get(f"{side}_ppp_mu")
        opp_allow = row.get(("away_ppp_allowed_mu" if side == "home" else "home_ppp_allowed_mu"))
        vals: list[float] = []
        for v in [own, opp_allow]:
            if v is not None and pd.notna(v):
                try:
                    fv = float(v)
                    if fv > 0:
                        vals.append(fv)
                except Exception:
                    continue
        if not vals:
            return None
        p = float(np.mean(vals))
        # Typical college PPP range
        return float(np.clip(p, 0.75, 1.35))

    ppp_home = _ppp("home")
    ppp_away = _ppp("away")

    if ppp_home is None or ppp_away is None:
        # Fallback: opponent-adjusted off/def ratings (per 100 possessions)
        ho = row.get("home_off_rating")
        ao = row.get("away_off_rating")
        hd = row.get("home_def_rating")
        ad = row.get("away_def_rating")
        if all(v is not None and pd.notna(v) for v in [ho, ao, hd, ad]):
            try:
                ho_f = float(ho)
                ao_f = float(ao)
                hd_f = float(hd)
                ad_f = float(ad)

                # In this codebase, `src/ncaab_model/features/ratings.py` produces opponent-adjusted
                # Off/Def ratings as centered deltas (mean ~0), where:
                #   off_eff ≈ (BASE + O_home - D_away)
                # not simply (O_home - D_away).
                # If values look like deltas, add a league baseline in points/100 possessions.
                is_delta_scale = (
                    max(abs(ho_f), abs(ao_f), abs(hd_f), abs(ad_f)) < 40.0
                )
                if is_delta_scale:
                    base_off_eff = 102.0  # ~1.02 PPP baseline; keeps totals on realistic scale
                    home_off_eff = base_off_eff + ho_f - ad_f
                    away_off_eff = base_off_eff + ao_f - hd_f
                else:
                    # If ratings are already on an absolute efficiency scale (points/100),
                    # approximate matchup efficiency as mean of offense and opponent defense.
                    home_off_eff = 0.5 * (ho_f + ad_f)
                    away_off_eff = 0.5 * (ao_f + hd_f)

                ppp_home = float(np.clip(home_off_eff / 100.0, 0.75, 1.35))
                ppp_away = float(np.clip(away_off_eff / 100.0, 0.75, 1.35))
            except Exception:
                ppp_home, ppp_away = None, None

    if ppp_home is None or ppp_away is None:
        return None, None

    mu_home = float(poss) * float(ppp_home)
    mu_away = float(poss) * float(ppp_away)
    total = float(mu_home + mu_away)
    margin = float(mu_home - mu_away)

    if not (70.0 <= total <= 250.0) or not (-80.0 <= margin <= 80.0):
        return None, None
    return total, margin


def _resolve_mean_total_margin(
    row: pd.Series,
    mean_source: str = "auto",
    allow_market_guardrails: bool = True,
) -> Tuple[Optional[float], Optional[float]]:
    # Mean selection is intentionally configurable:
    # - auto (default): prefers blend/model columns and applies market guardrails
    # - features: uses pace + PPP/off/def ratings only
    # - market: uses market_total + spread_home only
    src = (mean_source or "auto").strip().lower()
    features_strict = False
    if src in {"features_strict", "features-only", "features_only", "features!"}:
        src = "features"
        features_strict = True

    if src == "features":
        total, margin = _resolve_mean_total_margin_from_features(row)
        if total is not None and margin is not None:
            return total, margin
        if features_strict:
            return None, None
        # If feature inputs are missing, fall back to auto so the pipeline stays robust.
        src = "auto"

    market_total = _resolve_market_total(row)

    spread_home = None
    for scol in ["spread_home", "closing_spread_home", "home_spread", "closing_spread"]:
        if scol in row and pd.notna(row[scol]):
            try:
                spread_home = float(row[scol])
                break
            except Exception:
                continue
    market_margin = (-float(spread_home)) if spread_home is not None else None

    if src == "market":
        total = float(market_total) if market_total is not None else None
        margin = float(market_margin) if market_margin is not None else None
        return total, margin

    if src == "blend":
        total_cols = ["pred_total_blend", "pred_total_base", "pred_total"]
        margin_cols = ["pred_margin_blend", "pred_margin_base", "pred_margin"]
    elif src == "model":
        total_cols = [
            "pred_total_model_unified",
            "pred_total_model",
            "pred_total_calibrated",
            "pred_total",
            "pred_total_raw",
            "pred_total_seg",
            "pred_total_interval_mean",
            "total_pred",
        ]
        margin_cols = [
            "pred_margin_model",
            "pred_margin_calibrated",
            "pred_margin",
            "pred_margin_seg",
            "pred_margin_interval_mean",
            "margin_pred",
        ]
    else:
        # auto: Choose a preferred mean total/margin from available model/blend columns.
        # Use market lines only as a *guardrail* to avoid rare-but-deadly wrong-scale values,
        # not as a primary selector (which would implicitly hug the market).
        total_cols = [
            "pred_total_blend",
            "pred_total_base",
            "pred_total_model_unified",
            "pred_total_model",
            "pred_total_calibrated",
            "pred_total",
            "pred_total_raw",
            "pred_total_seg",
            "pred_total_interval_mean",
            "total_pred",
        ]
        margin_cols = [
            "pred_margin_blend",
            "pred_margin_base",
            "pred_margin_model",
            "pred_margin_calibrated",
            "pred_margin",
            "pred_margin_seg",
            "pred_margin_interval_mean",
            "margin_pred",
        ]

    total_candidates: list[tuple[str, float]] = []
    for tot_col in total_cols:
        if tot_col in row and pd.notna(row[tot_col]):
            try:
                v = float(row[tot_col])
            except Exception:
                continue
            if 70.0 <= v <= 250.0:
                total_candidates.append((tot_col, v))

    if total_candidates:
        total = total_candidates[0][1]
        if allow_market_guardrails and market_total is not None and abs(float(total) - float(market_total)) > 35.0:
            alt = next((v for _, v in total_candidates if abs(float(v) - float(market_total)) <= 35.0), None)
            if alt is not None:
                total = float(alt)
            else:
                total = float(market_total)
    else:
        total = None

    margin_candidates: list[tuple[str, float]] = []
    for mar_col in margin_cols:
        if mar_col in row and pd.notna(row[mar_col]):
            try:
                v = float(row[mar_col])
            except Exception:
                continue
            if -80.0 <= v <= 80.0:
                margin_candidates.append((mar_col, v))

    if margin_candidates:
        margin = margin_candidates[0][1]
        if allow_market_guardrails and market_margin is not None and abs(float(margin) - float(market_margin)) > 25.0:
            alt = next((v for _, v in margin_candidates if abs(float(v) - float(market_margin)) <= 25.0), None)
            if alt is not None:
                margin = float(alt)
            else:
                margin = float(market_margin)
    else:
        margin = None

    return total, margin


def _resolve_mean_total_margin_1h(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    for tot_col in [
        "pred_total_1h",
        "pred_total_model_1h",
        "pred_total_blend_1h",
        "pred_total_sim_1h",
    ]:
        if tot_col in row and pd.notna(row[tot_col]):
            try:
                total = float(row[tot_col])
            except Exception:
                continue
            if 25.0 <= total <= 125.0:
                break
    else:
        total = None

    for mar_col in [
        "pred_margin_1h",
        "pred_margin_model_1h",
        "pred_margin_blend_1h",
        "pred_margin_sim_1h",
    ]:
        if mar_col in row and pd.notna(row[mar_col]):
            try:
                margin = float(row[mar_col])
            except Exception:
                continue
            if -50.0 <= margin <= 50.0:
                break
    else:
        margin = None

    return total, margin


def _resolve_pace_mu(row: pd.Series) -> Optional[float]:
    # Prefer explicit pace estimate, then tempo ratings.
    for col in [
        "pace_game_est",
        "possessions_game_est",
    ]:
        if col in row and pd.notna(row[col]):
            try:
                v = float(row[col])
                if v > 0:
                    return v
            except Exception:
                continue

    ht = row.get("home_tempo_rating")
    at = row.get("away_tempo_rating")
    if pd.notna(ht) and pd.notna(at):
        try:
            v = (float(ht) + float(at)) / 2.0
            if v > 0:
                return v
        except Exception:
            pass
    # Some feature files also include a sum
    ts = row.get("tempo_rating_sum")
    if pd.notna(ts):
        try:
            v = float(ts) / 2.0
            if v > 0:
                return v
        except Exception:
            pass
    return None


def _resolve_pace_sigma(row: pd.Series, default_sigma: float) -> float:
    for col in [
        "pace_sigma_game_est",
        "pace_sigma",
    ]:
        if col in row and pd.notna(row[col]):
            try:
                v = float(row[col])
                if v >= 0:
                    return float(v)
            except Exception:
                continue
    return float(default_sigma)


def _load_injury_overrides(path: Path, date: str) -> Dict[str, dict]:
    """Load optional injury/availability overrides.

    Expected CSV columns (case-insensitive):
      - team (required)
      - date (optional; if present, filters to matching YYYY-MM-DD)
      - delta_total (optional; additive adjustment to game total expectation)
      - delta_margin (optional; additive adjustment to team's strength in home-margin space)
      - sigma_total_mult, sigma_margin_mult (optional; multiplicative)
      - pace_mult (optional; multiplicative)

    Returns mapping keyed by normalized team name.
    """
    try:
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        if df.empty:
            return {}

        cols = {c.lower(): c for c in df.columns}
        team_col = cols.get("team")
        if not team_col:
            return {}

        if "date" in cols:
            df = df[df[cols["date"]].astype(str) == str(date)]

        out: Dict[str, dict] = {}
        for _, r in df.iterrows():
            key = _norm_team_key(r.get(team_col))
            if not key:
                continue
            out[key] = {
                "delta_total": float(r.get(cols.get("delta_total"), 0.0) or 0.0) if cols.get("delta_total") else 0.0,
                "delta_margin": float(r.get(cols.get("delta_margin"), 0.0) or 0.0) if cols.get("delta_margin") else 0.0,
                "sigma_total_mult": float(r.get(cols.get("sigma_total_mult"), 1.0) or 1.0) if cols.get("sigma_total_mult") else 1.0,
                "sigma_margin_mult": float(r.get(cols.get("sigma_margin_mult"), 1.0) or 1.0) if cols.get("sigma_margin_mult") else 1.0,
                "pace_mult": float(r.get(cols.get("pace_mult"), 1.0) or 1.0) if cols.get("pace_mult") else 1.0,
            }
        return out
    except Exception:
        return {}


def _apply_overrides(
    row: pd.Series,
    total_mean: float,
    margin_mean: float,
    sigma_total: float,
    sigma_margin: Optional[float],
    pace_mu: Optional[float],
    injury_overrides: Dict[str, dict],
) -> Tuple[float, float, float, Optional[float], Optional[float], dict]:
    """Apply team-level overrides to game-level means/sigmas/pace."""
    home_key = _norm_team_key(row.get("home_team"))
    away_key = _norm_team_key(row.get("away_team"))

    h = injury_overrides.get(home_key or "", {})
    a = injury_overrides.get(away_key or "", {})

    delta_total = float(h.get("delta_total", 0.0)) + float(a.get("delta_total", 0.0))
    # Interpret delta_margin as *team strength* in home-margin space.
    # So away delta_margin increases away strength => decreases home margin.
    delta_margin = float(h.get("delta_margin", 0.0)) - float(a.get("delta_margin", 0.0))

    sigma_total_mult = float(max(h.get("sigma_total_mult", 1.0), a.get("sigma_total_mult", 1.0)))
    sigma_margin_mult = float(max(h.get("sigma_margin_mult", 1.0), a.get("sigma_margin_mult", 1.0)))
    pace_mult = float((h.get("pace_mult", 1.0) + a.get("pace_mult", 1.0)) / 2.0)

    total_mean2 = float(total_mean) + delta_total
    margin_mean2 = float(margin_mean) + delta_margin
    sigma_total2 = float(max(1e-6, float(sigma_total) * sigma_total_mult))
    sigma_margin2 = float(max(1e-6, float(sigma_margin) * sigma_margin_mult)) if sigma_margin is not None else None
    pace_mu2 = float(pace_mu * pace_mult) if pace_mu is not None else None

    meta = {
        "delta_total": delta_total,
        "delta_margin": delta_margin,
        "sigma_total_mult": sigma_total_mult,
        "sigma_margin_mult": sigma_margin_mult,
        "pace_mult": pace_mult,
    }
    return total_mean2, margin_mean2, sigma_total2, sigma_margin2, pace_mu2, meta


def _load_sim_calibration(path: Path) -> tuple[dict, Optional[str]]:
    try:
        if not path.exists():
            return {}, None
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj, None
        return {}, f"unexpected_calibration_type:{type(obj).__name__}"
    except Exception as e:
        return {}, f"{type(e).__name__}:{e}"


def _apply_sim_calibration(
    total_mean: float,
    margin_mean: float,
    sigma_total: float,
    sigma_margin: Optional[float],
    calibration: dict,
) -> Tuple[float, float, float, Optional[float], dict]:
    if not calibration:
        return total_mean, margin_mean, sigma_total, sigma_margin, {}

    delta_total = float(calibration.get("delta_total", 0.0) or 0.0)
    delta_margin = float(calibration.get("delta_margin", 0.0) or 0.0)
    sigma_total_mult = float(calibration.get("sigma_total_mult", 1.0) or 1.0)
    sigma_margin_mult = float(calibration.get("sigma_margin_mult", 1.0) or 1.0)

    # Guardrails: calibration artifacts can drift/accumulate; never allow extreme
    # global uncertainty inflation to dominate outputs.
    try:
        sigma_total_mult = float(np.clip(sigma_total_mult, 0.5, 5.0))
    except Exception:
        sigma_total_mult = 1.0
    try:
        sigma_margin_mult = float(np.clip(sigma_margin_mult, 0.5, 5.0))
    except Exception:
        sigma_margin_mult = 1.0

    total_mean2 = float(total_mean) + delta_total
    margin_mean2 = float(margin_mean) + delta_margin
    sigma_total2 = float(max(1e-6, float(sigma_total) * sigma_total_mult))
    sigma_margin2 = float(max(1e-6, float(sigma_margin) * sigma_margin_mult)) if sigma_margin is not None else None

    meta = {
        "delta_total": delta_total,
        "delta_margin": delta_margin,
        "sigma_total_mult": sigma_total_mult,
        "sigma_margin_mult": sigma_margin_mult,
    }
    return total_mean2, margin_mean2, sigma_total2, sigma_margin2, meta


def _resolve_total_sigma(row: pd.Series) -> float:
    for sig_col in [
        "interval_total_std",
        "interval_total_sigma",
        "tot_sigma",
        "sigma_total",
        "sigma_total_adj",
        # Main pipeline per-game uncertainty
        "pred_total_sigma",
        "pred_total_sigma_bootstrap",
    ]:
        if sig_col in row and pd.notna(row[sig_col]):
            try:
                return float(row[sig_col])
            except Exception:
                continue

    # Derive sigma from prediction intervals if available.
    try:
        # Enriched quantile bands (often p10/p90)
        if pd.notna(row.get("pred_total_p10")) and pd.notna(row.get("pred_total_p90")):
            low = float(row.get("pred_total_p10"))
            high = float(row.get("pred_total_p90"))
            return float(max(1e-6, (high - low) / 2.563103131089195))
        if pd.notna(row.get("pred_total_ci90_low")) and pd.notna(row.get("pred_total_ci90_high")):
            low = float(row.get("pred_total_ci90_low"))
            high = float(row.get("pred_total_ci90_high"))
            # 90% central interval => +/- z0.95
            return float(max(1e-6, (high - low) / (2.0 * 1.6448536269514722)))
        if pd.notna(row.get("pred_total_ci75_low")) and pd.notna(row.get("pred_total_ci75_high")):
            low = float(row.get("pred_total_ci75_low"))
            high = float(row.get("pred_total_ci75_high"))
            # 75% central interval => +/- z0.875
            return float(max(1e-6, (high - low) / (2.0 * 1.1503493803760082)))
    except Exception:
        pass
    return DEFAULT_TOTAL_SIGMA


def _resolve_margin_sigma(row: pd.Series) -> Optional[float]:
    for sig_col in [
        "interval_margin_std",
        "interval_margin_sigma",
        "mar_sigma",
        "sigma_margin",
        "sigma_margin_adj",
        # Main pipeline per-game uncertainty
        "pred_margin_sigma",
        "pred_margin_sigma_bootstrap",
    ]:
        if sig_col in row and pd.notna(row[sig_col]):
            try:
                v = float(row[sig_col])
                if v > 0:
                    return v
            except Exception:
                continue

    # Derive sigma from prediction intervals if available.
    try:
        # Enriched quantile bands (often p10/p90)
        if pd.notna(row.get("pred_margin_p10")) and pd.notna(row.get("pred_margin_p90")):
            low = float(row.get("pred_margin_p10"))
            high = float(row.get("pred_margin_p90"))
            return float(max(1e-6, (high - low) / 2.563103131089195))
        if pd.notna(row.get("pred_margin_ci90_low")) and pd.notna(row.get("pred_margin_ci90_high")):
            low = float(row.get("pred_margin_ci90_low"))
            high = float(row.get("pred_margin_ci90_high"))
            return float(max(1e-6, (high - low) / (2.0 * 1.6448536269514722)))
        if pd.notna(row.get("pred_margin_ci75_low")) and pd.notna(row.get("pred_margin_ci75_high")):
            low = float(row.get("pred_margin_ci75_low"))
            high = float(row.get("pred_margin_ci75_high"))
            return float(max(1e-6, (high - low) / (2.0 * 1.1503493803760082)))
    except Exception:
        pass
    return None


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


def _resolve_market_total_1h(row: pd.Series) -> Optional[float]:
    for mcol in [
        "market_total_1h",
        "closing_total_1h",
        "total_1h",
    ]:
        if mcol in row and pd.notna(row[mcol]):
            try:
                return float(row[mcol])
            except Exception:
                continue
    return None


def _resolve_spread_home_1h(row: pd.Series) -> Optional[float]:
    for scol in [
        "spread_home_1h",
        "closing_spread_home_1h",
        "home_spread_1h",
    ]:
        if scol in row and pd.notna(row[scol]):
            try:
                return float(row[scol])
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


def simulate_game_row(
    row: pd.Series,
    rho: float = DEFAULT_RHO,
    samples: int = DEFAULT_SAMPLES,
    use_pace: bool = False,
    pace_sigma: float = DEFAULT_PACE_SIGMA,
    injury_overrides: Optional[Dict[str, dict]] = None,
    sim_calibration: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
    mean_source: str = "auto",
    allow_market_guardrails: bool = True,
    engine: str = "auto",
    segment_probs_half1: Optional[np.ndarray] = None,
    segment_probs_half2: Optional[np.ndarray] = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng()
    mean_source_used = (mean_source or "auto").strip().lower() or "auto"
    engine_used = _resolve_sim_engine(engine, "features" if mean_source_used in {"features", "features_strict", "features-only", "features_only", "features!"} else mean_source_used)

    def _infer_first_matching_col(row: pd.Series, candidates: list[str], selected: float) -> Optional[str]:
        try:
            sel = float(selected)
        except Exception:
            return None
        for c in candidates:
            try:
                if c in row and pd.notna(row[c]):
                    v = float(row[c])
                    if np.isfinite(v) and abs(v - sel) <= 1e-6:
                        return c
            except Exception:
                continue
        return None

    # For the possession/event simulator, default to the same mean-selection policy
    # as other engines (model/blend first, with market guardrails). If those inputs
    # are missing, fall back to feature-derived means.
    total_mean, margin_mean = _resolve_mean_total_margin(
        row,
        mean_source=mean_source_used,
        allow_market_guardrails=bool(allow_market_guardrails),
    )
    if (total_mean is None or margin_mean is None) and engine_used == "events" and mean_source_used == "auto":
        ft_total, ft_margin = _resolve_mean_total_margin_from_features(row)
        if ft_total is not None and ft_margin is not None:
            total_mean, margin_mean = ft_total, ft_margin
            mean_source_used = "features_auto"
    if total_mean is None or margin_mean is None:
        return {
            "sim_ok": False,
            "mu_total": total_mean,
            "mu_margin": margin_mean,
            "mean_source": (mean_source or "auto"),
            "mean_source_used": mean_source_used,
        }

    # Trace the mean-selection path for diagnostics (helps debug guardrail overrides).
    try:
        market_total_dbg = _resolve_market_total(row)
        spread_home_dbg = None
        for scol in ["spread_home", "closing_spread_home", "home_spread", "closing_spread"]:
            if scol in row and pd.notna(row[scol]):
                try:
                    spread_home_dbg = float(row[scol])
                    break
                except Exception:
                    continue
        market_margin_dbg = (-float(spread_home_dbg)) if spread_home_dbg is not None else None

        # Candidate ordering should mirror _resolve_mean_total_margin
        src_dbg = (mean_source_used or "auto").strip().lower() or "auto"
        if src_dbg == "blend":
            total_cols_dbg = ["pred_total_blend", "pred_total_base", "pred_total"]
            margin_cols_dbg = ["pred_margin_blend", "pred_margin_base", "pred_margin"]
        elif src_dbg == "model":
            total_cols_dbg = [
                "pred_total_model_unified",
                "pred_total_model",
                "pred_total_calibrated",
                "pred_total",
                "pred_total_raw",
                "pred_total_seg",
                "pred_total_interval_mean",
                "total_pred",
            ]
            margin_cols_dbg = [
                "pred_margin_model",
                "pred_margin_calibrated",
                "pred_margin",
                "pred_margin_seg",
                "pred_margin_interval_mean",
                "margin_pred",
            ]
        else:
            total_cols_dbg = [
                "pred_total_blend",
                "pred_total_base",
                "pred_total_model_unified",
                "pred_total_model",
                "pred_total_calibrated",
                "pred_total",
                "pred_total_raw",
                "pred_total_seg",
                "pred_total_interval_mean",
                "total_pred",
            ]
            margin_cols_dbg = [
                "pred_margin_blend",
                "pred_margin_base",
                "pred_margin_model",
                "pred_margin_calibrated",
                "pred_margin",
                "pred_margin_seg",
                "pred_margin_interval_mean",
                "margin_pred",
            ]

        # First raw candidate (pre-guardrail), if present.
        first_total_val = None
        first_total_col = None
        for c in total_cols_dbg:
            if c in row and pd.notna(row[c]):
                try:
                    vv = float(row[c])
                except Exception:
                    continue
                if 70.0 <= vv <= 250.0:
                    first_total_val = float(vv)
                    first_total_col = c
                    break

        first_margin_val = None
        first_margin_col = None
        for c in margin_cols_dbg:
            if c in row and pd.notna(row[c]):
                try:
                    vv = float(row[c])
                except Exception:
                    continue
                if -80.0 <= vv <= 80.0:
                    first_margin_val = float(vv)
                    first_margin_col = c
                    break

        mean_total_col = _infer_first_matching_col(row, total_cols_dbg, float(total_mean))
        mean_margin_col = _infer_first_matching_col(row, margin_cols_dbg, float(margin_mean))

        guardrail_total_applied = False
        try:
            if first_total_val is not None and abs(float(first_total_val) - float(total_mean)) > 1e-6:
                guardrail_total_applied = True
        except Exception:
            guardrail_total_applied = False

        guardrail_margin_applied = False
        try:
            if first_margin_val is not None and abs(float(first_margin_val) - float(margin_mean)) > 1e-6:
                guardrail_margin_applied = True
        except Exception:
            guardrail_margin_applied = False

        mean_trace = {
            "mean_total_selected": float(total_mean),
            "mean_margin_selected": float(margin_mean),
            "mean_total_first_candidate": float(first_total_val) if first_total_val is not None else None,
            "mean_margin_first_candidate": float(first_margin_val) if first_margin_val is not None else None,
            "mean_total_first_col": first_total_col,
            "mean_margin_first_col": first_margin_col,
            "mean_total_col_used": mean_total_col,
            "mean_margin_col_used": mean_margin_col,
            "market_total_resolved": float(market_total_dbg) if market_total_dbg is not None else None,
            "market_margin_resolved": float(market_margin_dbg) if market_margin_dbg is not None else None,
            "mean_total_guardrail_applied": bool(guardrail_total_applied),
            "mean_margin_guardrail_applied": bool(guardrail_margin_applied),
        }
    except Exception:
        mean_trace = {}
    sigma_total = _resolve_total_sigma(row)
    sigma_margin = _resolve_margin_sigma(row)

    pace_mu = _resolve_pace_mu(row)
    try:
        pace_sigma = _resolve_pace_sigma(row, float(pace_sigma))
    except Exception:
        pass
    applied = {}
    if injury_overrides:
        total_mean, margin_mean, sigma_total, sigma_margin, pace_mu, applied = _apply_overrides(
            row,
            float(total_mean),
            float(margin_mean),
            float(sigma_total),
            sigma_margin,
            pace_mu,
            injury_overrides,
        )

    calib_applied = {}
    if sim_calibration:
        total_mean, margin_mean, sigma_total, sigma_margin, calib_applied = _apply_sim_calibration(
            float(total_mean),
            float(margin_mean),
            float(sigma_total),
            sigma_margin,
            sim_calibration,
        )

    # Preserve trace of what ultimately became the targets.
    try:
        mean_trace["mean_total_after_overrides_calib"] = float(total_mean)
        mean_trace["mean_margin_after_overrides_calib"] = float(margin_mean)
    except Exception:
        pass

    # Allow sim calibration to override rho for the fallback path that infers
    # per-team sigma from total sigma.
    rho_eff = float(rho)
    try:
        if sim_calibration and "rho" in sim_calibration and sim_calibration.get("rho") is not None:
            rho_eff = float(sim_calibration.get("rho"))
    except Exception:
        rho_eff = float(rho)

    mu_home = (total_mean + margin_mean) / 2.0
    mu_away = (total_mean - margin_mean) / 2.0

    # 1H mean targets when present; otherwise scale full-game.
    half_frac = _resolve_half_frac(row)
    # 1H mean targets when present; otherwise scale full-game.
    # For feature-only mean mode, always scale full-game (keeps it independent from
    # any downstream prediction columns that might be present on the row).
    total_mean_1h, margin_mean_1h = (None, None)
    if mean_source_used not in {"features", "market"}:
        total_mean_1h, margin_mean_1h = _resolve_mean_total_margin_1h(row)
    elif mean_source_used == "market":
        try:
            mt1 = _resolve_market_total_1h(row)
            sp1 = _resolve_spread_home_1h(row)
            total_mean_1h = float(mt1) if mt1 is not None else None
            margin_mean_1h = (-float(sp1)) if sp1 is not None else None
        except Exception:
            total_mean_1h, margin_mean_1h = (None, None)
    if total_mean_1h is None:
        total_mean_1h = float(total_mean) * half_frac
    if margin_mean_1h is None:
        margin_mean_1h = float(margin_mean) * 0.5

    # Guardrail: if an explicit 1H total mean is wildly inconsistent with the full-game
    # mean, treat it as untrusted and fall back to proportional scaling.
    # This avoids pathological implied 2H totals (e.g., 1H ~ 75 with full-game ~ 106).
    try:
        if total_mean_1h is not None and total_mean is not None:
            frac_1h = float(total_mean_1h) / float(max(float(total_mean), 1e-6))
            if not np.isfinite(frac_1h) or frac_1h < 0.30 or frac_1h > 0.70:
                total_mean_1h = float(total_mean) * float(half_frac)
    except Exception:
        pass

    # Apply same override/calibration deltas to 1H means (scaled by minutes).
    try:
        delta_total_applied = float(applied.get("delta_total", 0.0)) if applied else 0.0
        delta_margin_applied = float(applied.get("delta_margin", 0.0)) if applied else 0.0
        total_mean_1h = float(total_mean_1h) + delta_total_applied * half_frac
        margin_mean_1h = float(margin_mean_1h) + delta_margin_applied * 0.5
    except Exception:
        pass

    # 1H calibration: if explicit 1H params exist, use them; otherwise fall back
    # to scaling full-game deltas by half.
    has_1h_cal = bool(
        sim_calibration
        and any(
            k in sim_calibration
            for k in [
                "delta_total_1h",
                "delta_margin_1h",
                "sigma_total_1h_mult",
                "sigma_margin_1h_mult",
            ]
        )
    )

    calib_delta_total_1h_applied = 0.0
    calib_delta_margin_1h_applied = 0.0
    calib_sigma_total_1h_mult_applied = 1.0
    calib_sigma_margin_1h_mult_applied = 1.0
    if has_1h_cal:
        try:
            calib_delta_total_1h_applied = float(sim_calibration.get("delta_total_1h", 0.0) or 0.0)
            calib_delta_margin_1h_applied = float(sim_calibration.get("delta_margin_1h", 0.0) or 0.0)
            total_mean_1h = float(total_mean_1h) + calib_delta_total_1h_applied
            margin_mean_1h = float(margin_mean_1h) + calib_delta_margin_1h_applied
        except Exception:
            pass
    else:
        try:
            delta_total_cal = float(calib_applied.get("delta_total", 0.0)) if calib_applied else 0.0
            delta_margin_cal = float(calib_applied.get("delta_margin", 0.0)) if calib_applied else 0.0
            calib_delta_total_1h_applied = float(delta_total_cal) * float(half_frac)
            calib_delta_margin_1h_applied = float(delta_margin_cal) * 0.5
            total_mean_1h = float(total_mean_1h) + delta_total_cal * half_frac
            margin_mean_1h = float(margin_mean_1h) + delta_margin_cal * 0.5
        except Exception:
            pass

    mu_home_1h = (float(total_mean_1h) + float(margin_mean_1h)) / 2.0
    mu_away_1h = (float(total_mean_1h) - float(margin_mean_1h)) / 2.0
    mu_home_1h = float(max(mu_home_1h, 0.0))
    mu_away_1h = float(max(mu_away_1h, 0.0))

    # Uncertainty scaling for 1H: empirically, the second half tends to be more variable
    # (end-game fouling/late pace changes). Allocate slightly less than proportional
    # variance to 1H to keep 1H intervals realistic while preserving full-game calibration.
    half_var_frac = float(0.45 + 0.4 * float(half_frac - 0.5))
    half_var_frac = float(np.clip(half_var_frac, 0.35, 0.55))
    sigma_total_1h = float(max(1e-6, float(sigma_total) * float(np.sqrt(half_var_frac))))
    sigma_margin_1h = float(max(1e-6, float(sigma_margin) * float(np.sqrt(half_var_frac)))) if sigma_margin is not None else None
    if has_1h_cal:
        try:
            m = float(sim_calibration.get("sigma_total_1h_mult", 1.0) or 1.0)
            # Hard cap: 1H variance should not exceed full-game variance.
            # Since base scaling is sigma_full*sqrt(half_frac), the max additional
            # multiplier is ~1/sqrt(half_frac).
            m_cap = float(0.99 / float(np.sqrt(max(half_frac, 1e-6))))
            # Empirically, 1H total bands are already close with sqrt-scaling from full game;
            # do not further inflate them globally.
            m = float(min(m, 1.0, m_cap))
            calib_sigma_total_1h_mult_applied = float(m)
            sigma_total_1h = float(max(1e-6, float(sigma_total_1h) * m))
        except Exception:
            pass
        if sigma_margin_1h is not None:
            try:
                m = float(sim_calibration.get("sigma_margin_1h_mult", 1.0) or 1.0)
                m_cap = float(0.99 / float(np.sqrt(max(half_frac, 1e-6))))
                # Allow some additional 1H margin spread, but keep it modest.
                m = float(min(m, 1.15, m_cap))
                calib_sigma_margin_1h_mult_applied = float(m)
                sigma_margin_1h = float(max(1e-6, float(sigma_margin_1h) * m))
            except Exception:
                pass

    # Basic sanity clamps (avoid pathological inputs creating NaNs)
    try:
        mu_home = float(mu_home)
        mu_away = float(mu_away)
    except Exception:
        return {
            "sim_ok": False,
            "mu_total": total_mean,
            "mu_margin": margin_mean,
            "mean_source": (mean_source or "auto"),
            "mean_source_used": mean_source_used,
        }
    mu_home = float(max(mu_home, 0.0))
    mu_away = float(max(mu_away, 0.0))

    # Optional pace/possessions path: simulate possessions and per-team PPP.
    # This behaves more like a basketball simulator:
    #   - one shared possessions draw per sim
    #   - per-team scoring = possessions * PPP + (noise scaled by possessions)
    #   - an additional shared (or anti-shared) shock is used to match the target
    #     covariance implied by (sigma_total, sigma_margin)
    if engine_used == "events":
        # Event sim needs a possessions estimate. If we have pace columns but they're
        # NaN (common early in the day), fall back to a sane default rather than
        # failing the simulation.
        if pace_mu is None or pace_mu <= 0:
            pace_mu = float(DEFAULT_PACE)

        ev = _simulate_events_samples(
            row=row,
            samples=int(samples),
            pace_mu=float(pace_mu),
            pace_sigma=float(pace_sigma) if pace_sigma is not None else DEFAULT_PACE_SIGMA,
            half_frac=float(half_frac),
            mu_home=float(mu_home),
            mu_away=float(mu_away),
            rng=rng,
        )
        home_pts = ev["home"]
        away_pts = ev["away"]
        home_1h = ev["home_1h"]
        away_1h = ev["away_1h"]

        # Apply 1H calibration to the event-sim by reallocating points between halves.
        # This preserves full-game totals while shifting 1H (and 2H as the residual).
        # Note: values can become non-integer; that's fine for probabilistic outputs.
        try:
            dT = float(calib_delta_total_1h_applied)
            dM = float(calib_delta_margin_1h_applied)
            if np.isfinite(dT) and np.isfinite(dM) and (abs(dT) > 1e-9 or abs(dM) > 1e-9):
                d_home = 0.5 * (dT + dM)
                d_away = 0.5 * (dT - dM)
                home_1h = np.clip(home_1h.astype(float) + float(d_home), 0.0, home_pts.astype(float))
                away_1h = np.clip(away_1h.astype(float) + float(d_away), 0.0, away_pts.astype(float))
        except Exception:
            pass
        totals = home_pts + away_pts
        margins = home_pts - away_pts
        totals_1h = home_1h + away_1h
        margins_1h = home_1h - away_1h

        q10_t = float(np.quantile(totals, 0.10))
        q50_t = float(np.quantile(totals, 0.50))
        q90_t = float(np.quantile(totals, 0.90))
        q10_m = float(np.quantile(margins, 0.10))
        q50_m = float(np.quantile(margins, 0.50))
        q90_m = float(np.quantile(margins, 0.90))

        q10_t1 = float(np.quantile(totals_1h, 0.10))
        q50_t1 = float(np.quantile(totals_1h, 0.50))
        q90_t1 = float(np.quantile(totals_1h, 0.90))
        q10_m1 = float(np.quantile(margins_1h, 0.10))
        q50_m1 = float(np.quantile(margins_1h, 0.50))
        q90_m1 = float(np.quantile(margins_1h, 0.90))

        # Derive 2H as residual
        totals_2h = totals - totals_1h
        margins_2h = margins - margins_1h
        q10_t2 = float(np.quantile(totals_2h, 0.10))
        q50_t2 = float(np.quantile(totals_2h, 0.50))
        q90_t2 = float(np.quantile(totals_2h, 0.90))
        q10_m2 = float(np.quantile(margins_2h, 0.10))
        q50_m2 = float(np.quantile(margins_2h, 0.50))
        q90_m2 = float(np.quantile(margins_2h, 0.90))

        market_total = _resolve_market_total(row)
        p_over_market = float(np.mean(totals > market_total)) if market_total is not None else None
        market_total_1h = _resolve_market_total_1h(row)
        p_over_market_1h = float(np.mean(totals_1h > market_total_1h)) if market_total_1h is not None else None

        spread_home_1h = _resolve_spread_home_1h(row)
        p_cover_home_1h = float(np.mean(margins_1h + float(spread_home_1h) > 0)) if spread_home_1h is not None else None

        spread_home = None
        try:
            for scol in ["spread_home", "closing_spread_home", "home_spread"]:
                if scol in row and pd.notna(row[scol]):
                    spread_home = float(row[scol])
                    break
        except Exception:
            spread_home = None
        p_cover_home = float(np.mean(margins + float(spread_home) > 0)) if spread_home is not None else None

        try:
            p_home_win = float(np.mean(margins > 0))
        except Exception:
            p_home_win = None
        try:
            p_home_win_1h = float(np.mean(margins_1h > 0))
        except Exception:
            p_home_win_1h = None

        sigma_total_s = float(max(1e-6, float(np.std(totals, ddof=1))))
        sigma_margin_s = float(max(1e-6, float(np.std(margins, ddof=1))))
        sigma_total_1h_s = float(max(1e-6, float(np.std(totals_1h, ddof=1))))
        sigma_margin_1h_s = float(max(1e-6, float(np.std(margins_1h, ddof=1))))
        sigma_total_2h_s = float(max(1e-6, float(np.std(totals_2h, ddof=1))))
        sigma_margin_2h_s = float(max(1e-6, float(np.std(margins_2h, ddof=1))))

        use_time_aware_segments = True
        segments_grid_min_used = None
        try:
            # Set NCAAB_SEGMENTS_TIME_AWARE explicitly to override.
            raw = os.environ.get("NCAAB_SEGMENTS_TIME_AWARE")
            v = (raw or "").strip()
            if v:
                use_time_aware_segments = _safe_bool(v)
            else:
                # If tuned 2-min weights are provided, default to the point-allocation
                # splitter so the weights actually affect the segment shape.
                if segment_probs_half1 is not None or segment_probs_half2 is not None:
                    size10 = False
                    try:
                        if segment_probs_half1 is not None and np.asarray(segment_probs_half1, dtype=float).reshape(-1).size == 10:
                            size10 = True
                        if segment_probs_half2 is not None and np.asarray(segment_probs_half2, dtype=float).reshape(-1).size == 10:
                            size10 = True
                    except Exception:
                        size10 = False
                    try:
                        grid_min = int(_segments_grid_min_from_env())
                        segments_grid_min_used = int(grid_min)
                    except Exception:
                        grid_min = 5
                        segments_grid_min_used = int(grid_min)
                    if size10 or int(grid_min) == 2:
                        use_time_aware_segments = False
        except Exception:
            use_time_aware_segments = True

        if segments_grid_min_used is None:
            try:
                segments_grid_min_used = int(_segments_grid_min_from_env())
            except Exception:
                segments_grid_min_used = 5

        abs_margin_proxy_source = None
        abs_margin_proxy_value = None
        try:
            if spread_home is not None and np.isfinite(float(spread_home)):
                abs_margin_proxy_source = "spread"
                abs_margin_proxy_value = abs(float(spread_home))
            else:
                abs_margin_proxy_source = "expected"
                abs_margin_proxy_value = abs(float(mu_home) - float(mu_away))
        except Exception:
            abs_margin_proxy_source = "expected"
            try:
                abs_margin_proxy_value = abs(float(mu_home) - float(mu_away))
            except Exception:
                abs_margin_proxy_value = None

        # If we have explicit 1H calibration deltas, time-aware segments would ignore the
        # calibrated 1H distribution (it re-simulates its own possession timeline).
        # Force point-allocation segments so end_min=20 matches calibrated 1H, unless
        # the user explicitly opts into time-aware segments with 1H calibration.
        try:
            allow_time_aware_with_1h_cal = _safe_bool(os.environ.get("NCAAB_ALLOW_TIME_AWARE_SEGMENTS_WITH_1H_CAL"))
        except Exception:
            allow_time_aware_with_1h_cal = False
        try:
            if (not allow_time_aware_with_1h_cal) and has_1h_cal:
                dT = float(calib_delta_total_1h_applied)
                dM = float(calib_delta_margin_1h_applied)
                if np.isfinite(dT) and np.isfinite(dM) and (abs(dT) > 1e-9 or abs(dM) > 1e-9):
                    use_time_aware_segments = False
        except Exception:
            pass

        if use_time_aware_segments:
            # Recompute event params here to avoid threading through ev payload.
            to_h, ft_h, three_h = _derive_event_rates(row, "home")
            to_a, ft_a, three_a = _derive_event_rates(row, "away")
            ft_pct_h, p2_h, p3_h = _calibrate_shooting_to_ppp(
                float(np.clip(float(mu_home) / max(1.0, float(np.clip(float(pace_mu), PACE_MIN, PACE_MAX))), 0.75, 1.35)),
                to_h,
                ft_h,
                three_h,
            )
            ft_pct_a, p2_a, p3_a = _calibrate_shooting_to_ppp(
                float(np.clip(float(mu_away) / max(1.0, float(np.clip(float(pace_mu), PACE_MIN, PACE_MAX))), 0.75, 1.35)),
                to_a,
                ft_a,
                three_a,
            )
            try:
                grid_min = _segments_grid_min_from_env()
            except Exception:
                grid_min = 5
            seg_fn = _segment_2min_quantiles_from_events_timeline if int(grid_min) == 2 else _segment_5min_quantiles_from_events_timeline
            segment_rows = seg_fn(
                poss_1h=ev.get("home_poss_1h") if ev.get("home_poss_1h") is not None else np.zeros(int(samples), dtype=int),
                poss_2h=(ev.get("home_poss") - ev.get("home_poss_1h")) if (ev.get("home_poss") is not None and ev.get("home_poss_1h") is not None) else np.zeros(int(samples), dtype=int),
                home_params=(to_h, ft_h, three_h, ft_pct_h, p2_h, p3_h),
                away_params=(to_a, ft_a, three_a, ft_pct_a, p2_a, p3_a),
                rng=rng,
                enable_late_foul=True,
            )
        else:
            try:
                grid_min_pts = _segments_grid_min_from_env()
            except Exception:
                grid_min_pts = 5
            segment_rows = _segment_grid_quantiles_from_points(
                home_pts=home_pts,
                away_pts=away_pts,
                home_1h=home_1h,
                away_1h=away_1h,
                grid_min=int(grid_min_pts),
                abs_margin_proxy=abs_margin_proxy_value,
                seg_probs_half1=segment_probs_half1,
                seg_probs_half2=segment_probs_half2,
                home_stats={
                    "poss": ev.get("home_poss"),
                    "tov": ev.get("home_tov"),
                    "fta": ev.get("home_fta"),
                    "fga2": ev.get("home_fga2"),
                    "fga3": ev.get("home_fga3"),
                    "poss_1h": ev.get("home_poss_1h"),
                    "tov_1h": ev.get("home_tov_1h"),
                    "fta_1h": ev.get("home_fta_1h"),
                    "fga2_1h": ev.get("home_fga2_1h"),
                    "fga3_1h": ev.get("home_fga3_1h"),
                },
                away_stats={
                    "poss": ev.get("away_poss"),
                    "tov": ev.get("away_tov"),
                    "fta": ev.get("away_fta"),
                    "fga2": ev.get("away_fga2"),
                    "fga3": ev.get("away_fga3"),
                    "poss_1h": ev.get("away_poss_1h"),
                    "tov_1h": ev.get("away_tov_1h"),
                    "fta_1h": ev.get("away_fta_1h"),
                    "fga2_1h": ev.get("away_fga2_1h"),
                    "fga3_1h": ev.get("away_fga3_1h"),
                },
                rng=rng,
            )

        poss_mu_used = float(np.clip(float(pace_mu), PACE_MIN, PACE_MAX))
        return {
            "sim_ok": True,
            "sim_method": "events",
            "sim_engine": "events",
            "segments_grid_min": int(segments_grid_min_used) if segments_grid_min_used is not None else None,
            "segments_mode": "time_aware" if bool(use_time_aware_segments) else "points_alloc",
            "abs_margin_proxy": float(abs_margin_proxy_value) if abs_margin_proxy_value is not None and np.isfinite(float(abs_margin_proxy_value)) else None,
            "abs_margin_proxy_source": abs_margin_proxy_source,
            "segment_probs_half1_len": _prob_vec_len(segment_probs_half1),
            "segment_probs_half2_len": _prob_vec_len(segment_probs_half2),
            "segment_probs_half1_hash": _hash_prob_vec_short(segment_probs_half1),
            "segment_probs_half2_hash": _hash_prob_vec_short(segment_probs_half2),
            "mean_source": (mean_source or "auto"),
            "mean_source_used": mean_source_used,
            **mean_trace,
            "pace_mu": poss_mu_used,
            "pace_sigma": float(pace_sigma) if pace_sigma is not None else DEFAULT_PACE_SIGMA,
            "rho_used": None,
            "sigma_total": sigma_total_s,
            "sigma_margin": sigma_margin_s,
            "mu_home": float(np.mean(home_pts)),
            "mu_away": float(np.mean(away_pts)),
            "ppp_home_mu": float(np.mean(home_pts)) / float(max(poss_mu_used, 1e-6)),
            "ppp_away_mu": float(np.mean(away_pts)) / float(max(poss_mu_used, 1e-6)),
            "cov_target": None,
            "cov_from_possessions": None,
            "cov_residual": None,
            "mu_total": float(np.mean(totals)),
            "mu_margin": float(np.mean(margins)),
            "q10_total": q10_t,
            "q50_total": q50_t,
            "q90_total": q90_t,
            "q10_margin": q10_m,
            "q50_margin": q50_m,
            "q90_margin": q90_m,
            "pace_mu_1h": float(poss_mu_used) * float(half_frac),
            "pace_sigma_1h": float((pace_sigma if pace_sigma is not None else DEFAULT_PACE_SIGMA) * float(np.sqrt(max(half_frac, 1e-6)))),
            "sigma_total_1h": sigma_total_1h_s,
            "sigma_margin_1h": sigma_margin_1h_s,
            "mu_home_1h": float(np.mean(home_1h)),
            "mu_away_1h": float(np.mean(away_1h)),
            "mu_total_1h": float(np.mean(totals_1h)),
            "mu_margin_1h": float(np.mean(margins_1h)),
            "q10_total_1h": q10_t1,
            "q50_total_1h": q50_t1,
            "q90_total_1h": q90_t1,
            "q10_margin_1h": q10_m1,
            "q50_margin_1h": q50_m1,
            "q90_margin_1h": q90_m1,
            "sigma_total_2h": sigma_total_2h_s,
            "sigma_margin_2h": sigma_margin_2h_s,
            "mu_total_2h": float(np.mean(totals_2h)),
            "mu_margin_2h": float(np.mean(margins_2h)),
            "q10_total_2h": q10_t2,
            "q50_total_2h": q50_t2,
            "q90_total_2h": q90_t2,
            "q10_margin_2h": q10_m2,
            "q50_margin_2h": q50_m2,
            "q90_margin_2h": q90_m2,
            "p_over_market_1h": p_over_market_1h,
            "market_total_1h": market_total_1h,
            "p_cover_home_1h": p_cover_home_1h,
            "spread_home_1h": spread_home_1h,
            "p_over_market": p_over_market,
            "market_total": market_total,
            "p_cover_home": p_cover_home,
            "spread_home": spread_home,
            "p_home_win": p_home_win,
            "p_home_win_1h": p_home_win_1h,
            "override_delta_total": float(applied.get("delta_total", 0.0)) if applied else 0.0,
            "override_delta_margin": float(applied.get("delta_margin", 0.0)) if applied else 0.0,
            "override_sigma_total_mult": float(applied.get("sigma_total_mult", 1.0)) if applied else 1.0,
            "override_sigma_margin_mult": float(applied.get("sigma_margin_mult", 1.0)) if applied else 1.0,
            "override_pace_mult": float(applied.get("pace_mult", 1.0)) if applied else 1.0,
            "calib_delta_total": float(calib_applied.get("delta_total", 0.0)) if calib_applied else 0.0,
            "calib_delta_margin": float(calib_applied.get("delta_margin", 0.0)) if calib_applied else 0.0,
            "calib_sigma_total_mult": float(calib_applied.get("sigma_total_mult", 1.0)) if calib_applied else 1.0,
            "calib_sigma_margin_mult": float(calib_applied.get("sigma_margin_mult", 1.0)) if calib_applied else 1.0,
            "calib_delta_total_1h": float(calib_delta_total_1h_applied),
            "calib_delta_margin_1h": float(calib_delta_margin_1h_applied),
            "calib_sigma_total_1h_mult": float(calib_sigma_total_1h_mult_applied),
            "calib_sigma_margin_1h_mult": float(calib_sigma_margin_1h_mult_applied),
            "_segments_rows": segment_rows,
            **{f"event_{k}": v for k, v in (ev.get("agg") or {}).items()},
        }

    if use_pace and pace_mu is not None and pace_mu > 0:
        def _sim_half_from_possessions(
            poss_half: np.ndarray,
            e_poss_half: float,
            var_poss_half: float,
            mu_home_half: float,
            mu_away_half: float,
            sigma_total_half: float,
            sigma_margin_half: Optional[float],
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
            total_mean_half = float(mu_home_half) + float(mu_away_half)

            var_total_pts_half = float(max(float(sigma_total_half) ** 2, 1e-6))
            var_margin_pts_half = float(sigma_margin_half) ** 2 if sigma_margin_half is not None else (DEFAULT_TOTAL_SIGMA / 2.0) ** 2
            var_sum_half = float(max((var_total_pts_half + var_margin_pts_half) / 2.0, 1e-6))
            cov_target_half = float((var_total_pts_half - var_margin_pts_half) / 4.0)

            w_home_half = float(mu_home_half / float(max(total_mean_half, 1e-6)))
            w_home_half = float(np.clip(w_home_half, 0.2, 0.8))
            var_home_target_half = float(max(var_sum_half * w_home_half, 1e-6))
            var_away_target_half = float(max(var_sum_half - var_home_target_half, 1e-6))

            ppp_home_mu_half = float(mu_home_half) / float(max(e_poss_half, 1e-6))
            ppp_away_mu_half = float(mu_away_half) / float(max(e_poss_half, 1e-6))
            ppp_home_mu_half = float(max(ppp_home_mu_half, 1e-6))
            ppp_away_mu_half = float(max(ppp_away_mu_half, 1e-6))

            cov_from_poss_half = float(ppp_home_mu_half * ppp_away_mu_half * float(var_poss_half))
            cov_res_half = float(cov_target_half - cov_from_poss_half)
            var_shared_half = float(abs(cov_res_half))

            var_home_ind_target_half = float(max(var_home_target_half - var_shared_half, 1e-6))
            var_away_ind_target_half = float(max(var_away_target_half - var_shared_half, 1e-6))

            var_ppp_home_half = float(max((var_home_ind_target_half - (ppp_home_mu_half ** 2) * float(var_poss_half)) / float(max(e_poss_half, 1e-6)), 1e-6))
            var_ppp_away_half = float(max((var_away_ind_target_half - (ppp_away_mu_half ** 2) * float(var_poss_half)) / float(max(e_poss_half, 1e-6)), 1e-6))

            eps_home_half = rng.normal(0.0, np.sqrt(var_ppp_home_half) * np.sqrt(poss_half), size=samples)
            eps_away_half = rng.normal(0.0, np.sqrt(var_ppp_away_half) * np.sqrt(poss_half), size=samples)

            if var_shared_half > 0:
                shared_half = rng.normal(0.0, float(np.sqrt(var_shared_half)), size=samples)
                if cov_res_half >= 0:
                    sh_h = shared_half
                    sh_a = shared_half
                else:
                    sh_h = shared_half
                    sh_a = -shared_half
            else:
                sh_h = 0.0
                sh_a = 0.0

            home_half = poss_half * ppp_home_mu_half + eps_home_half + sh_h
            away_half = poss_half * ppp_away_mu_half + eps_away_half + sh_a
            home_half = np.clip(home_half, 0.0, None)
            away_half = np.clip(away_half, 0.0, None)
            home_half = np.rint(home_half)
            away_half = np.rint(away_half)
            totals_half = home_half + away_half
            margins_half = home_half - away_half

            meta = {
                "ppp_home_mu": float(ppp_home_mu_half),
                "ppp_away_mu": float(ppp_away_mu_half),
                "cov_target": float(cov_target_half),
                "cov_from_possessions": float(cov_from_poss_half),
                "cov_residual": float(cov_res_half),
            }
            return home_half, away_half, totals_half, margins_half, meta

        pace_mu_used = float(pace_mu)
        pace_sigma_used = float(pace_sigma) if pace_sigma is not None else DEFAULT_PACE_SIGMA
        pace_sigma_used = float(max(1e-6, pace_sigma_used))

        poss = rng.normal(pace_mu_used, pace_sigma_used, size=samples)
        poss = np.clip(poss, PACE_MIN, PACE_MAX)
        e_poss = float(max(pace_mu_used, 1e-6))
        var_poss = float(pace_sigma_used) ** 2

        # Simulate 1H + 2H jointly so that full-game totals are coherent with halves.
        frac_k = 80.0
        alpha = float(max(1.0, half_frac * frac_k))
        beta = float(max(1.0, (1.0 - half_frac) * frac_k))
        frac = rng.beta(alpha, beta, size=samples)
        frac = np.clip(frac, 0.25, 0.75)
        poss_1h = np.clip(poss * frac, PACE_MIN * 0.25, PACE_MAX * 0.75)
        poss_2h = np.clip(poss - poss_1h, 0.0, None)

        pace_mu_1h = float(pace_mu_used) * half_frac
        pace_mu_2h = float(pace_mu_used) * float(1.0 - half_frac)
        e_poss_1h = float(max(pace_mu_1h, 1e-6))
        e_poss_2h = float(max(pace_mu_2h, 1e-6))
        var_poss_1h = float(pace_sigma_used) ** 2 * float(max(half_frac, 1e-6))
        var_poss_2h = float(pace_sigma_used) ** 2 * float(max(1.0 - half_frac, 1e-6))

        # Make 2H mean/sigma the residual so halves add up to the calibrated full-game.
        mu_home_2h = float(max(float(mu_home) - float(mu_home_1h), 0.0))
        mu_away_2h = float(max(float(mu_away) - float(mu_away_1h), 0.0))

        sigma_total_full = float(max(1e-6, float(sigma_total)))
        sigma_margin_full = float(max(1e-6, float(sigma_margin))) if sigma_margin is not None else None
        var_total_full = float(sigma_total_full) ** 2
        var_total_1h = float(sigma_total_1h) ** 2
        var_total_2h = float(max(var_total_full - var_total_1h, 1e-6))
        sigma_total_2h = float(np.sqrt(var_total_2h))

        if sigma_margin_full is not None and sigma_margin_1h is not None:
            var_margin_full = float(sigma_margin_full) ** 2
            var_margin_1h = float(sigma_margin_1h) ** 2
            var_margin_2h = float(max(var_margin_full - var_margin_1h, 1e-6))
            sigma_margin_2h = float(np.sqrt(var_margin_2h))
        else:
            sigma_margin_2h = None

        home_1h, away_1h, totals_1h, margins_1h, meta_1h = _sim_half_from_possessions(
            poss_1h,
            e_poss_1h,
            var_poss_1h,
            float(mu_home_1h),
            float(mu_away_1h),
            float(sigma_total_1h),
            float(sigma_margin_1h) if sigma_margin_1h is not None else None,
        )
        home_2h, away_2h, totals_2h, margins_2h, meta_2h = _sim_half_from_possessions(
            poss_2h,
            e_poss_2h,
            var_poss_2h,
            float(mu_home_2h),
            float(mu_away_2h),
            float(sigma_total_2h),
            float(sigma_margin_2h) if sigma_margin_2h is not None else None,
        )

        home_pts = home_1h + home_2h
        away_pts = away_1h + away_2h
        totals = totals_1h + totals_2h
        margins = margins_1h + margins_2h

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

        q10_t1 = float(np.quantile(totals_1h, 0.10))
        q50_t1 = float(np.quantile(totals_1h, 0.50))
        q90_t1 = float(np.quantile(totals_1h, 0.90))
        q10_m1 = float(np.quantile(margins_1h, 0.10))
        q50_m1 = float(np.quantile(margins_1h, 0.50))
        q90_m1 = float(np.quantile(margins_1h, 0.90))

        q10_t2 = float(np.quantile(totals_2h, 0.10))
        q50_t2 = float(np.quantile(totals_2h, 0.50))
        q90_t2 = float(np.quantile(totals_2h, 0.90))
        q10_m2 = float(np.quantile(margins_2h, 0.10))
        q50_m2 = float(np.quantile(margins_2h, 0.50))
        q90_m2 = float(np.quantile(margins_2h, 0.90))

        market_total_1h = _resolve_market_total_1h(row)
        p_over_market_1h = None
        if market_total_1h is not None:
            p_over_market_1h = float(np.mean(totals_1h > market_total_1h))

        spread_home_1h = _resolve_spread_home_1h(row)
        p_cover_home_1h = None
        if spread_home_1h is not None:
            p_cover_home_1h = float(np.mean(margins_1h + float(spread_home_1h) > 0))

        # Full-game ATS cover probability when spread_home is available.
        spread_home = None
        try:
            for scol in [
                "spread_home",
                "closing_spread_home",
                "home_spread",
            ]:
                if scol in row and pd.notna(row[scol]):
                    spread_home = float(row[scol])
                    break
        except Exception:
            spread_home = None
        p_cover_home = None
        if spread_home is not None:
            try:
                p_cover_home = float(np.mean(margins + float(spread_home) > 0))
            except Exception:
                p_cover_home = None

        # Win probabilities derived from simulated margins.
        try:
            p_home_win = float(np.mean(margins > 0))
        except Exception:
            p_home_win = None
        try:
            p_home_win_1h = float(np.mean(margins_1h > 0))
        except Exception:
            p_home_win_1h = None

        segment_rows = None
        segments_grid_min_used = None
        abs_margin_proxy_source = None
        abs_margin_proxy_value = None
        try:
            try:
                grid_min_pts = _segments_grid_min_from_env()
            except Exception:
                grid_min_pts = 5
            segments_grid_min_used = int(grid_min_pts)

            try:
                if spread_home is not None and np.isfinite(float(spread_home)):
                    abs_margin_proxy_source = "spread"
                    abs_margin_proxy_value = abs(float(spread_home))
                else:
                    abs_margin_proxy_source = "expected"
                    abs_margin_proxy_value = abs(float(mu_home) - float(mu_away))
            except Exception:
                abs_margin_proxy_source = "expected"
                try:
                    abs_margin_proxy_value = abs(float(mu_home) - float(mu_away))
                except Exception:
                    abs_margin_proxy_value = None

            segment_rows = _segment_grid_quantiles_from_points(
                home_pts=home_pts,
                away_pts=away_pts,
                home_1h=home_1h,
                away_1h=away_1h,
                grid_min=int(grid_min_pts),
                abs_margin_proxy=abs_margin_proxy_value,
                seg_probs_half1=segment_probs_half1,
                seg_probs_half2=segment_probs_half2,
                rng=rng,
            )
        except Exception:
            segment_rows = None

        return {
            "sim_ok": True,
            "sim_method": "pace",
            "mean_source": (mean_source or "auto"),
            "mean_source_used": mean_source_used,
            "segments_grid_min": int(segments_grid_min_used) if segments_grid_min_used is not None else None,
            "segments_mode": "points_alloc",
            "abs_margin_proxy": float(abs_margin_proxy_value) if abs_margin_proxy_value is not None and np.isfinite(float(abs_margin_proxy_value)) else None,
            "abs_margin_proxy_source": abs_margin_proxy_source,
            "segment_probs_half1_len": _prob_vec_len(segment_probs_half1),
            "segment_probs_half2_len": _prob_vec_len(segment_probs_half2),
            "segment_probs_half1_hash": _hash_prob_vec_short(segment_probs_half1),
            "segment_probs_half2_hash": _hash_prob_vec_short(segment_probs_half2),
            **mean_trace,
            "pace_mu": pace_mu_used,
            "pace_sigma": pace_sigma_used,
            "rho_used": None,
            "sigma_total": float(sigma_total),
            "sigma_margin": float(sigma_margin) if sigma_margin is not None else None,
            "mu_home": float(np.mean(home_pts)),
            "mu_away": float(np.mean(away_pts)),
            "ppp_home_mu": float(meta_1h.get("ppp_home_mu", 0.0) + meta_2h.get("ppp_home_mu", 0.0)) / 2.0,
            "ppp_away_mu": float(meta_1h.get("ppp_away_mu", 0.0) + meta_2h.get("ppp_away_mu", 0.0)) / 2.0,
            "cov_target": float(meta_1h.get("cov_target", 0.0) + meta_2h.get("cov_target", 0.0)),
            "cov_from_possessions": float(meta_1h.get("cov_from_possessions", 0.0) + meta_2h.get("cov_from_possessions", 0.0)),
            "cov_residual": float(meta_1h.get("cov_residual", 0.0) + meta_2h.get("cov_residual", 0.0)),
            "mu_total": float(np.mean(totals)),
            "mu_margin": float(np.mean(margins)),
            "q10_total": q10_t,
            "q50_total": q50_t,
            "q90_total": q90_t,
            "q10_margin": q10_m,
            "q50_margin": q50_m,
            "q90_margin": q90_m,
            "pace_mu_1h": pace_mu_1h,
            "pace_sigma_1h": float(pace_sigma_used) * float(np.sqrt(half_frac)),
            "sigma_total_1h": float(sigma_total_1h),
            "sigma_margin_1h": float(sigma_margin_1h) if sigma_margin_1h is not None else None,
            "mu_home_1h": float(np.mean(home_1h)),
            "mu_away_1h": float(np.mean(away_1h)),
            "mu_total_1h": float(np.mean(totals_1h)),
            "mu_margin_1h": float(np.mean(margins_1h)),
            "q10_total_1h": q10_t1,
            "q50_total_1h": q50_t1,
            "q90_total_1h": q90_t1,
            "q10_margin_1h": q10_m1,
            "q50_margin_1h": q50_m1,
            "q90_margin_1h": q90_m1,
            "sigma_total_2h": float(sigma_total_2h),
            "sigma_margin_2h": float(sigma_margin_2h) if sigma_margin_2h is not None else None,
            "mu_total_2h": float(np.mean(totals_2h)),
            "mu_margin_2h": float(np.mean(margins_2h)),
            "q10_total_2h": q10_t2,
            "q50_total_2h": q50_t2,
            "q90_total_2h": q90_t2,
            "q10_margin_2h": q10_m2,
            "q50_margin_2h": q50_m2,
            "q90_margin_2h": q90_m2,
            "p_over_market_1h": p_over_market_1h,
            "market_total_1h": market_total_1h,
            "p_cover_home_1h": p_cover_home_1h,
            "spread_home_1h": spread_home_1h,
            "p_over_market": p_over_market,
            "market_total": market_total,
            "p_cover_home": p_cover_home,
            "spread_home": spread_home,
            "p_home_win": p_home_win,
            "p_home_win_1h": p_home_win_1h,
            "override_delta_total": float(applied.get("delta_total", 0.0)) if applied else 0.0,
            "override_delta_margin": float(applied.get("delta_margin", 0.0)) if applied else 0.0,
            "override_sigma_total_mult": float(applied.get("sigma_total_mult", 1.0)) if applied else 1.0,
            "override_sigma_margin_mult": float(applied.get("sigma_margin_mult", 1.0)) if applied else 1.0,
            "override_pace_mult": float(applied.get("pace_mult", 1.0)) if applied else 1.0,
            "calib_delta_total": float(calib_applied.get("delta_total", 0.0)) if calib_applied else 0.0,
            "calib_delta_margin": float(calib_applied.get("delta_margin", 0.0)) if calib_applied else 0.0,
            "calib_sigma_total_mult": float(calib_applied.get("sigma_total_mult", 1.0)) if calib_applied else 1.0,
            "calib_sigma_margin_mult": float(calib_applied.get("sigma_margin_mult", 1.0)) if calib_applied else 1.0,
            "calib_delta_total_1h": float(calib_delta_total_1h_applied),
            "calib_delta_margin_1h": float(calib_delta_margin_1h_applied),
            "calib_sigma_total_1h_mult": float(calib_sigma_total_1h_mult_applied),
            "calib_sigma_margin_1h_mult": float(calib_sigma_margin_1h_mult_applied),
            "_segments_rows": segment_rows,
        }

    if sigma_margin is not None:
        # Use both total and margin uncertainty (if available) to infer covariance.
        # total = H + A; margin = H - A
        # Var(total) = Var(H)+Var(A)+2Cov; Var(margin)=Var(H)+Var(A)-2Cov
        var_total = float(sigma_total) ** 2
        var_margin = float(sigma_margin) ** 2
        sum_var = max((var_total + var_margin) / 2.0, 1e-6)  # Var(H)+Var(A)

        # Split per-team variance proportional to expected scoring share.
        # This does not change Var(total) or Var(margin) (those depend only on sum_var and cov),
        # but yields more realistic home/away point distributions.
        mu_total = float(mu_home) + float(mu_away)
        w_home = float(mu_home) / float(max(mu_total, 1e-6))
        w_home = float(np.clip(w_home, 0.2, 0.8))
        var_home = float(max(sum_var * w_home, 1e-6))
        var_away = float(max(sum_var - var_home, 1e-6))
        cov = (var_total - var_margin) / 4.0
        # Clamp covariance to keep matrix PSD
        cov_max = float(np.sqrt(var_home * var_away))
        cov = float(np.clip(cov, -0.99 * cov_max, 0.99 * cov_max))
        cov_mat = np.array([[var_home, cov], [cov, var_away]], dtype=float)
        means = np.array([mu_home, mu_away], dtype=float)
        rho_used = float(cov / cov_max) if cov_max > 0 else 0.0
    else:
        # Infer equal per-team sigma from total variance and assumed correlation
        var_total = float(sigma_total) ** 2
        s2 = var_total / (2.0 * (1.0 + float(rho_eff)))
        sigma_team = float(np.sqrt(max(s2, 1e-6)))
        cov = float(rho_eff) * (sigma_team ** 2)
        cov_mat = np.array([[sigma_team ** 2, cov], [cov, sigma_team ** 2]], dtype=float)
        means = np.array([mu_home, mu_away], dtype=float)
        rho_used = float(rho_eff)

    sigma_home = float(np.sqrt(max(float(cov_mat[0, 0]), 1e-12)))
    sigma_away = float(np.sqrt(max(float(cov_mat[1, 1]), 1e-12)))

    try:
        samples_arr = rng.multivariate_normal(means, cov_mat, size=samples)
    except np.linalg.LinAlgError:
        # Fallback: independent normals
        samples_arr = np.column_stack([
            rng.normal(mu_home, sigma_home, size=samples),
            rng.normal(mu_away, sigma_away, size=samples),
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

    # Full-game ATS cover probability when spread_home is available.
    spread_home = None
    try:
        for scol in [
            "spread_home",
            "closing_spread_home",
            "home_spread",
        ]:
            if scol in row and pd.notna(row[scol]):
                spread_home = float(row[scol])
                break
    except Exception:
        spread_home = None
    p_cover_home = None
    if spread_home is not None:
        try:
            p_cover_home = float(np.mean(margins + float(spread_home) > 0))
        except Exception:
            p_cover_home = None

    # Full-game win probability derived from simulated margin.
    try:
        p_home_win = float(np.mean(margins > 0))
    except Exception:
        p_home_win = None

    # 1H points-only fallback: scale mean and sigma by half.
    means_1h = np.array([mu_home_1h, mu_away_1h], dtype=float)
    if sigma_margin_1h is not None:
        var_total_1h = float(sigma_total_1h) ** 2
        var_margin_1h = float(sigma_margin_1h) ** 2
        sum_var_1h = max((var_total_1h + var_margin_1h) / 2.0, 1e-6)
        mu_total_1h = float(mu_home_1h) + float(mu_away_1h)
        w_home_1h = float(mu_home_1h) / float(max(mu_total_1h, 1e-6))
        w_home_1h = float(np.clip(w_home_1h, 0.2, 0.8))
        var_home_1h = float(max(sum_var_1h * w_home_1h, 1e-6))
        var_away_1h = float(max(sum_var_1h - var_home_1h, 1e-6))
        cov_1h = (var_total_1h - var_margin_1h) / 4.0
        cov_max_1h = float(np.sqrt(var_home_1h * var_away_1h))
        cov_1h = float(np.clip(cov_1h, -0.99 * cov_max_1h, 0.99 * cov_max_1h))
        cov_mat_1h = np.array([[var_home_1h, cov_1h], [cov_1h, var_away_1h]], dtype=float)
    else:
        var_total_1h = float(sigma_total_1h) ** 2
        s2_1h = var_total_1h / (2.0 * (1.0 + float(rho_eff)))
        sigma_team_1h = float(np.sqrt(max(s2_1h, 1e-6)))
        cov_1h = float(rho_eff) * (sigma_team_1h ** 2)
        cov_mat_1h = np.array([[sigma_team_1h ** 2, cov_1h], [cov_1h, sigma_team_1h ** 2]], dtype=float)

    try:
        s1 = rng.multivariate_normal(means_1h, cov_mat_1h, size=samples)
    except Exception:
        s1 = np.column_stack([
            rng.normal(mu_home_1h, float(np.sqrt(cov_mat_1h[0, 0])), size=samples),
            rng.normal(mu_away_1h, float(np.sqrt(cov_mat_1h[1, 1])), size=samples),
        ])
    home_1h = np.clip(s1[:, 0], 0.0, None)
    away_1h = np.clip(s1[:, 1], 0.0, None)

    # Enforce coherence between 1H and full-game samples for point-allocation segments.
    # Without this, independent sampling can yield 1H > full, producing zeroed 2H
    # residuals and implausibly low 2H projections.
    try:
        home_1h = np.minimum(home_1h, home_pts)
        away_1h = np.minimum(away_1h, away_pts)
    except Exception:
        pass
    totals_1h = home_1h + away_1h
    margins_1h = home_1h - away_1h

    q10_t1 = float(np.quantile(totals_1h, 0.10))
    q50_t1 = float(np.quantile(totals_1h, 0.50))
    q90_t1 = float(np.quantile(totals_1h, 0.90))
    q10_m1 = float(np.quantile(margins_1h, 0.10))
    q50_m1 = float(np.quantile(margins_1h, 0.50))
    q90_m1 = float(np.quantile(margins_1h, 0.90))

    market_total_1h = _resolve_market_total_1h(row)
    p_over_market_1h = None
    if market_total_1h is not None:
        p_over_market_1h = float(np.mean(totals_1h > market_total_1h))

    spread_home_1h = _resolve_spread_home_1h(row)
    p_cover_home_1h = None
    if spread_home_1h is not None:
        p_cover_home_1h = float(np.mean(margins_1h + float(spread_home_1h) > 0))

    segment_rows = None
    segments_grid_min_used = None
    abs_margin_proxy_source = None
    abs_margin_proxy_value = None
    try:
        try:
            grid_min_pts = _segments_grid_min_from_env()
        except Exception:
            grid_min_pts = 5
        segments_grid_min_used = int(grid_min_pts)

        try:
            if spread_home is not None and np.isfinite(float(spread_home)):
                abs_margin_proxy_source = "spread"
                abs_margin_proxy_value = abs(float(spread_home))
            else:
                abs_margin_proxy_source = "expected"
                abs_margin_proxy_value = abs(float(mu_home) - float(mu_away))
        except Exception:
            abs_margin_proxy_source = "expected"
            try:
                abs_margin_proxy_value = abs(float(mu_home) - float(mu_away))
            except Exception:
                abs_margin_proxy_value = None

        segment_rows = _segment_grid_quantiles_from_points(
            home_pts=home_pts,
            away_pts=away_pts,
            home_1h=home_1h,
            away_1h=away_1h,
            grid_min=int(grid_min_pts),
            abs_margin_proxy=abs_margin_proxy_value,
            seg_probs_half1=segment_probs_half1,
            seg_probs_half2=segment_probs_half2,
            rng=rng,
        )
    except Exception:
        segment_rows = None

    return {
        "sim_ok": True,
        "sim_method": "points",
        "mean_source": (mean_source or "auto"),
        "mean_source_used": mean_source_used,
        "segments_grid_min": int(segments_grid_min_used) if segments_grid_min_used is not None else None,
        "segments_mode": "points_alloc",
        "abs_margin_proxy": float(abs_margin_proxy_value) if abs_margin_proxy_value is not None and np.isfinite(float(abs_margin_proxy_value)) else None,
        "abs_margin_proxy_source": abs_margin_proxy_source,
        "segment_probs_half1_len": _prob_vec_len(segment_probs_half1),
        "segment_probs_half2_len": _prob_vec_len(segment_probs_half2),
        "segment_probs_half1_hash": _hash_prob_vec_short(segment_probs_half1),
        "segment_probs_half2_hash": _hash_prob_vec_short(segment_probs_half2),
        **mean_trace,
        "rho_used": rho_used,
        "sigma_total": float(sigma_total),
        "sigma_margin": float(sigma_margin) if sigma_margin is not None else None,
        "mu_home": float(mu_home),
        "mu_away": float(mu_away),
        "mu_total": float(np.mean(totals)),
        "mu_margin": float(np.mean(margins)),
        "q10_total": q10_t,
        "q50_total": q50_t,
        "q90_total": q90_t,
        "q10_margin": q10_m,
        "q50_margin": q50_m,
        "q90_margin": q90_m,
        "sigma_total_1h": float(sigma_total_1h),
        "sigma_margin_1h": float(sigma_margin_1h) if sigma_margin_1h is not None else None,
        "mu_home_1h": float(np.mean(home_1h)),
        "mu_away_1h": float(np.mean(away_1h)),
        "mu_total_1h": float(np.mean(totals_1h)),
        "mu_margin_1h": float(np.mean(margins_1h)),
        "q10_total_1h": q10_t1,
        "q50_total_1h": q50_t1,
        "q90_total_1h": q90_t1,
        "q10_margin_1h": q10_m1,
        "q50_margin_1h": q50_m1,
        "q90_margin_1h": q90_m1,
        "p_over_market_1h": p_over_market_1h,
        "market_total_1h": market_total_1h,
        "p_cover_home_1h": p_cover_home_1h,
        "spread_home_1h": spread_home_1h,
        "p_over_market": p_over_market,
        "market_total": market_total,
        "p_cover_home": p_cover_home,
        "spread_home": spread_home,
        "p_home_win": p_home_win,
        "override_delta_total": float(applied.get("delta_total", 0.0)) if applied else 0.0,
        "override_delta_margin": float(applied.get("delta_margin", 0.0)) if applied else 0.0,
        "override_sigma_total_mult": float(applied.get("sigma_total_mult", 1.0)) if applied else 1.0,
        "override_sigma_margin_mult": float(applied.get("sigma_margin_mult", 1.0)) if applied else 1.0,
        "override_pace_mult": float(applied.get("pace_mult", 1.0)) if applied else 1.0,
        "calib_delta_total": float(calib_applied.get("delta_total", 0.0)) if calib_applied else 0.0,
        "calib_delta_margin": float(calib_applied.get("delta_margin", 0.0)) if calib_applied else 0.0,
        "calib_sigma_total_mult": float(calib_applied.get("sigma_total_mult", 1.0)) if calib_applied else 1.0,
        "calib_sigma_margin_mult": float(calib_applied.get("sigma_margin_mult", 1.0)) if calib_applied else 1.0,
        "calib_delta_total_1h": float(calib_delta_total_1h_applied),
        "calib_delta_margin_1h": float(calib_delta_margin_1h_applied),
        "calib_sigma_total_1h_mult": float(calib_sigma_total_1h_mult_applied),
        "calib_sigma_margin_1h_mult": float(calib_sigma_margin_1h_mult_applied),
        "_segments_rows": segment_rows,
    }


def run_simulations_for_date(out_dir: Path, date: str,
                             preds_path: Optional[Path] = None,
                             lines_path: Optional[Path] = None,
                             samples: int = DEFAULT_SAMPLES,
                             rho: float = DEFAULT_RHO,
                             use_pace: Optional[bool] = None,
                             pace_sigma: float = DEFAULT_PACE_SIGMA,
                             injuries_path: Optional[Path] = None,
                             seed: Optional[int] = None,
                             mean_source: str = "auto",
                             allow_market_guardrails: bool = True,
                             engine: str = "auto",
                             quantiles_out_prefix: str = "sim_quantiles_",
                             segments_out_prefix: str = "sim_segments_",
                             meta_out_prefix: str = "sim_meta_") -> Path:
    out_dir = Path(out_dir)
    # Simulation inputs default to the unified enriched rows for a given date.
    # Those rows carry the market-derived mean total + spread-derived margin plus
    # optional uncertainty columns used to derive sigmas.
    enr_path = out_dir / f"predictions_unified_enriched_{date}.csv"
    games_path = out_dir / f"games_{date}.csv"

    # Prefer the primary per-date predictions artifact when present.
    # In practice, the unified/display artifacts can be heavily sanitized (near-constant
    # means across the slate), which collapses per-game segment trajectories and breaks
    # 5-minute backtests.
    default_preds_path = out_dir / f"predictions_{date}.csv"

    if preds_path is None:
        # Choose the best available default inputs.
        for cand in [default_preds_path, enr_path]:
            if cand.exists():
                preds_path = cand
                break
        if preds_path is None:
            preds_path = enr_path
    if lines_path is None:
        # Use rolling last-odds file by default; per-date odds can be sparse.
        lines_path = out_dir / "games_with_last.csv"

    calib_path = out_dir / "sim_calibration.json"
    sim_calibration, sim_calibration_load_error = _load_sim_calibration(calib_path)
    # If the file exists but couldn't be read/parsed (e.g., transient write), retry once.
    if (not sim_calibration) and sim_calibration_load_error and calib_path.exists():
        try:
            import time

            time.sleep(0.05)
        except Exception:
            pass
        sim_calibration2, sim_calibration_load_error2 = _load_sim_calibration(calib_path)
        if sim_calibration2:
            sim_calibration = sim_calibration2
            sim_calibration_load_error = sim_calibration_load_error2

    # Effective rho can be overridden via sim_calibration.json (used for fallback
    # covariance construction when sigma_margin is missing).
    rho_eff = float(rho)
    try:
        if isinstance(sim_calibration, dict) and "rho" in sim_calibration and sim_calibration.get("rho") is not None:
            rho_eff = float(sim_calibration.get("rho"))
    except Exception:
        rho_eff = float(rho)

    def _stable_u32_from_str(s: str) -> int:
        try:
            h = hashlib.sha256(s.encode("utf-8", errors="ignore")).digest()
            return int.from_bytes(h[:4], byteorder="little", signed=False)
        except Exception:
            return 0

    if seed is None:
        try:
            import os
            senv = (os.environ.get("NCAAB_SIM_SEED") or "").strip()
            seed = int(senv) if senv else None
        except Exception:
            seed = None
    if seed is None:
        # Predictable default: stable seed per date.
        seed = _stable_u32_from_str(f"date:{date}")

    if engine is None or str(engine).strip() == "":
        engine = "auto"
    try:
        import os
        env_engine = (os.environ.get("NCAAB_SIM_ENGINE") or "").strip()
        if env_engine:
            engine = env_engine
    except Exception:
        pass

    used_fallback_preds = False
    if not preds_path.exists():
        # Backtest/regeneration robustness: historical runs may not have all
        # intermediate artifacts persisted.
        fallback_candidates = [
            default_preds_path,
            enr_path,
            out_dir / f"predictions_unified_{date}.csv",
            out_dir / f"predictions_enriched_{date}.csv",
        ]
        alt = next((p for p in fallback_candidates if p.exists()), None)
        if alt is not None:
            preds_path = alt
            used_fallback_preds = True
        else:
            raise FileNotFoundError(
                f"Predictions file not found for date={date}. Tried: {', '.join(str(p) for p in fallback_candidates)}"
            )

    preds = pd.read_csv(preds_path)
    if "date" in preds.columns:
        preds = preds[preds["date"].astype(str) == str(date)]

    # If we had to fall back from a missing default, do not overwrite the
    # unified-enriched artifact: it may intentionally contain additional columns.

    # Normalize id dtype for stable merges/output
    if "game_id" in preds.columns:
        preds["game_id"] = preds["game_id"].map(_to_game_id_str)

    # Enrich with per-team features when available (drives feature-based/event sims)
    tf_path = out_dir / "team_features.csv"
    if tf_path.exists() and {"home_team", "away_team"}.issubset(preds.columns):
        try:
            tf = pd.read_csv(tf_path)
            if "date" in tf.columns:
                tf = tf[tf["date"].astype(str) == str(date)]
            if "team" in tf.columns:
                tf["_team_norm"] = tf["team"].map(_norm_team_key)
            preds["_home_norm"] = preds["home_team"].map(_norm_team_key)
            preds["_away_norm"] = preds["away_team"].map(_norm_team_key)

            keep_tf = [
                "_team_norm",
                "season_off_ppg",
                "season_def_ppg",
                "season_margin_std",
                "season_total_std",
                "rest_days",
                "back_to_back",
                "ewm_off_ppg",
                "ewm_def_ppg",
                "ewm_margin_avg",
            ]
            keep_tf = [c for c in keep_tf if c in tf.columns]
            tf_small = tf[keep_tf].drop_duplicates(subset=["_team_norm"]) if keep_tf else None
            if tf_small is not None and not tf_small.empty:
                home_tf = tf_small.rename(columns={c: f"home_team_{c}" for c in tf_small.columns if c != "_team_norm"})
                home_tf["_home_norm"] = tf_small["_team_norm"].values
                away_tf = tf_small.rename(columns={c: f"away_team_{c}" for c in tf_small.columns if c != "_team_norm"})
                away_tf["_away_norm"] = tf_small["_team_norm"].values
                preds = preds.merge(home_tf, on="_home_norm", how="left")
                preds = preds.merge(away_tf, on="_away_norm", how="left")

            # Derive PPP-style columns expected by the feature-only mean constructor.
            # Use the best available possession estimate.
            poss = None
            for c in ["pace_game_est", "possessions_game_est"]:
                if c in preds.columns:
                    poss = pd.to_numeric(preds[c], errors="coerce")
                    break
            if poss is None:
                ph = pd.to_numeric(preds.get("possessions_home_est"), errors="coerce")
                pa = pd.to_numeric(preds.get("possessions_away_est"), errors="coerce")
                if ph is not None and pa is not None:
                    poss = (ph + pa) / 2.0
            if poss is not None:
                poss = poss.clip(lower=PACE_MIN, upper=PACE_MAX)

                def _pick_ppg(prefix: str, which: str) -> pd.Series:
                    for k in [f"{prefix}_ewm_{which}", f"{prefix}_season_{which}"]:
                        if k in preds.columns:
                            return pd.to_numeric(preds[k], errors="coerce")
                    return pd.Series(np.nan, index=preds.index)

                home_off = _pick_ppg("home_team", "off_ppg")
                away_off = _pick_ppg("away_team", "off_ppg")
                home_def = _pick_ppg("home_team", "def_ppg")
                away_def = _pick_ppg("away_team", "def_ppg")

                # PPP means
                preds["home_ppp_mu"] = (home_off / poss).clip(lower=0.75, upper=1.35)
                preds["away_ppp_mu"] = (away_off / poss).clip(lower=0.75, upper=1.35)
                preds["home_ppp_allowed_mu"] = (home_def / poss).clip(lower=0.75, upper=1.35)
                preds["away_ppp_allowed_mu"] = (away_def / poss).clip(lower=0.75, upper=1.35)
        except Exception:
            pass

    # Derive rolling event-rate features from historical boxscores when available.
    # This feeds the event-driven simulator with empirical TO/3P/FTA distributions.
    try:
        import os
        from src.augment_features import augment_boxscores

        lookback_games = 15
        try:
            lg = (os.environ.get("NCAAB_SIM_EVENT_LOOKBACK_GAMES") or "").strip()
            if lg:
                lookback_games = int(lg)
        except Exception:
            lookback_games = 15
        lookback_games = int(np.clip(lookback_games, 5, 60))

        bs_path = None
        for cand in [out_dir / "boxscores.csv", out_dir / "boxscores_last2.csv"]:
            if cand.exists():
                bs_path = cand
                break

        if bs_path is not None and {"home_team", "away_team"}.issubset(preds.columns):
            bs = pd.read_csv(bs_path)
            if "date" in bs.columns:
                bs["date"] = pd.to_datetime(bs["date"], errors="coerce")
            aug = augment_boxscores(bs)
            if not aug.empty and "date" in aug.columns:
                aug["date"] = pd.to_datetime(aug["date"], errors="coerce")
                cutoff = pd.to_datetime(date, errors="coerce")
                if pd.notna(cutoff):
                    # IMPORTANT: avoid same-day leakage. When simulating date D,
                    # only use boxscores strictly before D.
                    aug = aug[aug["date"] < cutoff]

            def _longify(side: str) -> pd.DataFrame:
                cols = {
                    "team": f"{side}_team",
                    "pace": f"{side}_pace",
                    "to_rate": f"{side}_to_rate",
                    "3p_rate": f"{side}_3p_rate",
                    "fta_rate": f"{side}_fta_rate",
                }
                base_cols = ["date", "home_team", "away_team"]
                use = [c for c in base_cols if c in aug.columns]
                for k in cols.values():
                    if k in aug.columns:
                        use.append(k)
                sub = aug[use].copy() if use else pd.DataFrame()
                if sub.empty or cols["team"] not in sub.columns:
                    return pd.DataFrame()
                out = pd.DataFrame(
                    {
                        "date": pd.to_datetime(sub.get("date"), errors="coerce"),
                        "team": sub[cols["team"]].map(_norm_team_key),
                        "pace": pd.to_numeric(sub.get(cols["pace"]), errors="coerce"),
                        "to_rate": pd.to_numeric(sub.get(cols["to_rate"]), errors="coerce"),
                        "three_rate": pd.to_numeric(sub.get(cols["3p_rate"]), errors="coerce"),
                        "fta_rate": pd.to_numeric(sub.get(cols["fta_rate"]), errors="coerce"),
                    }
                )
                return out.dropna(subset=["team"]).copy()

            long = pd.concat([_longify("home"), _longify("away")], ignore_index=True)
            if not long.empty:
                long = long.sort_values(["team", "date"])
                def _tail_mean(g: pd.DataFrame) -> pd.Series:
                    gg = g.tail(lookback_games)
                    return pd.Series(
                        {
                            "event_pace": float(pd.to_numeric(gg["pace"], errors="coerce").dropna().mean()) if "pace" in gg.columns else np.nan,
                            "event_to_rate": float(pd.to_numeric(gg["to_rate"], errors="coerce").dropna().mean()) if "to_rate" in gg.columns else np.nan,
                            "event_3p_rate": float(pd.to_numeric(gg["three_rate"], errors="coerce").dropna().mean()) if "three_rate" in gg.columns else np.nan,
                            "event_fta_rate": float(pd.to_numeric(gg["fta_rate"], errors="coerce").dropna().mean()) if "fta_rate" in gg.columns else np.nan,
                            "event_games": int(len(gg)),
                        }
                    )

                team_rates = long.groupby("team", dropna=True).apply(_tail_mean).reset_index()
                preds["_home_norm"] = preds["home_team"].map(_norm_team_key)
                preds["_away_norm"] = preds["away_team"].map(_norm_team_key)

                h = team_rates.rename(
                    columns={
                        "team": "_home_norm",
                        "event_pace": "home_team_event_pace",
                        "event_to_rate": "home_team_event_to_rate",
                        "event_3p_rate": "home_team_event_3p_rate",
                        "event_fta_rate": "home_team_event_fta_rate",
                        "event_games": "home_team_event_games",
                    }
                )
                a = team_rates.rename(
                    columns={
                        "team": "_away_norm",
                        "event_pace": "away_team_event_pace",
                        "event_to_rate": "away_team_event_to_rate",
                        "event_3p_rate": "away_team_event_3p_rate",
                        "event_fta_rate": "away_team_event_fta_rate",
                        "event_games": "away_team_event_games",
                    }
                )
                preds = preds.merge(h, on="_home_norm", how="left")
                preds = preds.merge(a, on="_away_norm", how="left")
    except Exception:
        pass

    # If we still don't have team metadata, try games_<date>.csv.
    if ("home_team" not in preds.columns or "away_team" not in preds.columns) and games_path.exists():
        try:
            g = pd.read_csv(games_path)
            if "date" in g.columns:
                g = g[g["date"].astype(str) == str(date)]
            if "game_id" in g.columns:
                g["game_id"] = g["game_id"].map(_to_game_id_str)
            keep = [c for c in [
                "game_id",
                "home_team",
                "away_team",
                "start_time_iso",
                "start_time_local",
                "start_tz_abbr",
                "neutral_site",
                "venue",
                "home_tempo_rating",
                "away_tempo_rating",
                "tempo_rating_sum",
            ] if c in g.columns]
            if keep and "game_id" in preds.columns:
                preds = preds.merge(g[keep].drop_duplicates(subset=["game_id"]), on="game_id", how="left")
        except Exception:
            pass

    # If no predictions for the date, write a header-only CSV to avoid empty-file parse errors downstream
    if preds.shape[0] == 0:
        out_path = out_dir / f"{quantiles_out_prefix}{date}.csv"
        header_cols = [
            "date","game_id","home_team","away_team",
            "sim_ok","mu_total","mu_margin",
            "q10_total","q50_total","q90_total",
            "q10_margin","q50_margin","q90_margin",
            "p_home_win","p_cover_home",
            "p_over_market","market_total",
        ]
        pd.DataFrame(columns=header_cols).to_csv(out_path, index=False)
        return out_path

    # Try to enrich with market lines/odds (optional)
    if lines_path.exists():
        try:
            market_df = pd.read_csv(lines_path, low_memory=False)

            if "date" in market_df.columns:
                market_df = market_df[market_df["date"].astype(str) == str(date)]
            elif "date_game" in market_df.columns:
                market_df = market_df[market_df["date_game"].astype(str) == str(date)]

            if "game_id" in market_df.columns:
                market_df["game_id"] = market_df["game_id"].map(_to_game_id_str)

            # Normalize common line columns across sources
            if "market_total" not in market_df.columns:
                if "total" in market_df.columns:
                    market_df["market_total"] = market_df["total"]
                elif "closing_total" in market_df.columns:
                    market_df["market_total"] = market_df["closing_total"]

            if "spread_home" not in market_df.columns:
                if "home_spread" in market_df.columns:
                    market_df["spread_home"] = market_df["home_spread"]
                elif "closing_spread_home" in market_df.columns:
                    market_df["spread_home"] = market_df["closing_spread_home"]

            if "game_id" in preds.columns and "game_id" in market_df.columns:
                def _agg_first(s: pd.Series):
                    s2 = s.dropna()
                    return s2.iloc[0] if len(s2) else np.nan

                def _agg_median_num(s: pd.Series):
                    s2 = pd.to_numeric(s, errors="coerce").dropna()
                    return float(s2.median()) if len(s2) else np.nan

                first_fields = [
                    "home_team",
                    "away_team",
                    "start_time_iso",
                    "start_time_local",
                    "start_tz_abbr",
                    "neutral_site",
                    "venue",
                ]
                num_fields = [
                    "market_total",
                    "spread_home",
                    "moneyline_home",
                    "moneyline_away",
                    "home_spread_price",
                    "away_spread_price",
                    "over_price",
                    "under_price",
                    "market_total_1h",
                    "spread_home_1h",
                ]

                def _needs_fill(col: str, numeric: bool) -> bool:
                    if col not in preds.columns:
                        return True
                    try:
                        s = preds[col]
                        if numeric:
                            return int(pd.to_numeric(s, errors="coerce").notna().sum()) == 0
                        # string-ish: consider empty/"nan" as missing
                        sv = s.astype(str).str.strip()
                        sv = sv.replace({"nan": "", "NaN": "", "None": "", "NULL": "", "null": ""})
                        return int((sv != "").sum()) == 0
                    except Exception:
                        return True

                select_fields: list[str] = []
                agg_spec: dict[str, object] = {}

                rename_map: dict[str, str] = {}
                for c in first_fields:
                    if c in market_df.columns and _needs_fill(c, numeric=False):
                        select_fields.append(c)
                        agg_spec[c] = _agg_first
                        if c in preds.columns:
                            rename_map[c] = f"{c}__mkt"

                for c in num_fields:
                    if c in market_df.columns and _needs_fill(c, numeric=True):
                        select_fields.append(c)
                        agg_spec[c] = _agg_median_num
                        if c in preds.columns:
                            rename_map[c] = f"{c}__mkt"

                if select_fields:
                    m = market_df[["game_id"] + select_fields].copy()
                    m = m.dropna(subset=["game_id"])
                    m = m.groupby("game_id", as_index=False).agg(agg_spec)
                    if rename_map:
                        m = m.rename(columns=rename_map)
                    preds = preds.merge(m, on="game_id", how="left")
                    # Fill any existing-but-empty fields from market-derived columns
                    for src, dst in rename_map.items():
                        if (src in preds.columns) and (dst in preds.columns):
                            try:
                                if src in num_fields:
                                    base = pd.to_numeric(preds[src], errors="coerce")
                                    alt = pd.to_numeric(preds[dst], errors="coerce")
                                    preds[src] = base.where(base.notna(), alt)
                                else:
                                    base = preds[src]
                                    alt = preds[dst]
                                    preds[src] = base.where(base.notna() & (base.astype(str).str.strip() != ""), alt)
                            except Exception:
                                pass
                    # Drop helper columns
                    try:
                        for dst in rename_map.values():
                            if dst in preds.columns:
                                preds = preds.drop(columns=[dst])
                    except Exception:
                        pass
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
                    join_cols = ["_h", "_a"]
                    cols_to_keep = ["market_total", "home_team", "away_team"]
                    preds = preds.merge(
                        market_df[join_cols + [c for c in cols_to_keep if c in market_df.columns]],
                        on=join_cols,
                        how="left",
                    )
        except Exception:
            pass

    # Merge tempo/off/def features if present (enables pace simulation)
    try:
        feats_path = out_dir / f"features_{date}.csv"
        if feats_path.exists() and "game_id" in preds.columns:
            feats = pd.read_csv(feats_path)
            feats["game_id"] = feats["game_id"].map(_to_game_id_str)
            keep = [
                "game_id",
                # Pace/possessions estimates (enables pace-based simulation)
                "pace_game_est",
                "possessions_game_est",
                "pace_sigma_game_est",
                "home_off_rating",
                "away_off_rating",
                "home_def_rating",
                "away_def_rating",
                "home_tempo_rating",
                "away_tempo_rating",
                "tempo_rating_sum",
                "rest_home",
                "rest_away",
                "b2b_home",
                "b2b_away",
                "neutral_site",
            ]
            keep = [c for c in keep if c in feats.columns]
            feats = feats[keep].drop_duplicates(subset=["game_id"])
            preds = preds.merge(feats, on="game_id", how="left")
    except Exception:
        pass

    # Derive 1H market total/spread when absent (used for 1H sim market probabilities).
    try:
        if "market_total_1h" not in preds.columns:
            preds["market_total_1h"] = np.nan
        if "spread_home_1h" not in preds.columns:
            preds["spread_home_1h"] = np.nan

        mt_full = pd.to_numeric(preds.get("market_total"), errors="coerce")
        sp_full = pd.to_numeric(preds.get("spread_home"), errors="coerce")

        # projection-based halftime scoring ratio when available; else 0.5
        if {"proj_home", "proj_away", "proj_home_1h", "proj_away_1h"}.issubset(preds.columns):
            denom = pd.to_numeric(preds["proj_home"], errors="coerce") + pd.to_numeric(preds["proj_away"], errors="coerce")
            numer = pd.to_numeric(preds["proj_home_1h"], errors="coerce") + pd.to_numeric(preds["proj_away_1h"], errors="coerce")
            hratio = (numer / denom.replace(0, np.nan)).clip(lower=0.35, upper=0.65).fillna(0.5)
        else:
            hratio = pd.Series(0.5, index=preds.index)

        mt1 = pd.to_numeric(preds.get("market_total_1h"), errors="coerce")
        sp1 = pd.to_numeric(preds.get("spread_home_1h"), errors="coerce")
        preds["market_total_1h"] = mt1.where(mt1.notna(), mt_full * hratio)
        preds["spread_home_1h"] = sp1.where(sp1.notna(), sp_full * 0.5)
    except Exception:
        pass

    # Determine whether to run pace simulation.
    if use_pace is None:
        use_pace = bool(
            any(c in preds.columns for c in ["pace_game_est", "possessions_game_est", "home_tempo_rating", "away_tempo_rating", "tempo_rating_sum"])
        )

    if injuries_path is None:
        injuries_path = Path("data") / "injuries_overrides.csv"
    injury_overrides = _load_injury_overrides(injuries_path, date) if injuries_path else {}

    # Simulate per row
    results = []
    segment_rows_all: list[dict] = []

    try:
        grid_min_cfg = int(_segments_grid_min_from_env())
    except Exception:
        grid_min_cfg = 5

    # Optional tuned segment weights.
    # - 5-min grid: 4 segments per half.
    # - 2-min grid: either 10 segments per half (direct) or 4 (expanded internally).
    seg_probs_half1 = None
    seg_probs_half2 = None
    team_seg_global_h1 = None
    team_seg_global_h2 = None
    team_seg_by_team: dict[str, dict] = {}
    seg_weights_path_used = None
    seg_weights_load_error = None
    team_weights_path_used = None
    team_weights_load_error = None
    try:
        import os

        wpath = (os.environ.get("NCAAB_SEGMENT_WEIGHTS_PATH") or "").strip()
        candidates: list[Path] = []
        if wpath:
            try:
                candidates.append(Path(wpath))
            except Exception:
                pass
        candidates.extend([Path(out_dir) / "segment_weights.json", Path("data") / "segment_weights.json"])

        wobj = None
        for p in candidates:
            try:
                if p.exists():
                    wobj = json.loads(p.read_text(encoding="utf-8"))
                    seg_weights_path_used = str(p)
                    break
            except Exception as e:
                if seg_weights_load_error is None:
                    seg_weights_load_error = f"{p}: {repr(e)}"
                continue

        if wobj is None and seg_weights_load_error is None:
            seg_weights_load_error = "segment_weights_not_found"

        if isinstance(wobj, dict):
            h1 = wobj.get("half1")
            h2 = wobj.get("half2")
            if isinstance(h1, list) and len(h1) in {4, 10}:
                if int(grid_min_cfg) == 2 or len(h1) == 4:
                    seg_probs_half1 = np.asarray(h1, dtype=float)
            if isinstance(h2, list) and len(h2) in {4, 10}:
                if int(grid_min_cfg) == 2 or len(h2) == 4:
                    seg_probs_half2 = np.asarray(h2, dtype=float)
    except Exception as e:
        try:
            seg_weights_load_error = repr(e)
        except Exception:
            seg_weights_load_error = "segment_weights_load_failed"

    # Optional per-team tuned 2-min segment weights.
    # Format: {"global": {"half1": [10], "half2": [10]}, "teams": {"team": {"half1": [10], "half2": [10]}}}
    # Can be disabled via env for A/B testing.
    if int(grid_min_cfg) == 2:
        try:
            import os

            # Back-compat / convenience flag: if explicitly provided and falsey, disable.
            # (Primary switch remains NCAAB_DISABLE_TEAM_SEGMENT_WEIGHTS.)
            try:
                raw_use = (os.environ.get("NCAAB_USE_TEAM_SEGMENT_WEIGHTS_2MIN") or "").strip()
                if raw_use and (not _safe_bool(raw_use)):
                    team_weights_load_error = "team_segment_weights_disabled"
                    raise StopIteration("skip")
            except StopIteration:
                raise
            except Exception:
                pass

            if _safe_bool(os.environ.get("NCAAB_DISABLE_TEAM_SEGMENT_WEIGHTS")):
                team_weights_load_error = "team_segment_weights_disabled"
                raise StopIteration("skip")

            twpath = (os.environ.get("NCAAB_TEAM_SEGMENT_WEIGHTS_2MIN_PATH") or "").strip()
            candidates_tw: list[Path] = []
            if twpath:
                try:
                    candidates_tw.append(Path(twpath))
                except Exception:
                    pass
            candidates_tw.extend([
                Path(out_dir) / "team_segment_weights_2min.json",
                Path("data") / "team_segment_weights_2min.json",
            ])

            twobj = None
            for p in candidates_tw:
                try:
                    if p.exists():
                        twobj = json.loads(p.read_text(encoding="utf-8"))
                        team_weights_path_used = str(p)
                        break
                except Exception as e:
                    if team_weights_load_error is None:
                        team_weights_load_error = f"{p}: {repr(e)}"
                    continue

            if twobj is None and team_weights_load_error is None:
                team_weights_load_error = "team_segment_weights_not_found"

            if isinstance(twobj, dict):
                g = twobj.get("global")
                teams = twobj.get("teams")
                if isinstance(g, dict):
                    gh1 = g.get("half1")
                    gh2 = g.get("half2")
                    if isinstance(gh1, list) and len(gh1) == 10:
                        team_seg_global_h1 = np.asarray(gh1, dtype=float)
                    if isinstance(gh2, list) and len(gh2) == 10:
                        team_seg_global_h2 = np.asarray(gh2, dtype=float)
                if isinstance(teams, dict):
                    # keys are already normalized lowercase in the tuner
                    team_seg_by_team = {str(k).strip().lower(): v for k, v in teams.items() if k is not None}
        except StopIteration:
            # Expected skip when disabled.
            pass
        except Exception as e:
            if team_weights_load_error is None:
                try:
                    team_weights_load_error = repr(e)
                except Exception:
                    team_weights_load_error = "team_segment_weights_load_failed"
    id_col, home_col, away_col = _resolve_keys(preds)
    for _, r in preds.iterrows():

        def _pick_first_nonnull(*keys: str):
            for k in keys:
                if k in r and pd.notna(r.get(k)):
                    return r.get(k)
            for k in keys:
                if k in r:
                    return r.get(k)
            return None

        # Order-independent per-game RNG: stable across runs even if row order changes.
        gid_s = _to_game_id_str(r.get(id_col))
        game_seed = _stable_u32_from_str(f"{seed}:{gid_s}")
        rng = np.random.default_rng(game_seed)

        seg_probs_half1_row = seg_probs_half1
        seg_probs_half2_row = seg_probs_half2
        seg_probs_source = "base"
        if int(grid_min_cfg) == 2 and team_seg_global_h1 is not None and team_seg_global_h2 is not None:
            try:
                ht_key = _norm_team_key(r.get("home_team") if "home_team" in preds.columns else r.get(home_col))
                at_key = _norm_team_key(r.get("away_team") if "away_team" in preds.columns else r.get(away_col))

                def _get_team_half(team_key: Optional[str], half_key: str, gvec: np.ndarray) -> np.ndarray:
                    if not team_key:
                        return gvec
                    t = team_seg_by_team.get(str(team_key).strip().lower())
                    if isinstance(t, dict):
                        v = t.get(half_key)
                        if isinstance(v, list) and len(v) == 10:
                            return np.asarray(v, dtype=float)
                    return gvec

                h1_home = _get_team_half(ht_key, "half1", team_seg_global_h1)
                h1_away = _get_team_half(at_key, "half1", team_seg_global_h1)
                h2_home = _get_team_half(ht_key, "half2", team_seg_global_h2)
                h2_away = _get_team_half(at_key, "half2", team_seg_global_h2)

                w1 = 0.5 * (h1_home + h1_away)
                w2 = 0.5 * (h2_home + h2_away)
                w1 = np.where(np.isfinite(w1), w1, 0.0)
                w2 = np.where(np.isfinite(w2), w2, 0.0)
                w1 = np.clip(w1, 0.0, None)
                w2 = np.clip(w2, 0.0, None)
                s1 = float(w1.sum())
                s2 = float(w2.sum())
                if s1 > 0:
                    w1 = (w1 / s1).astype(float)
                if s2 > 0:
                    w2 = (w2 / s2).astype(float)
                seg_probs_half1_row = w1
                seg_probs_half2_row = w2
                seg_probs_source = "team"
            except Exception:
                seg_probs_half1_row = seg_probs_half1
                seg_probs_half2_row = seg_probs_half2
                seg_probs_source = "base"

        # Observability: capture which vectors we actually passed for this row.
        seg_h1_hash = _hash_prob_vec_short(seg_probs_half1_row)
        seg_h2_hash = _hash_prob_vec_short(seg_probs_half2_row)
        seg_h1_len = _prob_vec_len(seg_probs_half1_row)
        seg_h2_len = _prob_vec_len(seg_probs_half2_row)
        sim_res = simulate_game_row(
            r,
            rho=rho_eff,
            samples=samples,
            use_pace=bool(use_pace),
            pace_sigma=pace_sigma,
            injury_overrides=injury_overrides,
            sim_calibration=sim_calibration,
            rng=rng,
            mean_source=mean_source,
            allow_market_guardrails=allow_market_guardrails,
            engine=engine,
            segment_probs_half1=seg_probs_half1_row,
            segment_probs_half2=seg_probs_half2_row,
        )

        abs_margin_proxy_value = sim_res.get("abs_margin_proxy") if isinstance(sim_res, dict) else None
        abs_margin_proxy_source = sim_res.get("abs_margin_proxy_source") if isinstance(sim_res, dict) else None

        seg_rows = None
        try:
            seg_rows = sim_res.pop("_segments_rows", None)
        except Exception:
            seg_rows = None

        if seg_rows:
            base_seg = {
                "date": date,
                "game_id": gid_s,
                "home_team": r.get("home_team") if "home_team" in preds.columns else r.get(home_col),
                "away_team": r.get("away_team") if "away_team" in preds.columns else r.get(away_col),
                "start_time_local": r.get("start_time_local"),
                "start_time_iso": r.get("start_time_iso"),
                "start_tz_abbr": r.get("start_tz_abbr"),
                "abs_margin_proxy": abs_margin_proxy_value,
                "abs_margin_proxy_source": abs_margin_proxy_source,
                "segment_probs_source": seg_probs_source,
                "segment_probs_half1_len_passed": seg_h1_len,
                "segment_probs_half2_len_passed": seg_h2_len,
                "segment_probs_half1_hash_passed": seg_h1_hash,
                "segment_probs_half2_hash_passed": seg_h2_hash,
            }
            for sr in seg_rows:
                if isinstance(sr, dict):
                    segment_rows_all.append({**base_seg, **sr})
        out = {
            "date": date,
            "game_id": gid_s,
            "sim_seed": int(seed) if seed is not None else None,
            "sim_game_seed": int(game_seed) if game_seed is not None else None,
            "home_team": r.get("home_team") if "home_team" in preds.columns else r.get(home_col),
            "away_team": r.get("away_team") if "away_team" in preds.columns else r.get(away_col),
            "start_time_iso": r.get("start_time_iso"),
            "start_time_local": r.get("start_time_local"),
            "start_tz_abbr": r.get("start_tz_abbr"),
            "segments_grid_min_cfg": int(grid_min_cfg),
            "segment_weights_path_used": seg_weights_path_used,
            "segment_weights_load_error": seg_weights_load_error,
            "team_segment_weights_path_used": team_weights_path_used,
            "team_segment_weights_load_error": team_weights_load_error,
            "segment_probs_source": seg_probs_source,
            "segment_probs_half1_len_passed": seg_h1_len,
            "segment_probs_half2_len_passed": seg_h2_len,
            "segment_probs_half1_hash_passed": seg_h1_hash,
            "segment_probs_half2_hash_passed": seg_h2_hash,
            "neutral_site": _pick_first_nonnull("neutral_site", "neutral_site_y", "neutral_site_x"),
            "venue": r.get("venue"),
            "rest_home": r.get("rest_home"),
            "rest_away": r.get("rest_away"),
            "b2b_home": r.get("b2b_home"),
            "b2b_away": r.get("b2b_away"),
            "spread_home": r.get("spread_home"),
            "moneyline_home": r.get("moneyline_home"),
            "moneyline_away": r.get("moneyline_away"),
            "home_spread_price": r.get("home_spread_price"),
            "away_spread_price": r.get("away_spread_price"),
            "over_price": r.get("over_price"),
            "under_price": r.get("under_price"),
            "home_tempo_rating": r.get("home_tempo_rating"),
            "away_tempo_rating": r.get("away_tempo_rating"),
            "tempo_rating_sum": r.get("tempo_rating_sum"),
        }
        out.update(sim_res)
        results.append(out)

    sim_df = pd.DataFrame(results)
    out_path = out_dir / f"{quantiles_out_prefix}{date}.csv"
    try:
        sim_df = sim_df.replace([np.inf, -np.inf], np.nan)
    except Exception:
        pass
    sim_df.to_csv(out_path, index=False, na_rep="")

    if segment_rows_all:
        try:
            seg_df = pd.DataFrame(segment_rows_all)
            seg_path = out_dir / f"{segments_out_prefix}{date}.csv"
            try:
                seg_df = seg_df.replace([np.inf, -np.inf], np.nan)
            except Exception:
                pass

            # Optional post-hoc calibration of cumulative segment endpoints.
            # Prefer outputs/segment_calibration_5min.json (affine per endpoint minute: a*value+b).
            # Fall back to outputs/segment_bias_5min.json (additive bias per endpoint minute).
            try:
                if "end_min" in seg_df.columns:
                    try:
                        import os

                        disable_cal = (os.environ.get("NCAAB_DISABLE_SEGMENT_CALIB") or "").strip().lower()
                        if disable_cal in {"1", "true", "yes"}:
                            disable_cal = "1"
                        else:
                            disable_cal = "0"
                    except Exception:
                        disable_cal = "0"

                    try:
                        import os

                        debug_seg_cal = (os.environ.get("NCAAB_DEBUG_SEGMENT_CALIB") or "").strip().lower() in {"1", "true", "yes"}
                    except Exception:
                        debug_seg_cal = False

                    if disable_cal == "1":
                        raise RuntimeError("Segment calibration disabled via NCAAB_DISABLE_SEGMENT_CALIB")

                    try:
                        import os

                        bias_only = (os.environ.get("NCAAB_SEGMENT_BIAS_ONLY") or "").strip().lower() in {"1", "true", "yes"}
                    except Exception:
                        bias_only = False

                    seg_df["end_min"] = pd.to_numeric(seg_df["end_min"], errors="coerce")

                    cols = [
                        c
                        for c in (
                            "mu_total_score_end",
                            "q10_total_score_end",
                            "q50_total_score_end",
                            "q90_total_score_end",
                        )
                        if c in seg_df.columns
                    ]
                    for col in cols:
                        seg_df[col] = pd.to_numeric(seg_df[col], errors="coerce")

                    calib_path = out_dir / "segment_calibration_5min.json"
                    bias_path = out_dir / "segment_bias_5min.json"
                    stage2_path = out_dir / "segment_calibration_stage2_5min.json"

                    stage1_affine_applied = False
                    stage1_bias_applied = False
                    stage2_applied = False
                    stage2_skipped_reason: str | None = None
                    if (not bias_only) and calib_path.exists() and cols:
                        calib = read_json(calib_path)
                        a_map = calib.get("a_by_end_min") if isinstance(calib, dict) else None
                        b_map = calib.get("b_by_end_min") if isinstance(calib, dict) else None
                        if isinstance(a_map, dict) and isinstance(b_map, dict) and a_map:
                            norm_a = {}
                            norm_b = {}
                            for k, v in a_map.items():
                                try:
                                    kk = float(k)
                                    vv = float(v)
                                except Exception:
                                    continue
                                # Guardrail: extremely small slopes collapse per-game variance.
                                # Skip those entries rather than destroying signal.
                                if np.isfinite(kk) and np.isfinite(vv) and vv >= 0.50:
                                    norm_a[kk] = vv
                            for k, v in b_map.items():
                                try:
                                    kk = float(k)
                                    vv = float(v)
                                except Exception:
                                    continue
                                if np.isfinite(kk) and np.isfinite(vv):
                                    norm_b[kk] = vv

                            if norm_a:
                                # Apply affine correction only for endpoints that passed guardrails.
                                # For skipped endpoints, keep the original signal (do NOT NaN it out).
                                a_hat = seg_df["end_min"].map(norm_a)
                                apply_mask = a_hat.notna()
                                if apply_mask.any():
                                    b_hat = seg_df["end_min"].map(norm_b).fillna(0.0)
                                    for col in cols:
                                        seg_df.loc[apply_mask, col] = a_hat.loc[apply_mask] * seg_df.loc[apply_mask, col] + b_hat.loc[apply_mask]
                                stage1_affine_applied = True

                    # Stage1 fallback (or forced mode): simple bias-only correction.
                    if (bias_only or (not stage1_affine_applied)) and bias_path.exists() and cols:
                        bias_payload = read_json(bias_path)
                        bias_map = bias_payload.get("bias_by_end_min") if isinstance(bias_payload, dict) else None
                        if isinstance(bias_map, dict) and bias_map:
                            norm_bias = {}
                            for k, v in bias_map.items():
                                try:
                                    kk = float(k)
                                    vv = float(v)
                                except Exception:
                                    continue
                                if np.isfinite(kk) and np.isfinite(vv):
                                    norm_bias[kk] = vv

                            if norm_bias:
                                # IMPORTANT: the bias map is typically defined only on the 5-min grid.
                                # When we emit 2-min endpoints, most rows will not have an entry.
                                # Treat missing bias as 0.0 (no correction) rather than NaN-ing out
                                # the entire column via arithmetic with NaNs.
                                seg_df["_bias_hat"] = seg_df["end_min"].map(norm_bias).fillna(0.0)
                                for col in cols:
                                    seg_df[col] = seg_df[col] - seg_df["_bias_hat"]
                                seg_df = seg_df.drop(columns=["_bias_hat"], errors="ignore")
                                stage1_bias_applied = True

                    # Optional second-stage residual bias correction.
                    # Note: Stage1 affine calibration may be skipped by guardrails (e.g., very small slopes).
                    # In that case we often fall back to bias-only Stage1; stage2 is still useful and should
                    # be allowed to run on top of whatever stage1 was applied.
                    if stage2_path.exists() and cols:
                        import os

                        disable_stage2 = (os.environ.get("NCAAB_DISABLE_SEGMENT_CALIB_STAGE2") or "").strip().lower()
                        if disable_stage2 not in {"1", "true", "yes"}:
                            stage2 = read_json(stage2_path)
                            bias_map = stage2.get("bias_by_end_min") if isinstance(stage2, dict) else None
                            if isinstance(bias_map, dict) and bias_map:
                                norm_bias2 = {}
                                for k, v in bias_map.items():
                                    try:
                                        kk = float(k)
                                        vv = float(v)
                                    except Exception:
                                        continue
                                    if np.isfinite(kk) and np.isfinite(vv):
                                        norm_bias2[kk] = vv

                                if norm_bias2:
                                    seg_df["_bias2_hat"] = seg_df["end_min"].map(norm_bias2).fillna(0.0)
                                    for col in cols:
                                        seg_df[col] = seg_df[col] - seg_df["_bias2_hat"]
                                    seg_df = seg_df.drop(columns=["_bias2_hat"], errors="ignore")
                                    stage2_applied = True
                        else:
                            stage2_skipped_reason = "env_disabled"

                    if debug_seg_cal:
                        try:
                            print(
                                {
                                    "segment_calib_debug": {
                                        "date": str(date),
                                        "bias_only": bool(bias_only),
                                        "stage1_affine_applied": bool(stage1_affine_applied),
                                        "stage1_bias_applied": bool(stage1_bias_applied),
                                        "stage2_path_exists": bool(stage2_path.exists()),
                                        "stage2_applied": bool(stage2_applied),
                                        "stage2_skipped_reason": stage2_skipped_reason,
                                    }
                                }
                            )
                        except Exception:
                            pass

                    # Enforce monotonicity within each game for cumulative columns.
                    if cols and "game_id" in seg_df.columns:
                        seg_df = seg_df.sort_values(["game_id", "end_min"], kind="mergesort")
                        for col in cols:
                            seg_df[col] = seg_df.groupby("game_id")[col].cummax()

                    # Re-anchor calibrated segment cumulative totals to the primary sim totals.
                    # The segment calibration is an endpoint-specific transform (a*value+b - bias),
                    # which can introduce constant shifts at 20/40 that make the trajectory disagree
                    # with the main 1H/full-game totals used elsewhere. We preserve internal
                    # consistency by ensuring:
                    #   end_min=20 matches *_total_1h
                    #   end_min=40 matches *_total
                    # while applying a smooth piecewise-linear adjustment across endpoints.
                    try:
                        if cols and "game_id" in seg_df.columns and "end_min" in seg_df.columns:
                            anchor_map = {
                                "mu_total_score_end": ("mu_total_1h", "mu_total"),
                                "q10_total_score_end": ("q10_total_1h", "q10_total"),
                                "q50_total_score_end": ("q50_total_1h", "q50_total"),
                                "q90_total_score_end": ("q90_total_1h", "q90_total"),
                            }

                            usable = [c for c in cols if c in anchor_map]
                            if usable:
                                needed_sim = ["game_id"]
                                for seg_col in usable:
                                    s1, sfull = anchor_map[seg_col]
                                    needed_sim.extend([s1, sfull])

                                if all((c in sim_df.columns) for c in needed_sim):
                                    anchors = sim_df[needed_sim].copy()
                                    seg20 = seg_df.loc[seg_df["end_min"] == 20, ["game_id"] + usable].copy()
                                    seg40 = seg_df.loc[seg_df["end_min"] == 40, ["game_id"] + usable].copy()

                                    seg20 = seg20.rename(columns={c: f"{c}_seg20" for c in usable})
                                    seg40 = seg40.rename(columns={c: f"{c}_seg40" for c in usable})

                                    a = anchors.merge(seg20, on="game_id", how="inner").merge(seg40, on="game_id", how="inner")

                                    for seg_col in usable:
                                        s1, sfull = anchor_map[seg_col]
                                        a[f"delta20__{seg_col}"] = pd.to_numeric(a[s1], errors="coerce") - pd.to_numeric(
                                            a.get(f"{seg_col}_seg20"), errors="coerce"
                                        )
                                        a[f"delta40__{seg_col}"] = pd.to_numeric(a[sfull], errors="coerce") - pd.to_numeric(
                                            a.get(f"{seg_col}_seg40"), errors="coerce"
                                        )

                                    deltas = a[["game_id"] + [f"delta20__{c}" for c in usable] + [f"delta40__{c}" for c in usable]].copy()
                                    seg_df = seg_df.merge(deltas, on="game_id", how="left")

                                    end_min = pd.to_numeric(seg_df["end_min"], errors="coerce")
                                    frac1 = (end_min / 20.0).clip(lower=0.0, upper=1.0)
                                    frac2 = ((end_min - 20.0) / 20.0).clip(lower=0.0, upper=1.0)

                                    half1_mask = end_min <= 20.0
                                    for seg_col in usable:
                                        d20 = pd.to_numeric(seg_df.get(f"delta20__{seg_col}"), errors="coerce")
                                        d40 = pd.to_numeric(seg_df.get(f"delta40__{seg_col}"), errors="coerce")
                                        d20 = d20.fillna(0.0)
                                        d40 = d40.fillna(d20)

                                        delta = np.where(
                                            half1_mask,
                                            d20.to_numpy(dtype=float) * frac1.to_numpy(dtype=float),
                                            d20.to_numpy(dtype=float) + (d40.to_numpy(dtype=float) - d20.to_numpy(dtype=float)) * frac2.to_numpy(dtype=float),
                                        )

                                        seg_df[seg_col] = pd.to_numeric(seg_df[seg_col], errors="coerce") + delta

                                    seg_df = seg_df.drop(
                                        columns=[f"delta20__{c}" for c in usable] + [f"delta40__{c}" for c in usable],
                                        errors="ignore",
                                    )

                                    # Re-enforce monotonicity after anchoring.
                                    seg_df = seg_df.sort_values(["game_id", "end_min"], kind="mergesort")
                                    for seg_col in usable:
                                        seg_df[seg_col] = pd.to_numeric(seg_df[seg_col], errors="coerce")
                                        seg_df[seg_col] = seg_df.groupby("game_id")[seg_col].cummax()
                    except Exception:
                        pass
            except Exception:
                pass
            seg_df.to_csv(seg_path, index=False, na_rep="")
        except Exception:
            pass

    try:
        meta_path = out_dir / f"{meta_out_prefix}{date}.json"
        sim_calibration_sha256 = None
        try:
            if calib_path.exists():
                raw = calib_path.read_bytes()
                sim_calibration_sha256 = hashlib.sha256(raw).hexdigest()
        except Exception:
            sim_calibration_sha256 = None
        meta = {
            "date": date,
            "sim_seed": int(seed) if seed is not None else None,
            "per_game_seed": "sha256_u32(f'{seed}:{game_id}')",
            "samples": int(samples),
            "rho": float(rho),
            "rho_effective": float(rho_eff),
            "use_pace": bool(use_pace),
            "pace_sigma": float(pace_sigma),
            "segments_grid_min_cfg": int(grid_min_cfg) if 'grid_min_cfg' in locals() else None,
            "segments_time_aware_env": (os.environ.get("NCAAB_SEGMENTS_TIME_AWARE") if "os" in globals() else None),
            "segments_grid_min_env": (os.environ.get("NCAAB_SEGMENTS_GRID_MIN") if "os" in globals() else None),
            "team_segment_weights_disabled_env": (os.environ.get("NCAAB_DISABLE_TEAM_SEGMENT_WEIGHTS") if "os" in globals() else None),
            "segment_weights_path_used": seg_weights_path_used if 'seg_weights_path_used' in locals() else None,
            "segment_weights_load_error": seg_weights_load_error if 'seg_weights_load_error' in locals() else None,
            "team_segment_weights_path_used": team_weights_path_used if 'team_weights_path_used' in locals() else None,
            "team_segment_weights_load_error": team_weights_load_error if 'team_weights_load_error' in locals() else None,
            "injuries_path": str(injuries_path) if injuries_path is not None else None,
            "sim_calibration_path": str(calib_path),
            "sim_calibration_sha256": sim_calibration_sha256,
            "sim_calibration_load_error": sim_calibration_load_error,
            "sim_calibration": sim_calibration,
            "mean_source": str(mean_source),
            "allow_market_guardrails": bool(allow_market_guardrails),
        }

        # Diagnostics: how often we had spread vs expected-margin proxy available.
        try:
            games_n = int(len(sim_df)) if 'sim_df' in locals() and sim_df is not None else 0
            spread_used_n = None
            if games_n > 0 and 'sim_df' in locals() and sim_df is not None:
                if "abs_margin_proxy_source" in sim_df.columns:
                    src = sim_df["abs_margin_proxy_source"].astype(str).str.lower().str.strip()
                    spread_used_n = int((src == "spread").sum())
                elif "spread_home" in sim_df.columns:
                    sh = pd.to_numeric(sim_df["spread_home"], errors="coerce")
                    spread_used_n = int((sh.notna() & np.isfinite(sh)).sum())

            if spread_used_n is not None:
                meta["abs_margin_proxy_usage"] = {
                    "games": games_n,
                    "spread_used": int(spread_used_n),
                    "expected_used": int(max(0, games_n - int(spread_used_n))),
                    "spread_share": (float(spread_used_n) / float(games_n)) if games_n > 0 else None,
                }
        except Exception:
            pass
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass
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
    ap.add_argument("--use-pace", action="store_true", help="Enable pace/possessions-based simulation when tempo/pace inputs exist")
    ap.add_argument("--no-pace", action="store_true", help="Force-disable pace simulation")
    ap.add_argument("--pace-sigma", type=float, default=DEFAULT_PACE_SIGMA, help="Std dev for possessions (per 40)")
    ap.add_argument("--injuries-file", type=str, default="", help="Optional CSV of injury/availability overrides")
    ap.add_argument("--seed", type=int, default=None, help="Deterministic simulation seed (default: stable per-date, or env NCAAB_SIM_SEED)")
    args = ap.parse_args()

    out_dir = Path(args.outputs)
    preds_path = Path(args.preds_file) if args.preds_file else None
    lines_path = Path(args.lines_file) if args.lines_file else None

    use_pace = None
    if args.use_pace:
        use_pace = True
    if args.no_pace:
        use_pace = False

    injuries_path = Path(args.injuries_file) if args.injuries_file else None
    path = run_simulations_for_date(
        out_dir,
        args.date,
        preds_path,
        lines_path,
        args.samples,
        args.rho,
        use_pace=use_pace,
        pace_sigma=args.pace_sigma,
        injuries_path=injuries_path,
        seed=args.seed,
    )
    print({"wrote": str(path)})
