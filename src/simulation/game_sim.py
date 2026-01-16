import json

import numpy as np
import pandas as pd

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

# Lightweight Monte Carlo simulator using baseline totals/margins.
# Assumes per-game means for total and margin and estimates per-team variance
# with a shared correlation parameter.

DEFAULT_RHO = 0.3  # positive correlation between team scores
DEFAULT_TOTAL_SIGMA = 14.0  # fallback spread of total points
DEFAULT_SAMPLES = 4000

# Pace/possessions modeling (used when tempo/pace inputs are present)
DEFAULT_PACE = 69.0
DEFAULT_PACE_SIGMA = 3.5
PACE_MIN = 55.0
PACE_MAX = 85.0

HALF_FRAC_DEFAULT = 0.5


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


def _resolve_mean_total_margin(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    # Choose a preferred mean total/margin from available model/blend columns.
    # Use market lines only as a *guardrail* to avoid rare-but-deadly wrong-scale values,
    # not as a primary selector (which would implicitly hug the market).
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

    total_candidates: list[tuple[str, float]] = []
    for tot_col in [
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
    ]:
        if tot_col in row and pd.notna(row[tot_col]):
            try:
                v = float(row[tot_col])
            except Exception:
                continue
            if 70.0 <= v <= 250.0:
                total_candidates.append((tot_col, v))

    if total_candidates:
        # Preferred candidate is the first in our ordered list.
        total = total_candidates[0][1]
        # Guardrail: if wildly off market, try the first candidate within tolerance; else anchor.
        if market_total is not None and abs(float(total) - float(market_total)) > 35.0:
            alt = next((v for _, v in total_candidates if abs(float(v) - float(market_total)) <= 35.0), None)
            if alt is not None:
                total = float(alt)
            else:
                total = float(market_total)
    else:
        total = None

    margin_candidates: list[tuple[str, float]] = []
    for mar_col in [
        "pred_margin_blend",
        "pred_margin_base",
        "pred_margin_model",
        "pred_margin_calibrated",
        "pred_margin",
        "pred_margin_seg",
        "pred_margin_interval_mean",
        "margin_pred",
    ]:
        if mar_col in row and pd.notna(row[mar_col]):
            try:
                v = float(row[mar_col])
            except Exception:
                continue
            if -80.0 <= v <= 80.0:
                margin_candidates.append((mar_col, v))

    if margin_candidates:
        margin = margin_candidates[0][1]
        if market_margin is not None and abs(float(margin) - float(market_margin)) > 25.0:
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


def _load_sim_calibration(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


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
) -> dict:
    if rng is None:
        rng = np.random.default_rng()
    total_mean, margin_mean = _resolve_mean_total_margin(row)
    if total_mean is None or margin_mean is None:
        return {
            "sim_ok": False,
            "mu_total": total_mean,
            "mu_margin": margin_mean,
        }
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
    total_mean_1h, margin_mean_1h = _resolve_mean_total_margin_1h(row)
    if total_mean_1h is None:
        total_mean_1h = float(total_mean) * half_frac
    if margin_mean_1h is None:
        margin_mean_1h = float(margin_mean) * 0.5

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
    if has_1h_cal:
        try:
            total_mean_1h = float(total_mean_1h) + float(sim_calibration.get("delta_total_1h", 0.0) or 0.0)
            margin_mean_1h = float(margin_mean_1h) + float(sim_calibration.get("delta_margin_1h", 0.0) or 0.0)
        except Exception:
            pass
    else:
        try:
            delta_total_cal = float(calib_applied.get("delta_total", 0.0)) if calib_applied else 0.0
            delta_margin_cal = float(calib_applied.get("delta_margin", 0.0)) if calib_applied else 0.0
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
            sigma_total_1h = float(max(1e-6, float(sigma_total_1h) * m))
        except Exception:
            pass
        if sigma_margin_1h is not None:
            try:
                m = float(sim_calibration.get("sigma_margin_1h_mult", 1.0) or 1.0)
                m_cap = float(0.99 / float(np.sqrt(max(half_frac, 1e-6))))
                # Allow some additional 1H margin spread, but keep it modest.
                m = float(min(m, 1.15, m_cap))
                sigma_margin_1h = float(max(1e-6, float(sigma_margin_1h) * m))
            except Exception:
                pass

    # Basic sanity clamps (avoid pathological inputs creating NaNs)
    try:
        mu_home = float(mu_home)
        mu_away = float(mu_away)
    except Exception:
        return {"sim_ok": False, "mu_total": total_mean, "mu_margin": margin_mean}
    mu_home = float(max(mu_home, 0.0))
    mu_away = float(max(mu_away, 0.0))

    # Optional pace/possessions path: simulate possessions and per-team PPP.
    # This behaves more like a basketball simulator:
    #   - one shared possessions draw per sim
    #   - per-team scoring = possessions * PPP + (noise scaled by possessions)
    #   - an additional shared (or anti-shared) shock is used to match the target
    #     covariance implied by (sigma_total, sigma_margin)
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

        return {
            "sim_ok": True,
            "sim_method": "pace",
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

    return {
        "sim_ok": True,
        "sim_method": "points",
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
    }


def run_simulations_for_date(out_dir: Path, date: str,
                             preds_path: Optional[Path] = None,
                             lines_path: Optional[Path] = None,
                             samples: int = DEFAULT_SAMPLES,
                             rho: float = DEFAULT_RHO,
                             use_pace: Optional[bool] = None,
                             pace_sigma: float = DEFAULT_PACE_SIGMA,
                             injuries_path: Optional[Path] = None,
                             seed: Optional[int] = None) -> Path:
    out_dir = Path(out_dir)
    # Simulation inputs default to the unified enriched rows for a given date.
    # Those rows carry the market-derived mean total + spread-derived margin plus
    # optional uncertainty columns used to derive sigmas.
    enr_path = out_dir / f"predictions_unified_enriched_{date}.csv"
    games_path = out_dir / f"games_{date}.csv"

    if preds_path is None:
        preds_path = enr_path
    if lines_path is None:
        # Use rolling last-odds file by default; per-date odds can be sparse.
        lines_path = out_dir / "games_with_last.csv"

    sim_calibration = _load_sim_calibration(out_dir / "sim_calibration.json")

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

    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")

    preds = pd.read_csv(preds_path)
    if "date" in preds.columns:
        preds = preds[preds["date"].astype(str) == str(date)]

    # Normalize id dtype for stable merges/output
    if "game_id" in preds.columns:
        preds["game_id"] = preds["game_id"].map(_to_game_id_str)

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
        out_path = out_dir / f"sim_quantiles_{date}.csv"
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
            market_df = pd.read_csv(lines_path)

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

                select_fields: list[str] = []
                agg_spec: dict[str, object] = {}

                for c in first_fields:
                    if c not in preds.columns and c in market_df.columns:
                        select_fields.append(c)
                        agg_spec[c] = _agg_first

                for c in num_fields:
                    if c not in preds.columns and c in market_df.columns:
                        select_fields.append(c)
                        agg_spec[c] = _agg_median_num

                if select_fields:
                    m = market_df[["game_id"] + select_fields].copy()
                    m = m.dropna(subset=["game_id"])
                    m = m.groupby("game_id", as_index=False).agg(agg_spec)
                    preds = preds.merge(m, on="game_id", how="left")
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
        sim_res = simulate_game_row(
            r,
            rho=rho_eff,
            samples=samples,
            use_pace=bool(use_pace),
            pace_sigma=pace_sigma,
            injury_overrides=injury_overrides,
            sim_calibration=sim_calibration,
            rng=rng,
        )
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
    out_path = out_dir / f"sim_quantiles_{date}.csv"
    sim_df.to_csv(out_path, index=False)

    try:
        meta_path = out_dir / f"sim_meta_{date}.json"
        meta = {
            "date": date,
            "sim_seed": int(seed) if seed is not None else None,
            "per_game_seed": "sha256_u32(f'{seed}:{game_id}')",
            "samples": int(samples),
            "rho": float(rho),
            "rho_effective": float(rho_eff),
            "use_pace": bool(use_pace),
            "pace_sigma": float(pace_sigma),
            "injuries_path": str(injuries_path) if injuries_path is not None else None,
            "sim_calibration": sim_calibration,
        }
        import json

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
