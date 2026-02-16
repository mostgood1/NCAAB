from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from .live_snapshot_features import market_blend_weight
from .pbp_synthetic_snapshots import _payload_date_utc, _play_elapsed_min, _iter_plays


@dataclass(frozen=True)
class BacktestConfig:
    step_sec: int = 30
    min_elapsed_min: float = 6.0
    max_elapsed_min: float = 39.5
    horizon_min: float = 40.0
    ft_weight: float = 0.44
    use_tuning_clamps: bool = True


@dataclass(frozen=True)
class LiveLensClamps:
    pace_lo: float
    pace_hi: float
    pps_lo: float
    pps_hi: float


@dataclass(frozen=True)
class IntervalCalibrationFitConfig:
    bucket_min: float = 2.0
    min_bucket_n: int = 250
    shrink_tau: float = 1500.0
    delta_cap: float = 18.0
    target_cov: float = 0.80
    sigma_mult_clip_lo: float = 0.70
    sigma_mult_clip_hi: float = 1.60


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def _coerce_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:
            return None
        return float(v)
    except Exception:
        return None


def _coerce_int(x: object) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _extract_pregame_total_line(payload: dict[str, Any]) -> Optional[float]:
    # Prefer pickcenter
    try:
        pc = payload.get("pickcenter")
        if isinstance(pc, list) and pc:
            pc0 = pc[0] if isinstance(pc[0], dict) else None
            if isinstance(pc0, dict):
                total = _coerce_float(pc0.get("overUnder"))
                if total is None:
                    total = _coerce_float(pc0.get("total"))
                return total
    except Exception:
        pass

    try:
        odds = payload.get("odds")
        if isinstance(odds, list) and odds:
            for o in odds:
                if not isinstance(o, dict):
                    continue
                total = _coerce_float(o.get("overUnder"))
                if total is None:
                    total = _coerce_float(o.get("total"))
                if total is not None:
                    return total
    except Exception:
        pass

    return None


def _final_total_from_plays(payload: dict[str, Any]) -> Optional[int]:
    last_total: Optional[int] = None
    for p in _iter_plays(payload):
        try:
            hs = p.get("homeScore")
            a_s = p.get("awayScore")
            if hs is None or a_s is None:
                continue
            t = int(hs) + int(a_s)
            last_total = int(t)
        except Exception:
            continue
    return last_total


def _is_shooting_play(p: dict[str, Any]) -> bool:
    try:
        if bool(p.get("shootingPlay")):
            return True
    except Exception:
        pass
    try:
        txt = str(p.get("text") or "").strip().lower()
        if not txt:
            return False
        return ("makes" in txt) or ("misses" in txt) or ("free throw" in txt)
    except Exception:
        return False


def _update_shot_counts(p: dict[str, Any], fga: int, fta: int) -> tuple[int, int]:
    if not _is_shooting_play(p):
        return fga, fta
    try:
        pa = p.get("pointsAttempted")
        if pa is None:
            # Heuristic from text
            txt = str(p.get("text") or "").lower()
            if "free throw" in txt:
                return fga, fta + 1
            if "3-pt" in txt or "three" in txt:
                return fga + 1, fta
            if "2-pt" in txt or "layup" in txt or "jumper" in txt or "dunk" in txt:
                return fga + 1, fta
            return fga, fta
        pa_i = int(pa)
        if pa_i == 1:
            fta += 1
        elif pa_i in (2, 3):
            fga += 1
    except Exception:
        pass
    return int(fga), int(fta)


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(v))))


def _period_from_elapsed(elapsed_min: float) -> int:
    return 1 if float(elapsed_min) <= 20.0 else 2


def _elapsed_bucket_min(elapsed_min: float, bucket_min: float) -> int:
    b = float(bucket_min)
    if not np.isfinite(b) or b <= 0:
        b = 2.0
    try:
        return int(np.floor(float(elapsed_min) / b) * b)
    except Exception:
        return int(float(elapsed_min))


def fit_interval_calibration(
    df: pd.DataFrame,
    *,
    cfg: IntervalCalibrationFitConfig | None = None,
    projection_col: str = "proj_blend",
    actual_col: str = "actual_total",
    elapsed_col: str = "elapsed_min",
) -> dict[str, Any]:
    """Fit a simple elapsed-bucket calibration for live projections.

    Model:
      mu_cal = mu + global_delta_add + bucket_delta_add
      sigma_cal = global_sigma * bucket_sigma_mult

    where bucket adjustments are shrunk towards 0 based on bucket sample size.
    """

    cfg2 = cfg or IntervalCalibrationFitConfig()
    if df is None or df.empty:
        return {"status": "empty", "message": "No rows", "projection_col": projection_col}

    d = df.copy()
    mu = pd.to_numeric(d.get(projection_col), errors="coerce")
    y = pd.to_numeric(d.get(actual_col), errors="coerce")
    em = pd.to_numeric(d.get(elapsed_col), errors="coerce")
    m = mu.notna() & y.notna() & em.notna()
    if int(m.sum()) <= 0:
        return {"status": "empty", "message": "No valid rows", "projection_col": projection_col}

    mu = mu.loc[m]
    y = y.loc[m]
    em = em.loc[m]
    err = (mu - y).astype(float)

    global_bias = float(err.mean())
    # Estimate sigma from residual 10/90 span so that a Normal(0,sigma)
    # would imply ~80% central mass between q10 and q90.
    z80 = 1.2815515655446004
    denom = float(2.0 * z80)
    resid_g = (err.values - global_bias).astype(float)
    try:
        q90 = float(np.quantile(resid_g, 0.90))
        q10 = float(np.quantile(resid_g, 0.10))
        span = float(q90 - q10)
        global_sigma = float(span / denom) if abs(span) > 1e-9 else float("nan")
    except Exception:
        global_sigma = float("nan")
    if not np.isfinite(global_sigma) or global_sigma <= 1e-6:
        global_sigma = float(np.std(resid_g))
    if not np.isfinite(global_sigma) or global_sigma <= 1e-6:
        global_sigma = 12.0

    global_delta_add = float(-global_bias)

    bucket_min = float(cfg2.bucket_min)
    bkeys = em.apply(lambda v: _elapsed_bucket_min(float(v), bucket_min))
    tmp = pd.DataFrame({"bucket": bkeys, "err": err.values})

    buckets: list[dict[str, Any]] = []
    for b, g in tmp.groupby("bucket"):
        n = int(len(g))
        if n < int(cfg2.min_bucket_n):
            continue
        b_bias = float(g["err"].mean())
        b_sigma = float("nan")
        try:
            resid_b = (g["err"].values.astype(float) - b_bias).astype(float)
            q90b = float(np.quantile(resid_b, 0.90))
            q10b = float(np.quantile(resid_b, 0.10))
            span_b = float(q90b - q10b)
            if abs(span_b) > 1e-9:
                b_sigma = float(span_b / denom)
        except Exception:
            b_sigma = float("nan")
        if not np.isfinite(b_sigma) or b_sigma <= 1e-6:
            try:
                b_sigma = float(np.std((g["err"].values.astype(float) - b_bias).astype(float)))
            except Exception:
                b_sigma = float("nan")
        if not np.isfinite(b_sigma) or b_sigma <= 1e-6:
            b_sigma = global_sigma

        # Shrink factor towards global based on sample size.
        shrink = float(n / (n + float(cfg2.shrink_tau)))
        # Bucket delta is relative to global correction.
        raw_bucket_delta = -(b_bias - global_bias)
        bucket_delta_add = float(shrink * raw_bucket_delta)
        if np.isfinite(cfg2.delta_cap) and cfg2.delta_cap > 0:
            bucket_delta_add = float(np.clip(bucket_delta_add, -float(cfg2.delta_cap), float(cfg2.delta_cap)))

        # Bucket sigma multiplier relative to global.
        raw_mult = float(b_sigma / global_sigma) if global_sigma > 1e-6 else 1.0
        raw_mult = float(np.clip(raw_mult, float(cfg2.sigma_mult_clip_lo), float(cfg2.sigma_mult_clip_hi)))
        sigma_mult = float(1.0 + shrink * (raw_mult - 1.0))

        buckets.append(
            {
                "elapsed_bucket_min": int(b),
                "n": int(n),
                "bucket_bias": float(b_bias),
                "delta_add": float(bucket_delta_add),
                "sigma_mult": float(sigma_mult),
            }
        )

    buckets.sort(key=lambda r: int(r.get("elapsed_bucket_min") or 0))
    return {
        "status": "ok",
        "version": 1,
        "projection_col": str(projection_col),
        "actual_col": str(actual_col),
        "elapsed_col": str(elapsed_col),
        "fit": {
            "bucket_min": float(bucket_min),
            "min_bucket_n": int(cfg2.min_bucket_n),
            "shrink_tau": float(cfg2.shrink_tau),
            "delta_cap": float(cfg2.delta_cap),
            "target_cov": float(cfg2.target_cov),
            "sigma_mult_clip": [float(cfg2.sigma_mult_clip_lo), float(cfg2.sigma_mult_clip_hi)],
            "n_rows": int(len(err)),
        },
        "global": {"delta_add": float(global_delta_add), "sigma": float(global_sigma)},
        "elapsed_buckets": buckets,
    }


def _apply_interval_calibration_row(
    *,
    elapsed_min: float | None,
    mu: float | None,
    calib: dict[str, Any] | None,
) -> tuple[float | None, float | None]:
    if mu is None or elapsed_min is None or calib is None:
        return mu, None
    try:
        g = calib.get("global") if isinstance(calib, dict) else None
        delta_g = float(g.get("delta_add")) if isinstance(g, dict) and g.get("delta_add") is not None else 0.0
        sigma_g = float(g.get("sigma")) if isinstance(g, dict) and g.get("sigma") is not None else None
        bucket_min = float(calib.get("fit", {}).get("bucket_min", 2.0))
        bkey = _elapsed_bucket_min(float(elapsed_min), bucket_min)
        delta_b = 0.0
        sigma_mult = 1.0
        for r in calib.get("elapsed_buckets") or []:
            if isinstance(r, dict) and int(r.get("elapsed_bucket_min") or -999) == int(bkey):
                if r.get("delta_add") is not None:
                    delta_b = float(r.get("delta_add"))
                if r.get("sigma_mult") is not None:
                    sigma_mult = float(r.get("sigma_mult"))
                break
        mu2 = float(mu) + float(delta_g) + float(delta_b)
        sigma2 = None
        if sigma_g is not None and np.isfinite(float(sigma_g)) and float(sigma_g) > 1e-6:
            sigma2 = float(sigma_g) * float(sigma_mult)
        return mu2, sigma2
    except Exception:
        return mu, None


def iter_interval_rows_for_game(
    *,
    event_id: str,
    payload: dict[str, Any],
    config: BacktestConfig,
    clamps: Optional[LiveLensClamps] = None,
) -> Iterable[dict[str, Any]]:
    eid = str(event_id)
    line_total = _extract_pregame_total_line(payload)
    actual_total = _final_total_from_plays(payload)
    if actual_total is None:
        return []

    # Build a timeline keyed by elapsed seconds.
    timeline: list[tuple[int, int, int, int]] = []  # (elapsed_sec, total_pts, fga, fta)
    fga = 0
    fta = 0
    for p in _iter_plays(payload):
        elapsed_min = _play_elapsed_min(p)
        if elapsed_min is None:
            continue
        if elapsed_min < 0:
            continue
        if elapsed_min > float(config.horizon_min) + 0.01:
            continue
        try:
            hs = p.get("homeScore")
            a_s = p.get("awayScore")
            if hs is None or a_s is None:
                continue
            total_pts = int(hs) + int(a_s)
        except Exception:
            continue

        fga, fta = _update_shot_counts(p, fga, fta)
        elapsed_sec = int(round(float(elapsed_min) * 60.0))
        if elapsed_sec < 0:
            elapsed_sec = 0
        if elapsed_sec > int(round(float(config.horizon_min) * 60.0)):
            elapsed_sec = int(round(float(config.horizon_min) * 60.0))
        timeline.append((int(elapsed_sec), int(total_pts), int(fga), int(fta)))

    if not timeline:
        return []

    try:
        timeline.sort(key=lambda t: t[0])
    except Exception:
        pass

    min_sec = int(round(float(config.min_elapsed_min) * 60.0))
    max_sec = int(round(float(config.max_elapsed_min) * 60.0))
    horizon_sec = int(round(float(config.horizon_min) * 60.0))
    step = max(1, int(config.step_sec))

    idx = 0
    cur_total = 0
    cur_fga = 0
    cur_fta = 0

    for s in range(min_sec, min(max_sec, horizon_sec) + 1, step):
        while idx < len(timeline) and timeline[idx][0] <= s:
            _, cur_total, cur_fga, cur_fta = timeline[idx]
            idx += 1

        elapsed_min = float(s) / 60.0
        remaining_min = max(0.0, float(config.horizon_min) - float(elapsed_min))
        period = _period_from_elapsed(elapsed_min)

        total_points = int(cur_total)
        shot_proxy = float(cur_fga) + float(config.ft_weight) * float(cur_fta)

        proj_pace = None
        if elapsed_min > 0.01:
            proj_pace = (float(total_points) / float(elapsed_min)) * float(config.horizon_min)

        proj_clamped = None
        poss_rate = None
        ppp = None
        if shot_proxy > 0.01 and elapsed_min > 0.01:
            poss_rate = float(shot_proxy) / float(elapsed_min)
            ppp = float(total_points) / float(shot_proxy)

        if proj_pace is not None:
            proj_clamped = float(proj_pace)
            if config.use_tuning_clamps and clamps is not None and poss_rate is not None and ppp is not None:
                pr = _clamp(float(poss_rate), clamps.pace_lo, clamps.pace_hi)
                eff = _clamp(float(ppp), clamps.pps_lo, clamps.pps_hi)
                proj_clamped = float(pr) * float(config.horizon_min) * float(eff)

        w = market_blend_weight(elapsed_min, horizon_min=float(config.horizon_min))
        proj_blend = None
        if line_total is not None and proj_clamped is not None:
            proj_blend = (1.0 - float(w)) * float(proj_clamped) + float(w) * float(line_total)
        else:
            proj_blend = proj_clamped

        err_pace = (float(proj_pace) - float(actual_total)) if (proj_pace is not None) else None
        err_clamped = (float(proj_clamped) - float(actual_total)) if (proj_clamped is not None) else None
        err_blend = (float(proj_blend) - float(actual_total)) if (proj_blend is not None) else None

        yield {
            "event_id": eid,
            "elapsed_sec": int(s),
            "elapsed_min": float(elapsed_min),
            "remaining_min": float(remaining_min),
            "period": int(period),
            "total_points": int(total_points),
            "shot_proxy": float(shot_proxy),
            "poss_rate": float(poss_rate) if poss_rate is not None else None,
            "ppp": float(ppp) if ppp is not None else None,
            "line_total": float(line_total) if line_total is not None else None,
            "actual_total": int(actual_total),
            "proj_pace": float(proj_pace) if proj_pace is not None else None,
            "proj_clamped": float(proj_clamped) if proj_clamped is not None else None,
            "proj_blend": float(proj_blend) if proj_blend is not None else None,
            "err_pace": float(err_pace) if err_pace is not None else None,
            "err_clamped": float(err_clamped) if err_clamped is not None else None,
            "err_blend": float(err_blend) if err_blend is not None else None,
        }


def _infer_clamps_from_tuning() -> LiveLensClamps:
    try:
        from ..live_lens_tuning import DEFAULT_TUNING

        t = DEFAULT_TUNING
        return LiveLensClamps(pace_lo=float(t.pace_lo), pace_hi=float(t.pace_hi), pps_lo=float(t.pps_lo), pps_hi=float(t.pps_hi))
    except Exception:
        return LiveLensClamps(pace_lo=2.75, pace_hi=3.25, pps_lo=0.95, pps_hi=1.18)


def _date_from_payload_or_fallback(payload: dict[str, Any]) -> Optional[str]:
    d = _payload_date_utc(payload)
    return d.isoformat() if isinstance(d, dt.date) else None


def run_interval_backtest(
    *,
    cache_dir: Path,
    start_date: dt.date,
    end_date: dt.date,
    out_csv: Path,
    step_sec: int = 30,
    min_elapsed_min: float = 6.0,
    max_elapsed_min: float = 39.5,
    max_files: int = 0,
    use_tuning_clamps: bool = True,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .pbp_synthetic_snapshots import index_pbp_cache_by_date

    cache_dir = Path(cache_dir)
    idx = index_pbp_cache_by_date(cache_dir=cache_dir, start_date=start_date, end_date=end_date, max_files=max_files)

    clamps = _infer_clamps_from_tuning() if use_tuning_clamps else None

    rows: list[dict[str, Any]] = []
    games = 0
    skipped = 0

    for ds, eids in sorted(idx.by_date.items()):
        for eid in eids:
            p = cache_dir / f"{eid}.json"
            payload = _safe_read_json(p)
            if not isinstance(payload, dict):
                skipped += 1
                continue
            games += 1
            cfg = BacktestConfig(step_sec=int(step_sec), min_elapsed_min=float(min_elapsed_min), max_elapsed_min=float(max_elapsed_min), use_tuning_clamps=bool(use_tuning_clamps))
            for r in iter_interval_rows_for_game(event_id=str(eid), payload=payload, config=cfg, clamps=clamps):
                r["date"] = ds
                rows.append(r)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    # Optional: apply elapsed-bucket calibration to blend projection, and attach sigma + 80% interval coverage.
    try:
        if isinstance(calibration, dict) and ("global" in calibration):
            mu = pd.to_numeric(df.get("proj_blend"), errors="coerce")
            em = pd.to_numeric(df.get("elapsed_min"), errors="coerce")
            y = pd.to_numeric(df.get("actual_total"), errors="coerce")
            mu_cal: list[float | None] = []
            sg_cal: list[float | None] = []
            for e0, m0 in zip(em.to_numpy(dtype=float, na_value=np.nan), mu.to_numpy(dtype=float, na_value=np.nan)):
                e1 = None if (not np.isfinite(e0)) else float(e0)
                m1 = None if (not np.isfinite(m0)) else float(m0)
                m2, s2 = _apply_interval_calibration_row(elapsed_min=e1, mu=m1, calib=calibration)
                mu_cal.append(m2)
                sg_cal.append(s2)
            df["proj_blend_cal"] = pd.to_numeric(pd.Series(mu_cal), errors="coerce")
            df["sigma_blend_cal"] = pd.to_numeric(pd.Series(sg_cal), errors="coerce")
            df["err_blend_cal"] = df["proj_blend_cal"] - y
            # Central 80% interval coverage based on calibrated sigma.
            z80 = 1.2815515655446004
            lo = df["proj_blend_cal"] - z80 * df["sigma_blend_cal"]
            hi = df["proj_blend_cal"] + z80 * df["sigma_blend_cal"]
            df["cov80_blend_cal"] = (y >= lo) & (y <= hi)
    except Exception:
        pass

    df.to_csv(out_csv, index=False)

    summary = summarize_interval_backtest(df)
    return {
        "status": "ok",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "cache_dir": str(cache_dir),
        "out_csv": str(out_csv),
        "rows": int(len(df)),
        "games": int(games),
        "skipped": int(skipped),
        "summary": summary,
    }


def _metric_block(err: pd.Series) -> dict[str, float | int | None]:
    try:
        e = pd.to_numeric(err, errors="coerce").dropna()
        if e.empty:
            return {"n": 0, "mae": None, "rmse": None, "bias": None}
        mae = float(np.mean(np.abs(e.values)))
        rmse = float(np.sqrt(np.mean((e.values) ** 2)))
        bias = float(np.mean(e.values))
        return {"n": int(len(e)), "mae": mae, "rmse": rmse, "bias": bias}
    except Exception:
        return {"n": 0, "mae": None, "rmse": None, "bias": None}


def summarize_interval_backtest(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {"overall": {}, "by_elapsed_bucket": []}

    out: dict[str, Any] = {}
    out["overall"] = {
        "pace": _metric_block(df.get("err_pace")),
        "clamped": _metric_block(df.get("err_clamped")),
        "blend": _metric_block(df.get("err_blend")),
    }

    # Optional calibrated projection summary.
    try:
        if "err_blend_cal" in df.columns:
            out["overall"]["blend_cal"] = _metric_block(df.get("err_blend_cal"))
        if "cov80_blend_cal" in df.columns:
            cov = df.get("cov80_blend_cal")
            cov2 = cov.dropna() if isinstance(cov, pd.Series) else None
            if cov2 is not None and not cov2.empty:
                out["overall"]["blend_cal_cov80"] = {"n": int(len(cov2)), "coverage": float(cov2.mean())}
    except Exception:
        pass

    # Bucket by elapsed minutes into 2-min bins for readability.
    try:
        em = pd.to_numeric(df.get("elapsed_min"), errors="coerce")
        bucket = (np.floor(em / 2.0) * 2.0).astype("Int64")
        df2 = df.copy()
        df2["elapsed_bucket_min"] = bucket
        rows: list[dict[str, Any]] = []
        for b, g in df2.dropna(subset=["elapsed_bucket_min"]).groupby("elapsed_bucket_min"):
            row = {
                "elapsed_bucket_min": int(b),
                "pace": _metric_block(g.get("err_pace")),
                "clamped": _metric_block(g.get("err_clamped")),
                "blend": _metric_block(g.get("err_blend")),
            }
            if "err_blend_cal" in g.columns:
                row["blend_cal"] = _metric_block(g.get("err_blend_cal"))
            if "cov80_blend_cal" in g.columns:
                try:
                    cov = g.get("cov80_blend_cal")
                    cov2 = cov.dropna() if isinstance(cov, pd.Series) else None
                    if cov2 is not None and not cov2.empty:
                        row["blend_cal_cov80"] = {"n": int(len(cov2)), "coverage": float(cov2.mean())}
                except Exception:
                    pass
            rows.append(row)
        rows.sort(key=lambda r: r["elapsed_bucket_min"])
        out["by_elapsed_bucket"] = rows
    except Exception:
        out["by_elapsed_bucket"] = []

    return out
