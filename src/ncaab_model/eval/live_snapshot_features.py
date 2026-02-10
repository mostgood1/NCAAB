from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import pandas as pd


@dataclass
class SegmentIndex:
    x_end_min: list[float]
    y_q50_total_end: list[float]


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _parse_jsonl(path: Path, max_lines: int = 400000) -> Iterator[dict[str, Any]]:
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if max_lines and n >= int(max_lines):
                break
            s = line.strip()
            if not s:
                continue
            n += 1
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if isinstance(rec, dict):
                yield rec


def _load_segments_index(segments_path: Path) -> dict[str, SegmentIndex]:
    # Only need total trajectory; keep minimal.
    df = pd.read_csv(segments_path, low_memory=True)
    need = {"game_id", "end_min", "q50_total_score_end"}
    if not need.issubset(set(df.columns)):
        missing = sorted(list(need - set(df.columns)))
        raise ValueError(f"segments missing columns: {missing}")

    df = df[["game_id", "end_min", "q50_total_score_end"]].copy()
    df["game_id"] = df["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["end_min"] = pd.to_numeric(df["end_min"], errors="coerce")
    df["q50_total_score_end"] = pd.to_numeric(df["q50_total_score_end"], errors="coerce")
    df = df.dropna(subset=["game_id", "end_min", "q50_total_score_end"])

    out: dict[str, SegmentIndex] = {}
    for gid, grp in df.groupby("game_id"):
        g2 = grp.sort_values("end_min")
        xs = g2["end_min"].astype(float).to_list()
        ys = g2["q50_total_score_end"].astype(float).to_list()
        if len(xs) >= 2:
            out[str(gid)] = SegmentIndex(x_end_min=xs, y_q50_total_end=ys)
    return out


def _interp_total(seg: dict[str, SegmentIndex], game_id: str, end_min: float) -> Optional[float]:
    tab = seg.get(str(game_id))
    if not tab:
        return None
    xs = tab.x_end_min
    ys = tab.y_q50_total_end
    if not xs or not ys:
        return None
    t = float(end_min)
    if t <= xs[0]:
        return float(ys[0])
    if t >= xs[-1]:
        return float(ys[-1])
    for i in range(1, len(xs)):
        if t <= xs[i]:
            x0, x1 = float(xs[i - 1]), float(xs[i])
            y0, y1 = float(ys[i - 1]), float(ys[i])
            w = 0.0 if x1 == x0 else (t - x0) / (x1 - x0)
            return float(y0 + w * (y1 - y0))
    return None


def market_blend_weight(elapsed_min: Optional[float], horizon_min: float = 40.0) -> float:
    # Keep aligned with templates/index.html.
    if elapsed_min is None:
        return 0.0
    e = float(elapsed_min)
    h = float(horizon_min)
    if not (e >= 0 and h > 1):
        return 0.0
    start = 3.0 if h <= 20.5 else 5.0
    if e <= start:
        return 0.0
    max_w = 0.55
    t = (e - start) / max(1e-6, (h - start))
    if t < 0:
        t = 0.0
    if t > 1:
        t = 1.0
    return float(max_w * t)


def _load_results_actual_total(results_path: Path) -> dict[str, Optional[float]]:
    df = pd.read_csv(results_path, low_memory=True)
    if "game_id" not in df.columns:
        raise ValueError("results missing game_id")
    df["game_id"] = df["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)

    if "actual_total" in df.columns:
        df["actual_total"] = pd.to_numeric(df["actual_total"], errors="coerce")
    else:
        hs = pd.to_numeric(df.get("home_score"), errors="coerce")
        as_ = pd.to_numeric(df.get("away_score"), errors="coerce")
        df["actual_total"] = hs + as_

    out: dict[str, Optional[float]] = {}
    for _, r in df.iterrows():
        gid = str(r.get("game_id") or "").strip()
        if not gid:
            continue
        v = _to_float(r.get("actual_total"))
        out[gid] = v
    return out


def build_live_feature_table(
    *,
    date: str,
    snapshots_path: Path,
    out_csv: Path,
    segments_path: Optional[Path] = None,
    results_path: Optional[Path] = None,
    horizon_min: float = 40.0,
    max_lines: int = 400000,
) -> dict[str, Any]:
    if not snapshots_path.exists():
        raise FileNotFoundError(str(snapshots_path))

    seg_idx: dict[str, SegmentIndex] = {}
    if segments_path is not None and segments_path.exists():
        seg_idx = _load_segments_index(segments_path)

    actual_total: dict[str, Optional[float]] = {}
    if results_path is not None and results_path.exists():
        actual_total = _load_results_actual_total(results_path)

    last_line: dict[str, dict[str, Any]] = {}
    last_pbp: dict[str, dict[str, Any]] = {}

    rows: list[dict[str, Any]] = []

    for rec in _parse_jsonl(snapshots_path, max_lines=max_lines):
        ep = str(rec.get("endpoint") or "").strip()
        gid = str(rec.get("event_id") or rec.get("game_id") or "").strip()
        if not gid:
            continue
        data = rec.get("data")
        if not isinstance(data, dict):
            continue

        if ep == "live_lines":
            last_line[gid] = dict(data)
            continue

        if ep == "live_pbp_stats":
            # Keep a compact rollup so table rows don't bloat.
            stats = data.get("stats") if isinstance(data.get("stats"), dict) else {}
            home = stats.get("home") if isinstance(stats.get("home"), dict) else {}
            away = stats.get("away") if isinstance(stats.get("away"), dict) else {}

            poss = (_to_float((home or {}).get("poss_est")) or 0.0) + (_to_float((away or {}).get("poss_est")) or 0.0)
            tov = (_to_int((home or {}).get("to")) or 0) + (_to_int((away or {}).get("to")) or 0)
            orb = (_to_int((home or {}).get("orb")) or 0) + (_to_int((away or {}).get("orb")) or 0)
            drb = (_to_int((home or {}).get("drb")) or 0) + (_to_int((away or {}).get("drb")) or 0)

            last_pbp[gid] = {
                "pbp_poss_est": float(poss),
                "pbp_to": int(tov),
                "pbp_orb": int(orb),
                "pbp_drb": int(drb),
                "pbp_fetched_from": data.get("fetched_from"),
            }
            continue

        if ep != "live_state":
            continue

        total_points = _to_int(data.get("total_points"))
        rem_reg_s = _to_float(data.get("remaining_reg_seconds"))
        period = _to_int(data.get("period"))
        ts = rec.get("ts")

        elapsed_min = None
        remaining_min = None
        if rem_reg_s is not None:
            remaining_min = max(0.0, float(rem_reg_s) / 60.0)
            elapsed_min = float(horizon_min) - remaining_min
            if elapsed_min < 0:
                elapsed_min = 0.0
            if elapsed_min > float(horizon_min):
                elapsed_min = float(horizon_min)

        exp_total = _interp_total(seg_idx, gid, float(elapsed_min)) if (elapsed_min is not None and seg_idx) else None
        sim_final_total = _interp_total(seg_idx, gid, float(horizon_min)) if seg_idx else None

        proj_model_total = None
        if total_points is not None and exp_total is not None and sim_final_total is not None:
            proj_model_total = float(total_points) + (float(sim_final_total) - float(exp_total))

        line_rec = last_line.get(gid) or {}
        live_line_total = _to_float(line_rec.get("total"))
        live_line_book = line_rec.get("book")
        live_line_provider_event_id = line_rec.get("event_id_provider")
        live_line_last_update = line_rec.get("last_update")

        w = market_blend_weight(elapsed_min, horizon_min=horizon_min) if (live_line_total is not None and proj_model_total is not None) else 0.0
        proj_blend_total = None
        if live_line_total is not None and proj_model_total is not None:
            proj_blend_total = (1.0 - float(w)) * float(proj_model_total) + float(w) * float(live_line_total)
        else:
            proj_blend_total = proj_model_total

        pbp = last_pbp.get(gid) or {}
        pbp_poss = _to_float(pbp.get("pbp_poss_est"))
        pbp_poss_per_min = (float(pbp_poss) / float(elapsed_min)) if (pbp_poss is not None and elapsed_min and elapsed_min > 0) else None
        pbp_ppp = (float(total_points) / float(pbp_poss)) if (pbp_poss is not None and pbp_poss > 0 and total_points is not None) else None

        act_total = actual_total.get(gid)
        err_model = (float(proj_model_total) - float(act_total)) if (proj_model_total is not None and act_total is not None) else None
        err_blend = (float(proj_blend_total) - float(act_total)) if (proj_blend_total is not None and act_total is not None) else None

        rows.append(
            {
                "date": str(date),
                "ts": ts,
                "game_id": str(gid),
                "period": period,
                "elapsed_min": elapsed_min,
                "remaining_min": remaining_min,
                "total_points": total_points,
                "home_score": _to_int(data.get("home_score")),
                "away_score": _to_int(data.get("away_score")),
                "pbp_poss_est": pbp_poss,
                "pbp_poss_per_min": pbp_poss_per_min,
                "pbp_ppp": pbp_ppp,
                "pbp_to": _to_int(pbp.get("pbp_to")),
                "pbp_orb": _to_int(pbp.get("pbp_orb")),
                "pbp_drb": _to_int(pbp.get("pbp_drb")),
                "pbp_fetched_from": pbp.get("pbp_fetched_from"),
                "live_line_total": live_line_total,
                "live_line_book": live_line_book,
                "live_line_provider_event_id": live_line_provider_event_id,
                "live_line_last_update": live_line_last_update,
                "exp_total_at_elapsed": exp_total,
                "sim_final_total": sim_final_total,
                "proj_model_total": proj_model_total,
                "blend_w": float(w),
                "proj_blend_total": proj_blend_total,
                "actual_total": act_total,
                "err_model": err_model,
                "err_blend": err_blend,
            }
        )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    return {
        "status": "ok",
        "date": str(date),
        "snapshots": str(snapshots_path),
        "segments": str(segments_path) if segments_path is not None else None,
        "results": str(results_path) if results_path is not None else None,
        "rows": int(len(df)),
        "games": int(df["game_id"].nunique()) if not df.empty and "game_id" in df.columns else 0,
        "out_csv": str(out_csv),
    }
