from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ncaab_model.data.adapters.espn_playbyplay import fetch_playbyplay, extract_cum_totals_5min


@dataclass(frozen=True)
class TuneSegments2MinTeamConfig:
    out_dir: Path
    start: str
    end: str
    endpoints: tuple[int, ...] = tuple(range(2, 41, 2))
    use_cache: bool = True
    sleep_seconds: float = 0.15
    max_games: int = 0
    min_games_per_team: int = 8
    shrink_to_global: float = 0.35  # 0=no shrink, 1=all global
    shrink_to_uniform: float = 0.02  # 0=no shrink, 1=all uniform
    out_path: Path | None = None


def _date_range(start_iso: str, end_iso: str) -> Iterable[dt.date]:
    s = dt.date.fromisoformat(start_iso)
    e = dt.date.fromisoformat(end_iso)
    cur = s
    one = dt.timedelta(days=1)
    while cur <= e:
        yield cur
        cur += one


def _normalize(v: np.ndarray, uniform_n: int) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    v = np.where(np.isfinite(v), v, 0.0)
    v = np.clip(v, 0.0, None)
    s = float(v.sum())
    if s <= 0:
        return np.array([1.0 / float(uniform_n)] * int(uniform_n), dtype=float)
    return (v / s).astype(float)


def _norm_team_key(v: object) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    return s if s else None


def _pick_game_team_map(out_dir: Path, date_iso: str) -> dict[str, tuple[str | None, str | None]]:
    """Return mapping game_id -> (home_team, away_team) for a date.

    Prefers sim_segments_2min_<date>.csv, falls back to sim_segments_<date>.csv.
    """
    candidates = [out_dir / f"sim_segments_2min_{date_iso}.csv", out_dir / f"sim_segments_{date_iso}.csv"]
    for p in candidates:
        if not p.exists():
            continue
        try:
            import pandas as pd

            df = pd.read_csv(p)
            if df.empty or "game_id" not in df.columns:
                continue
            for c in ("game_id", "home_team", "away_team"):
                if c in df.columns:
                    df[c] = df[c].astype(str)
            # Keep first row per game_id
            g = df.groupby("game_id", as_index=False).first()
            out: dict[str, tuple[str | None, str | None]] = {}
            for _, r in g.iterrows():
                gid = str(r.get("game_id") or "").strip().replace(".0", "")
                if not gid:
                    continue
                ht = _norm_team_key(r.get("home_team"))
                at = _norm_team_key(r.get("away_team"))
                out[gid] = (ht, at)
            if out:
                return out
        except Exception:
            continue
    return {}


def tune_segment_weights_2min_by_team(cfg: TuneSegments2MinTeamConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    endpoints = [int(x) for x in cfg.endpoints]
    if endpoints != list(range(2, 41, 2)):
        endpoints = sorted({int(x) for x in endpoints if int(x) > 0})

    # 10 segments per half for 2-minute grid.
    segs_per_half = 10
    uniform = np.array([1.0 / float(segs_per_half)] * segs_per_half, dtype=float)

    # Global accumulation (per-game shares)
    sum_share_g_h1 = np.zeros(segs_per_half, dtype=float)
    sum_share_g_h2 = np.zeros(segs_per_half, dtype=float)
    n_g_h1 = 0
    n_g_h2 = 0

    # Team accumulators
    sum_share_team_h1: dict[str, np.ndarray] = {}
    sum_share_team_h2: dict[str, np.ndarray] = {}
    n_team_h1: dict[str, int] = {}
    n_team_h2: dict[str, int] = {}

    n_games = 0
    n_missing = 0

    for d in _date_range(cfg.start, cfg.end):
        date_iso = d.isoformat()

        game_team = _pick_game_team_map(out_dir, date_iso)
        game_ids = sorted(game_team.keys())
        if cfg.max_games and int(cfg.max_games) > 0:
            game_ids = game_ids[: int(cfg.max_games)]

        for gid in game_ids:
            payload = fetch_playbyplay(gid, use_cache=bool(cfg.use_cache))
            fetched_from = payload.get("_fetched_from") if isinstance(payload, dict) else None
            did_network = isinstance(fetched_from, str) and fetched_from.startswith("network")

            plays = payload.get("plays") if isinstance(payload, dict) else None
            if payload is None or not isinstance(plays, list) or len(plays) == 0:
                n_missing += 1
                if did_network and cfg.sleep_seconds and float(cfg.sleep_seconds) > 0:
                    time.sleep(float(cfg.sleep_seconds))
                continue

            cum = extract_cum_totals_5min(payload, endpoints=endpoints)
            m_h = {int(x.end_min): int(x.home_score) for x in cum}
            m_a = {int(x.end_min): int(x.away_score) for x in cum}
            if not all(k in m_h and k in m_a for k in endpoints):
                n_missing += 1
                if did_network and cfg.sleep_seconds and float(cfg.sleep_seconds) > 0:
                    time.sleep(float(cfg.sleep_seconds))
                continue

            # Convert to per-segment increments for each team.
            ends = endpoints
            prev_h = 0
            prev_a = 0
            h_inc = []
            a_inc = []
            for em in ends:
                h = int(m_h.get(int(em), prev_h))
                a = int(m_a.get(int(em), prev_a))
                h_inc.append(max(0, h - prev_h))
                a_inc.append(max(0, a - prev_a))
                prev_h = h
                prev_a = a
            h_inc = np.asarray(h_inc, dtype=float)
            a_inc = np.asarray(a_inc, dtype=float)
            t_inc = h_inc + a_inc

            # Split into halves (2..20 inclusive is 10 segments; 22..40 is 10 segments)
            h1 = h_inc[:segs_per_half]
            h2 = h_inc[segs_per_half: 2 * segs_per_half]
            a1 = a_inc[:segs_per_half]
            a2 = a_inc[segs_per_half: 2 * segs_per_half]
            t1 = t_inc[:segs_per_half]
            t2 = t_inc[segs_per_half: 2 * segs_per_half]

            # Global per-game shares
            if float(t1.sum()) > 0:
                sum_share_g_h1 += (t1 / float(t1.sum()))
                n_g_h1 += 1
            if float(t2.sum()) > 0:
                sum_share_g_h2 += (t2 / float(t2.sum()))
                n_g_h2 += 1

            # Per-team shares
            ht, at = game_team.get(gid, (None, None))
            if ht:
                if float(h1.sum()) > 0:
                    sum_share_team_h1.setdefault(ht, np.zeros(segs_per_half, dtype=float))
                    sum_share_team_h1[ht] += (h1 / float(h1.sum()))
                    n_team_h1[ht] = int(n_team_h1.get(ht, 0) + 1)
                if float(h2.sum()) > 0:
                    sum_share_team_h2.setdefault(ht, np.zeros(segs_per_half, dtype=float))
                    sum_share_team_h2[ht] += (h2 / float(h2.sum()))
                    n_team_h2[ht] = int(n_team_h2.get(ht, 0) + 1)

            if at:
                if float(a1.sum()) > 0:
                    sum_share_team_h1.setdefault(at, np.zeros(segs_per_half, dtype=float))
                    sum_share_team_h1[at] += (a1 / float(a1.sum()))
                    n_team_h1[at] = int(n_team_h1.get(at, 0) + 1)
                if float(a2.sum()) > 0:
                    sum_share_team_h2.setdefault(at, np.zeros(segs_per_half, dtype=float))
                    sum_share_team_h2[at] += (a2 / float(a2.sum()))
                    n_team_h2[at] = int(n_team_h2.get(at, 0) + 1)

            n_games += 1
            if did_network and cfg.sleep_seconds and float(cfg.sleep_seconds) > 0:
                time.sleep(float(cfg.sleep_seconds))

    global_h1 = _normalize(sum_share_g_h1 / float(max(n_g_h1, 1)), uniform_n=segs_per_half)
    global_h2 = _normalize(sum_share_g_h2 / float(max(n_g_h2, 1)), uniform_n=segs_per_half)

    shrink_g = float(np.clip(float(cfg.shrink_to_global), 0.0, 1.0))
    shrink_u = float(np.clip(float(cfg.shrink_to_uniform), 0.0, 1.0))

    def _shrink(w: np.ndarray, g: np.ndarray) -> np.ndarray:
        w2 = (1.0 - shrink_g) * w + shrink_g * g
        w3 = (1.0 - shrink_u) * w2 + shrink_u * uniform
        return _normalize(w3, uniform_n=segs_per_half)

    teams_out: dict[str, dict] = {}
    all_teams = sorted(set(sum_share_team_h1.keys()) | set(sum_share_team_h2.keys()))
    for team in all_teams:
        n1 = int(n_team_h1.get(team, 0))
        n2 = int(n_team_h2.get(team, 0))
        if max(n1, n2) < int(cfg.min_games_per_team):
            continue

        w1_raw = (sum_share_team_h1.get(team) / float(max(n1, 1))) if team in sum_share_team_h1 else global_h1
        w2_raw = (sum_share_team_h2.get(team) / float(max(n2, 1))) if team in sum_share_team_h2 else global_h2
        w1 = _shrink(_normalize(w1_raw, uniform_n=segs_per_half), global_h1)
        w2 = _shrink(_normalize(w2_raw, uniform_n=segs_per_half), global_h2)

        teams_out[team] = {
            "half1": [float(x) for x in w1.tolist()],
            "half2": [float(x) for x in w2.tolist()],
            "games_h1": n1,
            "games_h2": n2,
        }

    out = {
        "start": cfg.start,
        "end": cfg.end,
        "grid_min": 2,
        "segments_per_half": segs_per_half,
        "endpoints": endpoints,
        "method": "empirical_points_share_from_pbp_team_split",
        "games_used": int(n_games),
        "global_half1_games": int(n_g_h1),
        "global_half2_games": int(n_g_h2),
        "pbp_missing_or_incomplete": int(n_missing),
        "min_games_per_team": int(cfg.min_games_per_team),
        "shrink_to_global": shrink_g,
        "shrink_to_uniform": shrink_u,
        "global": {
            "half1": [float(x) for x in global_h1.tolist()],
            "half2": [float(x) for x in global_h2.tolist()],
        },
        "teams": teams_out,
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
    }

    out_path = cfg.out_path
    if out_path is None:
        out_path = out_dir / "team_segment_weights_2min.json"

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass

    out["out_path"] = str(out_path)
    out["teams_written"] = int(len(teams_out))
    return out
