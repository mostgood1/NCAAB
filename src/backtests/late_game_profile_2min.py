from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ncaab_model.config import settings
from ncaab_model.data.cache import read_json
from ncaab_model.data.adapters.espn_playbyplay import _clock_to_remaining_seconds, _iter_plays


@dataclass(frozen=True)
class LateGame2MinProfileConfig:
    start: Optional[str]
    end: Optional[str]
    out_dir: Path
    cache_dir: Path
    out_prefix: str = "late_game_2min_profile"


def _safe_get(obj: object, *path: str) -> object:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _parse_game_date(summary: dict) -> Optional[dt.date]:
    # Prefer header.competitions[0].date (ISO)
    comp0 = None
    header = summary.get("header") if isinstance(summary, dict) else None
    if isinstance(header, dict):
        comps = header.get("competitions")
        if isinstance(comps, list) and comps:
            comp0 = comps[0]

    if isinstance(comp0, dict):
        ds = comp0.get("date")
        if ds:
            try:
                # Example: 2024-01-20T19:00Z
                return dt.datetime.fromisoformat(str(ds).replace("Z", "+00:00")).date()
            except Exception:
                pass

    # Fallback: gameInfo.startTime
    ds2 = _safe_get(summary, "gameInfo", "startTime")
    if ds2:
        try:
            return dt.datetime.fromisoformat(str(ds2).replace("Z", "+00:00")).date()
        except Exception:
            return None

    return None


def _extract_regulation_scores_2min_and_final(summary: dict) -> Optional[dict]:
    # We want the score at the start of the last 2:00 of the 2H (regulation)
    # and the final score at the end of regulation.
    plays = summary.get("plays") if isinstance(summary, dict) else None
    if not isinstance(plays, list) or not plays:
        return None

    best_baseline_rem: Optional[int] = None
    baseline_home = None
    baseline_away = None

    best_final_rem: Optional[int] = None
    final_home = None
    final_away = None

    had_overtime = False

    for p in _iter_plays(summary):
        try:
            per = p.get("period") or {}
            per_num = per.get("number") or per.get("value")
            per_num = int(per_num)
        except Exception:
            continue

        if per_num and per_num > 2:
            had_overtime = True

        if per_num != 2:
            continue

        try:
            clock = p.get("clock") or {}
            disp = clock.get("displayValue") if isinstance(clock, dict) else None
            rem = _clock_to_remaining_seconds(disp)
        except Exception:
            rem = None

        if rem is None:
            rem = _clock_to_remaining_seconds(p.get("clock"))

        if rem is None:
            continue

        try:
            hs = p.get("homeScore")
            a_s = p.get("awayScore")
            if hs is None or a_s is None:
                continue
            hs_i = int(hs)
            as_i = int(a_s)
        except Exception:
            continue

        # Baseline is the play closest to 2:00 remaining but still ABOVE 2:00.
        if rem > 120:
            if best_baseline_rem is None or rem < best_baseline_rem:
                best_baseline_rem = int(rem)
                baseline_home = hs_i
                baseline_away = as_i

        # Final regulation is the play closest to 0 remaining (smallest rem).
        if best_final_rem is None or rem < best_final_rem:
            best_final_rem = int(rem)
            final_home = hs_i
            final_away = as_i

    if baseline_home is None or baseline_away is None or final_home is None or final_away is None:
        return None

    base_total = int(baseline_home + baseline_away)
    fin_total = int(final_home + final_away)

    return {
        "home_score_2m": int(baseline_home),
        "away_score_2m": int(baseline_away),
        "total_score_2m": int(base_total),
        "home_score_final_reg": int(final_home),
        "away_score_final_reg": int(final_away),
        "total_score_final_reg": int(fin_total),
        "margin_2m": int(baseline_home - baseline_away),
        "last2_home_points": int(final_home - baseline_home),
        "last2_away_points": int(final_away - baseline_away),
        "last2_total_points": int(fin_total - base_total),
        "had_overtime": bool(had_overtime),
    }


def _bucket_margin(m: int) -> str:
    if m == 0:
        return "tie"
    side = "home" if m > 0 else "away"
    a = abs(int(m))
    if a <= 2:
        return f"{side}_lead_1_2"
    if a <= 5:
        return f"{side}_lead_3_5"
    if a <= 9:
        return f"{side}_lead_6_9"
    return f"{side}_lead_10p"


def run_late_game_2min_profile(cfg: LateGame2MinProfileConfig) -> dict:
    start_d = dt.date.fromisoformat(cfg.start) if cfg.start else None
    end_d = dt.date.fromisoformat(cfg.end) if cfg.end else None

    rows: list[dict] = []
    n_files = 0
    n_used = 0

    cache_dir = Path(cfg.cache_dir)
    if not cache_dir.exists():
        return {
            "start": cfg.start,
            "end": cfg.end,
            "files_seen": 0,
            "games_used": 0,
            "out_csv": None,
            "out_json": None,
            "error": f"Missing cache dir: {cache_dir}",
        }

    for p in sorted(cache_dir.glob("*.json")):
        n_files += 1
        try:
            payload = read_json(p)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        game_date = _parse_game_date(payload)
        if game_date is None:
            continue
        if start_d and game_date < start_d:
            continue
        if end_d and game_date > end_d:
            continue

        extracted = _extract_regulation_scores_2min_and_final(payload)
        if extracted is None:
            continue

        event_id = p.stem
        rows.append(
            {
                "event_id": str(event_id),
                "date": game_date.isoformat(),
                "margin_2m": int(extracted["margin_2m"]),
                "bucket": _bucket_margin(int(extracted["margin_2m"])),
                "total_2m": int(extracted["total_score_2m"]),
                "total_final_reg": int(extracted["total_score_final_reg"]),
                "last2_total_points": int(extracted["last2_total_points"]),
                "last2_home_points": int(extracted["last2_home_points"]),
                "last2_away_points": int(extracted["last2_away_points"]),
                "had_overtime": bool(extracted["had_overtime"]),
            }
        )
        n_used += 1

    df = pd.DataFrame(rows)

    out_bt = Path(cfg.out_dir) / "backtests"
    out_bt.mkdir(parents=True, exist_ok=True)

    tag_parts = []
    if cfg.start:
        tag_parts.append(cfg.start)
    if cfg.end:
        tag_parts.append(cfg.end)
    tag = ("_to_".join(tag_parts) if tag_parts else "all").replace(":", "-")

    out_csv = out_bt / f"{cfg.out_prefix}_{tag}.csv"
    out_json = out_bt / f"{cfg.out_prefix}_{tag}.json"

    summary: dict = {
        "start": cfg.start,
        "end": cfg.end,
        "files_seen": int(n_files),
        "games_used": int(n_used),
        "out_csv": str(out_csv),
        "out_json": str(out_json),
        "by_bucket": [],
    }

    if df.empty:
        summary["rows"] = 0
    else:
        summary["rows"] = int(len(df))
        summary["mean_last2_total_points"] = float(df["last2_total_points"].mean())
        summary["mean_last2_total_points_no_ot"] = float(df.loc[~df["had_overtime"], "last2_total_points"].mean()) if (~df["had_overtime"]).any() else None

        by_bucket = []
        for b, g in df.groupby("bucket"):
            by_bucket.append(
                {
                    "bucket": str(b),
                    "n": int(len(g)),
                    "mean_margin_2m": float(g["margin_2m"].mean()),
                    "mean_last2_total_points": float(g["last2_total_points"].mean()),
                    "p50_last2_total_points": float(np.quantile(g["last2_total_points"].to_numpy(dtype=float), 0.50)),
                    "p90_last2_total_points": float(np.quantile(g["last2_total_points"].to_numpy(dtype=float), 0.90)),
                }
            )
        by_bucket.sort(key=lambda x: (x["bucket"]))
        summary["by_bucket"] = by_bucket

        try:
            df.to_csv(out_csv, index=False)
        except Exception:
            pass

    try:
        out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass

    return summary


def default_config(
    start: Optional[str] = None,
    end: Optional[str] = None,
    out_prefix: str = "late_game_2min_profile",
) -> LateGame2MinProfileConfig:
    return LateGame2MinProfileConfig(
        start=start,
        end=end,
        out_dir=settings.outputs_dir,
        cache_dir=settings.data_dir / "cache" / "espn_summary",
        out_prefix=str(out_prefix),
    )
