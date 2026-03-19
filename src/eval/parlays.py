from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from src.eval.high_likelihood import HighLikelihoodConfig, build_high_likelihood


def _parse_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(str(value).strip()).isoformat()
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _norm_gid(value: Any) -> str:
    try:
        if value is None:
            return ""
        text = str(value).strip()
        return text[:-2] if text.endswith(".0") else text
    except Exception:
        return ""


def _american_profit_per_1_risk(odds: float | None) -> float:
    if odds is None:
        return 100.0 / 110.0
    try:
        value = float(odds)
        if not math.isfinite(value) or value == 0:
            return 100.0 / 110.0
        if value > 0:
            return value / 100.0
        return 100.0 / (-value)
    except Exception:
        return 100.0 / 110.0


def _american_implied_prob(odds: float | None) -> float | None:
    if odds is None:
        return None
    try:
        value = float(odds)
        if not math.isfinite(value) or value == 0:
            return None
        if value > 0:
            return 100.0 / (value + 100.0)
        return (-value) / ((-value) + 100.0)
    except Exception:
        return None


def _profit_to_american(profit_units: float | None) -> int | None:
    if profit_units is None:
        return None
    try:
        profit = float(profit_units)
    except Exception:
        return None
    if not math.isfinite(profit) or profit <= 0:
        return None
    if profit >= 1.0:
        return int(round(profit * 100.0))
    return int(round(-100.0 / profit))


def _current_local_naive() -> dt.datetime:
    return dt.datetime.now().astimezone().replace(tzinfo=None)


def _parse_start_local(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for candidate in (text, text.replace(" ", "T", 1)):
        try:
            parsed = dt.datetime.fromisoformat(candidate)
            return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed
        except Exception:
            continue
    return None


def _market_label(code: str) -> str:
    code = (code or "").upper()
    if code == "ATS":
        return "Spread"
    if code == "OU":
        return "Total"
    if code == "ML":
        return "Moneyline"
    return code or "Pick"


def _format_pick_label(row: dict[str, Any]) -> str:
    code = str(row.get("rec_code") or "").upper()
    away = str(row.get("away_team") or "").strip()
    home = str(row.get("home_team") or "").strip()
    matchup = f"{away} @ {home}".strip(" @")
    line = _to_float(row.get("line"))

    if code == "ML":
        selection = str(row.get("selection_team") or row.get("selection") or matchup).strip()
        return selection or matchup

    if code == "OU":
        selection = str(row.get("selection") or "").strip()
        if line is not None:
            return f"{selection} {line:.1f} - {matchup}" if matchup else f"{selection} {line:.1f}"
        return f"{selection} - {matchup}" if matchup else selection

    if code == "ATS":
        selection = str(row.get("selection") or "").strip().lower()
        if selection == "home":
            if line is not None:
                return f"{home} {line:+.1f} - {matchup}" if matchup else f"{home} {line:+.1f}"
            return home or matchup
        if selection == "away":
            if line is not None:
                return f"{away} {-line:+.1f} - {matchup}" if matchup else f"{away} {-line:+.1f}"
            return away or matchup

    selection = str(row.get("selection_team") or row.get("selection") or matchup).strip()
    return selection or matchup


def _decorate_pick(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    code = str(out.get("rec_code") or "").upper()
    away = str(out.get("away_team") or "").strip()
    home = str(out.get("home_team") or "").strip()
    out["game_id"] = _norm_gid(out.get("game_id"))
    out["market_label"] = _market_label(code)
    out["matchup"] = f"{away} @ {home}".strip(" @")
    out["display_pick"] = _format_pick_label(out)
    return out


def _pick_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    score = _to_float(row.get("score")) or 0.0
    p_win = _to_float(row.get("p_win")) or 0.0
    edge = abs(_to_float(row.get("edge")) or 0.0)
    implied = _american_implied_prob(_to_float(row.get("price"))) or 0.0
    return (-score, -p_win, -edge, -implied)


def _combo_profit_units(combo: tuple[dict[str, Any], ...]) -> float:
    total_return = 1.0
    for leg in combo:
        total_return *= 1.0 + _american_profit_per_1_risk(_to_float(leg.get("price")))
    return total_return - 1.0


@dataclass(frozen=True)
class BestBetsParlayConfig:
    out_dir: Path
    date: str
    min_score: float = 70.0
    best_bets: int = 8
    candidate_pool: int = 10
    parlay_size: int = 4
    max_parlays: int = 5
    future_only: bool = True
    include_markets: tuple[str, ...] = ("ATS", "OU", "ML")


def build_best_bets_and_parlays(cfg: BestBetsParlayConfig) -> dict[str, Any]:
    date = _parse_date(cfg.date)
    if not date:
        return {"status": "error", "message": f"invalid date: {cfg.date}"}

    markets = tuple(str(m or "").strip().upper() for m in (cfg.include_markets or ("ATS", "OU", "ML")) if str(m or "").strip())
    if not markets:
        markets = ("ATS", "OU", "ML")

    best_bets_count = max(1, min(25, int(cfg.best_bets)))
    parlay_size = max(2, min(6, int(cfg.parlay_size)))
    max_parlays = max(1, min(20, int(cfg.max_parlays)))
    candidate_pool = max(parlay_size, min(15, int(cfg.candidate_pool)))
    per_market_top_n = max(25, best_bets_count * 4, candidate_pool * 4, parlay_size * max_parlays)
    per_market_top_n = min(100, per_market_top_n)

    all_candidates: list[dict[str, Any]] = []
    source_summary: list[dict[str, Any]] = []
    errors: list[str] = []

    for market in markets:
        result = build_high_likelihood(
            HighLikelihoodConfig(
                out_dir=Path(cfg.out_dir),
                date=date,
                top_n=per_market_top_n,
                min_score=float(cfg.min_score),
                include_markets=(market,),
                market_preference=(market,),
                max_picks_per_game=1,
                max_ats_picks=per_market_top_n,
                max_ou_picks=per_market_top_n,
                max_ml_picks=per_market_top_n,
            )
        )
        picks = []
        status = str(result.get("status") or "error") if isinstance(result, dict) else "error"
        if status == "ok" and isinstance(result, dict):
            picks = list(result.get("picks") or [])
            all_candidates.extend(picks)
        else:
            message = result.get("message") if isinstance(result, dict) else None
            if message:
                errors.append(f"{market}: {message}")
        source_summary.append({"market": market, "status": status, "count": len(picks)})

    if not all_candidates:
        message = "; ".join(errors) if errors else f"no high-likelihood picks available for {date}"
        return {
            "status": "error",
            "date": date,
            "message": message,
            "source_summary": source_summary,
        }

    now_local = _current_local_naive()
    filtered_candidates: list[dict[str, Any]] = []
    for row in all_candidates:
        if bool(cfg.future_only):
            start_local = _parse_start_local(row.get("start_time_local"))
            if start_local is not None and start_local < now_local:
                continue
        filtered_candidates.append(row)

    ranked_unique: list[dict[str, Any]] = []
    seen_games: set[str] = set()
    for row in sorted(filtered_candidates, key=_pick_sort_key):
        gid = _norm_gid(row.get("game_id"))
        if not gid or gid in seen_games:
            continue
        seen_games.add(gid)
        ranked_unique.append(_decorate_pick(row))

    message: str | None = None
    if bool(cfg.future_only) and not ranked_unique:
        message = "No future picks remain after filtering games that have already started."

    best_bets = ranked_unique[:best_bets_count]
    best_bet = best_bets[0] if best_bets else None

    parlay_pool = ranked_unique[:candidate_pool]
    parlays: list[dict[str, Any]] = []
    if len(parlay_pool) >= parlay_size:
        for combo in combinations(parlay_pool, parlay_size):
            combined_p = 1.0
            avg_score = 0.0
            min_leg_score = 100.0
            for leg in combo:
                p_win = _to_float(leg.get("p_win")) or 0.0
                score = _to_float(leg.get("score")) or 0.0
                combined_p *= p_win
                avg_score += score
                min_leg_score = min(min_leg_score, score)
            avg_score /= float(parlay_size)
            profit_units = _combo_profit_units(combo)
            expected_units = combined_p * profit_units - (1.0 - combined_p)
            parlays.append(
                {
                    "legs": [dict(leg) for leg in combo],
                    "combined_p_win": combined_p,
                    "avg_leg_score": avg_score,
                    "min_leg_score": min_leg_score,
                    "profit_units_per_1_risk": profit_units,
                    "expected_units_per_1_risk": expected_units,
                    "approx_american_odds": _profit_to_american(profit_units),
                }
            )
        parlays.sort(
            key=lambda row: (
                -float(row.get("combined_p_win") or 0.0),
                -float(row.get("expected_units_per_1_risk") or 0.0),
                -float(row.get("min_leg_score") or 0.0),
                -float(row.get("avg_leg_score") or 0.0),
            )
        )
        parlays = parlays[:max_parlays]

    return {
        "status": "ok",
        "date": date,
        "generated_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "as_of_local": now_local.isoformat(timespec="seconds"),
        "params": {
            "min_score": float(cfg.min_score),
            "best_bets": best_bets_count,
            "candidate_pool": candidate_pool,
            "parlay_size": parlay_size,
            "max_parlays": max_parlays,
            "future_only": bool(cfg.future_only),
            "include_markets": list(markets),
        },
        "message": message,
        "source_summary": source_summary,
        "candidate_count": len(ranked_unique),
        "best_bet": best_bet,
        "best_bets": best_bets,
        "parlays": parlays,
    }