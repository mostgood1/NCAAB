from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def _normalize_name(value: Any) -> str:
    try:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        chars: list[str] = []
        last_space = False
        for ch in text:
            if ch.isalnum():
                chars.append(ch)
                last_space = False
            elif not last_space:
                chars.append(" ")
                last_space = True
        return " ".join("".join(chars).split())
    except Exception:
        return ""


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, dtype={"game_id": str}, low_memory=False)
        if "game_id" in df.columns:
            try:
                df["game_id"] = df["game_id"].map(_norm_gid)
            except Exception:
                pass
        return df
    except Exception:
        return pd.DataFrame()


def _matchup_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            out = float(value)
            return out if math.isfinite(out) else None
        text = str(value).strip()
        if not text or text.lower() in ("nan", "none", "null"):
            return None
        out = float(text)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _pair_key(home_team: Any, away_team: Any) -> str:
    home_key = _normalize_name(home_team)
    away_key = _normalize_name(away_team)
    return f"{home_key}::{away_key}" if home_key and away_key else ""


def _selected_side(row: dict[str, Any]) -> str | None:
    try:
        selection = str(
            row.get("selection_team")
            or row.get("selection")
            or row.get("display_pick")
            or ""
        ).strip()
        if not selection:
            return None
        selection_lower = selection.lower()
        if selection_lower in ("home", "h"):
            return "home"
        if selection_lower in ("away", "a"):
            return "away"
        selection_key = _normalize_name(selection)
        home_key = _normalize_name(row.get("home_team") or "")
        away_key = _normalize_name(row.get("away_team") or "")
        if home_key and selection_key.startswith(home_key):
            return "home"
        if away_key and selection_key.startswith(away_key):
            return "away"
    except Exception:
        return None
    return None


def _expected_combined_ppp(feature_row: dict[str, Any]) -> float | None:
    home_ppp = _matchup_float(feature_row.get("home_ppp_mu"))
    away_ppp = _matchup_float(feature_row.get("away_ppp_mu"))
    home_allowed = _matchup_float(feature_row.get("home_ppp_allowed_mu"))
    away_allowed = _matchup_float(feature_row.get("away_ppp_allowed_mu"))

    if home_ppp is not None and away_allowed is not None:
        exp_home = (home_ppp + away_allowed) / 2.0
    else:
        exp_home = home_ppp if home_ppp is not None else away_allowed

    if away_ppp is not None and home_allowed is not None:
        exp_away = (away_ppp + home_allowed) / 2.0
    else:
        exp_away = away_ppp if away_ppp is not None else home_allowed

    if exp_home is None or exp_away is None:
        return None
    return float(exp_home + exp_away)


def _load_matchup_context(out_dir: Path, date: str | None) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "by_game": {},
        "by_pair": {},
        "pace_median": None,
        "combined_ppp_median": None,
        "source": None,
    }
    try:
        candidates: list[Path] = []
        if date:
            candidates.append(Path(out_dir) / f"features_{date}.csv")
            candidates.append(Path(out_dir) / f"live_features_{date}.csv")
        candidates.append(Path(out_dir) / "features_curr.csv")

        feat_df = pd.DataFrame()
        src_name = None
        for path in candidates:
            tmp = _safe_read_csv(path)
            if not tmp.empty:
                feat_df = tmp.copy()
                src_name = path.name
                break
        if feat_df.empty:
            return ctx

        if "pace_game_est" in feat_df.columns:
            try:
                pace_vals = pd.to_numeric(feat_df["pace_game_est"], errors="coerce").dropna()
                if len(pace_vals):
                    ctx["pace_median"] = float(pace_vals.median())
            except Exception:
                pass

        combined_ppp_vals: list[float] = []
        for feat_row in feat_df.to_dict(orient="records"):
            combo = _expected_combined_ppp(feat_row)
            if combo is not None:
                combined_ppp_vals.append(combo)
            gid = _norm_gid(feat_row.get("game_id"))
            if gid:
                ctx["by_game"][gid] = feat_row
            pk = _pair_key(feat_row.get("home_team"), feat_row.get("away_team"))
            if pk:
                ctx["by_pair"][pk] = feat_row

        if combined_ppp_vals:
            ctx["combined_ppp_median"] = float(pd.Series(combined_ppp_vals).median())
        ctx["source"] = src_name
    except Exception:
        return ctx
    return ctx


def _build_basketball_matchup_logic(row: dict[str, Any], feature_ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        code = str(row.get("rec_code") or "").strip().upper()
        if code not in ("ATS", "ML", "OU"):
            return {}

        gid = _norm_gid(row.get("game_id"))
        feature_row = feature_ctx.get("by_game", {}).get(gid)
        if feature_row is None:
            feature_row = feature_ctx.get("by_pair", {}).get(
                _pair_key(row.get("home_team"), row.get("away_team"))
            )
        if not isinstance(feature_row, dict) or not feature_row:
            return {}

        pace_median = _matchup_float(feature_ctx.get("pace_median"))
        combined_ppp_median = _matchup_float(feature_ctx.get("combined_ppp_median"))
        source_name = str(feature_ctx.get("source") or "")

        home_team = str(row.get("home_team") or feature_row.get("home_team") or "").strip()
        away_team = str(row.get("away_team") or feature_row.get("away_team") or "").strip()
        pace_game = _matchup_float(feature_row.get("pace_game_est"))

        if code in ("ATS", "ML"):
            selected_side = _selected_side(row)
            if selected_side not in ("home", "away"):
                return {}

            if selected_side == "home":
                selected_team = home_team
                opp_team = away_team
                selected_off = _matchup_float(feature_row.get("home_off_rating"))
                opp_off = _matchup_float(feature_row.get("away_off_rating"))
                selected_def = _matchup_float(feature_row.get("home_def_rating"))
                opp_def = _matchup_float(feature_row.get("away_def_rating"))
                selected_ppp = _matchup_float(feature_row.get("home_ppp_mu"))
                opp_ppp = _matchup_float(feature_row.get("away_ppp_mu"))
                selected_allowed = _matchup_float(feature_row.get("home_ppp_allowed_mu"))
                opp_allowed = _matchup_float(feature_row.get("away_ppp_allowed_mu"))
                selected_rest = _matchup_float(feature_row.get("rest_home"))
                opp_rest = _matchup_float(feature_row.get("rest_away"))
                model_margin = _matchup_float(row.get("pred_margin"))
            else:
                selected_team = away_team
                opp_team = home_team
                selected_off = _matchup_float(feature_row.get("away_off_rating"))
                opp_off = _matchup_float(feature_row.get("home_off_rating"))
                selected_def = _matchup_float(feature_row.get("away_def_rating"))
                opp_def = _matchup_float(feature_row.get("home_def_rating"))
                selected_ppp = _matchup_float(feature_row.get("away_ppp_mu"))
                opp_ppp = _matchup_float(feature_row.get("home_ppp_mu"))
                selected_allowed = _matchup_float(feature_row.get("away_ppp_allowed_mu"))
                opp_allowed = _matchup_float(feature_row.get("home_ppp_allowed_mu"))
                selected_rest = _matchup_float(feature_row.get("rest_away"))
                opp_rest = _matchup_float(feature_row.get("rest_home"))
                base_margin = _matchup_float(row.get("pred_margin"))
                model_margin = (-base_margin) if base_margin is not None else None

            attack_edge = (selected_ppp - opp_allowed) if (selected_ppp is not None and opp_allowed is not None) else None
            defense_edge = (opp_ppp - selected_allowed) if (opp_ppp is not None and selected_allowed is not None) else None
            eff_attack = (selected_off - opp_def) if (selected_off is not None and opp_def is not None) else None
            eff_defense = (opp_off - selected_def) if (opp_off is not None and selected_def is not None) else None
            rest_edge = (selected_rest - opp_rest) if (selected_rest is not None and opp_rest is not None) else None

            reasons: list[str] = []
            if (attack_edge is not None) and (attack_edge > 0.025) and (defense_edge is not None) and (defense_edge > 0.025):
                reasons.append(
                    f"{selected_team} owns the cleaner efficiency matchup: its offense projects {attack_edge:.3f} PPP above what {opp_team} usually allows, and its defense grades {defense_edge:.3f} PPP better than {opp_team}'s scoring baseline."
                )
            else:
                if (attack_edge is not None) and (attack_edge > 0.03):
                    reasons.append(f"{selected_team}'s offense projects {attack_edge:.3f} PPP above {opp_team}'s defensive allowance.")
                if (defense_edge is not None) and (defense_edge > 0.03):
                    reasons.append(f"{selected_team}'s defense grades {defense_edge:.3f} PPP better than {opp_team}'s usual scoring rate.")
            if (not reasons) and (eff_attack is not None) and (eff_attack > 2.0) and (eff_defense is not None) and (eff_defense > 2.0):
                reasons.append(
                    f"{selected_team} grades better on both sides here, with a +{eff_attack:.1f} attack edge and a +{eff_defense:.1f} defensive edge in the rating matchup."
                )
            if (rest_edge is not None) and (rest_edge >= 2.0):
                reasons.append(f"{selected_team} also carries a {rest_edge:.0f}-day rest edge into the matchup.")
            if (pace_game is not None) and (pace_median is not None):
                if pace_game <= (pace_median - 1.5):
                    reasons.append(f"Tempo projects slower than the slate median at about {pace_game:.1f} possessions, which points to a more controlled half-court game.")
                elif pace_game >= (pace_median + 1.5):
                    reasons.append(f"Tempo projects above the slate median at about {pace_game:.1f} possessions, creating more chances for the stronger side to separate.")

            line_edge = None
            if code == "ATS":
                line_val = _matchup_float(row.get("line"))
                if (model_margin is not None) and (line_val is not None):
                    line_edge = model_margin + line_val
                    if line_edge > 0.75:
                        reasons.append(f"Projected margin clears the spread by about {line_edge:.1f} points.")
            elif (model_margin is not None) and (model_margin > 1.5):
                reasons.append(f"The win condition is backed by roughly a {model_margin:.1f}-point projected margin.")

            score = 0.0
            if attack_edge is not None:
                score += max(0.0, attack_edge) * 18.0
            if defense_edge is not None:
                score += max(0.0, defense_edge) * 14.0
            if eff_attack is not None:
                score += max(0.0, eff_attack) * 0.45
            if eff_defense is not None:
                score += max(0.0, eff_defense) * 0.35
            if rest_edge is not None:
                score += max(0.0, min(rest_edge, 4.0))
            if line_edge is not None:
                score += max(0.0, min(line_edge * 1.15, 4.0))
            elif model_margin is not None:
                score += max(0.0, min(model_margin * 0.35, 3.5))

            if not reasons:
                return {}
            return {
                "basketball_matchup_score": round(min(score, 25.0), 2),
                "basketball_summary": " ".join(reasons[:3]),
                "basketball_reasons": reasons[:3],
                "basketball_source": source_name,
            }

        selection = str(row.get("selection") or "").strip().lower()
        is_over = selection.startswith("over") or selection.startswith("o ")
        is_under = selection.startswith("under") or selection.startswith("u ")
        if (not is_over) and (not is_under):
            return {}

        line_val = _matchup_float(row.get("line"))
        pred_total = _matchup_float(row.get("pred_total"))
        combined_ppp = _expected_combined_ppp(feature_row)
        expected_total = (pace_game * combined_ppp) if (pace_game is not None and combined_ppp is not None) else None
        reasons = []
        if is_over:
            if (pace_game is not None) and (pace_median is not None) and (pace_game >= (pace_median + 1.5)):
                reasons.append(f"Tempo projects faster than the slate median at about {pace_game:.1f} possessions.")
            if (combined_ppp is not None) and (combined_ppp_median is not None) and (combined_ppp >= (combined_ppp_median + 0.04)):
                reasons.append(f"The combined scoring environment is strong at roughly {combined_ppp:.3f} expected PPP.")
            elif (expected_total is not None) and (line_val is not None) and (expected_total > (line_val + 3.0)):
                reasons.append(f"Feature-based scoring comes in around {expected_total:.1f}, above the market total.")
            if (pred_total is not None) and (line_val is not None) and (pred_total > (line_val + 2.0)):
                reasons.append(f"The model total sits {pred_total - line_val:.1f} points above the number.")
            score = 0.0
            if (pace_game is not None) and (pace_median is not None):
                score += max(0.0, pace_game - pace_median) * 0.8
            if (combined_ppp is not None) and (combined_ppp_median is not None):
                score += max(0.0, combined_ppp - combined_ppp_median) * 30.0
            if (pred_total is not None) and (line_val is not None):
                score += max(0.0, pred_total - line_val) * 0.3
            if (expected_total is not None) and (line_val is not None):
                score += max(0.0, expected_total - line_val) * 0.15
        else:
            if (pace_game is not None) and (pace_median is not None) and (pace_game <= (pace_median - 1.5)):
                reasons.append(f"Tempo projects slower than the slate median at about {pace_game:.1f} possessions.")
            if (combined_ppp is not None) and (combined_ppp_median is not None) and (combined_ppp <= (combined_ppp_median - 0.04)):
                reasons.append(f"The combined scoring environment lands on the lower end of the slate at roughly {combined_ppp:.3f} expected PPP.")
            elif (expected_total is not None) and (line_val is not None) and (expected_total < (line_val - 3.0)):
                reasons.append(f"Feature-based scoring lands around {expected_total:.1f}, below the market total.")
            if (pred_total is not None) and (line_val is not None) and (pred_total < (line_val - 2.0)):
                reasons.append(f"The model total sits {line_val - pred_total:.1f} points below the number.")
            score = 0.0
            if (pace_game is not None) and (pace_median is not None):
                score += max(0.0, pace_median - pace_game) * 0.8
            if (combined_ppp is not None) and (combined_ppp_median is not None):
                score += max(0.0, combined_ppp_median - combined_ppp) * 30.0
            if (pred_total is not None) and (line_val is not None):
                score += max(0.0, line_val - pred_total) * 0.3
            if (expected_total is not None) and (line_val is not None):
                score += max(0.0, line_val - expected_total) * 0.15

        if not reasons:
            return {}
        return {
            "basketball_matchup_score": round(min(score, 20.0), 2),
            "basketball_summary": " ".join(reasons[:3]),
            "basketball_reasons": reasons[:3],
            "basketball_source": source_name,
        }
    except Exception:
        return {}


def _priority_payload(row: dict[str, Any], *, matchup_present: bool = False) -> dict[str, Any]:
    try:
        code = str(row.get("rec_code") or "").upper()

        basketball_raw = _to_float(row.get("basketball_matchup_score"))
        if basketball_raw is None:
            basketball_norm = 0.0
        else:
            basketball_scale = 7.5 if code == "OU" else 10.0
            basketball_norm = min(1.0, max(0.0, basketball_raw / basketball_scale))

        p_win = _to_float(row.get("p_win"))
        prob_strength = min(1.0, max(0.0, abs((p_win or 0.5) - 0.5) * 2.0)) if p_win is not None else 0.0
        base_score = _to_float(row.get("score")) or 0.0
        score_norm = min(1.0, max(0.0, base_score / 100.0))

        model_support = 0.0
        if code == "OU":
            pred_total = _to_float(row.get("pred_total"))
            line_val = _to_float(row.get("line"))
            if (pred_total is not None) and (line_val is not None):
                model_support = min(1.0, max(0.0, pred_total - line_val) / 8.0)
        elif code == "ATS":
            pred_margin = _to_float(row.get("pred_margin"))
            line_val = _to_float(row.get("line"))
            sel_side = _selected_side(row)
            if (pred_margin is not None) and (line_val is not None):
                aligned = ((-pred_margin) + line_val) if sel_side == "away" else (pred_margin + line_val)
                model_support = min(1.0, max(0.0, aligned) / 5.0)
        elif code == "ML":
            pred_margin = _to_float(row.get("pred_margin"))
            sel_side = _selected_side(row)
            if pred_margin is not None:
                aligned = (-pred_margin) if sel_side == "away" else pred_margin
                model_support = min(1.0, max(0.0, aligned) / 6.0)

        sim_support = min(1.0, max(0.0, (0.45 * score_norm) + (0.30 * prob_strength) + (0.25 * model_support)))

        edge_val = abs(_to_float(row.get("edge")) or 0.0)
        if code == "OU":
            value_norm = min(1.0, max(0.0, edge_val) / 7.0)
        elif code == "ATS":
            value_norm = min(1.0, max(0.0, edge_val) / 4.5)
        elif code == "ML":
            implied = _american_implied_prob(_to_float(row.get("price")))
            if (p_win is not None) and (implied is not None):
                value_norm = min(1.0, max(0.0, p_win - implied) / 0.12)
            else:
                value_norm = 0.0
        else:
            value_norm = min(1.0, max(0.0, edge_val) / 5.0)

        priority = (0.60 * basketball_norm) + (0.30 * sim_support) + (0.10 * value_norm)
        if matchup_present and code in ("ATS", "ML", "OU") and basketball_raw is None:
            priority *= 0.65
        if sim_support < 0.25:
            priority *= 0.8

        return {
            "basketball_priority_score": round(basketball_norm * 100.0, 1),
            "sim_support_score": round(sim_support * 100.0, 1),
            "value_support_score": round(value_norm * 100.0, 1),
            "recommendation_priority_score": round(priority * 100.0, 1),
        }
    except Exception:
        return {}


def _pick_priority_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return (
        _to_float(row.get("recommendation_priority_score")) or 0.0,
        _to_float(row.get("basketball_priority_score")) or 0.0,
        _to_float(row.get("sim_support_score")) or 0.0,
        _to_float(row.get("score")) or 0.0,
        _to_float(row.get("p_win")) or 0.0,
        abs(_to_float(row.get("edge")) or 0.0),
    )


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
    reasons = out.get("reasons")
    out["model_reasons"] = list(reasons) if isinstance(reasons, list) else []
    return out


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

    matchup_ctx = _load_matchup_context(Path(cfg.out_dir), date)
    matchup_present = bool(matchup_ctx.get("by_game") or matchup_ctx.get("by_pair"))

    enriched_candidates: list[dict[str, Any]] = []
    for row in filtered_candidates:
        decorated = _decorate_pick(row)
        matchup_logic = _build_basketball_matchup_logic(decorated, matchup_ctx)
        if matchup_logic:
            decorated.update(matchup_logic)
        decorated.update(_priority_payload(decorated, matchup_present=matchup_present))

        basketball_reasons = decorated.get("basketball_reasons")
        if isinstance(basketball_reasons, list) and basketball_reasons:
            decorated["reasons"] = list(basketball_reasons) + list(decorated.get("model_reasons") or [])
        elif not isinstance(decorated.get("reasons"), list):
            decorated["reasons"] = list(decorated.get("model_reasons") or [])
        enriched_candidates.append(decorated)

    ranked_unique: list[dict[str, Any]] = []
    seen_games: set[str] = set()
    for row in sorted(enriched_candidates, key=_pick_priority_sort_key, reverse=True):
        gid = _norm_gid(row.get("game_id"))
        if not gid or gid in seen_games:
            continue
        seen_games.add(gid)
        ranked_unique.append(row)

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
            avg_priority = 0.0
            min_priority = 100.0
            for leg in combo:
                p_win = _to_float(leg.get("p_win")) or 0.0
                score = _to_float(leg.get("score")) or 0.0
                priority = _to_float(leg.get("recommendation_priority_score")) or 0.0
                combined_p *= p_win
                avg_score += score
                min_leg_score = min(min_leg_score, score)
                avg_priority += priority
                min_priority = min(min_priority, priority)
            avg_score /= float(parlay_size)
            avg_priority /= float(parlay_size)
            profit_units = _combo_profit_units(combo)
            expected_units = combined_p * profit_units - (1.0 - combined_p)
            parlays.append(
                {
                    "legs": [dict(leg) for leg in combo],
                    "combined_p_win": combined_p,
                    "avg_leg_score": avg_score,
                    "min_leg_score": min_leg_score,
                    "avg_recommendation_priority": avg_priority,
                    "min_recommendation_priority": min_priority,
                    "profit_units_per_1_risk": profit_units,
                    "expected_units_per_1_risk": expected_units,
                    "approx_american_odds": _profit_to_american(profit_units),
                }
            )
        parlays.sort(
            key=lambda row: (
                -float(row.get("avg_recommendation_priority") or 0.0),
                -float(row.get("min_recommendation_priority") or 0.0),
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