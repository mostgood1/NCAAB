"""Simulate alternative staking regimes on stored backtest picks.

Usage (PowerShell):
  python scripts/stake_simulation.py --date 2025-11-19 --mode kelly --kelly_fractions 0.25 0.5 1.0
  python scripts/stake_simulation.py --date 2025-11-19 --mode flat --flat_units 0.5 1 2

Reads outputs/backtest_metrics_<date>.json produced by daily_backtest.py (needs bets_detail section).
Reconstructs bet outcomes using outputs/daily_results/results_<date>.csv (or games_<date>.csv) and
closing medians (outputs/games_with_closing_<date>.csv).

Outputs: outputs/stake_simulation_<date>.json summarizing ROI/PNL per regime.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import datetime as dt
import pandas as pd

OUT = Path("outputs")


def _safe_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _load_games(date_str: str) -> pd.DataFrame:
    dr = _safe_csv(OUT / "daily_results" / f"results_{date_str}.csv")
    if not dr.empty:
        dr["game_id"] = dr.get("game_id", dr.get("id", pd.Series(range(len(dr)))))
        dr["game_id"] = dr["game_id"].astype(str)
        return dr
    g = _safe_csv(OUT / f"games_{date_str}.csv")
    if not g.empty:
        g["game_id"] = g.get("game_id", g.get("id", pd.Series(range(len(g)))))
        g["game_id"] = g["game_id"].astype(str)
        return g
    return pd.DataFrame()


def _load_closing(date_str: str) -> pd.DataFrame:
    gc = _safe_csv(OUT / f"games_with_closing_{date_str}.csv")
    if not gc.empty:
        gc["game_id"] = gc["game_id"].astype(str)
    return gc


def american_to_prob(odds_val):
    try:
        o = float(odds_val)
    except Exception:
        return None
    if o == 0:
        return None
    return (-o / (-o + 100)) if o < 0 else (100 / (o + 100))


def simulate(date_str: str, mode: str, kelly_fractions: list[float], flat_units: list[float]):
    bp = OUT / f"backtest_metrics_{date_str}.json"
    if not bp.exists():
        raise FileNotFoundError(f"Missing backtest metrics JSON for {date_str}")
    payload = json.loads(bp.read_text(encoding="utf-8"))
    bets = payload.get("bets_detail", {}) if isinstance(payload, dict) else {}

    games = _load_games(date_str)
    closing = _load_closing(date_str)

    # Merge closing medians onto games for evaluation
    if not games.empty and not closing.empty and "game_id" in games.columns and "game_id" in closing.columns:
        games = games.merge(closing, on="game_id", how="left")

    # Helper: evaluate outcomes
    def eval_totals(bet):
        gid = str(bet.get("game_id"))
        row = games[games["game_id"].astype(str) == gid]
        if row.empty:
            return None
        r = row.iloc[0]
        if {"home_score","away_score","closing_total"}.issubset(row.columns):
            try:
                total_actual = float(r.get("home_score")) + float(r.get("away_score"))
                closing_total = float(r.get("closing_total"))
                edge = float(bet.get("edge_total", 0))
                pick_over = edge > 0
                if total_actual == closing_total:
                    return 0
                return 1 if (total_actual > closing_total if pick_over else total_actual < closing_total) else -1
            except Exception:
                return None
        return None

    def eval_spread(bet):
        gid = str(bet.get("game_id"))
        row = games[games["game_id"].astype(str) == gid]
        if row.empty:
            return None
        r = row.iloc[0]
        needed = {"home_score","away_score","closing_spread_home"}
        if needed.issubset(row.columns):
            try:
                margin = float(r.get("home_score")) - float(r.get("away_score"))
                s_home = float(r.get("closing_spread_home"))
                edge = float(bet.get("edge_margin", 0))
                bet_home = edge > 0
                if margin == -s_home:
                    return 0
                return 1 if (margin > -s_home if bet_home else margin < s_home) else -1
            except Exception:
                return None
        return None

    def eval_moneyline(bet):
        gid = str(bet.get("game_id"))
        row = games[games["game_id"].astype(str) == gid]
        if row.empty:
            return None
        r = row.iloc[0]
        if {"home_score","away_score","closing_ml_home"}.issubset(row.columns):
            try:
                hs = float(r.get("home_score")); aw = float(r.get("away_score"))
                return 1 if hs > aw else -1
            except Exception:
                return None
        return None

    regimes = []
    if mode == "kelly":
        for kf in kelly_fractions:
            pnl = 0.0; bets_n = 0; resolved = 0; wins = 0
            # Totals
            for bet in bets.get("totals", []):
                outcome = eval_totals(bet)
                if outcome is None:
                    continue
                bets_n += 1
                resolved += 1 if outcome != 0 else 0
                # Approx p_win via logistic(edge/10)
                p = 1.0/(1.0 + math.exp(-float(bet.get("edge_total",0))/10.0))
                b = 0.909
                frac = kf * ((p*(b+1) - 1)/b)
                stake = max(0.0, min(5.0, frac))
                if outcome > 0:
                    wins += 1
                    pnl += stake * 0.909
                elif outcome < 0:
                    pnl -= stake
            # Spread
            for bet in bets.get("spread", []):
                outcome = eval_spread(bet)
                if outcome is None:
                    continue
                bets_n += 1
                resolved += 1 if outcome != 0 else 0
                margin_edge = float(bet.get("edge_margin",0))
                # Use spread logistic K default 0.115
                K = 0.115
                p_cover = 1.0/(1.0 + math.exp(-margin_edge/K))
                b = 0.909
                frac = kf * ((p_cover*(b+1) - 1)/b)
                stake = max(0.0, min(5.0, frac))
                if outcome > 0:
                    wins += 1
                    pnl += stake * 0.909
                elif outcome < 0:
                    pnl -= stake
            # Moneyline (home side only based on bet record)
            for bet in bets.get("moneyline", []):
                outcome = eval_moneyline(bet)
                if outcome is None:
                    continue
                bets_n += 1
                resolved += 1
                edge_p = float(bet.get("edge_prob",0))
                # Kelly using implied odds from JSON
                p_model = float(bet.get("p_model",0))
                p_imp = float(bet.get("p_implied",0))
                bprice = (1/p_imp) - 1 if p_imp > 0 else 1.0
                frac = kf * ((p_model*(bprice+1) - 1)/bprice) if bprice>0 else 0.0
                stake = max(0.0, min(10.0, frac))
                if outcome > 0:
                    # Profit with price derived from implied prob
                    if p_imp > 0:
                        pnl += stake * bprice
                else:
                    pnl -= stake
            roi = (pnl / bets_n) if bets_n else None
            regimes.append({"mode":"kelly","kelly_fraction":kf,"pnl_units":pnl,"roi":roi,"n_bets":bets_n,"n_resolved":resolved,"win_rate": (wins/resolved if resolved else None)})
    elif mode == "flat":
        for u in flat_units:
            pnl = 0.0; bets_n = 0; resolved = 0; wins = 0
            stake = float(u)
            for bet in bets.get("totals", []):
                outcome = eval_totals(bet)
                if outcome is None:
                    continue
                bets_n += 1
                resolved += 1 if outcome != 0 else 0
                if outcome > 0:
                    wins += 1
                    pnl += stake * 0.909
                elif outcome < 0:
                    pnl -= stake
            for bet in bets.get("spread", []):
                outcome = eval_spread(bet)
                if outcome is None:
                    continue
                bets_n += 1
                resolved += 1 if outcome != 0 else 0
                if outcome > 0:
                    wins += 1
                    pnl += stake * 0.909
                elif outcome < 0:
                    pnl -= stake
            for bet in bets.get("moneyline", []):
                outcome = eval_moneyline(bet)
                if outcome is None:
                    continue
                bets_n += 1
                resolved += 1
                # Approx price via implied prob stored
                p_imp = bet.get("p_implied")
                bprice = (1/float(p_imp)) - 1 if p_imp and float(p_imp)>0 else 0.0
                if outcome > 0:
                    wins += 1
                    pnl += stake * bprice
                else:
                    pnl -= stake
            roi = (pnl / bets_n) if bets_n else None
            regimes.append({"mode":"flat","flat_units":u,"pnl_units":pnl,"roi":roi,"n_bets":bets_n,"n_resolved":resolved,"win_rate": (wins/resolved if resolved else None)})
    else:
        raise ValueError("mode must be 'kelly' or 'flat'")

    out = {
        "date": date_str,
        "mode": mode,
        "generated_at": dt.datetime.now().isoformat(),
        "regimes": regimes,
    }
    (OUT / f"stake_simulation_{date_str}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote stake simulation to {OUT / f'stake_simulation_{date_str}.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Date YYYY-MM-DD to simulate")
    ap.add_argument("--mode", choices=["kelly","flat"], default="kelly")
    ap.add_argument("--kelly_fractions", nargs="*", type=float, default=[0.25,0.5,1.0])
    ap.add_argument("--flat_units", nargs="*", type=float, default=[1.0])
    args = ap.parse_args()
    simulate(args.date, args.mode, args.kelly_fractions, args.flat_units)
