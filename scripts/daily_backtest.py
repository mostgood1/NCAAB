"""Daily backtest metrics generator.

Usage (PowerShell):
  python scripts/daily_backtest.py --date 2025-11-19
If --date omitted, defaults to yesterday (local timezone) to ensure games resolved.

Outputs JSON at: outputs/backtest_metrics_<date>.json with keys:
  date, generated_at, totals_closing (BacktestSummary fields), spread_closing, moneyline_closing

Relies on prediction + closing line artifacts produced by existing pipeline:
  - outputs/predictions_unified_<date>.csv OR model predictions fallback
  - outputs/games_<date>.csv or daily_results/results_<date>.csv for scores
  - outputs/closing_lines.csv joined with game_id (must include market/period columns)

Safely skips any unavailable segment.
"""
from __future__ import annotations
import argparse
import json
import datetime as dt
from pathlib import Path
import pandas as pd

# Reuse existing backtest utilities
try:
    from src.ncaab_model.eval.backtest import (
        backtest_totals_with_closing,
        backtest_spread_with_closing,
        backtest_moneyline_with_closing,
        BacktestSummary,
    )
except ModuleNotFoundError:
    # Allow running without installed package by adjusting sys.path
    import sys
    from pathlib import Path as _P
    sys.path.append(str(_P(__file__).resolve().parent.parent))
    from src.ncaab_model.eval.backtest import (
        backtest_totals_with_closing,
        backtest_spread_with_closing,
        backtest_moneyline_with_closing,
        BacktestSummary,
    )

OUT = Path("outputs")

def _safe_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()

def _pick_games(date_str: str) -> pd.DataFrame:
    # Prefer daily_results for resolved scores; fallback to games_<date>.csv; finally games_<date>_fused variants.
    dr = _safe_csv(OUT / "daily_results" / f"results_{date_str}.csv")
    if not dr.empty and {"home_score","away_score"}.issubset(dr.columns):
        dr["game_id"] = dr.get("game_id", dr.get("id", pd.Series(range(len(dr)))))
        dr["game_id"] = dr["game_id"].astype(str)
        return dr
    g = _safe_csv(OUT / f"games_{date_str}.csv")
    if not g.empty:
        g["game_id"] = g.get("game_id", g.get("id", pd.Series(range(len(g)))))
        g["game_id"] = g["game_id"].astype(str)
        return g
    gf = _safe_csv(OUT / f"games_{date_str}_fused.csv")
    if not gf.empty:
        gf["game_id"] = gf.get("game_id", gf.get("id", pd.Series(range(len(gf)))))
        gf["game_id"] = gf["game_id"].astype(str)
        return gf
    return pd.DataFrame()

def _pick_preds(date_str: str) -> pd.DataFrame:
    uni = _safe_csv(OUT / f"predictions_unified_{date_str}.csv")
    if not uni.empty:
        uni["game_id"] = uni["game_id"].astype(str)
        return uni
    # Fallback to model predictions file pattern
    mp = _safe_csv(OUT / f"predictions_model_{date_str}.csv")
    if not mp.empty:
        mp["game_id"] = mp["game_id"].astype(str)
        # Standardize column names
        if "pred_total_model" in mp.columns and "pred_total" not in mp.columns:
            mp["pred_total"] = mp["pred_total_model"]
        if "pred_margin_model" in mp.columns and "pred_margin" not in mp.columns:
            mp["pred_margin"] = mp["pred_margin_model"]
        return mp
    return pd.DataFrame()

def _pick_closing(date_str: str) -> pd.DataFrame:
    cl = _safe_csv(OUT / "closing_lines.csv")
    if not cl.empty:
        # Attempt date filtering if commence_time/date present
        if "commence_time" in cl.columns:
            try:
                dtc = pd.to_datetime(cl["commence_time"], errors="coerce")
                cl = cl[dtc.dt.strftime("%Y-%m-%d") == date_str]
            except Exception:
                pass
        if "date" in cl.columns:
            cl = cl[cl["date"].astype(str) == date_str]
        # Ensure game_id exists; if missing, synthesize from teams and date
        if "game_id" not in cl.columns:
            home_col = next((c for c in ["home_team","home_team_name","home","home_name"] if c in cl.columns), None)
            away_col = next((c for c in ["away_team","away_team_name","away","away_name"] if c in cl.columns), None)
            if home_col and away_col:
                import re
                def _slug(v: str) -> str:
                    v = (str(v) if v is not None else '').lower()
                    return re.sub(r"[^a-z0-9]+","_", v).strip('_')
                home = cl[home_col].astype(str).map(_slug)
                away = cl[away_col].astype(str).map(_slug)
                cl["game_id"] = [f"{date_str}:{a}:{h}" for a,h in zip(away, home)]
        if "game_id" in cl.columns:
            cl["game_id"] = cl["game_id"].astype(str)
        # If filtering yielded no rows, fallback to synthetic from games_with_closing_<date>
        if cl.empty:
            gclose = _safe_csv(OUT / f"games_with_closing_{date_str}.csv")
            if not gclose.empty and "game_id" in gclose.columns:
                gclose["game_id"] = gclose["game_id"].astype(str)
                rows = []
                # Build minimal rows for totals/spreads/h2h as available
                for _, r in gclose.iterrows():
                    gid = r.get("game_id")
                    if "closing_total" in gclose.columns and pd.notna(r.get("closing_total")):
                        rows.append({"game_id": gid, "market": "totals", "period": "full_game", "total": r.get("closing_total")})
                    if "closing_spread_home" in gclose.columns and pd.notna(r.get("closing_spread_home")):
                        rows.append({"game_id": gid, "market": "spreads", "period": "full_game", "home_spread": r.get("closing_spread_home")})
                    if "closing_ml_home" in gclose.columns and pd.notna(r.get("closing_ml_home")):
                        rows.append({"game_id": gid, "market": "h2h", "period": "full_game", "moneyline_home": r.get("closing_ml_home"), "moneyline_away": r.get("closing_ml_away")})
                return pd.DataFrame(rows)
        return cl
    return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="Target date YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--threshold_totals", type=float, default=2.0, help="Edge threshold for totals bets")
    ap.add_argument("--threshold_spread", type=float, default=0.5, help="Edge threshold (pred_margin - closing_spread) for spread bets")
    ap.add_argument("--threshold_moneyline", type=float, default=0.02, help="Edge threshold (model_prob - implied_prob) for moneyline bets")
    ap.add_argument("--stake_mode", choices=["flat","kelly","fractional"], default="flat", help="Stake sizing regime")
    ap.add_argument("--kelly_fraction", type=float, default=0.5, help="Fraction of Kelly (for kelly mode)")
    ap.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll for Kelly sizing")
    ap.add_argument("--logistic_k", type=float, default=0.115, help="Logistic scale constant for spread win probability approximation")
    args = ap.parse_args()
    date_str = args.date or (dt.datetime.now().date() - dt.timedelta(days=1)).strftime("%Y-%m-%d")

    games = _pick_games(date_str)
    preds = _pick_preds(date_str)
    closing = _pick_closing(date_str)

    payload = {"date": date_str, "generated_at": dt.datetime.now().isoformat(),
               "stake_mode": args.stake_mode, "kelly_fraction": args.kelly_fraction, "bankroll_start": args.bankroll}

    # If games missing, synthesize minimal frame from games_with_closing_<date>.csv to enable bet selection
    try:
        if games.empty:
            gclose_path0 = OUT / f"games_with_closing_{date_str}.csv"
            if gclose_path0.exists():
                gc0 = pd.read_csv(gclose_path0)
                if not gc0.empty and "game_id" in gc0.columns:
                    gc0["game_id"] = gc0["game_id"].astype(str)
                    # Keep only relevant backtest columns
                    keep = [c for c in ["game_id","closing_total","closing_spread_home","closing_ml_home","closing_ml_away"] if c in gc0.columns]
                    games = gc0[keep].copy()
                    payload["games_synthesized_from_closing"] = int(len(games))
    except Exception:
        payload["games_synth_error"] = True

    # Ensure closing medians merged (if separate artifact exists)
    try:
        gclose_path = OUT / f"games_with_closing_{date_str}.csv"
        if gclose_path.exists() and not games.empty and "game_id" in games.columns:
            gclose = pd.read_csv(gclose_path)
            if not gclose.empty and "game_id" in gclose.columns:
                gclose["game_id"] = gclose["game_id"].astype(str)
                games["game_id"] = games["game_id"].astype(str)
                games = games.merge(gclose, on="game_id", how="left")
    except Exception:
        payload["closing_join_warning"] = "could_not_merge_games_with_closing"

    # Compute simple edges if absent (preds vs closing)
    try:
        if not preds.empty and "pred_total" in preds.columns and "game_id" in preds.columns and "closing_total" in games.columns:
            tmp = games[["game_id","closing_total"]].drop_duplicates()
            preds["game_id"] = preds["game_id"].astype(str)
            tmp["game_id"] = tmp["game_id"].astype(str)
            preds = preds.merge(tmp, on="game_id", how="left")
            if "edge_total" not in preds.columns:
                pt = pd.to_numeric(preds["pred_total"], errors="coerce")
                ct = pd.to_numeric(preds["closing_total"], errors="coerce")
                preds["edge_total"] = pt - ct
        if not preds.empty and "pred_margin" in preds.columns and "closing_spread_home" in games.columns:
            tmp2 = games[["game_id","closing_spread_home"]].drop_duplicates()
            preds = preds.merge(tmp2, on="game_id", how="left")
            if "edge_margin" not in preds.columns:
                pm = pd.to_numeric(preds["pred_margin"], errors="coerce")
                cs = pd.to_numeric(preds["closing_spread_home"], errors="coerce")
                preds["edge_margin"] = pm - cs
    except Exception:
        payload["edge_calc_warning"] = "failed_edge_derivation"

    # Helper probability/odds utilities
    def american_to_prob(odds_val):
        try:
            o = float(odds_val)
        except Exception:
            return None
        if o == 0:
            return None
        return (-o/( -o + 100)) if o < 0 else (100/(o + 100))

    def logistic(x):
        import math
        return 1.0/(1.0 + math.exp(-x))

    # Totals closing backtest
    try:
        if not games.empty and not preds.empty and "edge_total" in preds.columns:
            if not closing.empty:
                _, sum_tot = backtest_totals_with_closing(games, closing, preds, threshold=args.threshold_totals)
                payload["totals_closing"] = sum_tot.__dict__
            else:
                # Median-based fallback using games_with_closing merged columns
                g2 = games.copy()
                p2 = preds.copy()
                # Require closing_total and scores to resolve
                if ("closing_total" in g2.columns) and {"home_score","away_score"}.issubset(g2.columns):
                    try:
                        g2["game_id"] = g2["game_id"].astype(str)
                        p2["game_id"] = p2["game_id"].astype(str)
                    except Exception:
                        pass
                    mg = p2.merge(g2[["game_id","closing_total","home_score","away_score"]], on="game_id", how="left")
                    ct = pd.to_numeric(mg["closing_total"], errors="coerce")
                    et = pd.to_numeric(mg["edge_total"], errors="coerce")
                    hs = pd.to_numeric(mg["home_score"], errors="coerce")
                    aw = pd.to_numeric(mg["away_score"], errors="coerce")
                    # Select bets
                    sel = et.abs() >= args.threshold_totals
                    n_bets = int(sel.sum())
                    pnl = 0.0
                    wins = 0
                    resolved = 0
                    avg_edge = float(et[sel].mean()) if n_bets else None
                    for _, r in mg[sel].iterrows():
                        try:
                            total_actual = float(r.get("home_score", float("nan"))) + float(r.get("away_score", float("nan")))
                            closing = float(r.get("closing_total")) if pd.notna(r.get("closing_total")) else None
                            edge = float(r.get("edge_total"))
                            if closing is None or pd.isna(total_actual):
                                continue
                            # Direction: over if edge>0, under if edge<0
                            pick_over = edge > 0
                            # Resolve result (push when equal)
                            if total_actual == closing:
                                outcome = 0  # push
                            elif pick_over:
                                outcome = 1 if total_actual > closing else -1
                            else:
                                outcome = 1 if total_actual < closing else -1
                            if outcome != 0:
                                resolved += 1
                                if outcome > 0:
                                    wins += 1
                                    pnl += 0.909  # assume -110 price
                                else:
                                    pnl -= 1.0
                        except Exception:
                            continue
                    win_rate = (wins / resolved) if resolved else None
                    roi = (pnl / n_bets) if n_bets else None
                    payload["totals_closing"] = {
                        "n_games": int(len(g2)),
                        "n_books_rows": int(len(closing)),
                        "n_bets": n_bets,
                        "n_resolved": resolved,
                        "win_rate": win_rate,
                        "pnl_units": pnl,
                        "roi": roi,
                        "avg_edge": avg_edge,
                    }
                else:
                    payload["totals_closing"] = {"n_games": len(games), "n_books_rows": len(closing), "n_bets": 0, "n_resolved": 0, "win_rate": None, "pnl_units": 0.0, "roi": None, "avg_edge": None}
        else:
            payload["totals_closing"] = {"n_games": len(games), "n_books_rows": len(closing), "n_bets": 0, "n_resolved": 0, "win_rate": None, "pnl_units": 0.0, "roi": None, "avg_edge": None}
    except Exception as e:
        payload["totals_closing_error"] = str(e)

    # Spread closing backtest
    try:
        if not games.empty and not preds.empty and "pred_margin" in preds.columns and "edge_margin" in preds.columns:
            if not closing.empty:
                _, sum_spread = backtest_spread_with_closing(games, closing, preds)
                payload["spread_closing"] = sum_spread.__dict__
            else:
                # Median-based fallback using "closing_spread_home" on games
                g2 = games.copy()
                p2 = preds.copy()
                if ("closing_spread_home" in g2.columns) and {"home_score","away_score"}.issubset(g2.columns):
                    try:
                        g2["game_id"] = g2["game_id"].astype(str)
                        p2["game_id"] = p2["game_id"].astype(str)
                    except Exception:
                        pass
                    mg = p2.merge(g2[["game_id","closing_spread_home","home_score","away_score"]], on="game_id", how="left")
                    em = pd.to_numeric(mg["edge_margin"], errors="coerce")
                    cs = pd.to_numeric(mg["closing_spread_home"], errors="coerce")
                    hs = pd.to_numeric(mg["home_score"], errors="coerce")
                    aw = pd.to_numeric(mg["away_score"], errors="coerce")
                    sel = em.abs() >= args.threshold_spread
                    n_bets = int(sel.sum())
                    pnl = 0.0
                    wins = 0
                    resolved = 0
                    avg_edge = float(em[sel].mean()) if n_bets else None
                    for _, r in mg[sel].iterrows():
                        try:
                            margin = float(r.get("home_score", float("nan"))) - float(r.get("away_score", float("nan")))
                            s_home = float(r.get("closing_spread_home")) if pd.notna(r.get("closing_spread_home")) else None
                            edge = float(r.get("edge_margin"))
                            if s_home is None or pd.isna(margin):
                                continue
                            bet_home = edge > 0  # positive edge => model favors home relative to line
                            # Evaluate cover: home bet wins if margin > -s_home; away bet wins if margin < s_home
                            if margin == -s_home:
                                outcome = 0  # push (rare on halves of points)
                            elif bet_home:
                                outcome = 1 if margin > -s_home else -1
                            else:
                                outcome = 1 if margin < s_home else -1
                            if outcome != 0:
                                resolved += 1
                                if outcome > 0:
                                    wins += 1
                                    pnl += 0.909
                                else:
                                    pnl -= 1.0
                        except Exception:
                            continue
                    win_rate = (wins / resolved) if resolved else None
                    roi = (pnl / n_bets) if n_bets else None
                    payload["spread_closing"] = {
                        "n_games": int(len(g2)),
                        "n_books_rows": int(len(closing)),
                        "n_bets": n_bets,
                        "n_resolved": resolved,
                        "win_rate": win_rate,
                        "pnl_units": pnl,
                        "roi": roi,
                        "avg_edge": avg_edge,
                    }
                else:
                    payload["spread_closing"] = {"n_games": len(games), "n_books_rows": len(closing), "n_bets": 0, "n_resolved": 0, "win_rate": None, "pnl_units": 0.0, "roi": None, "avg_edge": None}
        else:
            payload["spread_closing"] = {"n_games": len(games), "n_books_rows": len(closing), "n_bets": 0, "n_resolved": 0, "win_rate": None, "pnl_units": 0.0, "roi": None, "avg_edge": None}
    except Exception as e:
        payload["spread_closing_error"] = str(e)

    # Moneyline closing backtest
    try:
        if not games.empty and not preds.empty and "ml_prob_model" in preds.columns:
            if not closing.empty:
                _, sum_ml = backtest_moneyline_with_closing(games, closing, preds)
                payload["moneyline_closing"] = sum_ml.__dict__
            else:
                # Median-based ML using closing_ml_* columns if available
                g2 = games.copy()
                p2 = preds.copy()
                have_ml = ("closing_ml_home" in g2.columns) | ("closing_ml_away" in g2.columns)
                if have_ml and {"home_score","away_score"}.issubset(g2.columns):
                    try:
                        g2["game_id"] = g2["game_id"].astype(str)
                        p2["game_id"] = p2["game_id"].astype(str)
                    except Exception:
                        pass
                    mg = p2.merge(g2[[c for c in ["game_id","closing_ml_home","closing_ml_away","home_score","away_score"] if c in g2.columns]], on="game_id", how="left")
                    n_bets = 0
                    wins = 0
                    resolved = 0
                    pnl = 0.0
                    edges = []
                    for _, r in mg.iterrows():
                        try:
                            p_model = float(r.get("ml_prob_model")) if pd.notna(r.get("ml_prob_model")) else None
                            ml_home = r.get("closing_ml_home")
                            if p_model is None or ml_home is None:
                                continue
                            p_imp = american_to_prob(ml_home)
                            if p_imp is None:
                                continue
                            edge_p = p_model - float(p_imp)
                            if edge_p >= args.threshold_moneyline:
                                n_bets += 1
                                edges.append(edge_p)
                                # Resolve (home side only here; extension: choose better side by model vs price)
                                hs = float(r.get("home_score", float("nan")))
                                aw = float(r.get("away_score", float("nan")))
                                if not (pd.isna(hs) or pd.isna(aw)):
                                    resolved += 1
                                    won = hs > aw
                                    if won:
                                        wins += 1
                                        # Payout for American odds
                                        o = float(ml_home)
                                        if o > 0:
                                            pnl += o/100.0
                                        else:
                                            pnl += 100.0/abs(o)
                                    else:
                                        pnl -= 1.0
                        except Exception:
                            continue
                    win_rate = (wins / resolved) if resolved else None
                    roi = (pnl / n_bets) if n_bets else None
                    payload["moneyline_closing"] = {
                        "n_games": int(len(g2)),
                        "n_books_rows": int(len(closing)),
                        "n_bets": int(n_bets),
                        "n_resolved": int(resolved),
                        "win_rate": win_rate,
                        "pnl_units": pnl,
                        "roi": roi,
                        "avg_edge": (sum(edges)/len(edges)) if edges else None,
                    }
                else:
                    payload["moneyline_closing"] = {"n_games": len(games), "n_books_rows": len(closing), "n_bets": 0, "n_resolved": 0, "win_rate": None, "pnl_units": 0.0, "roi": None, "avg_edge": None}
        else:
            payload["moneyline_closing"] = {"n_games": len(games), "n_books_rows": len(closing), "n_bets": 0, "n_resolved": 0, "win_rate": None, "pnl_units": 0.0, "roi": None, "avg_edge": None}
    except Exception as e:
        payload["moneyline_closing_error"] = str(e)

    # Enhanced bet & stake metrics (post primary summaries) ---------------------------------
    try:
        bets_detail = {"totals": [], "spread": [], "moneyline": []}
        # Merge predictions & games for enriched view
        merged = pd.DataFrame()
        if not preds.empty and not games.empty:
            try:
                merged = preds.merge(games, on="game_id", how="left", suffixes=("_p","_g"))
            except Exception:
                merged = preds.copy()
        # Totals bets selection
        if not merged.empty and "edge_total" in merged.columns and "closing_total" in merged.columns:
            mt = pd.to_numeric(merged.get("closing_total"), errors="coerce")
            et = pd.to_numeric(merged.get("edge_total"), errors="coerce")
            sel_mask = et.abs() >= args.threshold_totals
            for _, r in merged[sel_mask].iterrows():
                stake = 1.0
                if args.stake_mode == "fractional":
                    stake = min(3.0, max(0.25, abs(r.get("edge_total",0))/3.0))
                # Kelly for totals (approx) using normal assumption: p_win ≈ logistic(edge_total/10)
                if args.stake_mode == "kelly":
                    p = logistic(float(r.get("edge_total",0))/10.0)
                    b = 1.909 - 1.0  # assume -110 both sides; effective decimal price ~1.909
                    kf = args.kelly_fraction
                    frac = kf * ((p*(b+1) - 1)/b) if p is not None else 0.0
                    stake = max(0.0, min(5.0, frac * args.bankroll / 100.0))
                bets_detail["totals"].append({"game_id": r.get("game_id"), "edge_total": r.get("edge_total"), "stake": stake})
        # Spread bets selection
        if not merged.empty and "edge_margin" in merged.columns and "closing_spread_home" in merged.columns:
            em = pd.to_numeric(merged.get("edge_margin"), errors="coerce")
            sel_mask_sp = em.abs() >= args.threshold_spread
            for _, r in merged[sel_mask_sp].iterrows():
                stake = 1.0
                if args.stake_mode == "fractional":
                    stake = min(3.0, max(0.25, abs(r.get("edge_margin",0))/3.0))
                if args.stake_mode == "kelly":
                    # Approx cover probability
                    margin_edge = float(r.get("pred_margin",0)) - float(r.get("closing_spread_home",0))
                    p_cover = logistic(margin_edge/args.logistic_k)
                    b = 1.909 - 1.0
                    kf = args.kelly_fraction
                    frac = kf * ((p_cover*(b+1) - 1)/b) if p_cover is not None else 0.0
                    stake = max(0.0, min(5.0, frac * args.bankroll / 100.0))
                bets_detail["spread"].append({"game_id": r.get("game_id"), "edge_margin": r.get("edge_margin"), "stake": stake})
        # Moneyline bets selection
        if not merged.empty and "ml_prob_model" in merged.columns and ("closing_ml_home" in merged.columns or "closing_ml_away" in merged.columns):
            prob_col = merged.get("ml_prob_model")
            for _, r in merged.iterrows():
                p_model = r.get("ml_prob_model")
                ml_home = r.get("closing_ml_home")
                if p_model is None or ml_home is None:
                    continue
                p_imp = american_to_prob(ml_home)
                if p_imp is None:
                    continue
                edge_p = float(p_model) - float(p_imp)
                if edge_p >= args.threshold_moneyline:
                    stake = 1.0
                    if args.stake_mode == "fractional":
                        stake = min(5.0, max(0.25, edge_p*10))
                    if args.stake_mode == "kelly":
                        # Kelly fraction on American odds
                        price_prob = p_imp
                        b = (1/price_prob) - 1 if price_prob > 0 else 1.0
                        kf = args.kelly_fraction
                        frac = kf * ((p_model*(b+1) - 1)/b) if b > 0 else 0.0
                        stake = max(0.0, min(10.0, frac * args.bankroll / 50.0))
                    bets_detail["moneyline"].append({"game_id": r.get("game_id"), "edge_prob": edge_p, "stake": stake, "p_model": p_model, "p_implied": p_imp})
        payload["bets_detail"] = bets_detail
    except Exception as e:
        payload["bets_detail_error"] = str(e)

    # Adaptive thresholds: if no bets selected, choose top edges to surface candidates
    try:
        adaptive = {}
        # Reload merged frame if needed
        merged2 = None
        if ('bets_detail' in payload and isinstance(payload['bets_detail'], dict) and
            (len(payload['bets_detail'].get('totals', [])) == 0 or len(payload['bets_detail'].get('spread', [])) == 0)):
            if not preds.empty:
                try:
                    gcols = [c for c in ["game_id","closing_total","closing_spread_home"] if c in games.columns]
                    merged2 = preds.merge(games[gcols], on="game_id", how="left") if gcols else preds.copy()
                except Exception:
                    merged2 = preds.copy()
        # Totals adaptive
        if merged2 is not None and len(payload.get('bets_detail',{}).get('totals',[])) == 0 and 'edge_total' in merged2.columns:
            m = merged2[['game_id','edge_total']].dropna()
            if not m.empty:
                m['abs_edge'] = m['edge_total'].abs()
                top = m.sort_values('abs_edge', ascending=False).head(10)
                payload['bets_detail']['totals'] = [{"game_id": r.game_id, "edge_total": r.edge_total, "stake": 1.0} for r in top.itertuples()]
                adaptive['totals'] = {"strategy": "top_abs_edge", "count": int(len(top))}
        # Spread adaptive
        if merged2 is not None and len(payload.get('bets_detail',{}).get('spread',[])) == 0 and 'edge_margin' in merged2.columns:
            m = merged2[['game_id','edge_margin']].dropna()
            if not m.empty:
                m['abs_edge'] = m['edge_margin'].abs()
                top = m.sort_values('abs_edge', ascending=False).head(10)
                payload['bets_detail']['spread'] = [{"game_id": r.game_id, "edge_margin": r.edge_margin, "stake": 1.0} for r in top.itertuples()]
                adaptive['spread'] = {"strategy": "top_abs_edge", "count": int(len(top))}
        if adaptive:
            payload['adaptive_thresholds'] = adaptive
    except Exception:
        payload['adaptive_thresholds_error'] = True

    out_path = OUT / f"backtest_metrics_{date_str}.json"
    try:
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote backtest metrics to {out_path}")
    except Exception as e:
        print(f"Failed to write metrics JSON: {e}")

if __name__ == "__main__":
    main()
