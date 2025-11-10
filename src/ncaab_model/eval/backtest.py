from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Tuple


@dataclass
class BacktestSummary:
    n_games: int
    n_books_rows: int
    n_bets: int
    n_resolved: int
    win_rate: float | None
    pnl_units: float
    roi: float | None
    avg_edge: float | None


def american_to_prob(price: float) -> float:
    if price is None or np.isnan(price):
        return np.nan
    if price > 0:
        return 100.0 / (price + 100.0)
    else:
        return -price / (-price + 100.0)


def backtest_totals(games: pd.DataFrame, odds: pd.DataFrame, preds: pd.DataFrame, threshold: float = 2.0, default_price: float = -110.0):
    """Backtest totals bets based on predictions vs bookmaker total line.

    - games: DataFrame with columns [game_id, home_score, away_score]
    - odds: DataFrame with columns [game_id, book, total]
    - preds: DataFrame with columns [game_id, pred_total]
    """
    # Merge predictions to odds
    od = odds.copy()
    pr = preds[["game_id", "pred_total"]].copy()
    merged = od.merge(pr, on="game_id", how="inner")

    # Decide bet direction and filter by threshold
    merged["edge"] = merged["pred_total"] - merged["total"]
    bets = merged[merged["edge"].abs() >= threshold].copy()
    if bets.empty:
        return pd.DataFrame(columns=["game_id", "book", "bet", "line", "pred", "edge", "result", "price", "pnl"]), BacktestSummary(0, 0, 0, 0, None, 0.0, None, None)

    # Assign bet type
    bets["bet"] = np.where(bets["edge"] > 0, "over", "under")
    bets["line"] = bets["total"]
    bets["pred"] = bets["pred_total"]
    bets["price"] = default_price

    # Resolve results where scores available
    gm = games[["game_id", "home_score", "away_score"]].copy()
    # If any score columns already exist from previous joins, drop them to avoid suffix clashes
    for col in ("home_score", "away_score"):
        if col in bets.columns:
            bets = bets.drop(columns=[col])
    bets = bets.merge(gm, on="game_id", how="left")
    bets["scored_total"] = bets["home_score"] + bets["away_score"]
    # Mark unplayed (0-0) as unresolved
    bets.loc[(bets["home_score"] == 0) & (bets["away_score"] == 0), "scored_total"] = np.nan
    # Treat 0-0 as unresolved (common for not-yet-played games in ESPN data)
    bets.loc[(bets["home_score"] == 0) & (bets["away_score"] == 0), "scored_total"] = np.nan

    def settle(row):
        if pd.isna(row["scored_total"]):
            return None
        if row["bet"] == "over":
            if row["scored_total"] > row["line"]:
                return 1
            elif row["scored_total"] < row["line"]:
                return 0
            else:
                return 0.5  # push
        else:
            if row["scored_total"] < row["line"]:
                return 1
            elif row["scored_total"] > row["line"]:
                return 0
            else:
                return 0.5

    bets["result"] = bets.apply(settle, axis=1)

    # Compute pnl in units using American odds
    def pnl_units(row):
        price = row["price"] if not pd.isna(row["price"]) else default_price
        if row["result"] is None:
            return 0.0
        if row["result"] == 0.5:
            return 0.0
        if price > 0:
            win = price / 100.0
            lose = 1.0
        else:
            win = 1.0
            lose = -price / 100.0
        return win if row["result"] == 1 else -lose

    bets["pnl"] = bets.apply(pnl_units, axis=1)

    n_resolved = bets["result"].notna().sum()
    win_rate = None
    if n_resolved > 0:
        win_rate = float((bets["result"] == 1).sum() / n_resolved)
    pnl = float(bets["pnl"].sum())
    roi = None
    if len(bets) > 0:
        # stake 1 unit per bet
        roi = pnl / len(bets)

    summary = BacktestSummary(
        n_games=len(games),
        n_books_rows=len(odds),
        n_bets=len(bets),
        n_resolved=int(n_resolved),
        win_rate=win_rate,
        pnl_units=pnl,
        roi=roi,
        avg_edge=float(merged["edge"].abs().mean()) if len(merged) else None,
    )
    # Return per-bet view
    return bets[["game_id", "book", "bet", "line", "pred", "edge", "result", "price", "pnl"]], summary


def backtest_totals_with_closing(games: pd.DataFrame, closing: pd.DataFrame, preds: pd.DataFrame, threshold: float = 2.0):
    """Backtest totals using closing lines (totals market, full_game period) and predictions.

    - closing: expects columns [event_id/book/market/period/total, over_price, under_price, home_team_name, away_team_name, date]
    - preds: expects [game_id, pred_total]
    - games: provides resolution via final scores
    """
    cl = closing.copy()
    # filter to totals, full_game
    cl = cl[(cl["market"] == "totals") & (cl["period"] == "full_game")]
    # predictions merged by game_id after joining closing->games externally
    if "game_id" in cl.columns:
        merged = cl.merge(preds[["game_id", "pred_total"]], on="game_id", how="inner")
    else:
        # if not joined yet, cannot proceed robustly
        raise ValueError("Closing lines must be joined to games (include game_id). Use join-closing CLI.")

    merged["edge"] = merged["pred_total"] - merged["total"]
    bets = merged[merged["edge"].abs() >= threshold].copy()
    if bets.empty:
        return pd.DataFrame(columns=["game_id", "book", "bet", "line", "pred", "edge", "result", "price", "pnl"]), BacktestSummary(0, 0, 0, 0, None, 0.0, None, None)

    bets["bet"] = np.where(bets["edge"] > 0, "over", "under")
    bets["line"] = bets["total"]
    bets["pred"] = bets["pred_total"]
    # Use available side-specific prices if present; else default -110
    bets["price"] = np.where(bets["bet"] == "over", bets.get("over_price", np.nan), bets.get("under_price", np.nan))
    # Avoid chained assignment warnings by assigning the filled series back
    bets["price"] = bets["price"].fillna(-110.0)

    # Resolve
    gm = games[["game_id", "home_score", "away_score"]].copy()
    # If bets already has score columns from an earlier join, drop them to avoid suffixing
    for col in ("home_score", "away_score"):
        if col in bets.columns:
            bets = bets.drop(columns=[col])
    bets = bets.merge(gm, on="game_id", how="left")
    bets["scored_total"] = bets["home_score"] + bets["away_score"]
    # Treat 0-0 as unresolved (common for not-yet-played games in ESPN data)
    bets.loc[(bets["home_score"] == 0) & (bets["away_score"] == 0), "scored_total"] = np.nan

    def settle(row):
        if pd.isna(row["scored_total"]):
            return None
        if row["bet"] == "over":
            if row["scored_total"] > row["line"]:
                return 1
            elif row["scored_total"] < row["line"]:
                return 0
            else:
                return 0.5
        else:
            if row["scored_total"] < row["line"]:
                return 1
            elif row["scored_total"] > row["line"]:
                return 0
            else:
                return 0.5

    bets["result"] = bets.apply(settle, axis=1)

    # pnl as before
    def pnl_units(price: float, result):
        if result is None or result == 0.5:
            return 0.0
        if price > 0:
            win = price / 100.0
            lose = 1.0
        else:
            win = 1.0
            lose = -price / 100.0
        return win if result == 1 else -lose

    bets["pnl"] = [pnl_units(p, r) for p, r in zip(bets["price"], bets["result"])]

    n_resolved = bets["result"].notna().sum()
    win_rate = float((bets["result"] == 1).sum() / n_resolved) if n_resolved > 0 else None
    pnl = float(bets["pnl"].sum())
    roi = pnl / len(bets) if len(bets) else None
    summary = BacktestSummary(
        n_games=len(games),
        n_books_rows=len(closing),
        n_bets=len(bets),
        n_resolved=int(n_resolved),
        win_rate=win_rate,
        pnl_units=pnl,
        roi=roi,
        avg_edge=float(merged["edge"].abs().mean()) if len(merged) else None,
    )
    return bets[["game_id", "book", "bet", "line", "pred", "edge", "result", "price", "pnl"]], summary


def backtest_spread_with_closing(games: pd.DataFrame, closing: pd.DataFrame, preds: pd.DataFrame, threshold: float = 0.0) -> Tuple[pd.DataFrame, BacktestSummary]:
    """Backtest spread bets vs closing lines.

    Bets every game (threshold currently unused / placeholder) choosing side where model margin exceeds spread.
    Expects closing rows with columns: [game_id, market=='spread', period=='full_game', spread_home, home_team_name, away_team_name]
    preds must contain [game_id, pred_margin]. Games contain final scores for resolution.
    """
    cl = closing.copy()
    cl = cl[(cl.get("market") == "spread") & (cl.get("period") == "full_game")]
    if cl.empty or "game_id" not in cl.columns or "spread_home" not in cl.columns:
        return pd.DataFrame(columns=["game_id","book","bet","line","pred","edge","result","price","pnl"]), BacktestSummary(0,0,0,0,None,0.0,None,None)
    # Merge predictions
    if "pred_margin" not in preds.columns:
        return pd.DataFrame(columns=["game_id","book","bet","line","pred","edge","result","price","pnl"]), BacktestSummary(0,0,0,0,None,0.0,None,None)
    merged = cl.merge(preds[["game_id","pred_margin"]], on="game_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["game_id","book","bet","line","pred","edge","result","price","pnl"]), BacktestSummary(len(games), len(cl), 0, 0, None, 0.0, None, None)
    merged["edge"] = merged["pred_margin"] - merged["spread_home"]
    # Determine bet side (home ATS or away ATS)
    merged["bet"] = np.where(merged["edge"] > 0, "home", "away")
    merged["line"] = merged["spread_home"]
    merged["pred"] = merged["pred_margin"]
    # Price placeholder (standard -110)
    merged["price"] = -110.0
    bets = merged.copy()
    # Resolve using final scores
    gm = games[["game_id","home_score","away_score"]].copy()
    bets = bets.merge(gm, on="game_id", how="left")
    bets["margin_actual"] = bets["home_score"] - bets["away_score"]
    # Mark unresolved if no scores
    bets.loc[(bets["home_score"].isna()) | (bets["away_score"].isna()) | ((bets["home_score"]==0) & (bets["away_score"]==0)), "margin_actual"] = np.nan

    def settle(row):
        if pd.isna(row["margin_actual"]):
            return None
        # Spread logic: home spread_home value relative to 0
        # If bet = home, we need home to cover: margin_actual + spread_home > 0
        # If bet = away, we need away to cover: margin_actual + spread_home < 0
        # Push if margin_actual + spread_home == 0
        cover_metric = row["margin_actual"] + row["spread_home"]
        if cover_metric == 0:
            return 0.5
        if row["bet"] == "home":
            return 1 if cover_metric > 0 else 0
        else:
            return 1 if cover_metric < 0 else 0

    bets["result"] = bets.apply(settle, axis=1)

    def pnl_units(price: float, result):
        if result is None or result == 0.5:
            return 0.0
        if price > 0:
            win = price / 100.0
            lose = 1.0
        else:
            win = 1.0
            lose = -price / 100.0
        return win if result == 1 else -lose

    bets["pnl"] = [pnl_units(p, r) for p, r in zip(bets["price"], bets["result"])]
    n_resolved = bets["result"].notna().sum()
    win_rate = float((bets["result"] == 1).sum() / n_resolved) if n_resolved else None
    pnl = float(bets["pnl"].sum())
    roi = pnl / len(bets) if len(bets) else None
    summary = BacktestSummary(
        n_games=len(games),
        n_books_rows=len(cl),
        n_bets=len(bets),
        n_resolved=int(n_resolved),
        win_rate=win_rate,
        pnl_units=pnl,
        roi=roi,
        avg_edge=float(bets["edge"].abs().mean()) if len(bets) else None,
    )
    return bets[["game_id","book","bet","line","pred","edge","result","price","pnl"]], summary


def backtest_moneyline_with_closing(games: pd.DataFrame, closing: pd.DataFrame, preds: pd.DataFrame) -> Tuple[pd.DataFrame, BacktestSummary]:
    """Backtest simplistic moneyline bets vs closing lines.

    Expects closing with market=='moneyline', period=='full_game', columns: [game_id, book, ml_home, ml_away].
    preds needs ml_prob_model (prob home win). We bet home if prob>implied and away if prob<implied (value approach). One bet per game max.
    """
    cl = closing.copy()
    cl = cl[(cl.get("market") == "moneyline") & (cl.get("period") == "full_game")]
    if cl.empty or "game_id" not in cl.columns:
        return pd.DataFrame(columns=["game_id","book","bet","line","pred","edge","result","price","pnl"]), BacktestSummary(0,0,0,0,None,0.0,None,None)
    if "ml_prob_model" not in preds.columns:
        return pd.DataFrame(columns=["game_id","book","bet","line","pred","edge","result","price","pnl"]), BacktestSummary(0,0,0,0,None,0.0,None,None)
    merged = cl.merge(preds[["game_id","ml_prob_model"]], on="game_id", how="inner")
    # Compute implied probs
    merged["prob_home_implied"] = merged["ml_home"].apply(american_to_prob) if "ml_home" in merged.columns else np.nan
    merged["prob_away_implied"] = merged["ml_away"].apply(american_to_prob) if "ml_away" in merged.columns else np.nan
    # Determine edge for home side
    merged["edge_home"] = merged["ml_prob_model"] - merged["prob_home_implied"]
    # Determine away probability as 1 - model home probability
    merged["ml_prob_away"] = 1 - merged["ml_prob_model"]
    merged["edge_away"] = merged["ml_prob_away"] - merged["prob_away_implied"]
    # Select side with larger positive edge if any
    def choose(row):
        eh, ea = row["edge_home"], row["edge_away"]
        if pd.isna(eh) and pd.isna(ea):
            return None
        # Only bet if one of edges > 0
        best_edge = None
        side = None
        if eh is not None and eh > 0:
            best_edge = eh; side = "home"
        if ea is not None and ea > 0 and (best_edge is None or ea > best_edge):
            best_edge = ea; side = "away"
        return side
    merged["bet"] = merged.apply(choose, axis=1)
    bets = merged[merged["bet"].notna()].copy()
    if bets.empty:
        return pd.DataFrame(columns=["game_id","book","bet","line","pred","edge","result","price","pnl"]), BacktestSummary(len(games), len(cl), 0, 0, None, 0.0, None, None)
    # Price assignment and edge selection
    bets["price"] = np.where(bets["bet"] == "home", bets.get("ml_home", np.nan), bets.get("ml_away", np.nan))
    bets["edge"] = np.where(bets["bet"] == "home", bets["edge_home"], bets["edge_away"])
    bets["pred"] = np.where(bets["bet"] == "home", bets["ml_prob_model"], bets["ml_prob_away"])
    bets["line"] = bets["price"]
    # Resolve: need final scores
    gm = games[["game_id","home_score","away_score"]].copy()
    bets = bets.merge(gm, on="game_id", how="left")
    def settle(row):
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]) or ((row["home_score"]==0) and (row["away_score"]==0)):
            return None
        winner = "home" if row["home_score"] > row["away_score"] else ("away" if row["away_score"] > row["home_score"] else None)
        if winner is None:
            return 0.5
        return 1 if winner == row["bet"] else 0
    bets["result"] = bets.apply(settle, axis=1)
    def pnl_units(price: float, result):
        if result is None or result == 0.5:
            return 0.0
        if price > 0:
            return price / 100.0 if result == 1 else -1.0
        else:
            return 1.0 if result == 1 else price / -100.0 * -1.0
    bets["pnl"] = [pnl_units(p, r) for p, r in zip(bets["price"], bets["result"])]
    n_resolved = bets["result"].notna().sum()
    win_rate = float((bets["result"] == 1).sum() / n_resolved) if n_resolved else None
    pnl = float(bets["pnl"].sum())
    roi = pnl / len(bets) if len(bets) else None
    summary = BacktestSummary(
        n_games=len(games),
        n_books_rows=len(cl),
        n_bets=len(bets),
        n_resolved=int(n_resolved),
        win_rate=win_rate,
        pnl_units=pnl,
        roi=roi,
        avg_edge=float(bets["edge"].abs().mean()) if len(bets) else None,
    )
    return bets[["game_id","book","bet","line","pred","edge","result","price","pnl"]], summary
