from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketConsensusConfig:
    min_books: int = 1
    period: str = "full_game"
    book_title_filter: Optional[str] = None


def _norm_book_title(s: object) -> str:
    try:
        if s is None:
            return ""
        return str(s).strip().lower()
    except Exception:
        return ""


def _as_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _iqr(x: pd.Series) -> float:
    xs = _as_num(x).dropna()
    if xs.empty:
        return float("nan")
    try:
        return float(xs.quantile(0.75) - xs.quantile(0.25))
    except Exception:
        return float("nan")


def _range(x: pd.Series) -> float:
    xs = _as_num(x).dropna()
    if xs.empty:
        return float("nan")
    try:
        return float(xs.max() - xs.min())
    except Exception:
        return float("nan")


def build_market_consensus(df: pd.DataFrame, cfg: MarketConsensusConfig = MarketConsensusConfig()) -> pd.DataFrame:
    """Build one-row-per-game consensus + dispersion from a joined odds table.

    Intended input: outputs/games_with_last.csv (multi-row per game_id across books/markets/periods).

    Output columns:
      - game_id
      - market_total, market_total_books, market_total_std, market_total_iqr, market_total_range
      - spread_home, spread_home_books, spread_home_std, spread_home_iqr, spread_home_range

    Consensus is the median across books (robust to outliers).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    # Guard against inputs that already have an index level named 'game_id' in addition
    # to a 'game_id' column (pandas then considers the key ambiguous).
    try:
        work.index = pd.RangeIndex(len(work))
        work.index.name = None
    except Exception:
        pass
    if "game_id" not in work.columns:
        raise ValueError("input is missing required column: game_id")

    work["game_id"] = work["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

    # Optional filter by period.
    if "period" in work.columns and cfg.period:
        p = work["period"].astype(str).str.lower().str.strip()
        work = work[p == cfg.period.lower()].copy()

    # Optional filter by book title substring(s).
    if cfg.book_title_filter and "book" in work.columns:
        filt = [t.strip().lower() for t in str(cfg.book_title_filter).split(",") if t.strip()]
        if filt:
            b = work["book"].map(_norm_book_title)
            mask = pd.Series(False, index=work.index)
            for tok in filt:
                mask = mask | b.str.contains(tok, na=False)
            work = work[mask].copy()

    # Normalize key line columns.
    if "market_total" not in work.columns:
        if "total" in work.columns:
            work["market_total"] = work["total"]
        else:
            work["market_total"] = np.nan

    if "spread_home" not in work.columns:
        if "home_spread" in work.columns:
            work["spread_home"] = work["home_spread"]
        else:
            work["spread_home"] = np.nan

    if "market" in work.columns:
        m = work["market"].astype(str).str.lower().str.strip()
    else:
        m = pd.Series("", index=work.index)

    totals = work[m == "totals"].copy()
    spreads = work[m == "spreads"].copy()

    out = pd.DataFrame({"game_id": sorted(set(work["game_id"].dropna().astype(str).tolist()))})

    def _agg_lines(sub: pd.DataFrame, line_col: str, prefix: str) -> pd.DataFrame:
        if sub.empty:
            return pd.DataFrame({"game_id": out["game_id"].copy()})

        sub = sub[[c for c in ["game_id", "book", line_col] if c in sub.columns]].copy()
        try:
            sub.index = pd.RangeIndex(len(sub))
            sub.index.name = None
        except Exception:
            pass
        sub[line_col] = _as_num(sub[line_col])
        sub = sub[sub[line_col].notna()].copy()
        if sub.empty:
            return pd.DataFrame({"game_id": out["game_id"].copy()})

        # Book counts based on distinct non-null (game_id, book).
        if "book" in sub.columns:
            book_counts = (
                sub.dropna(subset=["book"]).groupby(sub["game_id"])["book"].nunique(dropna=True)
            )
        else:
            book_counts = sub.groupby(sub["game_id"])[line_col].size()

        grouped = sub.groupby(sub["game_id"])[line_col]
        game_ids = grouped.size().index.astype(str)
        res = pd.DataFrame({
            "game_id": game_ids,
            prefix: grouped.median().astype(float),
            f"{prefix}_std": grouped.std(ddof=0).astype(float),
            f"{prefix}_iqr": grouped.apply(_iqr).astype(float),
            f"{prefix}_range": grouped.apply(_range).astype(float),
        })
        try:
            res.index = pd.RangeIndex(len(res))
            res.index.name = None
        except Exception:
            pass
        res[f"{prefix}_books"] = book_counts.reindex(game_ids).fillna(0).astype(int).values

        # Apply min_books constraint (set to NaN when below threshold).
        minb = int(max(1, cfg.min_books))
        low = res[f"{prefix}_books"] < minb
        if low.any():
            res.loc[low, [prefix, f"{prefix}_std", f"{prefix}_iqr", f"{prefix}_range"]] = np.nan

        return res

    tot_ag = _agg_lines(totals, "market_total", "market_total")
    spr_ag = _agg_lines(spreads, "spread_home", "spread_home")

    out = out.merge(tot_ag, on="game_id", how="left")
    out = out.merge(spr_ag, on="game_id", how="left")
    return out


def make_market_consensus(
    in_path: Path,
    out_path: Path,
    cfg: MarketConsensusConfig = MarketConsensusConfig(),
) -> Path:
    if not Path(in_path).exists():
        raise FileNotFoundError(f"input not found: {in_path}")
    df = pd.read_csv(in_path, low_memory=False)
    out = build_market_consensus(df, cfg=cfg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path
