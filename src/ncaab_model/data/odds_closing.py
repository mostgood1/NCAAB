from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

# Columns produced by OddsHistoryRow normalization
REQUIRED_COLS = {
    "event_id", "book", "fetched_at", "commence_time", "market", "period"
}

EDGE_OUTPUT_COLS = [
    "event_id", "book", "market", "period", "commence_time",
    "home_team_name", "away_team_name",
    # market-specific raw odds columns will be preserved where present
    "moneyline_home", "moneyline_away",
    "home_spread", "home_spread_price", "away_spread", "away_spread_price",
    "total", "over_price", "under_price",
    # computed edge columns appended later
    "pred_total", "pred_margin", "edge_total", "edge_margin",
    "edge_total_pct", "edge_margin_pct", "kelly_fraction_total", "kelly_fraction_ml_home", "kelly_fraction_ml_away"
]


def _parse_times(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("fetched_at", "last_update", "commence_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df


def load_snapshots(input_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in input_paths:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = _parse_times(df)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df


def compute_closing_lines(df: pd.DataFrame, window_minutes: int | None = None) -> pd.DataFrame:
    """Compute closing lines per (event_id, book, market, period).

    Selection priority per (event_id, book, market, period):
      1. Latest row whose last_update is within the pre-tip window and <= commence_time
      2. If none, latest row whose last_update <= commence_time (any time before tip)
      3. If none, latest row whose fetched_at <= commence_time
      4. Fallback: latest observed row (even if post-tip)

    window_minutes: restrict primary eligibility to rows with last_update >= commence_time - window and <= commence_time.
      If None, treat all pre-tip last_update rows equally.
    """
    if df.empty:
        return df

    df = df.copy()
    df = _parse_times(df)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for closing lines: {missing}")

    # Guard window
    if window_minutes is not None:
        try:
            wdelta = pd.to_timedelta(int(window_minutes), unit="m")
        except Exception:
            wdelta = None
    else:
        wdelta = None

    keys = ["event_id", "book", "market", "period"]
    # Eligibility flags
    has_last = df["last_update"].notna()
    pre_tip_last = has_last & (df["last_update"] <= df["commence_time"])
    if wdelta is not None:
        in_window_last = pre_tip_last & (df["last_update"] >= (df["commence_time"] - wdelta))
    else:
        in_window_last = pre_tip_last
    pre_tip_fetch = df["fetched_at"] <= df["commence_time"]

    # Priority ranking
    # 3: in-window last_update
    # 2: any pre-tip last_update
    # 1: pre-tip fetched_at
    # 0: fallback
    df["prio"] = np.select([
        in_window_last,
        pre_tip_last,
        pre_tip_fetch,
    ], [3, 2, 1], default=0)
    # Timestamp for tie-break: prefer last_update where available else fetched_at
    df["tstamp"] = np.where(has_last, df["last_update"], df["fetched_at"])
    df_sorted = df.sort_values(keys + ["prio", "tstamp"]).reset_index(drop=True)
    picked = df_sorted.groupby(keys, as_index=False).tail(1)

    common_cols = [
        "event_id", "book", "market", "period", "commence_time", "last_update",
        "home_team_name", "away_team_name",
    ]
    # Add movement context columns (we'll populate after selecting rows)
    movement_cols = ["open_total", "close_total", "delta_total", "steam_total_flag",
                     "open_home_spread", "close_home_spread", "delta_home_spread", "steam_spread_flag",
                     "open_moneyline_home", "close_moneyline_home", "delta_moneyline_home", "steam_ml_home_flag"]
    h2h = picked[picked["market"] == "h2h"].copy()
    h2h = h2h[[c for c in common_cols + ["moneyline_home", "moneyline_away"] if c in h2h.columns]]
    spreads = picked[picked["market"] == "spreads"].copy()
    spreads = spreads[[c for c in common_cols + ["home_spread", "home_spread_price", "away_spread", "away_spread_price"] if c in spreads.columns]]
    totals = picked[picked["market"] == "totals"].copy()
    totals = totals[[c for c in common_cols + ["total", "over_price", "under_price"] if c in totals.columns]]
    out = pd.concat([h2h, spreads, totals], ignore_index=True)
    # Attach movement metrics by reconstructing earliest vs picked per key
    try:
        base_cols = ["event_id", "book", "market", "period"]
        earliest = df.sort_values([*base_cols, "tstamp"]).groupby(base_cols, as_index=False).head(1)
        latest = picked
        def _merge_mv(col_earliest: str, col_latest: str, out_open: str, out_close: str, out_delta: str, flag_name: str, frame_market: str):
            if col_latest not in latest.columns or col_earliest not in earliest.columns:
                return
            # Only compute within matching market subset
            e_sub = earliest[earliest["market"] == frame_market][base_cols + [col_earliest]]
            l_sub = latest[latest["market"] == frame_market][base_cols + [col_latest]]
            merged_mv = l_sub.merge(e_sub, on=base_cols, how="left", suffixes=("_latest","_earliest"))
            merged_mv[out_open] = merged_mv[col_earliest]
            merged_mv[out_close] = merged_mv[col_latest]
            merged_mv[out_delta] = merged_mv[out_close] - merged_mv[out_open]
            # Steam flag: large absolute move beyond heuristic thresholds
            thresh = 1.5 if frame_market == "totals" else (2.0 if frame_market == "spreads" else 40.0)
            merged_mv[flag_name] = merged_mv[out_delta].abs() >= thresh
            # Map back into out DataFrame
            out.merge(merged_mv[base_cols + [out_open, out_close, out_delta, flag_name]], on=base_cols, how="left", inplace=True)
        # Totals movement
        _merge_mv("total", "total", "open_total", "close_total", "delta_total", "steam_total_flag", "totals")
        # Spread movement (home perspective)
        _merge_mv("home_spread", "home_spread", "open_home_spread", "close_home_spread", "delta_home_spread", "steam_spread_flag", "spreads")
        # Moneyline movement (home side)
        _merge_mv("moneyline_home", "moneyline_home", "open_moneyline_home", "close_moneyline_home", "delta_moneyline_home", "steam_ml_home_flag", "h2h")
    except Exception:
        # Best-effort; leave movement columns empty on failure
        for c in movement_cols:
            if c not in out.columns:
                out[c] = np.nan
    return out


def compute_last_odds(df: pd.DataFrame, tolerance_seconds: int = 60) -> pd.DataFrame:
    """Compute last observed pre-tip odds per (event_id, book, market, period) without fallback.

    Rules (no synthetic data):
      - Determine a row timestamp: prefer last_update if present else fetched_at
      - Keep only rows where timestamp <= commence_time + tolerance_seconds
      - Select the latest timestamp per (event_id, book, market, period)
      - Do not backfill with post-tip or missing commence_time rows

    Returns tidy rows split per market similar to compute_closing_lines.
    """
    if df.empty:
        return df
    df = df.copy()
    df = _parse_times(df)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for last odds: {missing}")
    # Timestamp selection and pre-tip filter
    ts = df["last_update"].where(df.get("last_update").notna() if "last_update" in df.columns else False, df["fetched_at"])
    df["_ts"] = ts
    tol = pd.to_timedelta(max(0, int(tolerance_seconds)), unit="s")
    df["_pre_tip"] = (df["_ts"] <= (df["commence_time"] + tol))
    # Drop rows with unknown commence_time or post-tip (no fallback)
    df = df[df["commence_time"].notna() & df["_pre_tip"]].copy()
    if df.empty:
        return pd.DataFrame()
    keys = ["event_id", "book", "market", "period"]
    df_sorted = df.sort_values(keys + ["_ts"]).reset_index(drop=True)
    picked = df_sorted.groupby(keys, as_index=False).tail(1)
    common_cols = [
        "event_id", "book", "market", "period", "commence_time", "last_update",
        "home_team_name", "away_team_name",
    ]
    # Split by market keeping relevant fields
    h2h = picked[picked["market"] == "h2h"].copy()
    h2h = h2h[common_cols + ["moneyline_home", "moneyline_away"]]

    spreads = picked[picked["market"] == "spreads"].copy()
    spreads = spreads[common_cols + [
        "home_spread", "home_spread_price", "away_spread", "away_spread_price"
    ]]

    totals = picked[picked["market"] == "totals"].copy()
    totals = totals[common_cols + ["total", "over_price", "under_price"]]

    out = pd.concat([h2h, spreads, totals], ignore_index=True)
    return out


def read_directory_for_dates(in_dir: Path) -> list[Path]:
    """Collect all normalized odds snapshots under a root directory.

    Includes:
      - Flat files matching odds_YYYY-MM-DD.csv (daily aggregates)
      - Per-date subfolders containing snapshot_* files written by odds-snapshot
    """
    paths: list[Path] = []
    # Daily aggregate files
    paths.extend([p for p in in_dir.glob("odds_*.csv") if p.is_file()])
    # Per-date snapshot folders
    for sub in in_dir.iterdir():
        try:
            if not sub.is_dir():
                continue
            name = sub.name
            # simple YYYY-MM-DD check
            parts = name.split("-")
            if len(parts) == 3 and all(part.isdigit() for part in [parts[0], parts[1], parts[2]]):
                # snapshot files
                for f in sub.glob("snapshot_*.csv"):
                    if f.is_file():
                        paths.append(f)
        except Exception:
            continue
    return sorted(paths)


def make_closing_lines(in_dir: Path, out_path: Path) -> Path:
    in_paths = read_directory_for_dates(in_dir)
    if not in_paths:
        raise FileNotFoundError(f"No odds_*.csv files found in {in_dir}")
    df = load_snapshots(in_paths)
    closed = compute_closing_lines(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    closed.to_csv(out_path, index=False)
    return out_path


def make_last_odds(in_dir: Path, out_path: Path, tolerance_seconds: int = 60) -> Path:
    in_paths = read_directory_for_dates(in_dir)
    if not in_paths:
        raise FileNotFoundError(f"No odds_*.csv files found in {in_dir}")
    df = load_snapshots(in_paths)
    last_df = compute_last_odds(df, tolerance_seconds=tolerance_seconds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_df.to_csv(out_path, index=False)
    return out_path


def compute_edges(merged_games_last: pd.DataFrame) -> pd.DataFrame:
    """Compute prediction vs market edges from a merged games+last odds DataFrame.

    Expects columns:
      - pred_total, pred_margin (model outputs)
      - For totals: total
      - For spreads: home_spread (interpreted from home perspective)
      - For moneyline: moneyline_home, moneyline_away (American odds)

    Derived columns:
      - edge_total = pred_total - total
      - edge_margin = pred_margin - (-home_spread) if spread is home handicap (i.e., model margin minus market margin)
      - edge_total_pct: edge_total / total (guarded)
      - edge_margin_pct: edge_margin / abs(home_spread) (guarded)
      - kelly_fraction_total: simplistic fraction using assumption of fair variance (placeholder heuristic)
      - kelly_fraction_ml_home / ml_away: Kelly fractions from implied probabilities vs fair (derived from predicted margin approximated to ML prob)

    Note: The Kelly fractions here use a coarse conversion and should be refined with a proper win probability model later.
    """
    if merged_games_last.empty:
        return merged_games_last
    df = merged_games_last.copy()
    # Basic edges
    if "pred_total" in df.columns and "total" in df.columns:
        df["edge_total"] = pd.to_numeric(df["pred_total"], errors="coerce") - pd.to_numeric(df["total"], errors="coerce")
    else:
        df["edge_total"] = np.nan
    if "pred_margin" in df.columns and "home_spread" in df.columns:
        # Market home spread is points the home team gives (negative if favored). Convert to market margin.
        market_margin = pd.to_numeric(df["home_spread"], errors="coerce") * -1.0
        df["edge_margin"] = pd.to_numeric(df["pred_margin"], errors="coerce") - market_margin
    else:
        df["edge_margin"] = np.nan
    # Percent edges (guard division)
    # Use safe series extraction to avoid attribute errors when columns missing or scalar values present
    if "total" in df.columns:
        total_series = pd.to_numeric(df["total"], errors="coerce")
    else:
        total_series = pd.Series([np.nan] * len(df))
    if "home_spread" in df.columns:
        spread_series = pd.to_numeric(df["home_spread"], errors="coerce")
    else:
        spread_series = pd.Series([np.nan] * len(df))
    df["edge_total_pct"] = np.where(
        (np.abs(total_series) > 0) & df["edge_total"].notna(),
        df["edge_total"] / total_series.replace(0, np.nan),
        np.nan,
    )
    df["edge_margin_pct"] = np.where(
        (np.abs(spread_series) > 0) & df["edge_margin"].notna(),
        df["edge_margin"] / np.abs(spread_series.replace(0, np.nan)),
        np.nan,
    )
    # Kelly placeholder: treat edge_total as value vs fair line, scale by a volatility proxy
    # This is a heuristic; refined approach would integrate distribution assumptions.
    vol_proxy = 15.0  # stand-in for stdev of total outcome
    df["kelly_fraction_total"] = np.where(df["edge_total"].notna(), np.clip(df["edge_total"] / (vol_proxy * 2.0), -1, 1), np.nan)
    # Moneyline Kelly: Convert American odds and approximate fair probability from predicted margin (rough logistic)
    def american_to_prob(odds: float) -> float:
        if pd.isna(odds):
            return np.nan
        if odds < 0:
            return (-odds) / ((-odds) + 100.0)
        return 100.0 / (odds + 100.0)
    def american_to_b(odds: float) -> float:
        # payout multiplier b (decimal-1)
        if pd.isna(odds):
            return np.nan
        if odds < 0:
            return 100.0 / (-odds)
        return odds / 100.0
    def margin_to_prob(margin: float) -> float:
        # Approx logistic: P(home win) ~ 1/(1+exp(-margin/7))
        if pd.isna(margin):
            return np.nan
        return 1.0 / (1.0 + np.exp(-margin / 7.0))
    # Moneyline probabilities and Kelly only if columns present
    has_ml_cols = ("moneyline_home" in df.columns) and ("moneyline_away" in df.columns)
    has_margin = ("pred_margin" in df.columns)
    if has_ml_cols:
        df["home_ml_prob_market"] = df["moneyline_home"].map(american_to_prob)
        df["away_ml_prob_market"] = df["moneyline_away"].map(american_to_prob)
    else:
        df["home_ml_prob_market"] = np.nan
        df["away_ml_prob_market"] = np.nan
    if has_margin:
        df["home_ml_prob_fair"] = df["pred_margin"].map(margin_to_prob)
        df["away_ml_prob_fair"] = 1.0 - df["home_ml_prob_fair"]
    else:
        df["home_ml_prob_fair"] = np.nan
        df["away_ml_prob_fair"] = np.nan
    # Kelly fraction: (bp - q)/(b) where b = payout multiplier
    def kelly_fraction(american_odds: float, fair_p: float) -> float:
        if pd.isna(american_odds) or pd.isna(fair_p):
            return np.nan
        if american_odds < 0:
            b = 100.0 / (-american_odds)
        else:
            b = american_odds / 100.0
        q = 1.0 - fair_p
        k = (b * fair_p - q) / b
        return float(np.clip(k, -1.0, 1.0))
    if has_ml_cols and has_margin:
        df["kelly_fraction_ml_home"] = [kelly_fraction(o, p) for o, p in zip(df["moneyline_home"], df["home_ml_prob_fair"]) ]
        df["kelly_fraction_ml_away"] = [kelly_fraction(o, p) for o, p in zip(df["moneyline_away"], df["away_ml_prob_fair"]) ]
    else:
        df["kelly_fraction_ml_home"] = np.nan
        df["kelly_fraction_ml_away"] = np.nan
    # Moneyline EV using fair probabilities from margin model if available
    if has_ml_cols and has_margin:
        b_home = df["moneyline_home"].map(american_to_b)
        b_away = df["moneyline_away"].map(american_to_b)
        p_home = df["home_ml_prob_fair"]
        p_away = df["away_ml_prob_fair"]
        df["home_ml_ev"] = p_home * b_home - (1.0 - p_home)
        df["away_ml_ev"] = p_away * b_away - (1.0 - p_away)
        df["home_ml_b"] = b_home
        df["away_ml_b"] = b_away
    else:
        df["home_ml_ev"] = np.nan
        df["away_ml_ev"] = np.nan
        df["home_ml_b"] = np.nan
        df["away_ml_b"] = np.nan
    return df
