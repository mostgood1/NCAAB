from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _parse_date(s: str | None) -> str | None:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(str(s).strip()).isoformat()
    except Exception:
        return None


def _norm_gid(v: Any) -> str:
    try:
        if pd.isna(v):
            return ""
        s = str(v).strip()
        return s[:-2] if s.endswith(".0") else s
    except Exception:
        return str(v)


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        if isinstance(v, (float, int)):
            f = float(v)
            return f if np.isfinite(f) else None
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            return None
        f = float(s)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def _american_profit_per_1_risk(odds: float | None) -> float:
    """Profit (not return) for 1 unit risked."""
    if odds is None:
        # default -110-ish
        return 100.0 / 110.0
    try:
        o = float(odds)
        if not np.isfinite(o) or o == 0:
            return 100.0 / 110.0
        if o > 0:
            return o / 100.0
        return 100.0 / (-o)
    except Exception:
        return 100.0 / 110.0


def _expected_units_per_1_risk(p_win: float | None, odds: float | None) -> float | None:
    """Expected profit in units for risking 1 unit at the given odds.

    EV_units = p_win * profit_per_1_risk - (1 - p_win)
    """
    if p_win is None:
        return None
    try:
        p = float(p_win)
    except Exception:
        return None
    if not np.isfinite(p):
        return None
    p = max(0.0, min(1.0, p))
    prof = _american_profit_per_1_risk(odds)
    try:
        return float(p * prof - (1.0 - p))
    except Exception:
        return None


def _american_implied_prob(odds: float | None) -> float | None:
    """Convert American odds to implied probability in (0,1)."""
    if odds is None:
        return None
    try:
        o = float(odds)
        if not np.isfinite(o) or o == 0:
            return None
        if o > 0:
            return float(100.0 / (o + 100.0))
        return float((-o) / ((-o) + 100.0))
    except Exception:
        return None


def _norm_cdf(z: float) -> float:
    try:
        from math import erf, sqrt

        return 0.5 * (1.0 + erf(z / sqrt(2.0)))
    except Exception:
        return 0.5


def _clamp01(x: float) -> float:
    try:
        if x < 0:
            return 0.0
        if x > 1:
            return 1.0
        return float(x)
    except Exception:
        return 0.0


def _conf_strength(p: float | None, target: float = 0.65) -> float:
    """Map p in [0.5,1] to [0,1] with p==target at 1."""
    if p is None:
        return 0.0
    try:
        p = float(p)
    except Exception:
        return 0.0
    if not np.isfinite(p):
        return 0.0
    p = max(0.0, min(1.0, p))
    if p <= 0.5:
        return 0.0
    denom = max(1e-6, target - 0.5)
    return _clamp01((p - 0.5) / denom)


def _juice_penalty(price: float | None, max_abs: float = 120.0) -> float:
    """Penalty in [0,1] for heavily juiced negative prices."""
    if price is None:
        return 0.0
    try:
        p = float(price)
    except Exception:
        return 0.0
    if not np.isfinite(p):
        return 0.0
    if p >= 0:
        return 0.0
    a = abs(p)
    if a <= max_abs:
        return 0.0
    return _clamp01((a - max_abs) / 200.0)


@dataclass(frozen=True)
class HighLikelihoodConfig:
    out_dir: Path
    date: str
    top_n: int = 12
    min_score: float = 75.0
    sigma_total: float = 12.0
    sigma_margin: float = 8.0
    max_juice_abs: float = 120.0
    include_markets: tuple[str, ...] = ("ATS", "OU", "ML")
    max_picks_per_game: int = 1
    # Preference order used when limiting per-game picks and breaking ties.
    # Default: ATS/OU-first so the list isn't ML-dominated.
    market_preference: tuple[str, ...] = ("ATS", "OU", "ML")
    # Optional per-day caps by market (helps preserve hit-rate when ATS/OU are noisy).
    # Default: ATS/OU majority (ML is capped explicitly).
    max_ats_picks: int = 6
    max_ou_picks: int = 6
    max_ml_picks: int = 4
    # Hit-rate oriented gates for ML.
    ml_favorites_only: bool = False
    # Keep the hit-rate gate primarily on model probability; implied prob is no longer
    # used as a hard filter so modest underdogs can pass when EV is strong.
    min_ml_implied_prob: float = 0.0
    min_ml_model_prob: float = 0.55
    max_ml_underdog_price: float = 180.0
    # Unit-efficiency/ROI gates for ML.
    # - Avoid tying up units in extreme prices.
    # - Require positive expected units per 1 unit risk (and optionally a minimum).
    max_ml_favorite_abs_price: float = 200.0
    min_ml_ev_units: float = 0.02

    # Hit-rate oriented gates for ATS/OU.
    min_ats_model_prob: float = 0.60
    min_ou_model_prob: float = 0.60
    min_ats_edge_pts: float = 4.0
    min_ou_edge_pts: float = 4.0
    min_prob_edge_vs_implied: float = 0.05
    # Unit-efficiency/ROI gates for OU/ATS (expected units per 1 unit risk).
    min_ats_ev_units: float = 0.02
    min_ou_ev_units: float = 0.02


def _load_edges(out_dir: Path, date: str) -> pd.DataFrame:
    p = out_dir / f"align_period_{date}_edges.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"game_id": str}, low_memory=False)
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].map(_norm_gid)
    return df


def _load_results(out_dir: Path, date: str) -> pd.DataFrame:
    p = out_dir / "daily_results" / f"results_{date}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"game_id": str}, low_memory=False)
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].map(_norm_gid)
    return df


def _load_display(out_dir: Path, date: str) -> pd.DataFrame:
    p = out_dir / f"predictions_display_{date}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"game_id": str}, low_memory=False)
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].map(_norm_gid)
    return df


def _load_sim_quantiles(out_dir: Path, date: str) -> pd.DataFrame:
    p = out_dir / f"sim_quantiles_{date}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"game_id": str}, low_memory=False)
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].map(_norm_gid)
    return df


def _load_closing_movement(out_dir: Path, date: str) -> pd.DataFrame:
    """Load per-game closing line movement (open->close deltas + steam flags).

    Primary source is `games_with_closing_<date>.csv` (one row per book/market),
    aggregated down to one row per `game_id`.
    """

    candidates = [
        out_dir / f"games_with_closing_{date}.csv",
        out_dir / "games_with_closing_today.csv",
        out_dir / "games_with_closing_recent.csv",
        out_dir / "games_with_closing_prev.csv",
        out_dir / "games_with_closing.csv",
    ]

    p: Path | None = None
    for cand in candidates:
        if cand.exists():
            p = cand
            break

    if p is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(p, dtype={"game_id": str}, low_memory=False)
    except Exception:
        return pd.DataFrame()

    if df.empty or "game_id" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["game_id"] = df["game_id"].map(_norm_gid)

    # If the file includes multiple dates (e.g., *_recent), filter to the target day.
    try:
        if "date_game" in df.columns:
            df["date_game"] = df["date_game"].astype(str)
            df = df[df["date_game"] == date].copy()
    except Exception:
        pass

    if df.empty:
        return pd.DataFrame()

    def _med_num(s: pd.Series) -> float | None:
        x = pd.to_numeric(s, errors="coerce")
        x = x[np.isfinite(x.to_numpy())]
        if x.empty:
            return None
        try:
            return float(x.median())
        except Exception:
            return None

    def _any_flag(s: pd.Series) -> bool:
        try:
            x = pd.to_numeric(s, errors="coerce").fillna(0)
            return bool((x != 0).any())
        except Exception:
            return False

    want = {
        # Totals
        "open_total": ("open_total", _med_num),
        "close_total": ("close_total", _med_num),
        "delta_total": ("delta_total", _med_num),
        "steam_total_flag": ("steam_total_flag", _any_flag),
        # Spreads (home spread)
        "open_home_spread": ("open_home_spread", _med_num),
        "close_home_spread": ("close_home_spread", _med_num),
        "delta_home_spread": ("delta_home_spread", _med_num),
        "steam_spread_flag": ("steam_spread_flag", _any_flag),
        # Moneyline (home)
        "open_moneyline_home": ("open_moneyline_home", _med_num),
        "close_moneyline_home": ("close_moneyline_home", _med_num),
        "delta_moneyline_home": ("delta_moneyline_home", _med_num),
        "steam_ml_home_flag": ("steam_ml_home_flag", _any_flag),
    }

    agg_spec: dict[str, Any] = {}
    for out_col, (in_col, fn) in want.items():
        if in_col in df.columns:
            agg_spec[out_col] = (in_col, fn)

    if not agg_spec:
        return pd.DataFrame()

    try:
        g = df.groupby("game_id", as_index=False).agg(**agg_spec)
    except Exception:
        return pd.DataFrame()

    return g


def _load_live_snapshot_moves(
    out_dir: Path,
    date: str,
    *,
    cutoff_ts_by_event_id: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load material intraday moves from outputs/live_snapshots/live_<date>.jsonl.

    Returns mapping keyed by ESPN event_id (game_id).
    """

    # Late-night games can cause snapshot records to spill into the next day's file.
    # To keep pregame movement available for those games, read date and date+1 if present.
    paths: list[Path] = []
    try:
        p0 = out_dir / "live_snapshots" / f"live_{date}.jsonl"
        if p0.exists():
            paths.append(p0)
    except Exception:
        pass
    try:
        d2 = (dt.date.fromisoformat(str(date)) + dt.timedelta(days=1)).isoformat()
        p1 = out_dir / "live_snapshots" / f"live_{d2}.jsonl"
        if p1.exists():
            paths.append(p1)
    except Exception:
        pass
    if not paths:
        return {}

    def _to_num(v: Any) -> float | None:
        try:
            if v is None:
                return None
            if isinstance(v, (int, float)):
                f = float(v)
                return f if np.isfinite(f) else None
            s = str(v).strip()
            if not s or s.lower() in ("nan", "none", "null"):
                return None
            f = float(s)
            return f if np.isfinite(f) else None
        except Exception:
            return None

    # Track last/prev distinct values per (event,book) to avoid mixing.
    state: dict[tuple[str, str], dict[str, Any]] = {}

    book_priority = ["draftkings", "fanduel", "betmgm"]

    def _book_rank(book: str) -> int:
        b = str(book or "").strip().lower()
        try:
            return book_priority.index(b)
        except Exception:
            return 999

    def _ts_epoch(ts_iso: str | None) -> float | None:
        if not ts_iso:
            return None
        try:
            import datetime as _dt

            s = str(ts_iso).strip()
            if not s:
                return None
            d = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            if d.tzinfo is None:
                d = d.replace(tzinfo=_dt.timezone.utc)
            return float(d.timestamp())
        except Exception:
            return None

    cutoff_epoch_by_event_id: dict[str, float] = {}
    if isinstance(cutoff_ts_by_event_id, dict) and cutoff_ts_by_event_id:
        try:
            import datetime as _dt

            for eid, ts_str in cutoff_ts_by_event_id.items():
                try:
                    ee = _norm_gid(eid)
                    if not ee:
                        continue
                    s = str(ts_str or "").strip()
                    if not s:
                        continue
                    d = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
                    if d.tzinfo is None:
                        d = d.replace(tzinfo=_dt.timezone.utc)
                    cutoff_epoch_by_event_id[ee] = float(d.timestamp())
                except Exception:
                    continue
        except Exception:
            cutoff_epoch_by_event_id = {}
    try:
        import json

        for p in paths:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    if str(rec.get("endpoint") or "").strip() != "live_lines":
                        continue
                    eid = _norm_gid(rec.get("event_id"))
                    if not eid:
                        continue
                    data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
                    if not isinstance(data, dict):
                        data = {}
                    ts = str(rec.get("ts") or "").strip() or None

                    # Only use full-game pregame lines (not 1H/2H etc).
                    try:
                        if str(data.get("period") or "").strip().lower() != "full_game":
                            continue
                    except Exception:
                        continue

                    # Pregame cutoff (ignore snapshots after scheduled start_time).
                    cutoff_epoch = cutoff_epoch_by_event_id.get(eid)
                    if cutoff_epoch is not None:
                        te = _ts_epoch(ts)
                        if te is None or float(te) > float(cutoff_epoch):
                            continue

                    book = str(data.get("book") or "").strip() or ""

                    tot = _to_num(data.get("total"))
                    spr = _to_num(data.get("spread_home"))

                    st = state.setdefault(
                        (eid, book),
                        {
                            "total_prev": None,
                            "total_last": None,
                            "spread_prev": None,
                            "spread_last": None,
                            "ts_prev": None,
                            "ts_last": None,
                        },
                    )

                    if tot is not None:
                        last = st.get("total_last")
                        if last is None:
                            st["total_last"] = tot
                            st["ts_last"] = ts
                        elif float(tot) != float(last):
                            st["total_prev"] = last
                            st["total_last"] = tot
                            st["ts_prev"] = st.get("ts_last")
                            st["ts_last"] = ts

                    if spr is not None:
                        last = st.get("spread_last")
                        if last is None:
                            st["spread_last"] = spr
                            st["ts_last"] = ts
                        elif float(spr) != float(last):
                            st["spread_prev"] = last
                            st["spread_last"] = spr
                            st["ts_prev"] = st.get("ts_last")
                            st["ts_last"] = ts
    except Exception:
        return {}

    by_event: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for (eid, book), st in state.items():
        by_event.setdefault(eid, []).append((book, st))

    out: dict[str, dict[str, Any]] = {}
    for eid, items in by_event.items():
        if not items:
            continue

        def _item_sort_key(it: tuple[str, dict[str, Any]]) -> tuple[int, float]:
            book, st = it
            te = _ts_epoch(st.get("ts_last"))
            return (_book_rank(book), -(te if te is not None else -1.0))

        book, st = sorted(items, key=_item_sort_key)[0]
        d_total = None
        if st.get("total_prev") is not None and st.get("total_last") is not None:
            try:
                d_total = float(st["total_last"]) - float(st["total_prev"])
            except Exception:
                d_total = None
        d_spread = None
        if st.get("spread_prev") is not None and st.get("spread_last") is not None:
            try:
                d_spread = float(st["spread_last"]) - float(st["spread_prev"])
            except Exception:
                d_spread = None

        out[eid] = {
            "total_prev": st.get("total_prev"),
            "total_last": st.get("total_last"),
            "delta_total": d_total,
            "spread_prev": st.get("spread_prev"),
            "spread_last": st.get("spread_last"),
            "delta_spread_home": d_spread,
            "ts_last": st.get("ts_last"),
            "book": (str(book).strip() or None),
        }

    return out


def _best_rows(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Pick one row per game_id for a market kind (totals/spreads/ml)."""
    if df.empty or "game_id" not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    mkt = d.get("market")
    if mkt is not None:
        try:
            d["_mkt"] = mkt.astype(str).str.lower()
        except Exception:
            d["_mkt"] = ""
    else:
        d["_mkt"] = ""

    period = d.get("period")
    if period is not None:
        try:
            d["_period"] = period.astype(str).str.lower()
        except Exception:
            d["_period"] = ""
    else:
        d["_period"] = ""

    # Prefer full_game; keep others if nothing else.
    d_fg = d[d["_period"].isin(["full_game", "full", "game", "fg"])].copy()
    if d_fg.empty:
        d_fg = d

    if kind == "totals":
        keep = d_fg[d_fg["_mkt"].str.contains("total")].copy()
        if keep.empty:
            return pd.DataFrame()
        keep["pred_total"] = pd.to_numeric(keep.get("pred_total"), errors="coerce")
        keep["total"] = pd.to_numeric(keep.get("total"), errors="coerce")
        keep["edge_total"] = pd.to_numeric(keep.get("edge_total"), errors="coerce")
        keep["over_price"] = pd.to_numeric(keep.get("over_price"), errors="coerce")
        keep["under_price"] = pd.to_numeric(keep.get("under_price"), errors="coerce")
        keep["kelly_fraction_total"] = pd.to_numeric(keep.get("kelly_fraction_total"), errors="coerce")
        # Score candidates: have total+pred; prefer priced; then abs edge.
        keep["_has_price"] = keep[["over_price", "under_price"]].notna().any(axis=1)
        keep["_abs_edge"] = (keep["pred_total"] - keep["total"]).abs()
        keep = keep[keep["pred_total"].notna() & keep["total"].notna()].copy()
        if keep.empty:
            return pd.DataFrame()
        keep = keep.sort_values(["_has_price", "_abs_edge"], ascending=[False, False])
        return keep.drop_duplicates("game_id", keep="first")

    if kind == "spreads":
        keep = d_fg[d_fg["_mkt"].str.contains("spread")].copy()
        if keep.empty:
            return pd.DataFrame()
        keep["pred_margin"] = pd.to_numeric(keep.get("pred_margin"), errors="coerce")
        keep["home_spread"] = pd.to_numeric(keep.get("home_spread"), errors="coerce")
        keep["home_spread_price"] = pd.to_numeric(keep.get("home_spread_price"), errors="coerce")
        keep["away_spread_price"] = pd.to_numeric(keep.get("away_spread_price"), errors="coerce")
        keep = keep[keep["pred_margin"].notna() & keep["home_spread"].notna()].copy()
        if keep.empty:
            return pd.DataFrame()
        keep["_edge_home_cover"] = keep["pred_margin"] + keep["home_spread"]
        keep["_abs_edge"] = keep["_edge_home_cover"].abs()
        keep["_has_price"] = keep[["home_spread_price", "away_spread_price"]].notna().any(axis=1)
        keep = keep.sort_values(["_has_price", "_abs_edge"], ascending=[False, False])
        return keep.drop_duplicates("game_id", keep="first")

    if kind == "ml":
        keep = d_fg[d_fg["_mkt"].str.contains("h2h") | d_fg["_mkt"].str.contains("money")].copy()
        if keep.empty:
            return pd.DataFrame()
        keep["pred_margin"] = pd.to_numeric(keep.get("pred_margin"), errors="coerce")
        keep["moneyline_home"] = pd.to_numeric(keep.get("moneyline_home"), errors="coerce")
        keep["moneyline_away"] = pd.to_numeric(keep.get("moneyline_away"), errors="coerce")
        keep["home_ml_ev"] = pd.to_numeric(keep.get("home_ml_ev"), errors="coerce")
        keep["away_ml_ev"] = pd.to_numeric(keep.get("away_ml_ev"), errors="coerce")
        keep["home_ml_prob_fair"] = pd.to_numeric(keep.get("home_ml_prob_fair"), errors="coerce")
        keep["away_ml_prob_fair"] = pd.to_numeric(keep.get("away_ml_prob_fair"), errors="coerce")
        keep["kelly_fraction_ml_home"] = pd.to_numeric(keep.get("kelly_fraction_ml_home"), errors="coerce")
        keep["kelly_fraction_ml_away"] = pd.to_numeric(keep.get("kelly_fraction_ml_away"), errors="coerce")
        keep["_best_ev"] = np.nanmax(np.vstack([keep["home_ml_ev"].to_numpy(), keep["away_ml_ev"].to_numpy()]), axis=0)
        keep["_has_price"] = keep[["moneyline_home", "moneyline_away"]].notna().any(axis=1)
        keep = keep.sort_values(["_has_price", "_best_ev"], ascending=[False, False])
        return keep.drop_duplicates("game_id", keep="first")

    return pd.DataFrame()


def build_high_likelihood(cfg: HighLikelihoodConfig) -> dict[str, Any]:
    date = _parse_date(cfg.date)
    if not date:
        return {"status": "error", "message": f"invalid date: {cfg.date}"}

    out_dir = Path(cfg.out_dir)
    edges = _load_edges(out_dir, date)
    if edges.empty:
        return {
            "status": "error",
            "message": f"missing align_period_{date}_edges.csv",
            "date": date,
            "out_dir": str(out_dir),
        }

    inc = tuple((m or "").strip().upper() for m in (cfg.include_markets or ("ML",)))
    inc_set = set(inc)

    totals = _best_rows(edges, "totals") if ("OU" in inc_set or "TOTALS" in inc_set) else pd.DataFrame()
    spreads = _best_rows(edges, "spreads") if ("ATS" in inc_set or "SPREADS" in inc_set) else pd.DataFrame()
    ml = _best_rows(edges, "ml") if ("ML" in inc_set) else pd.DataFrame()

    # Fallback: many slates only have ML odds; synthesize ATS/OU candidates from predictions_display.
    disp = _load_display(out_dir, date) if (("OU" in inc_set or "ATS" in inc_set or "TOTALS" in inc_set or "SPREADS" in inc_set)) else pd.DataFrame()
    if (totals.empty and ("OU" in inc_set or "TOTALS" in inc_set)) and not disp.empty:
        d = disp.copy()
        d["total"] = pd.to_numeric(d.get("closing_total"), errors="coerce")
        d["pred_total"] = pd.to_numeric(d.get("pred_total"), errors="coerce")
        d["over_price"] = -110.0
        d["under_price"] = -110.0
        d["kelly_fraction_total"] = np.nan
        d["book"] = d.get("book") if "book" in d.columns else "display"
        totals = d

    if (spreads.empty and ("ATS" in inc_set or "SPREADS" in inc_set)) and not disp.empty:
        d = disp.copy()
        d["home_spread"] = pd.to_numeric(d.get("closing_spread_home"), errors="coerce")
        d["pred_margin"] = pd.to_numeric(d.get("pred_margin"), errors="coerce")
        d["home_spread_price"] = -110.0
        d["away_spread_price"] = -110.0
        d["book"] = d.get("book") if "book" in d.columns else "display"
        spreads = d

    # Optional sentiment / market movement context (open->close deltas + steam flags).
    mv = _load_closing_movement(out_dir, date)
    mv_by_gid: dict[str, dict[str, Any]] = {}
    if not mv.empty and "game_id" in mv.columns:
        for r in mv.to_dict(orient="records"):
            gid = _norm_gid(r.get("game_id"))
            if gid:
                mv_by_gid[gid] = r

    # Load optional intraday movement from live snapshots. We'll provide a per-game
    # cutoff after recs are built (start_time) so we only use *pregame* movement.
    live_mv_by_gid: dict[str, dict[str, Any]] = {}

    recs: list[dict[str, Any]] = []

    def _mk_common(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "date": date,
            "game_id": _norm_gid(r.get("game_id")),
            "home_team": r.get("home_team") or r.get("home_team_name"),
            "away_team": r.get("away_team") or r.get("away_team_name"),
            "book": r.get("book"),
            "start_time": r.get("start_time"),
            "start_time_iso": r.get("start_time_iso"),
            "start_time_local": r.get("start_time_local"),
            "start_tz_abbr": r.get("start_tz_abbr"),
            "display_date": r.get("display_date"),
        }

    # Prefer sim-based probabilities for ATS/OU when available.
    sim = _load_sim_quantiles(out_dir, date) if ("OU" in inc_set or "ATS" in inc_set or "TOTALS" in inc_set or "SPREADS" in inc_set) else pd.DataFrame()
    used_sim_ou = False
    used_sim_ats = False

    if not sim.empty:
        # Totals from sim
        if ("OU" in inc_set or "TOTALS" in inc_set) and {"market_total", "p_over_market"}.issubset(sim.columns):
            for r in sim.to_dict(orient="records"):
                line = _to_float(r.get("market_total"))
                p_over = _to_float(r.get("p_over_market"))
                pred_total = _to_float(r.get("mu_total"))
                if pred_total is None:
                    pred_total = _to_float(r.get("q50_total"))
                if line is None or p_over is None or pred_total is None:
                    continue

                sel = "Over" if float(p_over) >= 0.5 else "Under"
                p_side = float(p_over) if sel == "Over" else (1.0 - float(p_over))
                edge = float(pred_total) - float(line)

                price = -110.0
                implied = _american_implied_prob(price)
                if implied is None:
                    continue
                if abs(edge) < float(cfg.min_ou_edge_pts):
                    continue
                if float(p_side) < float(cfg.min_ou_model_prob):
                    continue
                ev_units = _expected_units_per_1_risk(p_side, price)
                if ev_units is None:
                    continue
                if float(ev_units) < float(cfg.min_ou_ev_units):
                    continue

                conf_model = _conf_strength(p_side, target=0.85)
                edge_s = _clamp01(abs(edge) / 6.0)
                ev_s = _clamp01((float(ev_units) - 0.0) / 0.08)
                juice_p = _juice_penalty(price, max_abs=cfg.max_juice_abs)
                score = 100.0 * (0.55 * conf_model + 0.15 * edge_s + 0.30 * ev_s) - 25.0 * juice_p
                score = float(max(0.0, min(100.0, score)))

                recs.append(
                    {
                        **_mk_common(r),
                        "rec_code": "OU",
                        "market": "totals",
                        "selection": sel,
                        "line": float(line),
                        "price": float(price),
                        "pred_total": float(pred_total),
                        "edge": float(edge),
                        "p_win": float(p_side),
                        "kelly_fraction": None,
                        "score": float(score),
                        "reasons": [
                            f"p_win={p_side:.3f}",
                            f"p_implied={float(implied):.3f}",
                            f"edge_pts={edge:+.1f}",
                            f"ev_units={ev_units:+.3f}" if ev_units is not None else "ev_units=–",
                            "src=sim_quantiles",
                        ],
                    }
                )
            used_sim_ou = True

        # Spreads from sim
        # NOTE: `sim_quantiles` provides both `spread_home` and `market_margin_resolved`.
        # Empirically, `market_margin_resolved == -spread_home` (i.e., opposite sign), and our grading
        # logic assumes `line` is the HOME spread (negative = home favored).
        if ("ATS" in inc_set or "SPREADS" in inc_set) and {"p_cover_home"}.issubset(sim.columns):
            for r in sim.to_dict(orient="records"):
                home = r.get("home_team") or r.get("home_team_name")
                away = r.get("away_team") or r.get("away_team_name")
                if not home or not away:
                    continue
                # Prefer `spread_home` when present; otherwise fall back to `-market_margin_resolved`.
                home_spread = _to_float(r.get("spread_home"))
                if home_spread is None:
                    mm = _to_float(r.get("market_margin_resolved"))
                    home_spread = (-float(mm)) if mm is not None else None
                p_home_cover = _to_float(r.get("p_cover_home"))
                pred_margin = _to_float(r.get("mu_margin"))
                if pred_margin is None:
                    pred_margin = _to_float(r.get("q50_margin"))
                if home_spread is None or p_home_cover is None or pred_margin is None:
                    continue

                edge_home_cover = float(pred_margin) + float(home_spread)
                pick_home = float(p_home_cover) >= 0.5
                sel = "Home" if pick_home else "Away"
                sel_team = home if pick_home else away
                p_side = float(p_home_cover) if pick_home else (1.0 - float(p_home_cover))

                price = -110.0
                implied = _american_implied_prob(price)
                if implied is None:
                    continue
                if abs(edge_home_cover) < float(cfg.min_ats_edge_pts):
                    continue
                if float(p_side) < float(cfg.min_ats_model_prob):
                    continue
                ev_units = _expected_units_per_1_risk(p_side, price)
                if ev_units is None:
                    continue
                if float(ev_units) < float(cfg.min_ats_ev_units):
                    continue

                conf_model = _conf_strength(p_side, target=0.85)
                edge_s = _clamp01(abs(edge_home_cover) / 3.0)
                ev_s = _clamp01((float(ev_units) - 0.0) / 0.08)
                juice_p = _juice_penalty(price, max_abs=cfg.max_juice_abs)
                score = 100.0 * (0.60 * conf_model + 0.10 * edge_s + 0.30 * ev_s) - 25.0 * juice_p
                score = float(max(0.0, min(100.0, score)))

                recs.append(
                    {
                        **_mk_common(r),
                        "rec_code": "ATS",
                        "market": "spreads",
                        "selection": sel,
                        "selection_team": sel_team,
                        # IMPORTANT: keep this as HOME spread (so grading logic is stable)
                        "line": float(home_spread),
                        "price": float(price),
                        "pred_margin": float(pred_margin),
                        "edge": float(edge_home_cover),
                        "p_win": float(p_side),
                        "kelly_fraction": None,
                        "score": float(score),
                        "reasons": [
                            f"p_win={p_side:.3f}",
                            f"p_implied={float(implied):.3f}",
                            f"edge_pts={edge_home_cover:+.1f}",
                            f"ev_units={ev_units:+.3f}" if ev_units is not None else "ev_units=–",
                            "src=sim_quantiles",
                        ],
                    }
                )
            used_sim_ats = True

    # If we have sim-driven OU/ATS, do not also add the normal-approx fallback versions.
    if used_sim_ou:
        totals = pd.DataFrame()
    if used_sim_ats:
        spreads = pd.DataFrame()

    # Totals
    for r in totals.to_dict(orient="records") if not totals.empty else []:
        pred_total = _to_float(r.get("pred_total"))
        line = _to_float(r.get("total"))
        if pred_total is None or line is None:
            continue
        edge = pred_total - line
        sel = "Over" if edge > 0 else "Under"
        price = _to_float(r.get("over_price" if sel == "Over" else "under_price"))
        if price is None:
            price = -110.0
        kelly = _to_float(r.get("kelly_fraction_total"))
        p_over = _norm_cdf((pred_total - line) / max(1e-6, cfg.sigma_total))
        p_side = p_over if sel == "Over" else (1.0 - p_over)

        # Conservative gates: require price and high model probability + meaningful edge.
        implied = _american_implied_prob(price)
        if implied is None:
            continue
        if abs(edge) < float(cfg.min_ou_edge_pts):
            continue
        if float(p_side) < float(cfg.min_ou_model_prob):
            continue

        ev_units = _expected_units_per_1_risk(p_side, price)
        if ev_units is None:
            continue
        if float(ev_units) < float(cfg.min_ou_ev_units):
            continue

        conf_model = _conf_strength(p_side, target=0.85)
        edge_s = _clamp01(abs(edge) / 6.0)
        ev_s = _clamp01(((ev_units or 0.0) - 0.0) / 0.08)
        juice_p = _juice_penalty(price, max_abs=cfg.max_juice_abs)
        score = 100.0 * (0.55 * conf_model + 0.15 * edge_s + 0.30 * ev_s) - 25.0 * juice_p
        score = float(max(0.0, min(100.0, score)))

        recs.append(
            {
                **_mk_common(r),
                "rec_code": "OU",
                "market": "totals",
                "selection": sel,
                "line": line,
                "price": price,
                "pred_total": pred_total,
                "edge": edge,
                "p_win": p_side,
                "kelly_fraction": kelly,
                "score": score,
                "reasons": [
                    f"p_win={p_side:.3f}",
                    f"p_implied={float(implied):.3f}",
                    f"edge_pts={edge:+.1f}",
                    f"ev_units={ev_units:+.3f}" if ev_units is not None else "ev_units=–",
                    f"price={int(price):+d}" if price is not None and float(price).is_integer() else (f"price={price:+.0f}" if price is not None else "price=–"),
                ],
            }
        )

    # Spreads
    for r in spreads.to_dict(orient="records") if not spreads.empty else []:
        pred_margin = _to_float(r.get("pred_margin"))
        home_spread = _to_float(r.get("home_spread"))
        if pred_margin is None or home_spread is None:
            continue
        edge_home_cover = pred_margin + home_spread
        pick_home = edge_home_cover > 0
        sel_team = (r.get("home_team") or r.get("home_team_name")) if pick_home else (r.get("away_team") or r.get("away_team_name"))
        sel = "Home" if pick_home else "Away"
        line = home_spread
        price = _to_float(r.get("home_spread_price" if pick_home else "away_spread_price"))
        if price is None:
            price = -110.0
        # Approx win prob from margin edge
        p_home_cover = _norm_cdf(edge_home_cover / max(1e-6, cfg.sigma_margin))
        p_side = p_home_cover if pick_home else (1.0 - p_home_cover)

        implied = _american_implied_prob(price)
        if implied is None:
            continue
        if abs(edge_home_cover) < float(cfg.min_ats_edge_pts):
            continue
        if float(p_side) < float(cfg.min_ats_model_prob):
            continue

        ev_units = _expected_units_per_1_risk(p_side, price)
        if ev_units is None:
            continue
        if float(ev_units) < float(cfg.min_ats_ev_units):
            continue

        conf_model = _conf_strength(p_side, target=0.85)
        edge_s = _clamp01(abs(edge_home_cover) / 3.0)
        ev_s = _clamp01(((ev_units or 0.0) - 0.0) / 0.08)
        juice_p = _juice_penalty(price, max_abs=cfg.max_juice_abs)
        score = 100.0 * (0.60 * conf_model + 0.10 * edge_s + 0.30 * ev_s) - 25.0 * juice_p
        score = float(max(0.0, min(100.0, score)))

        recs.append(
            {
                **_mk_common(r),
                "rec_code": "ATS",
                "market": "spreads",
                "selection": sel,
                "selection_team": sel_team,
                "line": line,
                "price": price,
                "pred_margin": pred_margin,
                "edge": edge_home_cover,
                "p_win": p_side,
                "kelly_fraction": None,
                "score": score,
                "reasons": [
                    f"p_win={p_side:.3f}",
                    f"p_implied={float(implied):.3f}",
                    f"edge_pts={edge_home_cover:+.1f}",
                    f"ev_units={ev_units:+.3f}" if ev_units is not None else "ev_units=–",
                    f"price={int(price):+d}" if price is not None and float(price).is_integer() else (f"price={price:+.0f}" if price is not None else "price=–"),
                ],
            }
        )

    # Moneyline
    for r in ml.to_dict(orient="records") if not ml.empty else []:
        home = r.get("home_team") or r.get("home_team_name")
        away = r.get("away_team") or r.get("away_team_name")

        home_ml = _to_float(r.get("moneyline_home"))
        away_ml = _to_float(r.get("moneyline_away"))
        if home_ml is None or away_ml is None:
            continue

        # Pick the side with higher model win probability.
        p_home = _to_float(r.get("home_ml_prob_fair"))
        p_away = _to_float(r.get("away_ml_prob_fair"))
        if p_home is None or p_away is None:
            pm = _to_float(r.get("pred_margin"))
            if pm is None:
                continue
            p_home = _norm_cdf(pm / max(1e-6, cfg.sigma_margin))
            p_away = 1.0 - p_home

        pick_home = bool(float(p_home) >= float(p_away))
        sel_team = home if pick_home else away
        price = home_ml if pick_home else away_ml
        p_win = float(p_home if pick_home else p_away)

        implied = _american_implied_prob(price)
        if implied is None:
            continue

        # Hit-rate gates.
        if bool(cfg.ml_favorites_only) and float(price) > 0:
            continue
        if (not bool(cfg.ml_favorites_only)) and float(price) > float(cfg.max_ml_underdog_price):
            continue
        # Avoid extreme favorites (unit efficiency guardrail).
        try:
            if float(price) < 0 and cfg.max_ml_favorite_abs_price is not None and float(cfg.max_ml_favorite_abs_price) > 0:
                if abs(float(price)) > float(cfg.max_ml_favorite_abs_price):
                    continue
        except Exception:
            pass
        if float(implied) < float(cfg.min_ml_implied_prob):
            continue
        if float(p_win) < float(cfg.min_ml_model_prob):
            continue

        # Unit-efficiency gate: require positive (or minimum) expected units per 1 unit risk.
        ev_units = _expected_units_per_1_risk(p_win, price)
        if ev_units is None:
            continue
        try:
            if float(ev_units) < float(cfg.min_ml_ev_units):
                continue
        except Exception:
            pass

        # Secondary signals.
        kelly = _to_float(r.get("kelly_fraction_ml_home" if pick_home else "kelly_fraction_ml_away"))

        # Score: prioritize win probability + implied probability + EV per unit; avoid long-shot drift.
        conf_model = _conf_strength(p_win, target=0.70)
        conf_implied = _conf_strength(float(implied), target=0.70)
        ev_s = _clamp01((float(ev_units) - 0.0) / 0.10)
        juice_p = _juice_penalty(price, max_abs=cfg.max_juice_abs)
        score = 100.0 * (0.55 * conf_model + 0.20 * conf_implied + 0.25 * ev_s) - 25.0 * juice_p
        score = float(max(0.0, min(100.0, score)))

        recs.append(
            {
                **_mk_common(r),
                "rec_code": "ML",
                "market": "h2h",
                "selection": sel_team,
                "selection_team": sel_team,
                "line": None,
                "price": price,
                "pred_margin": _to_float(r.get("pred_margin")),
                "edge": None,
                "p_win": p_win,
                "kelly_fraction": kelly,
                "score": score,
                "reasons": [
                    f"p_win={p_win:.3f}",
                    f"p_implied={float(implied):.3f}",
                    f"ev_units={ev_units:+.3f}",
                    f"price={price:+.0f}" if price is not None else "price=–",
                ],
                "ev": ev_units,
            }
        )

    # Apply sentiment adjustments (small score nudges + reasons) when line movement is available.
    # Conventions:
    # - Totals: delta_total > 0 means market total moved UP.
    # - Spreads: delta_home_spread < 0 means home spread moved MORE negative (market towards home).
    # - ML: delta_moneyline_home < 0 means home ML became MORE expensive (market towards home).
    for rec in recs:
        gid = _norm_gid(rec.get("game_id"))
        if not gid:
            continue
        mv_row = mv_by_gid.get(gid)
        if not mv_row:
            continue

        code = str(rec.get("rec_code") or "").upper()
        sel = str(rec.get("selection") or "")

        move_market: str | None = None
        move_open: float | None = None
        move_close: float | None = None
        move_delta: float | None = None
        steam = False
        align: bool | None = None
        adj = 0.0

        if code == "OU":
            move_market = "total"
            move_open = _to_float(mv_row.get("open_total"))
            move_close = _to_float(mv_row.get("close_total"))
            move_delta = _to_float(mv_row.get("delta_total"))
            steam = bool(mv_row.get("steam_total_flag") or False)
            if move_delta is not None:
                if sel.lower().startswith("over"):
                    align = float(move_delta) > 0
                elif sel.lower().startswith("under"):
                    align = float(move_delta) < 0

                base = min(2.0, abs(float(move_delta)) * 0.75)
                if align is True:
                    adj += base
                elif align is False:
                    adj -= base
                if steam and align is True:
                    adj += 2.0
                elif steam and align is False:
                    adj -= 2.0

        elif code == "ATS":
            move_market = "spread_home"
            move_open = _to_float(mv_row.get("open_home_spread"))
            move_close = _to_float(mv_row.get("close_home_spread"))
            move_delta = _to_float(mv_row.get("delta_home_spread"))
            steam = bool(mv_row.get("steam_spread_flag") or False)
            if move_delta is not None:
                if sel.lower().startswith("home"):
                    align = float(move_delta) < 0
                elif sel.lower().startswith("away"):
                    align = float(move_delta) > 0

                base = min(2.0, abs(float(move_delta)) * 0.60)
                if align is True:
                    adj += base
                elif align is False:
                    adj -= base
                if steam and align is True:
                    adj += 2.0
                elif steam and align is False:
                    adj -= 2.0

        elif code == "ML":
            move_market = "ml_home"
            move_open = _to_float(mv_row.get("open_moneyline_home"))
            move_close = _to_float(mv_row.get("close_moneyline_home"))
            move_delta = _to_float(mv_row.get("delta_moneyline_home"))
            steam = bool(mv_row.get("steam_ml_home_flag") or False)

            home_team = str(rec.get("home_team") or "")
            picked_home = bool(home_team) and sel.strip().lower() == home_team.strip().lower()
            if move_delta is not None:
                if picked_home:
                    align = float(move_delta) < 0
                else:
                    align = float(move_delta) > 0

                base = min(2.0, abs(float(move_delta)) / 30.0)
                if align is True:
                    adj += base
                elif align is False:
                    adj -= base
                if steam and align is True:
                    adj += 2.0
                elif steam and align is False:
                    adj -= 2.0

        if move_delta is None or move_market is None or align is None:
            continue

        rec["move_market"] = move_market
        rec["move_open"] = move_open
        rec["move_close"] = move_close
        rec["move_delta"] = move_delta
        rec["steam_flag"] = bool(steam)
        rec["sentiment_align"] = bool(align)
        rec["sentiment_adj"] = float(adj)

        try:
            rec["score"] = float(max(0.0, min(100.0, float(rec.get("score") or 0.0) + float(adj))))
        except Exception:
            pass

        reasons = rec.get("reasons")
        if not isinstance(reasons, list):
            reasons = []
        if move_open is not None and move_close is not None:
            reasons.append(f"move_{move_market}={move_open:+.1f}->{move_close:+.1f}")
        else:
            reasons.append(f"move_{move_market}_delta={float(move_delta):+.2f}")
        if steam:
            reasons.append(f"steam_{move_market}=1")
            try:
                if code == "OU":
                    reasons.append(f"steam_badge=STEAM TOTAL {'OVER' if float(move_delta) > 0 else 'UNDER'}")
                elif code == "ATS":
                    home_team = str(rec.get("home_team") or "").strip()
                    away_team = str(rec.get("away_team") or "").strip()
                    if float(move_delta) < 0:
                        reasons.append(f"steam_badge=STEAM {home_team or 'HOME'} ATS")
                    else:
                        reasons.append(f"steam_badge=STEAM {away_team or 'AWAY'} ATS")
                elif code == "ML":
                    home_team = str(rec.get("home_team") or "").strip()
                    away_team = str(rec.get("away_team") or "").strip()
                    if float(move_delta) < 0:
                        reasons.append(f"steam_badge=STEAM {home_team or 'HOME'} ML")
                    else:
                        reasons.append(f"steam_badge=STEAM {away_team or 'AWAY'} ML")
            except Exception:
                pass
        reasons.append("sent=align" if align else "sent=fade")
        reasons.append(f"sent_adj={float(adj):+.1f}")
        rec["reasons"] = reasons

        # Optional intraday live snapshot movement (only totals/spreads available).
        # Lazily load with a per-game cutoff (scheduled start_time) so we only use pregame movement.
        if not live_mv_by_gid:
            try:
                cutoff_by_gid: dict[str, str] = {}
                for rr in recs:
                    g2 = _norm_gid(rr.get("game_id"))
                    if not g2:
                        continue
                    st = rr.get("start_time")
                    if st is None or not str(st).strip():
                        continue
                    cutoff_by_gid[g2] = str(st)
                live_mv_by_gid = _load_live_snapshot_moves(out_dir, date, cutoff_ts_by_event_id=cutoff_by_gid)
            except Exception:
                live_mv_by_gid = {}

        lmv = live_mv_by_gid.get(gid) if live_mv_by_gid else None
        if isinstance(lmv, dict):
            live_adj = 0.0
            live_reasons: list[str] = []
            if code == "OU":
                dlt = _to_float(lmv.get("delta_total"))
                if dlt is not None and abs(float(dlt)) >= 1.5:
                    live_align = (float(dlt) > 0) if sel.lower().startswith("over") else ((float(dlt) < 0) if sel.lower().startswith("under") else None)
                    if live_align is not None:
                        live_adj = 2.0 if live_align else -2.0
                        live_reasons.append(f"live_move_total={float(dlt):+.1f}")
                        live_reasons.append("live_steam_total=1")
                        live_reasons.append("live_sent=align" if live_align else "live_sent=fade")
            elif code == "ATS":
                dls = _to_float(lmv.get("delta_spread_home"))
                if dls is not None and abs(float(dls)) >= 2.0:
                    live_align = (float(dls) < 0) if sel.lower().startswith("home") else ((float(dls) > 0) if sel.lower().startswith("away") else None)
                    if live_align is not None:
                        live_adj = 2.0 if live_align else -2.0
                        live_reasons.append(f"live_move_spread={float(dls):+.1f}")
                        live_reasons.append("live_steam_spread=1")
                        live_reasons.append("live_sent=align" if live_align else "live_sent=fade")

            if live_reasons:
                try:
                    rec["score"] = float(max(0.0, min(100.0, float(rec.get("score") or 0.0) + float(live_adj))))
                except Exception:
                    pass
                try:
                    rec["sentiment_live_adj"] = float(live_adj)
                    rec["sentiment_live_ts"] = lmv.get("ts_last")
                except Exception:
                    pass
                try:
                    rlist = rec.get("reasons")
                    if not isinstance(rlist, list):
                        rlist = []
                    rlist.extend(live_reasons)
                    rlist.append(f"live_adj={float(live_adj):+.1f}")
                    try:
                        if lmv.get("book"):
                            rlist.append(f"live_book={str(lmv.get('book'))}")
                    except Exception:
                        pass
                    rec["reasons"] = rlist
                except Exception:
                    pass

    # Label and filter
    for r in recs:
        s = float(r.get("score") or 0.0)
        if s >= 80:
            r["confidence_label"] = "high"
        elif s >= 70:
            r["confidence_label"] = "strong"
        elif s >= 60:
            r["confidence_label"] = "lean"
        else:
            r["confidence_label"] = "pass"

    recs = sorted(recs, key=lambda x: float(x.get("score") or 0.0), reverse=True)
    filtered = [r for r in recs if float(r.get("score") or 0.0) >= float(cfg.min_score)]

    # Per-game limiting with market preference (avoids correlated plays and prevents ATS/OU from crowding out ML).
    max_per_game = int(cfg.max_picks_per_game) if cfg.max_picks_per_game is not None else 0
    if max_per_game <= 0:
        max_per_game = 1
    pref = [str(x).upper() for x in (cfg.market_preference or ("ML", "ATS", "OU"))]
    pref_rank = {code: i for i, code in enumerate(pref)}

    by_gid: dict[str, list[dict[str, Any]]] = {}
    for r in filtered:
        gid = _norm_gid(r.get("game_id"))
        if not gid:
            continue
        by_gid.setdefault(gid, []).append(r)

    limited: list[dict[str, Any]] = []
    for gid, rows in by_gid.items():
        def _key(x: dict[str, Any]):
            code = str(x.get("rec_code") or "").upper()
            return (pref_rank.get(code, 999), -float(x.get("score") or 0.0))

        rows_sorted = sorted(rows, key=_key)
        limited.extend(rows_sorted[:max_per_game])

    # Preserve global ranking after limiting, but bias selection toward preferred markets.
    # Default preference is ATS/OU-first so the list isn't ML-dominated.
    def _global_key(x: dict[str, Any]):
        code = str(x.get("rec_code") or "").upper()
        return (pref_rank.get(code, 999), -float(x.get("score") or 0.0))

    filtered = sorted(limited, key=_global_key)

    # Optional per-market caps (per date).
    top_n = int(cfg.top_n)
    caps = {
        "ATS": int(cfg.max_ats_picks) if cfg.max_ats_picks is not None else 0,
        "OU": int(cfg.max_ou_picks) if cfg.max_ou_picks is not None else 0,
        "ML": int(cfg.max_ml_picks) if cfg.max_ml_picks is not None else top_n,
    }
    # Treat non-positive caps as "no picks" for that market.
    for k in ("ATS", "OU", "ML"):
        try:
            if int(caps.get(k) or 0) <= 0:
                caps[k] = 0
        except Exception:
            caps[k] = 0

    counts = {"ML": 0, "ATS": 0, "OU": 0}
    top: list[dict[str, Any]] = []
    for r in filtered:
        if len(top) >= top_n:
            break

        code = str(r.get("rec_code") or "").upper()
        if code not in counts:
            code = "ML" if code == "H2H" else code

        if code in caps and counts.get(code, 0) >= caps[code]:
            continue

        top.append(r)
        if code in counts:
            counts[code] += 1

    return {
        "status": "ok",
        "date": date,
        "generated_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "params": {
            "top_n": int(cfg.top_n),
            "min_score": float(cfg.min_score),
            "sigma_total": float(cfg.sigma_total),
            "sigma_margin": float(cfg.sigma_margin),
            "max_juice_abs": float(cfg.max_juice_abs),
            "include_markets": list(inc),
            "max_picks_per_game": int(cfg.max_picks_per_game),
            "market_preference": list(cfg.market_preference),
            "ml_favorites_only": bool(cfg.ml_favorites_only),
            "min_ml_implied_prob": float(cfg.min_ml_implied_prob),
            "min_ml_model_prob": float(cfg.min_ml_model_prob),
            "max_ml_underdog_price": float(cfg.max_ml_underdog_price),
            "min_ats_model_prob": float(cfg.min_ats_model_prob),
            "min_ou_model_prob": float(cfg.min_ou_model_prob),
            "min_ats_edge_pts": float(cfg.min_ats_edge_pts),
            "min_ou_edge_pts": float(cfg.min_ou_edge_pts),
            "min_prob_edge_vs_implied": float(cfg.min_prob_edge_vs_implied),
            "min_ats_ev_units": float(cfg.min_ats_ev_units),
            "min_ou_ev_units": float(cfg.min_ou_ev_units),
            "max_ats_picks": int(cfg.max_ats_picks),
            "max_ou_picks": int(cfg.max_ou_picks),
            "max_ml_picks": int(cfg.max_ml_picks),
        },
        "picks": top,
        "candidates": len(recs),
        "eligible": len(filtered),
    }


def reconcile_picks(out_dir: Path, date: str, picks: Iterable[dict[str, Any]]) -> dict[str, Any]:
    date = _parse_date(date) or date
    res = _load_results(Path(out_dir), date)
    if res.empty or "game_id" not in res.columns:
        return {
            "status": "error",
            "date": date,
            "message": "missing results file",
            "rows": 0,
        }

    res = res.copy()
    # Actual outcomes (prefer explicit columns if present; fall back to scores).
    hs = pd.to_numeric(res.get("home_score"), errors="coerce")
    aw = pd.to_numeric(res.get("away_score"), errors="coerce")
    res["_actual_margin"] = pd.to_numeric(res.get("actual_margin"), errors="coerce")
    res["_actual_total"] = pd.to_numeric(res.get("actual_total"), errors="coerce")

    missing_margin = res["_actual_margin"].isna() & hs.notna() & aw.notna()
    if missing_margin.any():
        res.loc[missing_margin, "_actual_margin"] = hs.loc[missing_margin] - aw.loc[missing_margin]

    missing_total = res["_actual_total"].isna() & hs.notna() & aw.notna()
    if missing_total.any():
        res.loc[missing_total, "_actual_total"] = hs.loc[missing_total] + aw.loc[missing_total]

    by_gid = {str(r.get("game_id")): r for r in res.to_dict(orient="records")}

    rows_out: list[dict[str, Any]] = []
    wins = losses = pushes = 0
    units = 0.0

    for p in picks:
        gid = _norm_gid(p.get("game_id"))
        rr = by_gid.get(gid)
        out = dict(p)
        out["result"] = None
        out["units"] = None

        if rr is None:
            rows_out.append(out)
            continue

        code = str(p.get("rec_code") or "").upper()
        price = _to_float(p.get("price"))

        if code == "OU":
            pick = str(p.get("selection") or "").strip().title()
            line = _to_float(p.get("line"))
            at = _to_float(rr.get("_actual_total"))

            if pick in ("Over", "Under") and line is not None and at is not None:
                diff = at - float(line)
                if abs(diff) < 1e-9:
                    pushes += 1
                    out["result"] = "P"
                    out["units"] = 0.0
                else:
                    outcome = "Over" if diff > 0 else "Under"
                    if outcome == pick:
                        wins += 1
                        prof = _american_profit_per_1_risk(price)
                        units += prof
                        out["result"] = "W"
                        out["units"] = float(prof)
                    else:
                        losses += 1
                        units -= 1.0
                        out["result"] = "L"
                        out["units"] = -1.0
            else:
                pushes += 1
                out["result"] = "P"
                out["units"] = 0.0

        elif code == "ATS":
            pick_side = str(p.get("selection") or "").strip().title()
            line = _to_float(p.get("line"))
            am = _to_float(rr.get("_actual_margin"))

            if pick_side in ("Home", "Away") and line is not None and am is not None:
                # line is assumed to be the home spread (e.g., -3.5 means home -3.5)
                diff = float(am) + float(line)
                if abs(diff) < 1e-9:
                    pushes += 1
                    out["result"] = "P"
                    out["units"] = 0.0
                else:
                    outcome = "Home" if diff > 0 else "Away"
                    if outcome == pick_side:
                        wins += 1
                        prof = _american_profit_per_1_risk(price)
                        units += prof
                        out["result"] = "W"
                        out["units"] = float(prof)
                    else:
                        losses += 1
                        units -= 1.0
                        out["result"] = "L"
                        out["units"] = -1.0
            else:
                pushes += 1
                out["result"] = "P"
                out["units"] = 0.0

        elif code == "ML":
            # Winner derived from actual margin sign
            am = _to_float(rr.get("_actual_margin"))
            home = rr.get("home_team")
            away = rr.get("away_team")
            if am is None or home is None or away is None:
                pushes += 1
                out["result"] = "P"
                out["units"] = 0.0
            else:
                winner = home if am > 0 else (away if am < 0 else None)
                pick = str(p.get("selection") or "")
                if winner is None:
                    pushes += 1
                    out["result"] = "P"
                    out["units"] = 0.0
                elif pick == winner:
                    wins += 1
                    prof = _american_profit_per_1_risk(price)
                    units += prof
                    out["result"] = "W"
                    out["units"] = float(prof)
                else:
                    losses += 1
                    units -= 1.0
                    out["result"] = "L"
                    out["units"] = -1.0

        rows_out.append(out)

    graded = wins + losses
    win_rate = float(wins / graded) if graded > 0 else None

    return {
        "status": "ok",
        "date": date,
        "rows": len(rows_out),
        "wins": int(wins),
        "losses": int(losses),
        "pushes": int(pushes),
        "win_rate": win_rate,
        "units": float(units),
        "picks": rows_out,
    }


def recent_results_dates(out_dir: Path, recent: int) -> list[str]:
    dr = Path(out_dir) / "daily_results"
    if not dr.exists():
        return []
    dates: list[str] = []
    for p in sorted(dr.glob("results_*.csv")):
        token = p.stem.replace("results_", "")
        d = _parse_date(token)
        if d:
            dates.append(d)
    dates = sorted(set(dates))
    return dates[-int(recent) :] if recent and recent > 0 else dates
