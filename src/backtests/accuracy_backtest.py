from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def _parse_date(s: str | None) -> Optional[str]:
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
        if s.endswith(".0"):
            s = s[:-2]
        return s
    except Exception:
        return str(v)


def _resolve_dates_from_results(out_dir: Path, start: Optional[str], end: Optional[str], recent: Optional[int]) -> list[str]:
    dr = out_dir / "daily_results"
    if not dr.exists():
        return []

    dates: list[str] = []
    for p in sorted(dr.glob("results_*.csv")):
        token = p.stem.replace("results_", "")
        d = _parse_date(token)
        if d:
            dates.append(d)

    dates = sorted(set(dates))

    if recent and recent > 0:
        return dates[-int(recent) :]

    if start and end:
        if start > end:
            start, end = end, start
        return [d for d in dates if start <= d <= end]

    if start and not end:
        return [d for d in dates if d == start]

    return dates


def _coalesce_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    a = pd.to_numeric(df.get(col), errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
    b = pd.to_numeric(df.get(f"{col}_disp"), errors="coerce") if f"{col}_disp" in df.columns else pd.Series(np.nan, index=df.index)
    out = a.copy()
    m = out.isna() & b.notna()
    out[m] = b[m]
    return out


def _load_results(out_dir: Path, date: str) -> pd.DataFrame:
    p = out_dir / "daily_results" / f"results_{date}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"game_id": str})
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].map(_norm_gid)

    # Filter to real finals (avoid 0/0 placeholders)
    if "home_score" in df.columns and "away_score" in df.columns:
        hs = pd.to_numeric(df.get("home_score"), errors="coerce")
        as_ = pd.to_numeric(df.get("away_score"), errors="coerce")
        final_mask = hs.notna() & as_.notna() & ((hs > 0) | (as_ > 0))
        df = df.loc[final_mask].copy()
        df["actual_total"] = hs.loc[final_mask].to_numpy() + as_.loc[final_mask].to_numpy()
        df["actual_margin"] = hs.loc[final_mask].to_numpy() - as_.loc[final_mask].to_numpy()

    return df


def _load_display(out_dir: Path, date: str) -> pd.DataFrame:
    p = out_dir / f"predictions_display_{date}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"game_id": str})
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].map(_norm_gid)
    return df


@dataclasses.dataclass
class AccuracyBacktestConfig:
    out_dir: Path
    start: Optional[str] = None
    end: Optional[str] = None
    recent: Optional[int] = None
    out_prefix: str = "accuracy"


def run_accuracy_backtest(cfg: AccuracyBacktestConfig) -> dict[str, Any]:
    cfg.out_dir = Path(cfg.out_dir)
    start = _parse_date(cfg.start)
    end = _parse_date(cfg.end)

    dates = _resolve_dates_from_results(cfg.out_dir, start, end, cfg.recent)
    if not dates:
        return {"error": "No results_*.csv dates found", "out_dir": str(cfg.out_dir)}

    bt_dir = cfg.out_dir / "backtests"
    bt_dir.mkdir(parents=True, exist_ok=True)

    per_game_rows: list[pd.DataFrame] = []
    per_date_rows: list[dict[str, Any]] = []

    for d in dates:
        res = _load_results(cfg.out_dir, d)
        disp = _load_display(cfg.out_dir, d)

        if res.empty or disp.empty or "game_id" not in res.columns or "game_id" not in disp.columns:
            per_date_rows.append(
                {
                    "date": d,
                    "skipped": True,
                    "reason": f"missing files or game_id (results_empty={res.empty} display_empty={disp.empty})",
                }
            )
            continue

        # Keep a small subset from display to avoid inflating columns
        keep_disp = [c for c in [
            "game_id",
            "pred_total",
            "pred_margin",
            "closing_total",
            "closing_spread_home",
        ] if c in disp.columns]

        df = res.merge(disp[keep_disp], on="game_id", how="left", suffixes=("", "_disp"))
        df["date"] = d

        pred_margin = _coalesce_numeric(df, "pred_margin")
        pred_total = _coalesce_numeric(df, "pred_total")

        actual_margin = pd.to_numeric(df.get("actual_margin"), errors="coerce")
        actual_total = pd.to_numeric(df.get("actual_total"), errors="coerce")

        # Lines
        market_total = pd.to_numeric(df.get("market_total"), errors="coerce")
        closing_total = pd.to_numeric(df.get("closing_total"), errors="coerce")
        line_total = market_total.copy()
        line_total[line_total.isna()] = closing_total[line_total.isna()]

        spread_home = pd.to_numeric(df.get("spread_home"), errors="coerce")
        closing_spread_home = pd.to_numeric(df.get("closing_spread_home"), errors="coerce")
        line_spread = spread_home.copy()
        line_spread[line_spread.isna()] = closing_spread_home[line_spread.isna()]

        # Outcomes
        ou = df["ou_result_full"] if "ou_result_full" in df.columns else pd.Series(pd.NA, index=df.index)
        ats = df["ats_result"] if "ats_result" in df.columns else pd.Series(pd.NA, index=df.index)

        # Winners accuracy: exclude ties (actual_margin == 0)
        mw = pred_margin.notna() & actual_margin.notna() & (actual_margin != 0)
        win_correct = ((pred_margin[mw] > 0).astype(int) == (actual_margin[mw] > 0).astype(int))

        # Totals accuracy vs line: exclude Pushes and missing line
        mt = pred_total.notna() & line_total.notna() & ou.notna() & ou.isin(["Over", "Under"])
        tot_correct = ((pred_total[mt] > line_total[mt]).astype(int) == (ou[mt] == "Over").astype(int))

        # ATS accuracy vs spread: compare predicted margin to implied threshold; exclude pushes
        ma = pred_margin.notna() & line_spread.notna() & ats.notna() & ats.isin(["Home Cover", "Away Cover"])
        # Home covers if actual_margin + spread_home > 0; predict home cover if pred_margin + spread_home > 0
        ats_correct = ((pred_margin[ma] > -line_spread[ma]).astype(int) == (ats[ma] == "Home Cover").astype(int))

        # MAE metrics
        m_total = pred_total.notna() & actual_total.notna() & (actual_total > 0)
        m_margin = pred_margin.notna() & actual_margin.notna()
        mae_total = float(np.mean(np.abs((pred_total[m_total] - actual_total[m_total]).astype(float)))) if m_total.any() else None
        mae_margin = float(np.mean(np.abs((pred_margin[m_margin] - actual_margin[m_margin]).astype(float)))) if m_margin.any() else None

        per_date_rows.append(
            {
                "date": d,
                "skipped": False,
                "winners_n": int(mw.sum()),
                "winners_acc": float(win_correct.mean()) if mw.any() else None,
                "totals_n": int(mt.sum()),
                "totals_acc": float(tot_correct.mean()) if mt.any() else None,
                "ats_n": int(ma.sum()),
                "ats_acc": float(ats_correct.mean()) if ma.any() else None,
                "mae_total": mae_total,
                "mae_margin": mae_margin,
            }
        )

        # Per-game outputs (keep compact)
        df_out = pd.DataFrame(
            {
                "date": d,
                "game_id": df.get("game_id"),
                "home_team": df.get("home_team"),
                "away_team": df.get("away_team"),
                "pred_total": pred_total,
                "pred_margin": pred_margin,
                "actual_total": actual_total,
                "actual_margin": actual_margin,
                "line_total": line_total,
                "line_spread_home": line_spread,
                "ou_result_full": ou,
                "ats_result": ats,
            }
        )
        if len(df_out):
            df_out["winner_correct"] = np.nan
            df_out.loc[mw, "winner_correct"] = win_correct.astype(float).to_numpy()

            df_out["total_correct"] = np.nan
            df_out.loc[mt, "total_correct"] = tot_correct.astype(float).to_numpy()

            df_out["ats_correct"] = np.nan
            df_out.loc[ma, "ats_correct"] = ats_correct.astype(float).to_numpy()
        per_game_rows.append(df_out)

    per_date = pd.DataFrame(per_date_rows).sort_values("date")
    per_game = pd.concat(per_game_rows, ignore_index=True) if per_game_rows else pd.DataFrame()

    # Weighted aggregate accuracy
    def _agg_acc(n_col: str, acc_col: str) -> dict[str, Any]:
        ok = per_date["skipped"].astype(bool) == False
        n = pd.to_numeric(per_date.loc[ok, n_col], errors="coerce")
        acc = pd.to_numeric(per_date.loc[ok, acc_col], errors="coerce")
        mask = n.notna() & acc.notna() & (n > 0)
        if not mask.any():
            return {"n": 0, "acc": None}
        # reconstruct correct ~= acc*n
        correct = float(np.sum((n[mask] * acc[mask]).astype(float)))
        total = float(np.sum(n[mask].astype(float)))
        return {"n": int(total), "acc": float(correct / total) if total > 0 else None}

    summary: dict[str, Any] = {
        "range": {"start": dates[0], "end": dates[-1], "n_dates": int(len(dates))},
        "dates_skipped": int(per_date["skipped"].sum()) if "skipped" in per_date.columns else 0,
        "winners": _agg_acc("winners_n", "winners_acc"),
        "totals": _agg_acc("totals_n", "totals_acc"),
        "ats": _agg_acc("ats_n", "ats_acc"),
        "mae_total": float(pd.to_numeric(per_date.get("mae_total"), errors="coerce").mean()) if "mae_total" in per_date.columns else None,
        "mae_margin": float(pd.to_numeric(per_date.get("mae_margin"), errors="coerce").mean()) if "mae_margin" in per_date.columns else None,
    }

    out_stem = f"{cfg.out_prefix}_{dates[0]}_{dates[-1]}"
    out_game = bt_dir / f"{out_stem}.csv"
    out_per_date = bt_dir / f"{out_stem}_per_date.csv"
    out_summary = bt_dir / f"{out_stem}_summary.json"

    if not per_game.empty:
        per_game.to_csv(out_game, index=False, na_rep="")
    per_date.to_csv(out_per_date, index=False, na_rep="")
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "wrote": {
            "per_game": str(out_game),
            "per_date": str(out_per_date),
            "summary": str(out_summary),
        },
        "n_dates": int(len(dates)),
        "n_games": int(len(per_game)) if not per_game.empty else 0,
        "dates_skipped": int(summary.get("dates_skipped") or 0),
    }
