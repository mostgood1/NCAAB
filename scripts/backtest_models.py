import argparse
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _daterange(start: datetime, end: datetime) -> List[str]:
    d = start
    out = []
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _find_file_for_date(folder: Path, prefix: str, date: str, suffix: str = ".csv") -> Optional[Path]:
    candidates = list(folder.glob(f"{prefix}{date}*{suffix}"))
    if not candidates:
        return None
    # Prefer calibrated if multiple
    calibrated = [p for p in candidates if "calibrated" in p.name]
    return calibrated[0] if calibrated else candidates[0]


def _safe_merge(pred: pd.DataFrame, res: pd.DataFrame) -> Optional[pd.DataFrame]:
    # Try common keys in order of likelihood
    keys = [
        ["game_id"],
        ["id"],
        ["event_id"],
        ["date", "home_team", "away_team"],
        ["date", "home", "away"],
    ]
    for k in keys:
        if all(c in pred.columns for c in k) and all(c in res.columns for c in k):
            return pred.merge(res[k + [c for c in res.columns if c not in k]], on=k, how="inner")
    # No reliable key overlap; avoid cross joins which explode row counts
    return None


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = y_true - y_pred
    return np.mean(np.maximum(q * e, (q - 1) * e))


def _compute_crps_from_quantiles(y: np.ndarray, quant_cols: Dict[float, np.ndarray]) -> Optional[float]:
    if not quant_cols:
        return None
    qs = sorted(quant_cols.keys())
    losses = []
    for q in qs:
        losses.append(_pinball_loss(y, quant_cols[q], q))
    # Approximate CRPS by integrating pinball loss across quantiles
    return float(np.trapz(losses, x=qs))


def _brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def _log_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _detect_numeric(colnames: List[str], patterns: List[str]) -> Optional[str]:
    rgxs = [re.compile(p, re.I) for p in patterns]
    for c in colnames:
        for r in rgxs:
            if r.search(c):
                return c
    return None


def evaluate_day(pred_path: Path, results_path: Optional[Path]) -> Tuple[Dict[str, float], int]:
    pred = pd.read_csv(pred_path)
    res = pd.read_csv(results_path) if results_path and results_path.exists() else None

    df = _safe_merge(pred, res) if res is not None else None
    n = len(pred)
    metrics: Dict[str, float] = {}

    # Actuals: try to detect final margin and/or total
    cols_src = df if df is not None else pred
    actual_margin_col = _detect_numeric(cols_src.columns.tolist(), [r"final_margin", r"actual_margin", r"margin_final", r"score_diff", r"home_score.*-.*away_score"])  # noqa: E501
    actual_total_col = _detect_numeric(cols_src.columns.tolist(), [r"final_total", r"actual_total", r"total_points", r"home_score", r"away_score"])  # noqa: E501

    # If separate scores exist, compute margin/total
    if df is not None and actual_margin_col is None and {"home_score", "away_score"}.issubset(df.columns):
        df["actual_margin"] = df["home_score"] - df["away_score"]
        actual_margin_col = "actual_margin"
    if df is not None and actual_total_col is None and {"home_score", "away_score"}.issubset(df.columns):
        df["actual_total"] = df["home_score"] + df["away_score"]
        actual_total_col = "actual_total"

    # Probability metrics (binary outcomes)
    # Try to detect home win indicator and home win probability
    outcome_col = _detect_numeric(cols_src.columns.tolist(), [r"home_win", r"is_home_win", r"win_binary"])  # 0/1
    prob_col = _detect_numeric(cols_src.columns.tolist(), [r"p_home", r"prob_home", r"win_prob_home", r"home_win_prob"])  # 0..1
    if df is not None and outcome_col and prob_col:
        y = df[outcome_col].astype(float).values
        p = df[prob_col].astype(float).values
        metrics["brier_home_win"] = _brier_score(y, p)
        metrics["logloss_home_win"] = _log_loss(y, p)

    # Quantile-based CRPS for margin
    qcols_margin: Dict[float, np.ndarray] = {}
    for q in [0.1, 0.5, 0.9]:
        # match q10, q50, q90 variants for margin
        cname = _detect_numeric(
            cols_src.columns.tolist(),
            [fr"q{int(q*100)}[_-]?(margin|spread)", fr"margin[_-]?q{int(q*100)}", fr"pred_margin_q{int(q*100)}"],
        )
        if cname is not None:
            qcols_margin[q] = cols_src[cname].astype(float).values

    if df is not None and actual_margin_col and qcols_margin:
        y = df[actual_margin_col].astype(float).values
        crps_m = _compute_crps_from_quantiles(y, qcols_margin)
        if crps_m is not None:
            metrics["crps_margin"] = crps_m

    # Interval coverage if we have lo/hi for margin
    lo_col = _detect_numeric(cols_src.columns.tolist(), [r"(margin|spread).*_?lo", r"lo_(margin|spread)", r"margin_lo"])
    hi_col = _detect_numeric(cols_src.columns.tolist(), [r"(margin|spread).*_?hi", r"hi_(margin|spread)", r"margin_hi"])
    if df is not None and actual_margin_col and lo_col and hi_col:
        y = df[actual_margin_col].astype(float).values
        lo = df[lo_col].astype(float).values
        hi = df[hi_col].astype(float).values
        covered = ((y >= lo) & (y <= hi)).mean()
        width = np.mean(hi - lo)
        metrics["coverage_margin"] = float(covered)
        metrics["width_margin"] = float(width)

    return metrics, n


def main():
    parser = argparse.ArgumentParser(description="Backtest models over a date range and summarize metrics.")
    parser.add_argument("--start-date", type=str, help="Start date YYYY-MM-DD", required=False)
    parser.add_argument("--end-date", type=str, help="End date YYYY-MM-DD", required=False)
    parser.add_argument("--days", type=int, help="If provided, use this many trailing days ending today.")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Directory containing prediction CSVs.")
    parser.add_argument("--results-dir", type=str, default="daily_results", help="Directory with realized results CSVs.")
    parser.add_argument(
        "--pred-prefix",
        type=str,
        default="align_period_",
        help="Prediction file prefix before the date (default: align_period_)",
    )
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="results_",
        help="Results file prefix before the date (default: results_)",
    )
    parser.add_argument("--save-dir", type=str, default="outputs", help="Where to write backtest summaries.")
    parser.add_argument("--name", type=str, default="default", help="Name tag for output filenames.")
    args = parser.parse_args()

    if args.days is not None and (args.start_date or args.end_date):
        raise SystemExit("Provide either --days or --start-date/--end-date, not both.")

    today = datetime.now().date()
    if args.days is not None:
        end = today
        start = today - timedelta(days=max(args.days - 1, 0))
    else:
        if not args.start_date or not args.end_date:
            raise SystemExit("Either provide --days or both --start-date and --end-date.")
        start = _parse_date(args.start_date).date()
        end = _parse_date(args.end_date).date()

    dates = _daterange(datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.min.time()))

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = Path(args.outputs_dir)
    results_dir = Path(args.results_dir)

    rows = []
    total_events = 0
    for d in dates:
        pred_path = _find_file_for_date(outputs_dir, args.pred_prefix, d)
        res_path = _find_file_for_date(results_dir, args.results_prefix, d)
        if pred_path is None:
            continue
        metrics, n = evaluate_day(pred_path, res_path)
        total_events += n
        row = {"date": d, **metrics, "n": n}
        rows.append(row)

    if not rows:
        raise SystemExit("No backtestable days found in the specified range.")

    daily = pd.DataFrame(rows).sort_values("date")
    summary = {k: (daily[k].dropna().mean() if k != "n" else int(daily[k].sum())) for k in daily.columns if k != "date"}
    summary["start_date"] = dates[0]
    summary["end_date"] = dates[-1]
    summary["events_total"] = total_events

    # Save artifacts
    daily_path = out_dir / f"backtest_daily_{args.name}_{dates[0]}_{dates[-1]}.csv"
    summary_path = out_dir / f"backtest_summary_{args.name}_{dates[0]}_{dates[-1]}.csv"
    daily.to_csv(daily_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(f"[ok] Backtest saved: {daily_path}")
    print(f"[ok] Summary saved: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
