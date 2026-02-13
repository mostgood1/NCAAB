from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ncaab_model.live_lens_accuracy import results_path, signals_path
from ncaab_model.live_lens_buckets import LiveLensBucketReportConfig, iter_date_range
from ncaab_model.live_lens_early_retune_search import EarlyOverSearchConfig, search_early_over_retune
from ncaab_model.live_lens_retune_search import LateOverSearchConfig, search_late_over_retune


def _has_bytes(p: Path) -> bool:
    try:
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


def _today_local() -> dt.date:
    # For script usage, local timezone is good enough.
    return dt.date.today()


def _safe_date(s: str) -> str:
    s2 = str(s or "").strip()
    dt.date.fromisoformat(s2)
    return s2


def _load_json(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _pick_best_early(df: pd.DataFrame, min_bucket_n: int, min_overall_n: int) -> dict[str, Any] | None:
    if df is None or df.empty:
        return None

    d = df.copy()
    for c in ["n_gt20", "overall_n"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    m = pd.Series([True] * len(d), index=d.index)
    if "n_gt20" in d.columns:
        m &= d["n_gt20"].fillna(0) >= int(min_bucket_n)
    if "overall_n" in d.columns:
        m &= d["overall_n"].fillna(0) >= int(min_overall_n)
    d = d[m].copy()
    if d.empty:
        return None

    # search_early_over_retune already sorts, but keep it explicit.
    d["roi_gt20_sort"] = pd.to_numeric(d.get("roi_gt20"), errors="coerce")
    d["overall_roi_sort"] = pd.to_numeric(d.get("overall_roi"), errors="coerce")
    d = d.sort_values(["roi_gt20_sort", "overall_roi_sort", "n_gt20"], ascending=[False, False, False], kind="stable")

    rec = d.iloc[0].to_dict()
    return {
        "penalty": float(rec.get("penalty")) if rec.get("penalty") is not None else None,
        "roi_gt20": rec.get("roi_gt20"),
        "n_gt20": int(rec.get("n_gt20") or 0),
        "overall_roi": rec.get("overall_roi"),
        "overall_n": int(rec.get("overall_n") or 0),
        "remaining_min": rec.get("remaining_min"),
        "period_max": rec.get("period_max"),
    }


def _pick_best_late(df: pd.DataFrame, min_bucket_n: int, min_overall_n: int) -> dict[str, Any] | None:
    if df is None or df.empty:
        return None

    d = df.copy()
    for c in ["n_5_10", "overall_n"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    m = pd.Series([True] * len(d), index=d.index)
    if "n_5_10" in d.columns:
        m &= d["n_5_10"].fillna(0) >= int(min_bucket_n)
    if "overall_n" in d.columns:
        m &= d["overall_n"].fillna(0) >= int(min_overall_n)
    d = d[m].copy()
    if d.empty:
        return None

    d["roi_5_10_sort"] = pd.to_numeric(d.get("roi_5_10"), errors="coerce")
    d["overall_roi_sort"] = pd.to_numeric(d.get("overall_roi"), errors="coerce")
    d = d.sort_values(["roi_5_10_sort", "overall_roi_sort", "n_5_10"], ascending=[False, False, False], kind="stable")

    rec = d.iloc[0].to_dict()
    return {
        "penalty": float(rec.get("penalty")) if rec.get("penalty") is not None else None,
        "margin_abs_min": float(rec.get("margin_abs_min") or 0.0),
        "roi_5_10": rec.get("roi_5_10"),
        "n_5_10": int(rec.get("n_5_10") or 0),
        "overall_roi": rec.get("overall_roi"),
        "overall_n": int(rec.get("overall_n") or 0),
        "period_min": rec.get("period_min"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep Live Lens early/late OVER penalties using logged signals + finalized results.")
    ap.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: yesterday local).")
    ap.add_argument("--start", default=None, help="Start date YYYY-MM-DD (overrides --days).")
    ap.add_argument("--days", type=int, default=21, help="Lookback days ending at --end (inclusive).")
    ap.add_argument("--assume-price", type=float, default=-110.0, help="Assumed bet price for ROI units.")
    ap.add_argument("--out-dir", type=str, default="outputs", help="Outputs dir (default: outputs).")
    ap.add_argument("--daily-results-dir", type=str, default=None, help="Override daily_results dir.")
    ap.add_argument("--tuning-json", type=str, default="outputs/live_lens_tuning.json", help="Tuning JSON to read/update.")
    ap.add_argument("--min-bucket-n", type=int, default=10, help="Min sample size in target bucket to accept best config.")
    ap.add_argument("--min-overall-n", type=int, default=25, help="Min overall sample size to accept best config.")
    ap.add_argument("--apply", action="store_true", help="Apply best penalties into tuning JSON.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    daily_results_dir = Path(args.daily_results_dir) if args.daily_results_dir else None
    tuning_path = Path(args.tuning_json)

    if args.end:
        end_date = _safe_date(args.end)
        end_dt = dt.date.fromisoformat(end_date)
    else:
        end_dt = _today_local() - dt.timedelta(days=1)
        end_date = end_dt.isoformat()

    if args.start:
        start_date = _safe_date(args.start)
        start_dt = dt.date.fromisoformat(start_date)
    else:
        days = int(args.days)
        if days < 1:
            days = 1
        start_dt = end_dt - dt.timedelta(days=days - 1)
        start_date = start_dt.isoformat()

    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt
        start_date, end_date = start_date, end_date

    # Load current Live Lens tuning (if present) so we can preserve defaults.
    tuning_obj = _load_json(tuning_path) or {}
    tuning = dict(tuning_obj.get("tuning") or {})

    base_late = {
        "late_over_remaining_lo": float(tuning.get("late_over_remaining_lo") or 5.0),
        "late_over_remaining_hi": float(tuning.get("late_over_remaining_hi") or 10.0),
        "late_over_margin_abs_min": float(tuning.get("late_over_margin_abs_min") or 0.0),
        "late_over_period_min": float(tuning.get("late_over_period_min") or 2.0),
    }
    base_early = {
        "early_over_remaining_min": float(tuning.get("early_over_remaining_min") or 20.0),
        "early_over_period_max": float(tuning.get("early_over_period_max") or 1.0),
    }

    # Pre-filter to dates where both signals and results exist locally.
    candidate_dates = iter_date_range(start_date, end_date)
    used_dates: list[str] = []
    skipped: list[dict[str, Any]] = []

    for d in candidate_dates:
        sig_p = signals_path(d, out_dir=out_dir)
        res_p = results_path(d, out_dir=out_dir, daily_results_dir=daily_results_dir)
        if not _has_bytes(sig_p):
            skipped.append({"date": d, "status": "missing_signals", "signals_path": str(sig_p), "results_path": str(res_p)})
            continue
        if not _has_bytes(res_p):
            skipped.append({"date": d, "status": "missing_results", "signals_path": str(sig_p), "results_path": str(res_p)})
            continue
        used_dates.append(d)

    summary: dict[str, Any] = {
        "status": "ok",
        "start": start_date,
        "end": end_date,
        "days": (end_dt - start_dt).days + 1,
        "candidate_dates": len(candidate_dates),
        "used_dates": used_dates,
        "used_n": len(used_dates),
        "skipped": skipped,
        "assume_price": float(args.assume_price),
        "min_bucket_n": int(args.min_bucket_n),
        "min_overall_n": int(args.min_overall_n),
        "base_late": base_late,
        "base_early": base_early,
    }

    if not used_dates:
        out_json = out_dir / f"live_lens_over_tuning_{start_date}_to_{end_date}.summary.json"
        _write_json(out_json, {**summary, "status": "missing", "message": "No local (signals + results) date pairs found."})
        print({"status": "missing", "summary_json": str(out_json)})
        return 2

    base_cfg = LiveLensBucketReportConfig(
        dates=list(used_dates),
        out_dir=out_dir,
        daily_results_dir=daily_results_dir,
        assume_price=float(args.assume_price),
        include_watch=False,
        full_game_only=True,
        apply_retune=True,
        late_over_strength_penalty=float(tuning.get("late_over_strength_penalty") or 0.0),
        late_over_remaining_lo=float(base_late["late_over_remaining_lo"]),
        late_over_remaining_hi=float(base_late["late_over_remaining_hi"]),
        late_over_margin_abs_min=float(base_late["late_over_margin_abs_min"]),
        late_over_period_min=float(base_late["late_over_period_min"]),
        early_over_strength_penalty=float(tuning.get("early_over_strength_penalty") or 0.0),
        early_over_remaining_min=float(base_early["early_over_remaining_min"]),
        early_over_period_max=float(base_early["early_over_period_max"]),
    )

    # Run searches
    early_payload = search_early_over_retune(
        base_cfg=base_cfg,
        search_cfg=EarlyOverSearchConfig(
            dates=list(used_dates),
            assume_price=float(args.assume_price),
            remaining_min=float(base_early["early_over_remaining_min"]),
            period_max=float(base_early["early_over_period_max"]),
        ),
    )

    late_payload = search_late_over_retune(
        base_cfg=base_cfg,
        search_cfg=LateOverSearchConfig(
            dates=list(used_dates),
            assume_price=float(args.assume_price),
            remaining_lo=float(base_late["late_over_remaining_lo"]),
            remaining_hi=float(base_late["late_over_remaining_hi"]),
            period_min=float(base_late["late_over_period_min"]),
        ),
    )

    early_df: pd.DataFrame = early_payload.get("table") if isinstance(early_payload.get("table"), pd.DataFrame) else pd.DataFrame()
    late_df: pd.DataFrame = late_payload.get("table") if isinstance(late_payload.get("table"), pd.DataFrame) else pd.DataFrame()

    out_early = out_dir / f"live_lens_over_tuning_{start_date}_to_{end_date}.early.csv"
    out_late = out_dir / f"live_lens_over_tuning_{start_date}_to_{end_date}.late.csv"
    out_json = out_dir / f"live_lens_over_tuning_{start_date}_to_{end_date}.summary.json"

    try:
        out_early.parent.mkdir(parents=True, exist_ok=True)
        early_df.to_csv(out_early, index=False)
    except Exception:
        pass

    try:
        out_late.parent.mkdir(parents=True, exist_ok=True)
        late_df.to_csv(out_late, index=False)
    except Exception:
        pass

    best_early = _pick_best_early(early_df, min_bucket_n=int(args.min_bucket_n), min_overall_n=int(args.min_overall_n))
    best_late = _pick_best_late(late_df, min_bucket_n=int(args.min_bucket_n), min_overall_n=int(args.min_overall_n))

    summary["artifacts"] = {"early_csv": str(out_early), "late_csv": str(out_late), "summary_json": str(out_json)}
    summary["best_early"] = best_early
    summary["best_late"] = best_late

    applied: dict[str, Any] | None = None
    if bool(args.apply):
        applied = {"status": "skipped", "reason": "no_best_config"}
        if best_early or best_late:
            now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
            tuning_obj2 = tuning_obj if isinstance(tuning_obj, dict) else {}
            if "tuning" not in tuning_obj2 or not isinstance(tuning_obj2.get("tuning"), dict):
                tuning_obj2["tuning"] = {}
            t2 = tuning_obj2["tuning"]

            if best_early and best_early.get("penalty") is not None:
                t2["early_over_strength_penalty"] = float(best_early["penalty"])
                t2["early_over_remaining_min"] = float(base_early["early_over_remaining_min"])
                t2["early_over_period_max"] = float(base_early["early_over_period_max"])

            if best_late and best_late.get("penalty") is not None:
                t2["late_over_strength_penalty"] = float(best_late["penalty"])
                t2["late_over_remaining_lo"] = float(base_late["late_over_remaining_lo"])
                t2["late_over_remaining_hi"] = float(base_late["late_over_remaining_hi"])
                t2["late_over_period_min"] = float(base_late["late_over_period_min"])
                # Keep margin_abs_min tuned if we have it.
                if best_late.get("margin_abs_min") is not None:
                    t2["late_over_margin_abs_min"] = float(best_late["margin_abs_min"])

            tuning_obj2["generated_at"] = now
            meta = tuning_obj2.get("meta") if isinstance(tuning_obj2.get("meta"), dict) else {}
            meta["over_tuning"] = {
                "generated_at": now,
                "start": start_date,
                "end": end_date,
                "used_n": len(used_dates),
                "min_bucket_n": int(args.min_bucket_n),
                "min_overall_n": int(args.min_overall_n),
                "best_early": best_early,
                "best_late": best_late,
            }
            tuning_obj2["meta"] = meta

            try:
                _write_json(tuning_path, tuning_obj2)
                applied = {
                    "status": "ok",
                    "tuning_json": str(tuning_path),
                    "early_over_strength_penalty": t2.get("early_over_strength_penalty"),
                    "late_over_strength_penalty": t2.get("late_over_strength_penalty"),
                    "late_over_margin_abs_min": t2.get("late_over_margin_abs_min"),
                }
            except Exception as e:
                applied = {"status": "error", "message": str(e), "tuning_json": str(tuning_path)}

    summary["applied"] = applied
    _write_json(out_json, summary)

    print(
        {
            "status": "ok",
            "start": start_date,
            "end": end_date,
            "used_n": len(used_dates),
            "early_csv": str(out_early),
            "late_csv": str(out_late),
            "summary_json": str(out_json),
            "best_early": best_early,
            "best_late": best_late,
            "applied": applied,
        }
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
