import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ncaab_model.live_lens_accuracy import LiveLensAccuracyRetunedConfig, compute_live_lens_accuracy_retuned


def _safe_date(s: str) -> str:
    s2 = str(s or "").strip()
    dt.date.fromisoformat(s2)
    return s2


def _iter_dates(start_date: str, end_date: str) -> list[str]:
    s = dt.date.fromisoformat(_safe_date(start_date))
    e = dt.date.fromisoformat(_safe_date(end_date))
    if s > e:
        s, e = e, s
    out: list[str] = []
    cur = s
    while cur <= e:
        out.append(cur.isoformat())
        cur = cur + dt.timedelta(days=1)
    return out


def _read_tuning(path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    raw = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    t = raw.get("tuning") if isinstance(raw, dict) and isinstance(raw.get("tuning"), dict) else raw
    if not isinstance(t, dict):
        t = {}

    pens: dict[str, float] = {}
    try:
        raw_p = t.get("driver_tag_strength_penalties")
        if isinstance(raw_p, dict):
            for k, v in raw_p.items():
                try:
                    kk = str(k).strip()
                    vv = float(v)
                    if not kk or vv <= 0:
                        continue
                    pens[kk] = float(vv)
                except Exception:
                    continue
    except Exception:
        pens = {}

    return t, pens


def _get_float(t: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(t.get(key, default))
    except Exception:
        return float(default)


def _eval_one(
    date: str,
    *,
    tuning: dict[str, Any],
    assume_price: float,
    full_game_only: bool,
    apply_tag_penalties: bool,
    tag_penalties: dict[str, float] | None,
) -> dict[str, Any]:
    cfg = LiveLensAccuracyRetunedConfig(
        date=str(date),
        assume_price=float(assume_price),
        full_game_only=bool(full_game_only),
        apply_retune=True,
        late_over_strength_penalty=_get_float(tuning, "late_over_strength_penalty", 0.0),
        late_over_remaining_lo=_get_float(tuning, "late_over_remaining_lo", 5.0),
        late_over_remaining_hi=_get_float(tuning, "late_over_remaining_hi", 10.0),
        late_over_margin_abs_min=_get_float(tuning, "late_over_margin_abs_min", 0.0),
        late_over_period_min=_get_float(tuning, "late_over_period_min", 2.0),
        early_over_strength_penalty=_get_float(tuning, "early_over_strength_penalty", 0.0),
        early_over_remaining_min=_get_float(tuning, "early_over_remaining_min", 20.0),
        early_over_period_max=_get_float(tuning, "early_over_period_max", 1.0),
        apply_driver_tag_penalties=bool(apply_tag_penalties),
        driver_tag_strength_penalties=(tag_penalties if (apply_tag_penalties and tag_penalties) else None),
    )

    payload = compute_live_lens_accuracy_retuned(cfg)

    if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
        s = payload["summary"]
        out = {
            "status": s.get("status"),
            "n_settled": int(s.get("n_settled") or 0),
            "roi_units_per_bet": float(s.get("roi_units_per_bet") or 0.0),
            "wins": int(s.get("wins") or 0),
            "losses": int(s.get("losses") or 0),
            "pushes": int(s.get("pushes") or 0),
        }
        # Derive units profit so we can aggregate exactly via sum.
        out["profit_units"] = out["roi_units_per_bet"] * out["n_settled"]
        return out

    # Missing/empty/error payload
    return {
        "status": str(payload.get("status") if isinstance(payload, dict) else "error"),
        "n_settled": 0,
        "roi_units_per_bet": None,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "profit_units": 0.0,
        "message": (payload.get("message") if isinstance(payload, dict) else None),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Live Lens counterfactual ROI over a window with vs without driver-tag penalties."
    )
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD inclusive. Defaults to end-date - days + 1.")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD inclusive. Defaults to yesterday local.")
    parser.add_argument("--days", type=int, default=14, help="Window size when start-date not provided (default 14).")
    parser.add_argument(
        "--tuning-json",
        type=Path,
        default=Path("outputs/live_lens_tuning.json"),
        help="Tuning JSON used by the web UI (source of retune + tag penalties).",
    )
    parser.add_argument("--price", type=float, default=-110.0, help="Assumed odds price for ROI (default -110).")
    parser.add_argument(
        "--full-game-only",
        action="store_true",
        help="Restrict to horizon>=39 signals (default true).",
    )
    parser.add_argument(
        "--all-lenses",
        action="store_true",
        help="Include all horizons/lenses (overrides --full-game-only).",
    )
    parser.add_argument("--out-json", type=Path, default=None, help="Output JSON path (default in outputs/).")
    parser.add_argument("--out-csv", type=Path, default=None, help="Output CSV path (default in outputs/).")

    args = parser.parse_args(argv)

    # Defaults: match CLI behavior (yesterday local).
    end_date = _safe_date(args.end_date) if args.end_date else (dt.date.today() - dt.timedelta(days=1)).isoformat()

    if args.start_date:
        start_date = _safe_date(args.start_date)
    else:
        d_end = dt.date.fromisoformat(end_date)
        start_date = (d_end - dt.timedelta(days=max(0, int(args.days)) - 1)).isoformat()

    full_game_only = True
    if args.all_lenses:
        full_game_only = False
    elif args.full_game_only:
        full_game_only = True

    tuning_json = Path(args.tuning_json)
    tuning, tag_pens = _read_tuning(tuning_json)

    dates = _iter_dates(start_date, end_date)

    rows: list[dict[str, Any]] = []
    for d in dates:
        base = _eval_one(
            d,
            tuning=tuning,
            assume_price=float(args.price),
            full_game_only=full_game_only,
            apply_tag_penalties=False,
            tag_penalties=None,
        )
        with_tags = _eval_one(
            d,
            tuning=tuning,
            assume_price=float(args.price),
            full_game_only=full_game_only,
            apply_tag_penalties=True,
            tag_penalties=tag_pens,
        )

        rec: dict[str, Any] = {
            "date": d,
            "status_base": base.get("status"),
            "n_base": int(base.get("n_settled") or 0),
            "roi_base": base.get("roi_units_per_bet"),
            "profit_base": float(base.get("profit_units") or 0.0),
            "status_tags": with_tags.get("status"),
            "n_tags": int(with_tags.get("n_settled") or 0),
            "roi_tags": with_tags.get("roi_units_per_bet"),
            "profit_tags": float(with_tags.get("profit_units") or 0.0),
            "delta_n": int(with_tags.get("n_settled") or 0) - int(base.get("n_settled") or 0),
            "delta_profit": float(with_tags.get("profit_units") or 0.0) - float(base.get("profit_units") or 0.0),
        }
        # ROI delta is noisy if either side missing.
        try:
            if rec.get("roi_base") is not None and rec.get("roi_tags") is not None:
                rec["delta_roi"] = float(rec["roi_tags"]) - float(rec["roi_base"])
            else:
                rec["delta_roi"] = None
        except Exception:
            rec["delta_roi"] = None

        if base.get("message"):
            rec["message_base"] = base.get("message")
        if with_tags.get("message"):
            rec["message_tags"] = with_tags.get("message")

        rows.append(rec)

    df = pd.DataFrame(rows)

    # Aggregate only dates where we have settled bets.
    base_profit = float(pd.to_numeric(df.get("profit_base"), errors="coerce").fillna(0.0).sum())
    tags_profit = float(pd.to_numeric(df.get("profit_tags"), errors="coerce").fillna(0.0).sum())
    base_n = int(pd.to_numeric(df.get("n_base"), errors="coerce").fillna(0).sum())
    tags_n = int(pd.to_numeric(df.get("n_tags"), errors="coerce").fillna(0).sum())

    overall = {
        "window": {"start_date": start_date, "end_date": end_date, "dates": dates},
        "tuning_json": str(tuning_json),
        "assume_price": float(args.price),
        "full_game_only": bool(full_game_only),
        "tag_penalties": tag_pens,
        "base": {
            "n": base_n,
            "profit_units": base_profit,
            "roi_units_per_bet": (base_profit / max(1, base_n)),
        },
        "with_tag_penalties": {
            "n": tags_n,
            "profit_units": tags_profit,
            "roi_units_per_bet": (tags_profit / max(1, tags_n)),
        },
        "delta": {
            "n": tags_n - base_n,
            "profit_units": tags_profit - base_profit,
            "roi_units_per_bet": (tags_profit / max(1, tags_n)) - (base_profit / max(1, base_n)),
        },
    }

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = Path(args.out_json) if args.out_json else out_dir / f"live_lens_tag_penalty_window_eval_{start_date}_{end_date}.json"
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / f"live_lens_tag_penalty_window_eval_{start_date}_{end_date}.csv"

    out_json.write_text(json.dumps({"overall": overall, "per_date": rows}, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print(json.dumps({"out_json": str(out_json), "out_csv": str(out_csv), "overall": overall}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
