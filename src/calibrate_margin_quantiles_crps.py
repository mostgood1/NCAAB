import argparse
import glob
import json
from pathlib import Path

import pandas as pd


OUT = Path("outputs")


def _read_results(date: str) -> pd.DataFrame:
    p = OUT / f"daily_results/results_{date}.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def _read_predictions(date: str) -> pd.DataFrame:
    p = OUT / f"predictions_model_margins_{date}.csv"
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    p2 = OUT / f"predictions_unified_enriched_{date}.csv"
    try:
        if p2.exists():
            return pd.read_csv(p2)
    except Exception:
        pass
    return pd.DataFrame()


def _collect_dates() -> list[str]:
    files = sorted(glob.glob(str(OUT / "daily_results" / "results_*.csv")))
    return [Path(f).stem.replace("results_", "") for f in files]


def _date_in_range(d: str, start: str | None, end: str | None) -> bool:
    if start and d < start:
        return False
    if end and d > end:
        return False
    return True


def _join(df_preds: pd.DataFrame, df_res: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ["game_id", "home_team", "away_team", "date"] if c in df_preds.columns and c in df_res.columns]
    if not keys:
        keys = [c for c in ["home_team", "away_team"] if c in df_preds.columns and c in df_res.columns]
    if not keys:
        return pd.DataFrame()

    keep = list(set(keys + [c for c in ["actual_margin", "final_margin"] if c in df_res.columns]))
    res_min = df_res[keep].copy()
    # Normalize join key dtypes to avoid merge dtype mismatches.
    if "game_id" in keys:
        try:
            df_preds = df_preds.copy()
            df_preds["game_id"] = df_preds["game_id"].astype(str)
        except Exception:
            pass
        try:
            res_min["game_id"] = res_min["game_id"].astype(str)
        except Exception:
            pass
    if "actual_margin" not in res_min.columns and "final_margin" in res_min.columns:
        res_min = res_min.rename(columns={"final_margin": "actual_margin"})
    return pd.merge(df_preds, res_min, on=keys, how="inner")


def _pinball_loss(q: float, y: pd.Series, yhat: pd.Series) -> float:
    e = y - yhat
    return float(((q - (e < 0).astype(float)) * e).abs().mean())


def _evaluate_params(df: pd.DataFrame, shift: float, scale: float) -> float:
    q10 = pd.to_numeric(df["pred_margin_q10"], errors="coerce")
    q50 = pd.to_numeric(df["pred_margin_q50"], errors="coerce")
    q90 = pd.to_numeric(df["pred_margin_q90"], errors="coerce")
    y = pd.to_numeric(df["actual_margin"], errors="coerce")
    mask = q10.notna() & q50.notna() & q90.notna() & y.notna()
    if not mask.any():
        return 1e9

    q10c = (q50 + shift) - scale * (q50 - q10)
    q50c = (q50 + shift)
    q90c = (q50 + shift) + scale * (q90 - q50)

    l10 = _pinball_loss(0.1, y[mask], q10c[mask])
    l50 = _pinball_loss(0.5, y[mask], q50c[mask])
    l90 = _pinball_loss(0.9, y[mask], q90c[mask])
    return l10 + l50 + l90


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit global CRPS-style (pinball) calibration for margin quantiles")
    ap.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument("--n-days", type=int, default=None, help="Use only last N results dates (ignored if --start/--end)")
    args = ap.parse_args()

    dates = _collect_dates()
    if args.n_days and not (args.start or args.end):
        dates = dates[-int(args.n_days) :]
    dates = [d for d in dates if _date_in_range(d, args.start, args.end)]

    frames: list[pd.DataFrame] = []
    for d in dates:
        preds = _read_predictions(d)
        res = _read_results(d)
        if preds.empty or res.empty:
            continue
        joined = _join(preds, res)
        if not joined.empty:
            frames.append(joined)

    if not frames:
        print(json.dumps({"error": "No data"}))
        return

    df = pd.concat(frames, ignore_index=True)
    if not {"pred_margin_q10", "pred_margin_q50", "pred_margin_q90", "actual_margin"}.issubset(df.columns):
        print(json.dumps({"error": "Missing required columns"}))
        return

    best = {"loss": 1e9, "shift": 0.0, "scale": 1.0}
    for shift in [x / 2.0 for x in range(-40, 41)]:  # -20..20 step 0.5
        for scale in [0.5 + 0.05 * k for k in range(0, 21)]:  # 0.5..1.5 step 0.05
            loss = _evaluate_params(df, shift, scale)
            if loss < best["loss"]:
                best = {"loss": loss, "shift": round(shift, 3), "scale": round(scale, 3)}

    payload = {
        "shift": best["shift"],
        "scale": best["scale"],
        "loss": best["loss"],
        "n": int(len(df)),
        "start": args.start,
        "end": args.end,
    }
    (OUT / "calibration_margin_quantiles_crps.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
