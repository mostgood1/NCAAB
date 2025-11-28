from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from joblib import dump

from src.modeling.datasets import load_training_data
from src.modeling.baselines import train_eval_classifier, choose_feature_columns


TARGET_MAP = {
    "win": "home_win",
    "ats": "ats_home_cover",
    "ou": "ou_over",
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate NCAAB win/ATS/OU classifiers with time-aware CV.")
    ap.add_argument("--outputs-dir", default="outputs", help="Path to outputs directory containing daily_results/")
    ap.add_argument("--targets", default="win,ats,ou", help="Comma-separated targets: win, ats, ou")
    ap.add_argument("--folds", type=int, default=5, help="Number of time-aware CV folds")
    ap.add_argument("--allow-market", action="store_true", help="Allow closing_* features (leakage risky)")
    ap.add_argument("--date-start", default=None, help="Filter training data start date (YYYY-MM-DD)")
    ap.add_argument("--date-end", default=None, help="Filter training data end date (YYYY-MM-DD)")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    models_dir = outputs_dir / "models"
    reports_dir = outputs_dir / "reports"
    ensure_dir(models_dir)
    ensure_dir(reports_dir)

    df = load_training_data(outputs_dir=outputs_dir, date_start=args.date_start, date_end=args.date_end)
    if df is None or df.empty:
        print("No training data found under", outputs_dir / "daily_results")
        return

    # Resolve targets
    targ_keys = [t.strip().lower() for t in args.targets.split(",") if t.strip()]
    unknown = [t for t in targ_keys if t not in TARGET_MAP]
    if unknown:
        raise SystemExit(f"Unknown targets: {unknown}. Valid: {list(TARGET_MAP.keys())}")

    summary = {
        "outputs_dir": str(outputs_dir),
        "n_rows": int(len(df)),
        "date_start": args.date_start,
        "date_end": args.date_end,
        "folds": args.folds,
        "allow_market": bool(args.allow_market),
        "targets": targ_keys,
        "trained": {},
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    for key in targ_keys:
        target_col = TARGET_MAP[key]
        if target_col not in df.columns:
            print(f"Skipping {key}: target column '{target_col}' missing in data")
            continue
        try:
            model, avg_metrics, fold_metrics = train_eval_classifier(
                df=df,
                target_col=target_col,
                allow_market=args.allow_market,
                n_splits=args.folds,
            )
        except Exception as e:
            print(f"Training failed for {key}: {e}")
            continue

        # Save model artifact
        model_path = models_dir / f"{key}_classifier.joblib"
        dump(model, model_path)

        summary["trained"][key] = {
            "model_path": str(model_path),
            "avg_metrics": avg_metrics,
            "fold_metrics": fold_metrics,
            "n_rows_used": int(len(df.dropna(subset=[target_col]))),
            "features": choose_feature_columns(df.dropna(subset=[target_col]), allow_market=args.allow_market),
        }

    # Save report
    report_path = reports_dir / f"train_report_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}Z.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
