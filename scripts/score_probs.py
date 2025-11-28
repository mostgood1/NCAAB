from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import pandas as pd
from joblib import load

from src.modeling.baselines import choose_feature_columns

TARGET_KEYS = ["win", "ats", "ou"]
TARGET_TO_COL = {
    "win": "home_win",
    "ats": "ats_home_cover",
    "ou": "ou_over",
}
OUT_COL_MAP = {
    "win": "p_home_win",
    "ats": "p_home_cover",
    "ou": "p_over",
}


def _find_enriched(outputs_dir: Path, date: str | None) -> Path | None:
    cands: list[Path] = []
    if date:
        cands.append(outputs_dir / f"predictions_unified_enriched_{date}.csv")
        cands.append(outputs_dir / f"predictions_unified_{date}.csv")
        cands.append(outputs_dir / f"predictions_{date}.csv")
    # Last resort: latest enriched
    if not cands:
        for p in sorted(outputs_dir.glob("predictions_unified_enriched_*.csv"), reverse=True):
            cands.append(p)
    for p in cands:
        if p.exists():
            return p
    return None


def _last_report(reports_dir: Path) -> Path | None:
    reports = sorted(reports_dir.glob("train_report_*Z.json"), reverse=True)
    return reports[0] if reports else None


def _features_for_target(report_path: Path | None, df: pd.DataFrame, target_key: str, allow_market: bool) -> list[str]:
    if report_path and report_path.exists():
        try:
            obj = json.loads(report_path.read_text(encoding="utf-8"))
            feats = obj.get("trained", {}).get(target_key, {}).get("features")
            if isinstance(feats, list) and feats:
                return [str(c) for c in feats]
        except Exception:
            pass
    # Fallback: derive from df directly (risk of mismatch vs training)
    return choose_feature_columns(df, allow_market=allow_market)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score win/ATS/OU probabilities for a given date using trained models.")
    ap.add_argument("--date", default=None, help="Target date YYYY-MM-DD; if omitted, picks latest enriched file")
    ap.add_argument("--outputs-dir", default="outputs", help="Outputs directory")
    ap.add_argument("--models-dir", default=None, help="Models directory; default = <outputs>/models")
    ap.add_argument("--reports-dir", default=None, help="Reports directory; default = <outputs>/reports")
    ap.add_argument("--targets", default="win,ats,ou", help="Comma-separated subset of targets to score")
    ap.add_argument("--allow-market", action="store_true", help="Allow closing_* features if models were trained with them")
    ap.add_argument("--write-merged", action="store_true", help="Also write merged enriched file including probabilities")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    models_dir = Path(args.models_dir) if args.models_dir else outputs_dir / "models"
    reports_dir = Path(args.reports_dir) if args.reports_dir else outputs_dir / "reports"

    enriched = _find_enriched(outputs_dir, args.date)
    if enriched is None or not enriched.exists():
        print("Could not locate enriched/unified predictions for date:", args.date)
        sys.exit(2)

    df = pd.read_csv(enriched)
    if df is None or df.empty:
        print("Input enriched file is empty:", enriched)
        sys.exit(2)
    if "game_id" not in df.columns:
        print("Missing game_id column; cannot merge outputs deterministically.")
        sys.exit(2)

    # Build output frame with identifiers
    out = df[[c for c in ["game_id", "home_team", "away_team", "date"] if c in df.columns]].copy()

    selected = [t.strip().lower() for t in args.targets.split(",") if t.strip()]
    selected = [t for t in selected if t in TARGET_KEYS]
    if not selected:
        print("No valid targets requested.")
        sys.exit(2)

    # Load most recent report to resolve features per target
    report = _last_report(reports_dir)

    for key in selected:
        model_path = models_dir / f"{key}_classifier.joblib"
        if not model_path.exists():
            print(f"Model not found for {key} at {model_path}; skipping")
            continue
        model = load(model_path)
        feats = _features_for_target(report, df, key, allow_market=args.allow_market)
        # Ensure all features present; create missing with NaN
        for f in feats:
            if f not in df.columns:
                df[f] = float("nan")
        X = df[feats]
        try:
            proba = model.predict_proba(X)[:, 1]
        except Exception:
            pred = model.predict(X)
            proba = pred.astype(float)
        out[OUT_COL_MAP[key]] = proba

    date_tag = args.date
    if date_tag is None:
        # Try read from filename
        try:
            date_tag = enriched.stem.split("_")[-1]
        except Exception:
            date_tag = "unknown"

    probs_path = outputs_dir / f"model_probs_{date_tag}.csv"
    out.to_csv(probs_path, index=False)
    print("Wrote:", probs_path)

    if args.write_merged:
        merged = df.merge(out, on="game_id", how="left")
        merged_path = outputs_dir / f"predictions_unified_enriched_{date_tag}_with_probs.csv"
        merged.to_csv(merged_path, index=False)
        print("Merged enriched + probs:", merged_path)


if __name__ == "__main__":
    main()
