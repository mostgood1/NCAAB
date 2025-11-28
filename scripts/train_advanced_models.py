from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from joblib import dump

from src.modeling.datasets import load_training_data
from src.modeling.features import attach_team_history_features
from src.modeling.metrics import regression_metrics, classification_metrics, reliability_curve, sharpness, dispersion

try:
    from lightgbm import LGBMRegressor, LGBMClassifier  # type: ignore
except Exception:  # pragma: no cover
    LGBMRegressor = None  # type: ignore
    LGBMClassifier = None  # type: ignore

try:
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier  # type: ignore
    from sklearn.isotonic import IsotonicRegression  # type: ignore
except Exception:  # pragma: no cover
    GradientBoostingRegressor = None  # type: ignore
    GradientBoostingClassifier = None  # type: ignore
    IsotonicRegression = None  # type: ignore


def _pick_regressor() -> object:
    if LGBMRegressor is not None:
        return LGBMRegressor(n_estimators=500, learning_rate=0.03, subsample=0.9, colsample_bytree=0.8)
    if GradientBoostingRegressor is not None:
        return GradientBoostingRegressor()
    raise RuntimeError("No regressor available (LightGBM / GradientBoosting missing)")


def _pick_classifier() -> object:
    if LGBMClassifier is not None:
        return LGBMClassifier(n_estimators=600, learning_rate=0.03, subsample=0.9, colsample_bytree=0.8)
    if GradientBoostingClassifier is not None:
        return GradientBoostingClassifier()
    raise RuntimeError("No classifier available (LightGBM / GradientBoosting missing)")


def _quantile_predictions(base: object, X: pd.DataFrame, quantiles=(0.25, 0.5, 0.75)) -> dict:
    out = {}
    # If LightGBM, we can re-fit with objective=quantile per q (simplest approach for now)
    if LGBMRegressor is not None and isinstance(base, LGBMRegressor):
        for q in quantiles:
            qr = LGBMRegressor(objective="quantile", alpha=q, n_estimators=200, learning_rate=0.05)
            qr.fit(X, base._y) if hasattr(base, '_y') else None  # type: ignore
            out[f"q_{int(q*100)}"] = qr.predict(X)
    else:
        # Fallback: approximate quantiles using residual std + normal assumption
        preds = base.predict(X)
        if hasattr(base, '_y'):
            resid = base._y - preds  # type: ignore
            sigma = float(np.nanstd(resid))
        else:
            sigma = 7.5  # heuristic
        for q in quantiles:
            # inverse normal CDF approximation using numpy (erfinv)
            from math import sqrt
            import scipy.special  # type: ignore
            z = sqrt(2) * scipy.special.erfinv(2*q - 1)
            out[f"q_{int(q*100)}"] = preds + z * sigma
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train advanced totals/margin models + probability classifiers with history features.")
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--date-start", default=None)
    ap.add_argument("--date-end", default=None)
    ap.add_argument("--save", action="store_true", help="Persist model artifacts")
    ap.add_argument("--calibrate-win", action="store_true", help="Apply isotonic calibration to win probability")
    ap.add_argument("--calibrate-ats", action="store_true")
    ap.add_argument("--calibrate-ou", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.outputs_dir)
    models_dir = out_dir / "models"
    reports_dir = out_dir / "reports"
    models_dir.mkdir(exist_ok=True, parents=True)
    reports_dir.mkdir(exist_ok=True, parents=True)

    base_df = load_training_data(out_dir, date_start=args.date_start, date_end=args.date_end)
    if base_df.empty:
        print("No training data found.")
        return
    # Attach history features
    enriched = attach_team_history_features(base_df, out_dir)

    # Totals regression (predict actual_total)
    if "actual_total" in enriched.columns:
        reg_feats = [c for c in enriched.columns if c not in {"actual_total","home_win","ats_home_cover","ou_over","home_score","away_score"} and pd.api.types.is_numeric_dtype(enriched[c])]
        reg_df = enriched.dropna(subset=["actual_total"])
        Xr = reg_df[reg_feats]
        yr = reg_df["actual_total"].astype(float)
        reg = _pick_regressor()
        try:
            reg.fit(Xr, yr)
            setattr(reg, '_y', yr)  # store for quantile residual usage
            preds = reg.predict(Xr)
            r_metrics = regression_metrics(yr.values, preds)
            quant = _quantile_predictions(reg, Xr)
        except Exception as e:
            r_metrics = {"error": str(e)}
            quant = {}
    else:
        r_metrics = {"skipped": True}
        quant = {}
        reg = None

    # Margin regression (home margin prediction)
    if {"home_score","away_score"}.issubset(enriched.columns):
        enriched["home_margin"] = pd.to_numeric(enriched["home_score"], errors="coerce") - pd.to_numeric(enriched["away_score"], errors="coerce")
        m_feats = [c for c in enriched.columns if c not in {"home_margin","home_win","ats_home_cover","ou_over","home_score","away_score","actual_total"} and pd.api.types.is_numeric_dtype(enriched[c])]
        m_df = enriched.dropna(subset=["home_margin"])
        Xm = m_df[m_feats]
        ym = m_df["home_margin"].astype(float)
        mreg = _pick_regressor()
        try:
            mreg.fit(Xm, ym)
            m_preds = mreg.predict(Xm)
            m_metrics = regression_metrics(ym.values, m_preds)
            resid = ym.values - m_preds
            margin_sigma = float(np.nanstd(resid))
        except Exception as e:
            m_metrics = {"error": str(e)}
            margin_sigma = float("nan")
    else:
        m_metrics = {"skipped": True}
        margin_sigma = float("nan")
        mreg = None

    # Win / ATS / OU classifiers
    cls_results = {}
    for key, col in {"win": "home_win", "ats": "ats_home_cover", "ou": "ou_over"}.items():
        if col not in enriched.columns:
            cls_results[key] = {"skipped": True}
            continue
        work = enriched.dropna(subset=[col])
        y = work[col].astype(int)
        cls_feats = [c for c in work.columns if c not in {col,"home_score","away_score","actual_total","home_margin","home_win","ats_home_cover","ou_over"} and pd.api.types.is_numeric_dtype(work[c])]
        X = work[cls_feats]
        model = _pick_classifier()
        try:
            model.fit(X, y)
            prob = model.predict_proba(X)[:, 1]
            c_metrics = classification_metrics(y.values, prob)
            rel = reliability_curve(y.values, prob)
            cal_model = None
            if ((key == 'win' and args.calibrate_win) or (key == 'ats' and args.calibrate_ats) or (key == 'ou' and args.calibrate_ou)) and IsotonicRegression is not None:
                cal_model = IsotonicRegression(out_of_bounds='clip')
                cal_model.fit(prob, y.values)
                prob_cal = cal_model.predict(prob)
                c_metrics['accuracy_cal'] = classification_metrics(y.values, prob_cal)['accuracy']
                c_metrics['ece_cal'] = classification_metrics(y.values, prob_cal)['ece']
            cls_results[key] = {
                "metrics": c_metrics,
                "reliability": rel,
                "sharpness": sharpness(prob),
                "dispersion": dispersion(prob),
                "features": cls_feats,
                "calibrated": bool(cal_model is not None),
            }
        except Exception as e:
            cls_results[key] = {"error": str(e)}
        if args.save and 'error' not in cls_results[key]:
            dump(model, models_dir / f"{key}_adv_classifier.joblib")
            if cls_results[key].get('calibrated') and IsotonicRegression is not None:
                dump(cal_model, models_dir / f"{key}_adv_calibrator.joblib")  # type: ignore

    summary = {
        "timestamp": datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        "n_rows": int(len(enriched)),
        "totals_regression_metrics": r_metrics,
        "totals_quantiles_keys": list(quant.keys()),
        "margin_regression_metrics": m_metrics,
        "margin_sigma": margin_sigma,
        "classifiers": cls_results,
        "date_start": args.date_start,
        "date_end": args.date_end,
    }

    report_path = reports_dir / f"advanced_report_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}Z.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print("Advanced report:", report_path)

    if args.save and reg is not None:
        dump(reg, models_dir / "totals_regressor.joblib")
    if args.save and mreg is not None:
        dump(mreg, models_dir / "margin_regressor.joblib")


if __name__ == '__main__':
    main()
