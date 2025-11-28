import json
import os
from pathlib import Path
import pandas as pd
import joblib  # type: ignore

OUT = Path(os.getenv("NCAAB_OUTPUTS_DIR", Path(__file__).resolve().parents[1] / "outputs"))

COVER_SIDEcar = OUT / "meta_features_cover.json"
TOTAL_SIDEcar = OUT / "meta_features_total.json"

CANDIDATE_PREFIXES_COVER = [
    "pred_margin", "spread_home", "closing_spread_home",
    "proj_home", "proj_away", "edge_margin", "kelly_frac_spread",
    "neutral_site", "volatility_tag", "edge_conf_tag"
]

CANDIDATE_PREFIXES_TOTAL = [
    "pred_total", "market_total", "closing_total", "derived_total",
    "pred_total_sigma", "edge_total", "kelly_fraction_total",
    "proj_home", "proj_away", "volatility_tag", "edge_conf_tag"
]

EXCLUDE_SUFFIXES = {"_1h", "_2h"}


def _pick_ordered(df: pd.DataFrame, prefixes: list[str]) -> list[str]:
    cols = list(df.columns)
    picked: list[str] = []
    for pref in prefixes:
        for c in cols:
            if c.startswith(pref) and not any(c.endswith(suf) for suf in EXCLUDE_SUFFIXES):
                if c not in picked:
                    picked.append(c)
    return picked


def _model_feature_names(model_path: Path) -> list[str] | None:
    try:
        if not model_path.exists():
            return None
        m = joblib.load(model_path)
        # Try LightGBM sklearn API
        if hasattr(m, 'booster_') and hasattr(m.booster_, 'feature_name'):
            try:
                names = m.booster_.feature_name()
                return list(names) if names else None
            except Exception:
                pass
        if hasattr(m, 'feature_name_') and isinstance(m.feature_name_, (list, tuple)):
            return list(m.feature_name_)
        return None
    except Exception:
        return None


def emit_sidecars(preds_unified_path: Path | None = None) -> None:
    if preds_unified_path is None:
        # Best-effort: pick latest unified predictions
        candidates = sorted(OUT.glob("predictions_unified_*.csv"), reverse=True)
        preds_unified_path = candidates[0] if candidates else None
    if preds_unified_path is None or not preds_unified_path.exists():
        print("No predictions_unified_<date>.csv found under outputs; cannot emit sidecars.")
        return
    df = pd.read_csv(preds_unified_path)
    if df.empty:
        print("Unified predictions file is empty; skipping.")
        return
    # Prefer model-embedded feature names when available to exactly match training
    cover_model_path = OUT / 'meta_cover_lgbm.joblib'
    over_model_path = OUT / 'meta_over_lgbm.joblib'
    cover_list = _model_feature_names(cover_model_path) or _pick_ordered(df, CANDIDATE_PREFIXES_COVER)
    total_list = _model_feature_names(over_model_path) or _pick_ordered(df, CANDIDATE_PREFIXES_TOTAL)
    if not cover_list:
        # Fallback minimal
        cover_list = [c for c in df.columns if c in ("pred_margin", "spread_home", "closing_spread_home")]
    if not total_list:
        total_list = [c for c in df.columns if c in ("pred_total", "market_total", "closing_total", "derived_total")]
    COVER_SIDEcar.write_text(json.dumps({"features": cover_list}, indent=2), encoding="utf-8")
    TOTAL_SIDEcar.write_text(json.dumps({"features": total_list}, indent=2), encoding="utf-8")
    print(f"Wrote {COVER_SIDEcar}")
    print(f"Wrote {TOTAL_SIDEcar}")


if __name__ == "__main__":
    emit_sidecars()
