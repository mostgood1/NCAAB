import json
from pathlib import Path
from typing import List, Optional

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

COVER_SCHEMA_FILE = OUT_DIR / "meta_features_cover.json"
TOTAL_SCHEMA_FILE = OUT_DIR / "meta_features_total.json"


def save_feature_schema(kind: str, features: List[str]) -> Path:
    """
    Persist ordered feature names for meta models.
    kind: "cover" or "total"
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = COVER_SCHEMA_FILE if kind == "cover" else TOTAL_SCHEMA_FILE
    payload = {
        "kind": kind,
        "features": features,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_feature_schema(kind: str) -> Optional[List[str]]:
    """
    Load ordered feature names for meta models if available.
    Returns list of feature names or None.
    """
    target = COVER_SCHEMA_FILE if kind == "cover" else TOTAL_SCHEMA_FILE
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        feats = data.get("features")
        if isinstance(feats, list) and all(isinstance(x, str) for x in feats):
            return feats
    except Exception:
        return None
    return None


def build_ordered_matrix(df, feature_names: List[str]):
    """
    Given a DataFrame and the ordered feature_names, construct an ordered matrix
    with missing columns filled by NaN. Returns a DataFrame.
    """
    import pandas as pd
    cols = []
    for name in feature_names:
        if name in df.columns:
            cols.append(df[name])
        else:
            cols.append(pd.Series([pd.NA] * len(df), index=df.index))
    return pd.concat(cols, axis=1)
