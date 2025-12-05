import json
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "outputs"

def _load_sidecar(fname: str) -> Optional[List[str]]:
    p = OUT / fname
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        # Accept either a raw list or an object with a "features" list
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
        if isinstance(data, dict):
            feats = data.get("features")
            if isinstance(feats, list) and all(isinstance(x, str) for x in feats):
                return feats
    except Exception:
        return None
    return None

def build_ordered_feature_frame(df: pd.DataFrame, sidecar_name: str) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """
    Build an ordered feature DataFrame matching the trained meta sidecar schema.
    - Reads sidecar list from outputs/<sidecar_name>
    - Ensures all required columns exist; pads missing with NaN
    - Orders columns exactly as the sidecar
    Returns (X, feature_names) where feature_names is the ordered sidecar list or None if not available.
    """
    side = _load_sidecar(sidecar_name)
    if not side:
        return pd.DataFrame(index=df.index), None
    # Ensure columns exist; pad missing
    X = pd.DataFrame(index=df.index)
    for col in side:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = pd.NA
    # Order columns exactly
    X = X[side]
    return X, side
