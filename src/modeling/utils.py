import json
import pathlib
from functools import lru_cache
from typing import Dict
import re

OUT = pathlib.Path(__file__).resolve().parents[2] / "outputs"
DATA = pathlib.Path(__file__).resolve().parents[2] / "data"
TEAM_MAP_PATH = DATA / "team_map.csv"

@lru_cache(maxsize=1)
def load_team_map() -> Dict[str, str]:
    """Load team_map.csv into a simple normalization dictionary.
    CSV expected columns: raw, canonical (flexible: will use first two columns if headers differ).
    """
    mp: Dict[str, str] = {}
    if TEAM_MAP_PATH.exists():
        try:
            import pandas as pd
            df = pd.read_csv(TEAM_MAP_PATH)
            # Heuristic columns
            if {"raw","canonical"}.issubset({c.lower() for c in df.columns}):
                raw_col = next(c for c in df.columns if c.lower()=="raw")
                canon_col = next(c for c in df.columns if c.lower()=="canonical")
            else:
                cols = list(df.columns)
                if len(cols) >= 2:
                    raw_col, canon_col = cols[0], cols[1]
                else:
                    return mp
            for _, r in df.iterrows():
                rv = str(r.get(raw_col, "")).strip()
                cv = str(r.get(canon_col, "")).strip()
                if rv:
                    mp[rv.lower()] = cv or rv
        except Exception:
            return mp
    return mp

_slug_clean_re = re.compile(r"[^a-z0-9]+")

@lru_cache(maxsize=4096)
def canon_slug(name: str) -> str:
    """Canonical slug compatible with existing app slugging logic.
    Applies team_map overrides, strips mascots/punctuation, lowers and underscores.
    """
    if not name:
        return ""
    s = str(name).strip()
    lower = s.lower()
    mp = load_team_map()
    mapped = mp.get(lower)
    if mapped:
        lower = mapped.lower()
    cleaned = _slug_clean_re.sub("_", lower)
    cleaned = cleaned.strip("_")
    return cleaned
