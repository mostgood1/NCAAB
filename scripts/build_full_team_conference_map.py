"""
Build a full team→conference mapping with near-100% coverage from observed artifacts.

Inputs (best-effort, optional):
- outputs/games_*.csv, outputs/games_curr.csv
- outputs/daily_results/results_*.csv
- outputs/predictions_unified_*.csv, outputs/predictions_display_*.csv
- data/d1_conferences.csv (existing team→conference map)
- data/provider_aliases.csv (alias→canonical name mapping)
- data/team_map.csv (raw→canonical slug overrides)
- data/d1_conferences_list.csv (canonical list of D1 conference names)

Outputs:
- outputs/diagnostics/team_conference_missing.csv: teams without a mapped conference
- outputs/diagnostics/team_conference_full.csv: complete list of observed teams with mapped conference
- data/d1_conferences.csv: optionally appended with placeholder rows for unmapped teams (flag --append-placeholders)

Usage:
    python scripts/build_full_team_conference_map.py [--append-placeholders]

Notes:
- Uses `normalize_name` and `_canon_slug` logic similar to the app to maximize alias coverage.
- Placeholder rows use conference "Unknown" for quick manual correction.
"""
from __future__ import annotations
import os
import glob
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = ROOT / "data"
DIAG = OUT / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)

# Lightweight normalize_name fallback if src import fails
try:
    import sys
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from ncaab_model.data.merge_odds import normalize_name  # type: ignore
except Exception:
    def normalize_name(x: str) -> str:
        return str(x).lower().strip().replace(" ", "-")

# Load custom team map: raw→canonical slug overrides
def load_custom_team_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    p = DATA / "team_map.csv"
    if not p.exists():
        return mapping
    try:
        df = pd.read_csv(p)
        raw_col = next((c for c in df.columns if c.lower() == "raw"), None)
        can_col = next((c for c in df.columns if c.lower() == "canonical"), None)
        if raw_col and can_col:
            for r in df[[raw_col, can_col]].itertuples(index=False):
                raw = normalize_name(getattr(r, raw_col))
                can = normalize_name(getattr(r, can_col))
                if raw and can:
                    mapping[raw] = can
    except Exception:
        pass
    return mapping

CUSTOM_MAP = load_custom_team_map()

def canon_slug(name: str) -> str:
    s = normalize_name(name)
    return CUSTOM_MAP.get(s, s)

# Load provider aliases alias→canonical
def load_alias_map() -> Dict[str, str]:
    m: Dict[str, str] = {}
    p = DATA / "provider_aliases.csv"
    if not p.exists():
        return m
    try:
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        a = cols.get("alias") or cols.get("raw")
        c = cols.get("canonical") or cols.get("name")
        if a and c:
            for r in df[[a, c]].itertuples(index=False):
                alias = canon_slug(getattr(r, a))
                canon = canon_slug(getattr(r, c))
                if alias and canon:
                    m[alias] = canon
    except Exception:
        pass
    return m

ALIAS_MAP = load_alias_map()

# Load existing team→conference map (D1 preferred)
def load_team_conference_map() -> Dict[str, str]:
    m: Dict[str, str] = {}
    for fname in ["d1_conferences.csv", "conferences.csv"]:
        p = DATA / fname
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            cols = {c.lower(): c for c in df.columns}
            tcol = cols.get("team") or cols.get("name") or cols.get("school")
            ccol = cols.get("conference") or cols.get("conf")
            if tcol and ccol:
                for r in df[[tcol, ccol]].itertuples(index=False):
                    team = canon_slug(getattr(r, tcol))
                    conf = str(getattr(r, ccol) or "").strip()
                    if team:
                        m[team] = conf or "Unknown"
        except Exception:
            pass
    return m

TEAM_CONF = load_team_conference_map()

# Canonical D1 conference set (optional)
def load_d1_conferences_list() -> set[str]:
    p = DATA / "d1_conferences_list.csv"
    if not p.exists():
        return set()
    try:
        df = pd.read_csv(p)
        cname = next((c for c in df.columns if c.lower() in ("conference", "name")), None)
        if cname:
            return set(df[cname].astype(str).str.strip())
    except Exception:
        pass
    return set()

D1_SET = load_d1_conferences_list()

# Collect observed team names from outputs
OBS_COLLECT_FILES = [
    *list(OUT.glob("games_*.csv")),
    OUT / "games_curr.csv",
    *list((OUT / "daily_results").glob("results_*.csv")),
    *list(OUT.glob("predictions_unified_*.csv")),
    *list(OUT.glob("predictions_display_*.csv")),
]

def collect_observed_teams() -> set[str]:
    teams: set[str] = set()
    for p in OBS_COLLECT_FILES:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        for col in ("home_team", "away_team", "home", "away"):
            if col in df.columns:
                vals = df[col].dropna().astype(str).tolist()
                for v in vals:
                    s = canon_slug(v)
                    s = ALIAS_MAP.get(s, s)
                    if s:
                        teams.add(s)
    return teams

def resolve_conference_for_team(slug: str) -> str:
    # Try direct map
    conf = TEAM_CONF.get(slug)
    if conf:
        return conf
    # Try alias resolution to canonical
    alias_canon = ALIAS_MAP.get(slug)
    if alias_canon:
        return TEAM_CONF.get(alias_canon, "Unknown")
    return "Unknown"

def main(append_placeholders: bool = False) -> None:
    observed = collect_observed_teams()
    rows_full: List[Tuple[str, str]] = []
    rows_missing: List[str] = []
    for t in sorted(observed):
        conf = resolve_conference_for_team(t)
        rows_full.append((t, conf))
        if conf == "Unknown" or conf.strip() == "":
            rows_missing.append(t)

    # Write diagnostics
    full_path = DIAG / "team_conference_full.csv"
    miss_path = DIAG / "team_conference_missing.csv"
    pd.DataFrame(rows_full, columns=["team_slug", "conference"]).to_csv(full_path, index=False)
    pd.DataFrame(rows_missing, columns=["team_slug"]).to_csv(miss_path, index=False)

    print(f"Observed unique teams: {len(observed)}")
    print(f"Mapped teams: {len(observed) - len(rows_missing)}")
    print(f"Missing teams: {len(rows_missing)} → {miss_path}")

    # Optionally append placeholders to d1_conferences.csv for manual fix-up
    if append_placeholders and rows_missing:
        d1_path = DATA / "d1_conferences.csv"
        existing = pd.read_csv(d1_path) if d1_path.exists() else pd.DataFrame(columns=["team", "conference"])
        # Build set of existing team slugs to avoid duplicates
        cols = {c.lower(): c for c in existing.columns}
        team_col = cols.get("team") or cols.get("name") or "team"
        conf_col = cols.get("conference") or "conference"
        if team_col not in existing.columns:
            existing[team_col] = []
        if conf_col not in existing.columns:
            existing[conf_col] = []
        existing_slugs = set(existing[team_col].astype(str).map(canon_slug)) if not existing.empty else set()
        to_append = []
        for t in rows_missing:
            if t not in existing_slugs:
                to_append.append({team_col: t, conf_col: "Unknown"})
        if to_append:
            existing = pd.concat([existing, pd.DataFrame(to_append)], ignore_index=True)
            existing.to_csv(d1_path, index=False)
            print(f"Appended {len(to_append)} placeholder rows to {d1_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--append-placeholders", action="store_true", help="Append Unknown conference placeholders to data/d1_conferences.csv")
    args = ap.parse_args()
    main(append_placeholders=args.append_placeholders)
