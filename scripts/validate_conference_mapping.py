import os
import sys
import csv
from pathlib import Path
from typing import Dict, Set, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "diagnostics"


def _load_canonical_conferences() -> Set[str]:
    p = DATA / "d1_conferences_list.csv"
    if not p.exists():
        return set()
    df = pd.read_csv(p)
    return set(df["conference"].astype(str).str.strip())


def _canon(s: str) -> str:
    return str(s or "").strip()


def _load_team_conference_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    # Prefer d1_conferences.csv, fallback to conferences.csv
    for fname in ["d1_conferences.csv", "conferences.csv"]:
        fpath = DATA / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        cols = {c.lower(): c for c in df.columns}
        # expected columns: team, conference (case-flexible)
        team_col = cols.get("team") or cols.get("school") or cols.get("name")
        conf_col = cols.get("conference") or cols.get("conf")
        if not team_col or not conf_col:
            continue
        for r in df[[team_col, conf_col]].itertuples(index=False):
            t = _canon(getattr(r, team_col))
            c = _canon(getattr(r, conf_col))
            if t:
                mapping[t] = c or "Unknown"
    return mapping


def _observe_teams_from_outputs() -> Set[str]:
    teams: Set[str] = set()
    out_dir = ROOT / "outputs"
    # Scan common artifacts for observed team names
    candidates = (
        list(out_dir.glob("results_*.csv"))
        + list(out_dir.glob("align_period_*.csv"))
        + list(out_dir.glob("predictions_unified_*.csv"))
        + list(out_dir.glob("predictions_display_*.csv"))
    )
    for p in candidates:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        for col in ["home_team", "away_team", "home", "away"]:
            if col in df.columns:
                teams.update(df[col].astype(str).str.strip())
    return set(t for t in teams if t)


def _write_csv(path: Path, rows: List[Dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = []
    # Determine header from union of keys
    keys: Set[str] = set()
    for r in rows:
        keys |= set(r.keys())
    hdr = sorted(keys) if keys else []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        if hdr:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    canonical = _load_canonical_conferences()
    team_map = _load_team_conference_map()
    observed = _observe_teams_from_outputs()

    # Missing teams or Unknown mappings
    missing: List[Dict[str, str]] = []
    unknown: List[Dict[str, str]] = []
    for t in sorted(observed):
        c = team_map.get(t)
        if c is None:
            missing.append({"team": t, "issue": "not_mapped"})
        else:
            if c.strip() in ("", "Unknown", "Non-D1", "Other", "Non-D1/Other"):
                unknown.append({"team": t, "conference": c})

    # Check conference list coverage
    present_confs: Set[str] = set(v for v in team_map.values() if v)
    not_in_canonical = sorted([c for c in present_confs if c not in canonical and c not in ("Unknown", "Non-D1/Other")])
    canonical_missing: List[Dict[str, str]] = []
    for c in sorted(canonical):
        # If canonical conference has zero teams in map, flag
        if c not in present_confs:
            canonical_missing.append({"conference": c, "issue": "no_teams_in_map"})

    # Write reports
    _write_csv(OUT / "conference_mapping_missing_teams.csv", missing)
    _write_csv(OUT / "conference_mapping_unknown_teams.csv", unknown)
    _write_csv(OUT / "conference_mapping_noncanonical_conferences.csv", [{"conference": c} for c in not_in_canonical])
    _write_csv(OUT / "conference_mapping_canonical_missing.csv", canonical_missing)

    summary = {
        "observed_teams": len(observed),
        "mapped_teams": len([t for t in observed if t in team_map]),
        "unknown_count": len(unknown),
        "unmapped_count": len(missing),
        "noncanonical_conferences": len(not_in_canonical),
        "canonical_conferences_without_teams": len(canonical_missing),
    }
    _write_csv(OUT / "conference_mapping_summary.csv", [{k: str(v) for k, v in summary.items()}])

    print("Conference mapping diagnostics written to", OUT)


if __name__ == "__main__":
    main()
