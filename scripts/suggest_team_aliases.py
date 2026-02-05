from __future__ import annotations

import difflib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import _canon_slug, _compute_accuracy_market_season_payload, _conference_map  # noqa: E402


def main() -> None:
    payload = _compute_accuracy_market_season_payload()
    unmapped = payload.get("unmapped_teams") or []
    print("unmapped", len(unmapped))

    cdf = pd.read_csv("data/d1_conferences.csv", dtype=str, comment="#")
    d1_teams = [t for t in cdf["team"].dropna().astype(str).tolist()]

    d1_norm_to_team: dict[str, list[str]] = {}
    for t in d1_teams:
        n = _canon_slug(t)
        d1_norm_to_team.setdefault(n, []).append(t)

    d1_norms = list(d1_norm_to_team.keys())
    conf_map = _conference_map()

    def conf_for(team: str) -> str | None:
        return conf_map.get(team) or conf_map.get(_canon_slug(team))

    def suggest(team: str):
        n = _canon_slug(team)
        if n in d1_norm_to_team and len(d1_norm_to_team[n]) == 1:
            canon = d1_norm_to_team[n][0]
            return ("exact_norm", canon, 1.0, conf_for(canon))

        matches = difflib.get_close_matches(n, d1_norms, n=3, cutoff=0.88)
        if not matches:
            return None
        best = matches[0]
        score = difflib.SequenceMatcher(a=n, b=best).ratio()
        canon = d1_norm_to_team[best][0]
        return ("fuzzy_norm", canon, score, conf_for(canon))

    rows = []
    for t in unmapped:
        s = suggest(t)
        if not s:
            rows.append((t, None, 0.0, None, None))
        else:
            kind, canon, score, conf = s
            rows.append((t, canon, float(score), conf, kind))

    rows_sorted = sorted(rows, key=lambda x: (x[2], x[0]), reverse=True)

    print("\nHigh confidence (>=0.92):")
    for t, canon, score, conf, kind in rows_sorted:
        if canon and score >= 0.92:
            print(f"{t} -> {canon} score={score:.3f} conf={conf} ({kind})")

    print("\nLow confidence / no match (<0.92):")
    for t, canon, score, conf, kind in rows_sorted:
        if (not canon) or score < 0.92:
            print(f"{t} -> {canon} score={score:.3f} ({kind})")


if __name__ == "__main__":
    main()
