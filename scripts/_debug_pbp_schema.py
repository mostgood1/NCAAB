from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="401858786", help="ESPN event id to inspect")
    ap.add_argument("--cache-dir", default=str(Path("data") / "cache" / "espn_pbp"))
    ap.add_argument(
        "--find-odds",
        action="store_true",
        help="Scan cache for first event with non-empty odds or pickcenter",
    )
    ap.add_argument("--max-files", type=int, default=2000)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if args.find_odds:
        checked = 0
        for p in cache_dir.glob("*.json"):
            checked += 1
            if args.max_files and checked > args.max_files:
                break
            try:
                d0 = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if (d0.get("odds") and len(d0.get("odds") or []) > 0) or (d0.get("pickcenter") and len(d0.get("pickcenter") or []) > 0):
                print("found", p.stem, "odds", len(d0.get("odds") or []), "pickcenter", len(d0.get("pickcenter") or []))
                args.event = p.stem
                break
        else:
            print("no odds found in first", checked, "files")
            return 3

    p = cache_dir / f"{args.event}.json"
    if not p.exists():
        print(f"missing: {p}")
        return 2

    d = json.loads(p.read_text(encoding="utf-8"))
    try:
        comps = (((d.get("header") or {}).get("competitions") or [None])[0]) or {}
        comps = comps if isinstance(comps, dict) else {}
        competitors = comps.get("competitors") or []
        if isinstance(competitors, list) and competitors:
            mini = []
            for c0 in competitors:
                if not isinstance(c0, dict):
                    continue
                team = c0.get("team") or {}
                tid = None
                if isinstance(team, dict):
                    tid = team.get("id")
                mini.append({"homeAway": c0.get("homeAway"), "team_id": tid})
            print("competitors", mini)
    except Exception:
        pass
    plays = d.get("plays") or []
    print("plays", len(plays))

    c = Counter()
    for play in plays:
        t = play.get("type") or {}
        tx = t.get("text") if isinstance(t, dict) else None
        tx = str(tx).strip() if tx else ""
        if tx:
            c[tx] += 1

    print("unique_types", len(c))
    for k, v in c.most_common(40):
        print(v, k)

    turn = next(
        (
            pl
            for pl in plays
            if "turnover" in str(((pl.get("type") or {}) if isinstance(pl, dict) else {}).get("text") or "").lower()
            or "turnover" in str((pl.get("text") if isinstance(pl, dict) else None) or "").lower()
        ),
        None,
    )
    reb = next(
        (
            pl
            for pl in plays
            if "rebound" in str(((pl.get("type") or {}) if isinstance(pl, dict) else {}).get("text") or "").lower()
            or "rebound" in str((pl.get("text") if isinstance(pl, dict) else None) or "").lower()
        ),
        None,
    )
    print("turnover_example", (turn.get("type") if isinstance(turn, dict) else None), "|", (turn.get("text") if isinstance(turn, dict) else None))
    print("rebound_example", (reb.get("type") if isinstance(reb, dict) else None), "|", (reb.get("text") if isinstance(reb, dict) else None))

    if isinstance(turn, dict):
        print("turnover_keys", sorted(list(turn.keys()))[:30])
        print("turnover_team", turn.get("team"))
    if isinstance(reb, dict):
        print("rebound_keys", sorted(list(reb.keys()))[:30])
        print("rebound_team", reb.get("team"))

    print("pickcenter", type(d.get("pickcenter")).__name__, len(d.get("pickcenter") or []))
    print("odds", type(d.get("odds")).__name__, len(d.get("odds") or []))

    if isinstance(d.get("pickcenter"), list) and d.get("pickcenter"):
        pc0 = d["pickcenter"][0]
        if isinstance(pc0, dict):
            print("pickcenter0_keys", sorted(list(pc0.keys())))
            for kk in [
                "provider",
                "details",
                "overUnder",
                "overUnderLine",
                "overUnderValue",
                "spread",
                "spreadLine",
                "spreadValue",
                "homeTeamOdds",
                "awayTeamOdds",
                "homeTeamSpread",
                "awayTeamSpread",
                "lastUpdated",
                "updateTime",
            ]:
                if kk in pc0:
                    print("pickcenter0", kk, pc0[kk])
        else:
            print("pickcenter0", type(pc0).__name__)

    if isinstance(d.get("odds"), list) and d.get("odds"):
        o0 = d["odds"][0]
        if isinstance(o0, dict):
            print("odds0_keys", sorted(list(o0.keys())))
            for kk in [
                "provider",
                "details",
                "overUnder",
                "overUnderLine",
                "overUnderValue",
                "overUnderSummary",
                "lastUpdated",
                "lastUpdate",
                "book",
                "name",
            ]:
                if kk in o0:
                    print("odds0", kk, o0[kk])
        else:
            print("odds0", type(o0).__name__)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
