"""Analyze Live Lens signal logs (NDJSON) for side balance and missed-under diagnostics.

This script is intentionally lightweight and can be run ad-hoc:
  python scripts/analyze_live_lens_signals.py --date 2026-02-12
  python scripts/analyze_live_lens_signals.py --path outputs/live_lens_signals_2026-02-12.jsonl

It reports:
- Over/Under distribution for BET and WATCH (the UI recommendations)
- Over/Under distribution for candidate signals (e.g., under_below_watch)
- Candidate "miss distance" to WATCH threshold when metadata exists

Note: candidate fields (`is_candidate`, `thr_watch`) only exist in logs produced
after the corresponding frontend update.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


def _fnum(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return float(int(v))
        return float(v)
    except Exception:
        return None


def _norm_side(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"over", "o"}:
        return "over"
    if s in {"under", "u"}:
        return "under"
    return None


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return v != 0
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _iter_ndjson(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


@dataclass(frozen=True)
class ClassKey:
    cls: str  # BET/WATCH/CANDIDATE
    side: str  # over/under


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--path", type=str, help="Path to NDJSON log file")
    g.add_argument("--date", type=str, help="Slate date YYYY-MM-DD (reads outputs/live_lens_signals_<date>.jsonl)")
    ap.add_argument("--full-game-only", action="store_true", help="Filter to horizon>=39 (default: keep all)")
    args = ap.parse_args()

    path = Path(args.path) if args.path else Path("outputs") / f"live_lens_signals_{args.date}.jsonl"
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    rows = list(_iter_ndjson(path))
    if not rows:
        print(f"No valid rows parsed from {path}")
        return 0

    # Core tallies
    counts = Counter()
    by_kind = defaultdict(Counter)  # kind -> Counter(side)
    cand_kind = Counter()

    # Candidate distance diagnostics
    cand_miss_deltas: list[float] = []
    cand_edges: list[float] = []
    cand_remaining: list[float] = []

    # Metadata coverage
    meta_has_candidate = 0
    meta_has_thresholds = 0

    for r in rows:
        side = _norm_side(r.get("side"))
        if side is None:
            continue

        hz = _fnum(r.get("horizon"))
        if args.full_game_only and (hz is None or hz < 39):
            continue

        is_bet = _truthy(r.get("is_bet"))
        is_watch = _truthy(r.get("is_watch"))
        is_cand = _truthy(r.get("is_candidate"))

        if is_cand:
            meta_has_candidate += 1
        if r.get("thr") is not None or r.get("thr_watch") is not None:
            meta_has_thresholds += 1

        if is_bet:
            by_kind["BET"][side] += 1
            counts[ClassKey("BET", side)] += 1
        if is_watch:
            by_kind["WATCH"][side] += 1
            counts[ClassKey("WATCH", side)] += 1
        if is_cand:
            by_kind["CANDIDATE"][side] += 1
            counts[ClassKey("CANDIDATE", side)] += 1
            ck = r.get("candidate_kind")
            if ck:
                cand_kind[str(ck)] += 1

            # Miss distance to threshold when present
            strength = _fnum(r.get("strength"))
            thr_watch = _fnum(r.get("thr_watch"))
            if strength is not None and thr_watch is not None:
                cand_miss_deltas.append(thr_watch - strength)
            edge = _fnum(r.get("edge"))
            if edge is not None:
                cand_edges.append(edge)
            rem = _fnum(r.get("remaining"))
            if rem is not None:
                cand_remaining.append(rem)

    def _share(ctr: Counter, side_key: str) -> float:
        total = sum(ctr.values())
        return (ctr.get(side_key, 0) / total) if total else 0.0

    def _pct(x: float) -> str:
        return f"{100.0 * x:.1f}%"

    print(f"File: {path}  (rows parsed: {len(rows)})")

    for kind in ("BET", "WATCH", "CANDIDATE"):
        ctr = by_kind.get(kind, Counter())
        total = sum(ctr.values())
        if total == 0:
            print(f"{kind}: (none)")
            continue
        ov = ctr.get("over", 0)
        un = ctr.get("under", 0)
        print(f"{kind}: total {total} | over {ov} | under {un} | over share {_pct(_share(ctr, 'over'))}")

    if meta_has_candidate == 0:
        print("\nNote: no `is_candidate` rows found in this file.")
        print("      That usually means the log predates the under-candidate logging change.")

    if meta_has_thresholds == 0:
        print("Note: no threshold metadata (`thr`, `thr_watch`) found in this file.")

    if cand_kind:
        print("\nCandidate kinds:")
        for k, n in cand_kind.most_common():
            print(f"  {k}: {n}")

    if cand_miss_deltas:
        cand_miss_deltas.sort()
        p50 = cand_miss_deltas[len(cand_miss_deltas) // 2]
        p90 = cand_miss_deltas[int(0.9 * (len(cand_miss_deltas) - 1))]
        print("\nCandidate miss distance (thr_watch - strength):")
        print(f"  n={len(cand_miss_deltas)} | p50={p50:.2f} | p90={p90:.2f} | min={cand_miss_deltas[0]:.2f} | max={cand_miss_deltas[-1]:.2f}")

    if cand_edges:
        # Expect candidate under edges to be negative; print a quick summary.
        cand_edges.sort()
        p50e = cand_edges[len(cand_edges) // 2]
        print("\nCandidate edges (projBlend - live_line):")
        print(f"  n={len(cand_edges)} | p50={p50e:.2f} | min={cand_edges[0]:.2f} | max={cand_edges[-1]:.2f}")

    if cand_remaining:
        cand_remaining.sort()
        p50r = cand_remaining[len(cand_remaining) // 2]
        print("\nCandidate remaining minutes:")
        print(f"  n={len(cand_remaining)} | p50={p50r:.1f} | min={cand_remaining[0]:.1f} | max={cand_remaining[-1]:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
