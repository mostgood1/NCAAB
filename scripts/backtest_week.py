from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _flatten_day_row(day: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"date": day.get("date"), "ok": day.get("ok", True)}

    comps = day.get("components") or {}
    all_up = day.get("all_up") or {}

    def get(d: dict[str, Any], path: list[str]) -> Any:
        cur: Any = d
        for k in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return cur

    for comp in ("model", "sim", "blend"):
        rec = comps.get(comp) or {}
        out[f"{comp}_tot_mae"] = get(rec, ["totals", "mae"])
        out[f"{comp}_tot_rmse"] = get(rec, ["totals", "rmse"])
        out[f"{comp}_tot_bias"] = get(rec, ["totals", "bias"])
        out[f"{comp}_tot_n"] = get(rec, ["totals", "n"])

        out[f"{comp}_mar_mae"] = get(rec, ["margins", "mae"])
        out[f"{comp}_mar_rmse"] = get(rec, ["margins", "rmse"])
        out[f"{comp}_mar_bias"] = get(rec, ["margins", "bias"])
        out[f"{comp}_mar_n"] = get(rec, ["margins", "n"])

        out[f"{comp}_win_acc"] = get(rec, ["winner", "acc"])
        out[f"{comp}_win_n"] = get(rec, ["winner", "n"])

    out["all_tot_mae"] = get(all_up, ["totals", "mae"])
    out["all_tot_n"] = get(all_up, ["totals", "n"])
    out["all_win_acc"] = get(all_up, ["winner", "acc"])
    out["all_win_n"] = get(all_up, ["winner", "n"])

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest last N days using /api/recap logic via Flask test_client")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--end", type=str, default="", help="YYYY-MM-DD (optional)")
    ap.add_argument("--out", type=str, default="", help="Write CSV output to this path")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    # Import Flask app from repo root
    import sys

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from app import app as flask_app  # type: ignore

    qs = [f"days={max(1, int(args.days))}"]
    if args.end:
        qs.append(f"end={args.end}")
    path = "/api/recap?" + "&".join(qs)

    with flask_app.test_client() as c:
        resp = c.get(path)
        if resp.status_code != 200:
            raise SystemExit(f"HTTP {resp.status_code}: {resp.get_data(as_text=True)[:500]}")
        payload: dict[str, Any] = resp.get_json(force=True)

    per_day = payload.get("per_day")
    if not isinstance(per_day, list):
        raise SystemExit("Bad payload: missing per_day")

    flat = [_flatten_day_row(d) for d in per_day]

    # Print a small console summary
    print("Window:", payload.get("meta"))
    print("Days:", len(flat))
    if flat:
        print("Latest day:", flat[-1])

    if args.out:
        out_path = (repo_root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in flat for k in r.keys()})
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in flat:
                w.writerow(r)
        print("Wrote:", str(out_path))


if __name__ == "__main__":
    main()
