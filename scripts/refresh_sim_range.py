from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


def _daterange(start: str, end: str) -> list[str]:
    d0 = dt.date.fromisoformat(start)
    d1 = dt.date.fromisoformat(end)
    if d1 < d0:
        raise ValueError("end must be >= start")
    out: list[str] = []
    d = d0
    while d <= d1:
        out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def _run(cmd: list[str], cwd: Path) -> int:
    p = subprocess.run(cmd, cwd=str(cwd))
    return int(p.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh features + sim artifacts for a date range")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out-dir", default="outputs", help="Outputs directory")
    ap.add_argument("--skip-upload", action="store_true", help="Do not upload to Render")
    ap.add_argument("--base-url", default="https://ncaab.onrender.com", help="Render base URL")
    ap.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional sleep between days")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    dates = _daterange(args.start, args.end)

    results: dict[str, dict] = {}
    for date_str in dates:
        day_res: dict[str, object] = {"date": date_str}
        # 1) features
        rc = _run([py, str(repo / "src" / "modeling" / "gen_features_today.py"), "--date", date_str, "--write-dated"], repo)
        day_res["gen_features_rc"] = rc
        if rc != 0:
            results[date_str] = day_res
            continue

        # 2) sims
        rc = _run([py, str(repo / "scripts" / "run_game_simulations.py"), date_str, str(out_dir)], repo)
        day_res["run_sims_rc"] = rc
        if rc != 0:
            results[date_str] = day_res
            continue

        # 3) sim_blend
        rc = _run([py, str(repo / "scripts" / "blend_sim_quantiles.py"), date_str, str(out_dir)], repo)
        day_res["blend_rc"] = rc
        if rc != 0:
            results[date_str] = day_res
            continue

        # 4) validate (writes sim_inputs_diagnostic_<date>.json)
        rc = _run([py, str(repo / "scripts" / "validate_sim_inputs.py"), date_str, str(out_dir)], repo)
        day_res["validate_rc"] = rc

        # 5) upload
        if not args.skip_upload:
            ps = [
                "powershell.exe",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(repo / "scripts" / "upload_artifacts_to_render.ps1"),
                "-Date",
                date_str,
                "-BaseUrl",
                args.base_url,
            ]
            rc = _run(ps, repo)
            day_res["upload_rc"] = rc

        results[date_str] = day_res

        if args.sleep_seconds and args.sleep_seconds > 0:
            import time

            time.sleep(float(args.sleep_seconds))

    summary = {
        "start": args.start,
        "end": args.end,
        "count": len(dates),
        "results": results,
    }
    print(json.dumps(summary))

    # Nonzero if any day had a nonzero validate_rc (or earlier stage).
    bad = 0
    for d, r in results.items():
        for k in ("gen_features_rc", "run_sims_rc", "blend_rc", "validate_rc", "upload_rc"):
            if k in r and isinstance(r[k], int) and r[k] != 0:
                bad += 1
                break
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
