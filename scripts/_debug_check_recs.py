from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from app import app as flask_app

    date = "2026-01-28"

    with flask_app.test_client() as c:
        resp = c.get(f"/api/recommendations?date={date}")
        if resp.status_code != 200:
            raise SystemExit(f"HTTP {resp.status_code}: {resp.get_data(as_text=True)[:500]}")
        payload: dict[str, Any] = resp.get_json(force=True)

    rows = payload.get("data")
    if not isinstance(rows, list):
        # Back-compat: some endpoints may return the list directly under `rows`
        rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    ats = [r for r in rows if str(r.get("rec_code", "")).upper() == "ATS"]

    def f(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    near_zero = sum(1 for r in ats if abs(f(r.get("line"))) < 0.05)
    missing_bet = sum(1 for r in ats if not r.get("bet"))

    print("ATS rows:", len(ats))
    print("ATS near-zero lines:", near_zero)
    print("ATS missing bet:", missing_bet)
    if ats:
        print("ATS sample:", {k: ats[0].get(k) for k in ["game_id", "home_team", "away_team", "bet", "line", "pred_margin", "edge"]})

    # Cards time-pill sanity check
    with flask_app.test_client() as c:
        resp = c.get(f"/api/display_predictions?date={date}&view=cards")
        if resp.status_code != 200:
            raise SystemExit(f"cards HTTP {resp.status_code}: {resp.get_data(as_text=True)[:500]}")
        payload2: dict[str, Any] = resp.get_json(force=True)

    rows2 = payload2.get("rows")
    if not isinstance(rows2, list):
        rows2 = payload2.get("data") if isinstance(payload2.get("data"), list) else []
    pills = [str(r.get("display_time_ampm") or r.get("display_time_str") or "") for r in rows2]
    pills = [p for p in pills if p]
    from collections import Counter
    cts = Counter(pills)
    top = cts.most_common(5)
    print("Cards rows:", len(rows2))
    print("Cards unique time pills:", len(cts))
    print("Cards top time pills:", top)

    noon_rows = [r for r in rows2 if str(r.get("display_time_ampm") or "") == "12:00 PM"]
    print("Cards 12:00 PM count:", len(noon_rows))
    if noon_rows:
        sample = noon_rows[0]
        # Show the time fields that influence display
        fields = [
            "game_id",
            "start_time_iso",
            "start_time",
            "start_time_local",
            "tz",
            "display_time_str",
            "display_time_ampm",
        ]
        print("Cards 12PM sample fields:", {k: sample.get(k) for k in fields if k in sample})

        # Compare against schedule artifact
        try:
            import pandas as pd
            from pathlib import Path

            gpath = Path(REPO_ROOT) / "outputs" / f"games_{date}.csv"
            if gpath.exists():
                gdf = pd.read_csv(gpath)
                if "game_id" in gdf.columns:
                    gdf["game_id"] = gdf["game_id"].astype(str).str.replace(r"\\.0$", "", regex=True)
                gmap = gdf.set_index("game_id").to_dict(orient="index") if "game_id" in gdf.columns else {}

                for rr in noon_rows[:5]:
                    gid = str(rr.get("game_id") or "")
                    gg = gmap.get(gid) or {}
                    print(
                        "NOON GAME",
                        gid,
                        "cards:",
                        rr.get("display_time_ampm"),
                        rr.get("start_time_iso"),
                        "| schedule:",
                        gg.get("start_time"),
                        gg.get("start_time_local"),
                    )
        except Exception as e:
            print("Schedule compare failed:", e)


if __name__ == "__main__":
    main()
