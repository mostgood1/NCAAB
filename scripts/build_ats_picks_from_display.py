import csv
import math
import sys
import requests
from datetime import datetime


def round_to_half(x: float) -> float:
    return math.copysign(round(abs(x) * 2.0) / 2.0, x)


def main(date_str: str):
    base = "https://ncaab.onrender.com"
    url = f"{base}/api/display_predictions?date={date_str}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows = data.get("rows") or data.get("data") or []
    out_rows = []
    for it in rows:
        gid = str(it.get("game_id") or "").strip()
        home = (it.get("home_team") or it.get("home_team_name") or "").strip()
        away = (it.get("away_team") or it.get("away_team_name") or "").strip()
        try:
            pm = float(it.get("pred_margin")) if it.get("pred_margin") is not None else float("nan")
        except Exception:
            pm = float("nan")
        try:
            sh = float(it.get("spread_home")) if it.get("spread_home") is not None else float("nan")
        except Exception:
            sh = float("nan")
        # Decide side: home if predicted margin >= 0 else away; default home on NaN
        side = "home"
        if not math.isnan(pm):
            side = "home" if pm >= 0 else "away"
        # Choose numeric spread for home perspective
        if math.isnan(sh):
            # Fall back to predicted margin converted to a 0.5 rounded spread from home perspective
            # If home favored (pm>=0), home line negative; else positive
            mag = round_to_half(abs(pm)) if not math.isnan(pm) else 0.0
            sh = -mag if (not math.isnan(pm) and pm >= 0) else mag
        out_rows.append({
            "game_id": gid,
            "home_team": home,
            "away_team": away,
            "ats_side": side,
            "closing_spread_home": sh,
            "spread_home": sh,
            "_pred_margin_blend": (pm if not math.isnan(pm) else "")
        })
    # Write CSV
    out_path = f"outputs/picks/ats_picks_{date_str}.csv"
    import os
    os.makedirs("outputs/picks", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "game_id","home_team","away_team","ats_side","closing_spread_home","spread_home","_pred_margin_blend"
        ])
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    print(f"wrote {len(out_rows)} rows -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        date_arg = sys.argv[1]
    else:
        date_arg = datetime.utcnow().strftime("%Y-%m-%d")
    main(date_arg)
