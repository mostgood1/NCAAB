import json

import pandas as pd

import app as ncaab


def main(date: str = "2026-01-28") -> None:
    path = f"outputs/predictions_display_{date}.csv"
    df = pd.read_csv(path)
    rows = df.to_dict(orient="records")

    missing = 0
    missing_rows = []
    for r in rows:
        try:
            r = ncaab._backfill_start_fields(r)
        except Exception:
            pass
        iso = ncaab._derive_start_iso(r)
        if not iso:
            missing += 1
            missing_rows.append(
                {
                    "game_id": r.get("game_id"),
                    "home_team": r.get("home_team"),
                    "away_team": r.get("away_team"),
                    "display_date": r.get("display_date"),
                    "display_time_str": r.get("display_time_str"),
                    "start_time_display": r.get("start_time_display"),
                    "start_time_local": r.get("start_time_local"),
                    "start_time_iso": r.get("start_time_iso"),
                    "commence_time": r.get("commence_time"),
                    "start_time": r.get("start_time"),
                }
            )
    print("rows", len(rows), "missing_start_iso_after_derive", missing)
    if missing_rows:
        print("missing_start_iso_examples:")
        for ex in missing_rows[:10]:
            print(ex)
        # Try to see if enriched artifact has times for these matchups
        try:
            p_en = f"outputs/predictions_unified_enriched_{date}.csv"
            df_en = pd.read_csv(p_en, low_memory=False)
            cols = set(df_en.columns)
            time_cols = [c for c in [
                'start_time_iso','start_time','commence_time','start_time_local','start_tz_abbr',
                'display_date','display_time_str','start_time_display'
            ] if c in cols]
            if time_cols:
                print('enriched time cols', time_cols)
            for ex in missing_rows[:10]:
                hn = ncaab.normalize_name(str(ex.get('home_team') or ''))
                an = ncaab.normalize_name(str(ex.get('away_team') or ''))
                if not hn or not an:
                    continue
                # Best-effort lookup by normalized home/away
                df2 = df_en.copy()
                if 'home_team' in cols and 'away_team' in cols:
                    df2['_hn'] = df2['home_team'].astype(str).map(ncaab.normalize_name)
                    df2['_an'] = df2['away_team'].astype(str).map(ncaab.normalize_name)
                    hit = df2[(df2['_hn']==hn) & (df2['_an']==an)]
                    if hit.empty:
                        hit = df2[(df2['_hn']==an) & (df2['_an']==hn)]
                    if not hit.empty:
                        rec = hit.iloc[0].to_dict()
                        print('enriched match', ex.get('game_id'), {k: rec.get(k) for k in time_cols})
        except Exception as e:
            print('enriched lookup failed', type(e).__name__, str(e)[:200])

    ncaab._apply_site_display_global(rows)
    missing_pill = sum(
        1
        for r in rows
        if not (r.get("display_time_ampm") or r.get("display_time_str") or r.get("start_time_display"))
    )
    print("missing_any_time_display_fields", missing_pill)

    client = ncaab.app.test_client()

    # Smoke cards payload time coverage
    resp_cards = client.get(f"/api/display_predictions?date={date}&view=cards")
    print("cards status", resp_cards.status_code)
    js_cards = resp_cards.get_json()
    rows_cards = None
    if isinstance(js_cards, dict):
        if isinstance(js_cards.get("rows"), list):
            rows_cards = js_cards.get("rows")
        elif isinstance(js_cards.get("data"), list):
            rows_cards = js_cards.get("data")
    elif isinstance(js_cards, list):
        rows_cards = js_cards
    if not isinstance(rows_cards, list):
        print("cards json type", type(js_cards).__name__)
    else:
        def _is_missing(v) -> bool:
            if v is None:
                return True
            if isinstance(v, float) and pd.isna(v):
                return True
            s = str(v).strip().lower()
            return (not s) or (s in ("nan", "none", "null"))

        def _missing_time(r: dict) -> bool:
            return (
                _is_missing(r.get("start_time_iso"))
                and _is_missing(r.get("start_time_display"))
                and _is_missing(r.get("display_time_str"))
                and _is_missing(r.get("display_time_ampm"))
            )

        missing_cards = [r for r in rows_cards if isinstance(r, dict) and _missing_time(r)]
        print("cards rows", len(rows_cards), "missing_time_rows", len(missing_cards))

    resp = client.get(f"/api/recommendations?date={date}")
    print("recommendations status", resp.status_code)
    js = resp.get_json()
    print("recommendations json type", type(js).__name__)
    if isinstance(js, dict):
        print("recommendations keys", sorted(js.keys()))
        rec_rows = js.get("data") if isinstance(js.get("data"), list) else js.get("rows")
    else:
        rec_rows = None

    if not isinstance(rec_rows, list):
        print("recommendations rows is not a list; type=", type(rec_rows).__name__)
        print("recommendations json preview", json.dumps(js, indent=2)[:1000])
        return

    ats_count = sum(
        1
        for r in rec_rows
        if isinstance(r, dict)
        and ((r.get("rec_code") == "ATS") or (r.get("market") in ("spread", "spreads")))
    )
    print("recommendations rows", len(rec_rows), "ats_count", ats_count)


if __name__ == "__main__":
    main()
