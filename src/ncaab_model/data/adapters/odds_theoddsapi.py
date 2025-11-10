from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable
import requests

from ..schemas import Odds, OddsHistoryRow
from ...config import settings


class TheOddsAPIAdapter:
    """Minimal adapter for TheOddsAPI (https://the-odds-api.com/). Requires API key.

    Note: Be sure to review and comply with their terms of service and rate limits.
    """

    def __init__(self, api_key: str | None = None, region: str = "us", sport_key: str = "basketball_ncaab"):
        # Load and sanitize API key (trim whitespace and ignore anything after a '|')
        raw_key = (api_key or settings.theodds_api_key or "").strip()
        if "|" in raw_key:
            # Some dashboards copy the key with a trailing '|' segment; keep only the actual key
            raw_key = raw_key.split("|", 1)[0].strip()
        self.api_key = raw_key
        self.region = region
        self.sport_key = sport_key
        if not self.api_key:
            raise ValueError("TheOddsAPI key not set. Provide NCAAB_THEODDS_API_KEY or pass api_key.")

    def iter_odds(self, season: int) -> Iterable[Odds]:
        # Endpoint for NCAA Basketball odds (sport key may change; verify docs)
        url = f"https://api.the-odds-api.com/v4/sports/{self.sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": self.region,
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        now = datetime.now(timezone.utc)
        for event in r.json():
            game_id = str(event.get("id"))
            commence_raw = event.get("commence_time")
            if isinstance(commence_raw, str):
                # TheOddsAPI uses ISO 8601 with Z; normalize for fromisoformat
                commence = datetime.fromisoformat(commence_raw.replace("Z", "+00:00"))
            else:
                commence = None
            home_name = event.get("home_team") or None
            away_name = event.get("away_team") or None
            # Normalize names at source to maximize join coverage downstream
            try:
                from ..team_normalize import canonical_slug
                home_key = canonical_slug(home_name or "")
                away_key = canonical_slug(away_name or "")
            except Exception:
                home_key = None
                away_key = None
            for book in event.get("bookmakers", []):
                book_title = book.get("title", "unknown")
                moneyline_home = None
                moneyline_away = None
                spread = None
                total = None
                for market in book.get("markets", []):
                    key = market.get("key")
                    outcomes = market.get("outcomes", [])
                    if key == "h2h" and len(outcomes) >= 2:
                        try:
                            for oc in outcomes:
                                t = (oc.get("name") or oc.get("team"))
                                price = oc.get("price") or oc.get("odds")
                                if t and home_name and t == home_name and price is not None:
                                    moneyline_home = float(price)
                                elif t and away_name and t == away_name and price is not None:
                                    moneyline_away = float(price)
                        except Exception:
                            pass
                    elif key == "spreads" and len(outcomes) >= 2:
                        try:
                            home_out = None
                            for oc in outcomes:
                                if (oc.get("name") or oc.get("team")) == home_name:
                                    home_out = oc
                                    break
                            tgt = home_out or outcomes[0]
                            spread = float(tgt.get("point"))
                        except Exception:
                            pass
                    elif key == "totals" and len(outcomes) >= 1:
                        try:
                            total = float(outcomes[0].get("point"))
                        except Exception:
                            pass
                yield Odds(
                    game_id=game_id,
                    book=book_title,
                    fetched_at=now,
                    moneyline_home=moneyline_home,
                    moneyline_away=moneyline_away,
                    spread=spread,
                    total=total,
                    commence_time=commence,
                    home_team_name=home_name,
                    away_team_name=away_name,
                )

    # -------- Premium/history helpers (scaffold) --------
    def list_events_by_date(self, date_iso: str) -> list[dict]:
        """List NCAAB events for a given ISO date.

        Note: Endpoint path/params may vary by TheOddsAPI version/plan. This uses the common
        v4 pattern. Adjust if your plan differs.
        """
        url = f"https://api.the-odds-api.com/v4/sports/{self.sport_key}/events"
        params = {
            "apiKey": self.api_key,
            "dateFormat": "iso",
            "date": date_iso,
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def list_events_no_date(self) -> list[dict]:
        """List events without a date filter (provider returns upcoming events across dates).

        Useful to detect events missing from the date-filtered endpoint and then filter locally by
        commence_time.
        """
        url = f"https://api.the-odds-api.com/v4/sports/{self.sport_key}/events"
        params = {
            "apiKey": self.api_key,
            "dateFormat": "iso",
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _infer_period_from_key(market_key: str) -> str:
        key = (market_key or "").lower()
        if "1st" in key or "first_half" in key or "1h" in key or "_1st_half" in key:
            return "1h"
        if "2nd" in key or "second_half" in key or "2h" in key or "_2nd_half" in key:
            return "2h"
        return "full_game"

    def _normalize_market_rows(self, event: dict, book: dict, market: dict, fetched_at: datetime) -> list[OddsHistoryRow]:
        rows: list[OddsHistoryRow] = []
        event_id = str(event.get("id"))
        commence_raw = event.get("commence_time")
        commence = None
        if isinstance(commence_raw, str):
            commence = datetime.fromisoformat(commence_raw.replace("Z", "+00:00"))
        home = event.get("home_team") or None
        away = event.get("away_team") or None
        book_title = book.get("title", "unknown")
        mkey = (market.get("key") or "").lower()
        period = self._infer_period_from_key(mkey)
        last_update_raw = market.get("last_update") or book.get("last_update")
        last_update = None
        if isinstance(last_update_raw, str):
            try:
                last_update = datetime.fromisoformat(last_update_raw.replace("Z", "+00:00"))
            except Exception:
                last_update = None

        outcomes = market.get("outcomes", []) or []
        base = dict(
            event_id=event_id,
            book=book_title,
            fetched_at=fetched_at,
            last_update=last_update,
            commence_time=commence,
            home_team_name=home,
            away_team_name=away,
            market="",
            period=period,
        )
        if "h2h" in mkey:
            ml_home = None
            ml_away = None
            for oc in outcomes:
                name = oc.get("name") or oc.get("team")
                price = oc.get("price") or oc.get("odds")
                try:
                    if name == home and price is not None:
                        ml_home = float(price)
                    elif name == away and price is not None:
                        ml_away = float(price)
                except Exception:
                    pass
            rows.append(OddsHistoryRow(**{**base, "market": "h2h", "moneyline_home": ml_home, "moneyline_away": ml_away}))
        elif "spreads" in mkey:
            home_spread = home_price = away_spread = away_price = None
            for oc in outcomes:
                name = oc.get("name") or oc.get("team")
                try:
                    pt = oc.get("point")
                    px = oc.get("price") or oc.get("odds")
                    if name == home:
                        home_spread = float(pt) if pt is not None else home_spread
                        home_price = float(px) if px is not None else home_price
                    elif name == away:
                        away_spread = float(pt) if pt is not None else away_spread
                        away_price = float(px) if px is not None else away_price
                except Exception:
                    pass
            rows.append(OddsHistoryRow(**{
                **base,
                "market": "spreads",
                "home_spread": home_spread,
                "home_spread_price": home_price,
                "away_spread": away_spread,
                "away_spread_price": away_price,
            }))
        elif "totals" in mkey:
            total = over_px = under_px = None
            for oc in outcomes:
                try:
                    pt = oc.get("point")
                    name = (oc.get("name") or oc.get("label") or "").lower()
                    px = oc.get("price") or oc.get("odds")
                    if pt is not None:
                        total = float(pt)
                    if "over" in name and px is not None:
                        over_px = float(px)
                    if "under" in name and px is not None:
                        under_px = float(px)
                except Exception:
                    pass
            rows.append(OddsHistoryRow(**{
                **base,
                "market": "totals",
                "total": total,
                "over_price": over_px,
                "under_price": under_px,
            }))
        return rows

    def iter_current_odds_expanded(self, markets: str = "h2h,spreads,totals", date_iso: str | None = None) -> Iterable[OddsHistoryRow]:
        """Fetch current odds with expanded markets (including halves if your plan supports market keys).

        markets can include variants like spreads_1st_half, totals_1st_half, spreads_2nd_half, totals_2nd_half.
        """
        url = f"https://api.the-odds-api.com/v4/sports/{self.sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": self.region,
            "markets": markets,
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        if date_iso:
            params["date"] = date_iso

        def do_request(p):
            resp = requests.get(url, params=p, timeout=30)
            try:
                resp.raise_for_status()
                return resp
            except requests.HTTPError as e:
                status = getattr(resp, "status_code", None)
                if status in (400, 422):
                    p2 = dict(p)
                    p2.pop("date", None)
                    p2["markets"] = "h2h,spreads,totals"
                    resp2 = requests.get(url, params=p2, timeout=30)
                    resp2.raise_for_status()
                    return resp2
                raise

        r = do_request(params)
        now = datetime.now(timezone.utc)
        data = r.json() or []
        for event in data:
            for book in event.get("bookmakers", []) or []:
                for market in book.get("markets", []) or []:
                    for row in self._normalize_market_rows(event, book, market, now):
                        yield row

    def iter_odds_history_for_events(self, event_ids: list[str], markets: str = "h2h,spreads,totals") -> Iterable[OddsHistoryRow]:
        """Fetch odds-history for one or more events (per-event calls as per v4 docs).

        Correct endpoint shape: /v4/sports/{sport_key}/events/{event_id}/odds-history
        Note: Many plans do not support batching; we loop per event ID and yield normalized rows.
        """
        if not event_ids:
            return
        now = datetime.now(timezone.utc)
        base = f"https://api.the-odds-api.com/v4/sports/{self.sport_key}/events"
        for eid in event_ids:
            if not eid:
                continue
            url = f"{base}/{eid}/odds-history"
            params = {
                "apiKey": self.api_key,
                "regions": self.region,
                "markets": markets,
                "oddsFormat": "american",
                "dateFormat": "iso",
            }
            try:
                r = requests.get(url, params=params, timeout=45)
                r.raise_for_status()
                event = r.json() or {}
                # Some responses wrap in a list; normalize to dict
                if isinstance(event, list):
                    # pick first item if list provided
                    event = event[0] if event else {}
            except requests.HTTPError as e:
                # Gracefully skip events not available for history on this plan/date
                status = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
                if status in (401, 403, 404, 422):
                    continue
                raise
            except Exception:
                continue
            if not isinstance(event, dict) or not event:
                continue
            for book in event.get("bookmakers", []) or []:
                for market in book.get("markets", []) or []:
                    for row in self._normalize_market_rows(event, book, market, now):
                        yield row

    def try_alternate_sport_keys(self, keys: list[str]) -> dict[str, int]:
        """Probe alternate sport keys and report event counts per key.

        Returns a dict of {sport_key: event_count} for quick diagnostics.
        """
        out: dict[str, int] = {}
        for sk in keys:
            try:
                url = f"https://api.the-odds-api.com/v4/sports/{sk}/events"
                params = {"apiKey": self.api_key, "dateFormat": "iso"}
                r = requests.get(url, params=params, timeout=15)
                if r.status_code == 404:
                    out[sk] = -1
                    continue
                r.raise_for_status()
                events = r.json() or []
                out[sk] = len(events)
            except Exception:
                out[sk] = -1
        return out
