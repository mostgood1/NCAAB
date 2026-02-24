from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional
import re
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

    @staticmethod
    def _raise_for_status_no_leak(resp: requests.Response, context: str) -> None:
        """Raise a simple error without embedding request URLs/params.

        This avoids leaking apiKey in rich tracebacks that print locals/URLs.
        """
        if getattr(resp, "status_code", 0) and int(resp.status_code) >= 400:
            reason = getattr(resp, "reason", "")
            code = int(resp.status_code)
            raise RuntimeError(f"TheOddsAPI {context} failed (HTTP {code}{(' ' + str(reason)) if reason else ''})")

    def iter_odds(self, season: int) -> Iterable[Odds]:
        # Endpoint for NCAA Basketball odds (sport key may change; verify docs)
        url = f"https://api.the-odds-api.com/v4/sports/{self.sport_key}/odds"
        # Do not store apiKey in a local params dict (rich tracebacks can print locals).
        r = requests.get(
            url,
            params={
                "apiKey": self.api_key,
                "regions": self.region,
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
            },
            timeout=15,
        )
        self._raise_for_status_no_leak(r, "odds")
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

        Uses commence-time bounds (UTC) to avoid relying on undocumented `date` params.
        """
        url = f"https://api.the-odds-api.com/v4/sports/{self.sport_key}/events"
        # UTC day bounds (best-effort). The Odds API uses ISO 8601 timestamps with Z.
        # This is intentionally simple; callers can always override by using list_events_no_date
        # and filtering locally.
        commence_from = f"{date_iso}T00:00:00Z"
        commence_to = f"{date_iso}T23:59:59Z"
        r = requests.get(
            url,
            params={
                "apiKey": self.api_key,
                "dateFormat": "iso",
                "commenceTimeFrom": commence_from,
                "commenceTimeTo": commence_to,
            },
            timeout=20,
        )
        self._raise_for_status_no_leak(r, "events")
        return r.json()

    def list_events_no_date(self) -> list[dict]:
        """List events without a date filter (provider returns upcoming events across dates).

        Useful to detect events missing from the date-filtered endpoint and then filter locally by
        commence_time.
        """
        url = f"https://api.the-odds-api.com/v4/sports/{self.sport_key}/events"
        r = requests.get(url, params={"apiKey": self.api_key, "dateFormat": "iso"}, timeout=20)
        self._raise_for_status_no_leak(r, "events")
        return r.json()

    def get_event_odds(
        self,
        event_id: str,
        *,
        markets: str = "h2h,spreads,totals",
        bookmakers: str | None = None,
        odds_format: str = "american",
        date_format: str = "iso",
    ) -> dict:
        """Fetch odds for a single event via the event-level endpoint.

        This endpoint supports additional markets such as period markets (e.g. totals_h1).

        Docs: GET /v4/sports/{sport}/events/{eventId}/odds
        """
        eid = str(event_id or "").strip()
        if not eid:
            return {}
        url = f"https://api.the-odds-api.com/v4/sports/{self.sport_key}/events/{eid}/odds"
        params_no_key: dict[str, object] = {
            "regions": self.region,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }
        if bookmakers:
            params_no_key["bookmakers"] = bookmakers

        # Be resilient to plans/market coverage: some combinations return 422.
        r = requests.get(url, params={**params_no_key, "apiKey": self.api_key}, timeout=30)
        if r.status_code in (400, 422):
            r2 = requests.get(
                url,
                params={**{**params_no_key, "markets": "h2h,spreads,totals"}, "apiKey": self.api_key},
                timeout=30,
            )
            self._raise_for_status_no_leak(r2, "event_odds")
            data = r2.json() or {}
            return data if isinstance(data, dict) else (data[0] if isinstance(data, list) and data else {})
        self._raise_for_status_no_leak(r, "event_odds")
        data = r.json() or {}
        return data if isinstance(data, dict) else (data[0] if isinstance(data, list) and data else {})

    def get_event_odds_with_diag(
        self,
        event_id: str,
        *,
        markets: str = "h2h,spreads,totals",
        bookmakers: str | None = None,
        odds_format: str = "american",
        date_format: str = "iso",
    ) -> tuple[dict, dict]:
        """Fetch event odds and return (event_json, diag).

        This is a non-raising wrapper intended for debugging/telemetry.
        """
        eid = str(event_id or "").strip()
        diag: dict[str, object] = {
            "event_id": eid,
            "requested_markets": markets,
            "requested_bookmakers": bookmakers,
            "http_status": None,
            "used_markets": markets,
            "fallback_used": False,
            "error": None,
            "bookmakers_count": None,
            "market_keys": None,
        }
        if not eid:
            diag["error"] = "missing_event_id"
            return {}, diag

        url = f"https://api.the-odds-api.com/v4/sports/{self.sport_key}/events/{eid}/odds"
        params_no_key: dict[str, object] = {
            "regions": self.region,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }
        if bookmakers:
            params_no_key["bookmakers"] = bookmakers

        def _coerce_event(data: object) -> dict:
            if isinstance(data, dict):
                return data
            if isinstance(data, list) and data:
                return data[0] if isinstance(data[0], dict) else {}
            return {}

        try:
            r = requests.get(url, params={**params_no_key, "apiKey": self.api_key}, timeout=30)
            diag["http_status"] = getattr(r, "status_code", None)
            if r.status_code in (400, 422):
                # Retry with featured markets only.
                diag["fallback_used"] = True
                params2 = dict(params_no_key)
                params2["markets"] = "h2h,spreads,totals"
                diag["used_markets"] = params2["markets"]
                r2 = requests.get(url, params={**params2, "apiKey": self.api_key}, timeout=30)
                diag["http_status"] = getattr(r2, "status_code", None)
                self._raise_for_status_no_leak(r2, "event_odds")
                event = _coerce_event(r2.json() or {})
            else:
                self._raise_for_status_no_leak(r, "event_odds")
                event = _coerce_event(r.json() or {})

            try:
                bks = event.get("bookmakers", []) if isinstance(event, dict) else []
                if isinstance(bks, list):
                    diag["bookmakers_count"] = len(bks)
                    keys = []
                    for b in bks:
                        for m in (b or {}).get("markets", []) or []:
                            k = (m or {}).get("key")
                            if k:
                                keys.append(str(k))
                    if keys:
                        # De-dupe preserve order
                        seen = set()
                        diag["market_keys"] = [k for k in keys if not (k in seen or seen.add(k))][:80]
            except Exception:
                pass

            return event, diag
        except Exception as e:
            diag["error"] = str(e)
            return {}, diag

    def iter_event_odds(
        self,
        event_id: str,
        *,
        markets: str = "h2h,spreads,totals",
        bookmakers: str | None = None,
        diag: dict | None = None,
    ) -> Iterable[OddsHistoryRow]:
        """Yield normalized odds rows for a single event (event-level endpoint)."""
        now = datetime.now(timezone.utc)
        event = {}
        d = None
        try:
            event, d = self.get_event_odds_with_diag(event_id, markets=markets, bookmakers=bookmakers)
        except Exception:
            event, d = {}, {"error": "get_event_odds_with_diag_failed"}
        if isinstance(diag, dict) and isinstance(d, dict):
            try:
                diag.update(d)
            except Exception:
                pass
        if not isinstance(event, dict) or not event:
            return
        for book in event.get("bookmakers", []) or []:
            for market in book.get("markets", []) or []:
                for row in self._normalize_market_rows(event, book, market, now):
                    yield row

    @staticmethod
    def _infer_period_from_key(market_key: str) -> str:
        key = (market_key or "").lower().strip()

        # Prefer explicit half markers first.
        if any(tok in key for tok in ("_1st_half", "1st_half", "first_half", "_1h", "1h_")) or re.search(r"(^|_)h1($|_)", key):
            return "1h"
        if any(tok in key for tok in ("_2nd_half", "2nd_half", "second_half", "_2h", "2h_")) or re.search(r"(^|_)h2($|_)", key):
            return "2h"

        # Special-case common full-game market keys to avoid substring false positives.
        # Example: "h2h" contains "2h" but is a full-game moneyline market.
        if key in {"h2h", "spreads", "totals"}:
            return "full_game"

        # Conservative defaults.
        if "first_half" in key or "1st_half" in key or "1h" in key or re.search(r"(^|_)h1($|_)", key):
            return "1h"
        if "second_half" in key or "2nd_half" in key or "2h" in key or re.search(r"(^|_)h2($|_)", key):
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

    def iter_current_odds_expanded(
        self,
        markets: str = "h2h,spreads,totals",
        date_iso: str | None = None,
        bookmakers: str | None = None,
    ) -> Iterable[OddsHistoryRow]:
        """Fetch current odds with expanded markets (including halves if your plan supports market keys).

        markets can include variants like spreads_1st_half, totals_1st_half, spreads_2nd_half, totals_2nd_half.
        """
        url = f"https://api.the-odds-api.com/v4/sports/{self.sport_key}/odds"
        params_no_key: dict[str, object] = {
            "regions": self.region,
            "markets": markets,
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        if bookmakers:
            # Optional bookmaker filter. Expects comma-separated bookmaker keys (e.g. draftkings,fanduel,betmgm).
            params_no_key["bookmakers"] = bookmakers
        if date_iso:
            params_no_key["date"] = date_iso

        def do_request(p0: dict[str, object]) -> requests.Response:
            resp = requests.get(url, params={**p0, "apiKey": self.api_key}, timeout=30)
            if resp.status_code < 400:
                return resp
            status = int(resp.status_code)
            if status in (400, 422):
                # Some plans reject the `date` parameter. Retry without `date` but keep requested markets.
                p2 = dict(p0)
                p2.pop("date", None)
                resp2 = requests.get(url, params={**p2, "apiKey": self.api_key}, timeout=30)
                if resp2.status_code < 400:
                    return resp2
                # As a last resort, fall back to core markets only.
                p3 = dict(p2)
                p3["markets"] = "h2h,spreads,totals"
                resp3 = requests.get(url, params={**p3, "apiKey": self.api_key}, timeout=30)
                self._raise_for_status_no_leak(resp3, "odds")
                return resp3
            self._raise_for_status_no_leak(resp, "odds")
            return resp

        r = do_request(params_no_key)
        now = datetime.now(timezone.utc)
        data = r.json() or []
        for event in data:
            for book in event.get("bookmakers", []) or []:
                for market in book.get("markets", []) or []:
                    for row in self._normalize_market_rows(event, book, market, now):
                        yield row

    def iter_odds_history_for_events(
        self,
        event_ids: list[str],
        markets: str = "h2h,spreads,totals",
        bookmakers: str | None = None,
    ) -> Iterable[OddsHistoryRow]:
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
            params_no_key: dict[str, object] = {
                "regions": self.region,
                "markets": markets,
                "oddsFormat": "american",
                "dateFormat": "iso",
            }
            if bookmakers:
                params_no_key["bookmakers"] = bookmakers
            try:
                r = requests.get(url, params={**params_no_key, "apiKey": self.api_key}, timeout=45)
                event = r.json() or {}
                # Some responses wrap in a list; normalize to dict
                if isinstance(event, list):
                    # pick first item if list provided
                    event = event[0] if event else {}
                if r.status_code >= 400:
                    # Gracefully skip events not available for history on this plan/date
                    if int(r.status_code) in (401, 403, 404, 422):
                        continue
                    self._raise_for_status_no_leak(r, "odds_history")
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
                r = requests.get(url, params={"apiKey": self.api_key, "dateFormat": "iso"}, timeout=15)
                if r.status_code == 404:
                    out[sk] = -1
                    continue
                self._raise_for_status_no_leak(r, "events")
                events = r.json() or []
                out[sk] = len(events)
            except Exception:
                out[sk] = -1
        return out
