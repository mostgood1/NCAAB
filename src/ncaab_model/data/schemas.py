from __future__ import annotations

from typing import Optional
from pydantic import BaseModel
from datetime import datetime


class Game(BaseModel):
    game_id: str
    season: int
    date: datetime
    start_time: Optional[datetime] = None  # tipoff/commence time when available
    # Localized schedule context (used for midnight UTC drift correction / display)
    start_time_local: Optional[str] = None  # 'YYYY-MM-DD HH:MM' in schedule timezone
    start_tz_abbr: Optional[str] = None     # e.g. CST, EST, HST
    home_team: str
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_score_1h: Optional[int] = None
    away_score_1h: Optional[int] = None
    home_score_2h: Optional[int] = None
    away_score_2h: Optional[int] = None
    neutral_site: Optional[bool] = None
    venue: Optional[str] = None


class Odds(BaseModel):
    game_id: str
    book: str
    fetched_at: datetime
    moneyline_home: Optional[float] = None
    moneyline_away: Optional[float] = None
    spread: Optional[float] = None  # home spread (negative = favorite)
    total: Optional[float] = None   # game total
    # Optional metadata for joining with game feeds
    commence_time: Optional[datetime] = None
    # Normalized team names used for joining with games
    home_team_name: Optional[str] = None
    away_team_name: Optional[str] = None


class OddsHistoryRow(BaseModel):
    """Normalized odds snapshot row for historical or current odds across markets and periods.

    This flattens TheOddsAPI bookmaker/market/outcomes objects into one row per
    (event, bookmaker, market, period) with key price/line fields.

    period: one of {"full_game", "1h", "2h"}
    market: one of {"h2h", "spreads", "totals"}
    """

    event_id: str
    book: str
    fetched_at: datetime  # when we fetched this snapshot
    last_update: Optional[datetime] = None  # provider's last update for this market

    # Event context
    commence_time: Optional[datetime] = None
    home_team_name: Optional[str] = None
    away_team_name: Optional[str] = None

    # Market + period
    market: str  # h2h|spreads|totals
    period: str  # full_game|1h|2h

    # Lines/prices
    # Moneyline prices
    moneyline_home: Optional[float] = None
    moneyline_away: Optional[float] = None

    # Spreads with prices (home/away points and juice)
    home_spread: Optional[float] = None
    home_spread_price: Optional[float] = None
    away_spread: Optional[float] = None
    away_spread_price: Optional[float] = None

    # Totals with over/under prices
    total: Optional[float] = None
    over_price: Optional[float] = None
    under_price: Optional[float] = None


class BoxScoreRow(BaseModel):
    """Minimal per-game summary derived from ESPN summary/box score.

    Includes possessions and four factors for home and away teams.
    """

    game_id: str
    date: datetime | None = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None

    # Final scores (if available from summary)
    home_score: Optional[float] = None
    away_score: Optional[float] = None

    # Possessions estimates
    home_possessions: Optional[float] = None
    away_possessions: Optional[float] = None
    pace: Optional[float] = None  # average possessions per team

    # Four factors (per team)
    home_efg: Optional[float] = None
    home_tov_rate: Optional[float] = None
    home_orb_rate: Optional[float] = None
    home_ftr: Optional[float] = None

    away_efg: Optional[float] = None
    away_tov_rate: Optional[float] = None
    away_orb_rate: Optional[float] = None
    away_ftr: Optional[float] = None

    # Halftime & second-half scores (optional; parsed from linescores if available)
    home_score_1h: Optional[float] = None
    away_score_1h: Optional[float] = None
    home_score_2h: Optional[float] = None
    away_score_2h: Optional[float] = None
