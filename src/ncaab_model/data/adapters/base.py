from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable
from ..schemas import Game, Odds


class GamesAdapter(ABC):
    @abstractmethod
    def iter_games(self, season: int) -> Iterable[Game]:
        ...


class OddsAdapter(ABC):
    @abstractmethod
    def iter_odds(self, season: int) -> Iterable[Odds]:
        ...
