from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Iterable

from ..schemas import Game, Odds


class CsvGamesAdapter:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def iter_games(self, season: int) -> Iterable[Game]:
        df = pd.read_csv(self.path)
        for _, r in df.iterrows():
            if int(r["season"]) != season:
                continue
            yield Game(**r.to_dict())


class CsvOddsAdapter:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def iter_odds(self, season: int) -> Iterable[Odds]:
        df = pd.read_csv(self.path)
        for _, r in df.iterrows():
            if int(r["season"]) != season:
                continue
            yield Odds(**r.to_dict())
