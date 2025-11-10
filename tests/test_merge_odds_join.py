import pandas as pd
from ncaab_model.data.merge_odds import join_odds_to_games


def make_game(game_id, date, home, away):
    return {"game_id": str(game_id), "date": date, "home_team": home, "away_team": away}


def make_odds(date, home, away, book="BookX", total=140.5):
    return {"commence_time": f"{date}T20:00:00Z", "home_team_name": home, "away_team_name": away, "book": book, "market": "totals", "period": "full", "total": total}


def test_join_basic_alias_and_mascot_strip():
    # Games with alias and mascot differences
    games = pd.DataFrame([
        make_game(1, "2025-11-08", "UAlbany Great Danes", "Massachusetts Minutemen"),
        make_game(2, "2025-11-08", "Norfolk State Spartans", "William & Mary Tribe"),
        make_game(3, "2025-11-08", "San Jose State Spartans", "UC Santa Barbara Gauchos"),
    ])
    odds = pd.DataFrame([
        # Albany listed without leading U; ETSU style variant not used here
        make_odds("2025-11-08", "Albany Great Danes", "Massachusetts Minutemen", book="A"),
        # Norfolk abbreviated as St
        make_odds("2025-11-08", "Norfolk St Spartans", "William & Mary Tribe", book="B"),
        # San Jose St abbreviation; UC Santa Barbara identical
        make_odds("2025-11-08", "San José St Spartans", "UC Santa Barbara Gauchos", book="C"),
    ])

    merged = join_odds_to_games(games, odds, use_fuzzy=True, fuzzy_threshold=88, date_tolerance_days=1)
    # Each game should have at least one attached book after our normalization/fallbacks
    covered_by_game = merged.groupby("game_id")["book"].apply(lambda s: s.notna().any())
    assert covered_by_game.all(), f"Some games lacked odds attachment: {covered_by_game}"
