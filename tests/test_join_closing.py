import pandas as pd

from ncaab_model.data.join_closing import join_games_with_closing


def test_join_closing_avoids_partial_for_known_alias_pairs():
    games = pd.DataFrame(
        [
            {
                "game_id": "401851395",
                "date": "2026-03-13",
                "home_team": "UC Irvine Anteaters",
                "away_team": "Cal State Northridge Matadors",
            },
            {
                "game_id": "401851414",
                "date": "2026-03-13",
                "home_team": "Sam Houston Bearkats",
                "away_team": "Kennesaw State Owls",
            },
        ]
    )

    closing = pd.DataFrame(
        [
            {
                "event_id": "fa9593dab05dbc9b57a9702b1d100ee6",
                "book": "BetMGM",
                "market": "totals",
                "period": "full_game",
                "commence_time": "2026-03-13T20:00:00Z",
                "home_team_name": "UC Irvine Anteaters",
                "away_team_name": "CSU Northridge Matadors",
                "total": 145.5,
            },
            {
                "event_id": "85e950ed2033a1e3e7c550552846f614",
                "book": "BetMGM",
                "market": "totals",
                "period": "full_game",
                "commence_time": "2026-03-13T20:00:00Z",
                "home_team_name": "Sam Houston St Bearkats",
                "away_team_name": "Kennesaw St Owls",
                "total": 136.5,
            },
            {
                "event_id": "cbe307cc93641e7198a5eefdbd935953",
                "book": "BetMGM",
                "market": "totals",
                "period": "full_game",
                "commence_time": "2026-03-13T20:00:00Z",
                "home_team_name": "Western Kentucky Hilltoppers",
                "away_team_name": "Kennesaw St Owls",
                "total": 149.5,
            },
        ]
    )

    merged = join_games_with_closing(games, closing, allow_partial=True)

    uc_irvine = merged[merged["game_id"] == "401851395"].copy()
    assert len(uc_irvine) == 1
    assert uc_irvine["event_id"].iloc[0] == "fa9593dab05dbc9b57a9702b1d100ee6"
    assert not bool(uc_irvine.get("partial_pair", pd.Series([False])).iloc[0])

    sam_houston = merged[merged["game_id"] == "401851414"].copy()
    assert len(sam_houston) == 1
    assert sam_houston["event_id"].iloc[0] == "85e950ed2033a1e3e7c550552846f614"
    assert not bool(sam_houston.get("partial_pair", pd.Series([False])).iloc[0])