import pathlib
import pandas as pd

OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"

REQUIRED_COLS = {
    "team_slug","date","season_games","season_off_ppg","season_def_ppg","last5_off_ppg","rolling15_off_ppg","rest_days","ewm_off_ppg"
}

def test_team_features_basic():
    p = OUT / "team_features.csv"
    assert p.exists(), "team_features.csv missing"
    df = pd.read_csv(p)
    assert len(df) > 1000, f"unexpected small team_features rows: {len(df)}"
    missing = REQUIRED_COLS - set(df.columns)
    assert not missing, f"missing required columns: {missing}"
    # Null coverage check
    null_fracs = df[list(REQUIRED_COLS)].isna().mean().to_dict()
    high_null = {k:v for k,v in null_fracs.items() if v > 0.25}
    assert not high_null, f"high null fraction in columns: {high_null}"
