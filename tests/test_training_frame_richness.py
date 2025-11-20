import pandas as pd
from src.modeling.data import build_training_frame

MIN_COLUMNS = 40


def test_training_frame_columns():
    X, y_total, y_margin, _dates = build_training_frame(return_dates=True)
    assert not X.empty, "Training frame empty"
    assert len(X.columns) >= MIN_COLUMNS, f"Too few columns: {len(X.columns)}"
    assert "diff_season_off_ppg" in X.columns, "Expected differential feature missing"
    assert y_total.notna().sum() == len(X), "Mismatch rows vs y_total"
    assert y_margin.notna().sum() == len(X), "Mismatch rows vs y_margin"
