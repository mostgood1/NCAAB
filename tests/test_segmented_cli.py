import pandas as pd
import numpy as np
from pathlib import Path
from typer.testing import CliRunner

from ncaab_model.cli import app as typer_app

runner = CliRunner()


def _write_csv(tmp: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp / name
    df.to_csv(p, index=False)
    return p


def test_train_and_predict_segmented_team(tmp_path: Path):
    # Minimal synthetic feature set with targets so baseline & segmented can train
    rows = []
    teams = ["Alpha A", "Beta B", "Gamma C"]
    dates = pd.date_range("2025-11-01", periods=90, freq="D")
    rng = np.random.default_rng(0)
    for d in dates:
        home, away = rng.choice(teams, size=2, replace=False)
        rows.append({
            "game_id": f"{home[:2]}-{away[:2]}-{d.date().isoformat()}",
            "date": d.date().isoformat(),
            "home_team": home,
            "away_team": away,
            "home_off_rating": rng.uniform(95,115),
            "away_off_rating": rng.uniform(95,115),
            "home_def_rating": rng.uniform(95,115),
            "away_def_rating": rng.uniform(95,115),
            "home_tempo_rating": rng.uniform(65,75),
            "away_tempo_rating": rng.uniform(65,75),
            "tempo_rating_sum": rng.uniform(130,150),
            "target_total": rng.uniform(125,155),
            "target_margin": rng.uniform(-15,15),
        })
    feat_df = pd.DataFrame(rows)
    feats_csv = _write_csv(tmp_path, "features.csv", feat_df)

    # Train segmented team models (lower min_rows for test speed)
    models_dir = tmp_path / "models"
    result = runner.invoke(typer_app, [
        "train-segmented", "--features-csv", str(feats_csv), "--segment", "team", "--out-dir", str(models_dir), "--min-rows", "20"
    ])
    assert result.exit_code == 0, result.output
    # Expect JSONL model file present with >=1 line
    jsonl_path = models_dir / "segmented_team_models.jsonl"
    assert jsonl_path.exists(), "segmented models JSONL missing"
    lines = [ln for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) > 0, "No segmented models trained despite sufficient rows"

    # Predict segmented on a small slice
    predict_slice = feat_df.sample(10, random_state=1).reset_index(drop=True)
    slice_csv = _write_csv(tmp_path, "slice.csv", predict_slice)
    out_csv = tmp_path / "pred_segmented.csv"
    result2 = runner.invoke(typer_app, [
        "predict-segmented", "--features-csv", str(slice_csv), "--segment", "team", "--models-dir", str(models_dir), "--out", str(out_csv)
    ])
    assert result2.exit_code == 0, result2.output
    assert out_csv.exists(), "Output predictions file missing"
    pred_df = pd.read_csv(out_csv)
    # Columns existence checks
    for col in ["game_id", "pred_total", "pred_margin", "segmented_total", "segmented_margin", "blend_weight"]:
        assert col in pred_df.columns, f"Missing column {col} in predictions"
    # At least one row should have a segmented prediction value
    assert pred_df["segmented_total"].notna().any(), "No segmented predictions computed; check models loading"
    # Basic sanity: totals within plausible range
    assert pred_df["pred_total"].between(80, 250).all(), "Predicted totals out of expected bounds"
