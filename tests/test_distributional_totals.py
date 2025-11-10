import pandas as pd
import numpy as np
from pathlib import Path
from typer.testing import CliRunner

from ncaab_model.cli import app as typer_app

runner = CliRunner()


def _mk_features(n=200, rng=None):
    rng = rng or np.random.default_rng(42)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    # Construct targets with noise
    w = np.array([1.2, -0.5, 0.7, 0.0, 0.3], dtype=np.float32)
    mu = X @ w + 140.0
    sigma = 8.0 + 2.0 * np.abs(X[:, 0])  # heteroskedastic
    y = mu + rng.normal(scale=sigma)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["target_total"] = y
    # add required non-feature columns
    df["game_id"] = [f"G{i}" for i in range(n)]
    df["date"] = "2025-11-01"
    df["home_team"] = "A"
    df["away_team"] = "B"
    return df


def test_distributional_train_and_predict(tmp_path: Path):
    feats = _mk_features(n=300)
    feats_csv = tmp_path / "features.csv"
    feats.to_csv(feats_csv, index=False)

    models_dir = tmp_path / "models"
    r = runner.invoke(typer_app, [
        "train-distributional", str(feats_csv), "--out-dir", str(models_dir), "--alpha-mu", "0.5", "--alpha-sigma", "0.5", "--min-sigma", "5.0"
    ])
    assert r.exit_code == 0, r.output

    pred_out = tmp_path / "pred_dist.csv"
    r2 = runner.invoke(typer_app, [
        "predict-distributional", str(feats_csv), "--models-dir", str(models_dir), "--out", str(pred_out)
    ])
    assert r2.exit_code == 0, r2.output
    dfp = pd.read_csv(pred_out)
    assert {"pred_total_mu", "pred_total_sigma"}.issubset(dfp.columns)
    assert (dfp["pred_total_sigma"] > 0).all()
    # Sanity: average sigma should be within a reasonable band
    avg_sigma = dfp["pred_total_sigma"].mean()
    assert 5.0 <= avg_sigma <= 20.0
