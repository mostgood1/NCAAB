import pandas as pd
from pathlib import Path
from ncaab_model.train.distributional import train_distributional_totals, predict_distributional_totals


def test_baseline_merge_feature(tmp_path: Path):
    # Create synthetic features with targets
    feats = pd.DataFrame({
        'game_id': ['g1','g2','g3','g4'],
        'date': ['2025-11-01']*4,
        'home_team': ['A','B','C','D'],
        'away_team': ['E','F','G','H'],
        'target_total': [150, 142, 158, 147],
        'target_margin': [5, -3, 10, -7],
        # minimal feature columns
        'home_pf5': [80,81,82,83],
        'away_pf5': [70,71,72,73],
        'home_pa5': [68,69,70,71],
        'away_pa5': [60,61,62,63],
    })
    feats_path = tmp_path / 'features.csv'
    feats.to_csv(feats_path, index=False)

    # Baseline predictions CSV
    baseline = pd.DataFrame({
        'game_id': ['g1','g2','g3','g4'],
        'pred_total': [149.0, 143.0, 157.5, 148.2],
        'pred_margin': [4.8, -2.7, 11.0, -6.5],
    })
    base_path = tmp_path / 'baseline.csv'
    baseline.to_csv(base_path, index=False)

    out_dir = tmp_path / 'models_dist'
    res = train_distributional_totals(
        feats_path,
        out_dir,
        alpha_mu=0.1,
        alpha_sigma=0.1,
        min_sigma=5.0,
        sigma_mode='log',
        baseline_preds_csv=base_path,
        baseline_pred_col='pred_total'
    )
    # Ensure baseline feature captured in feature_columns returned
    assert any('baseline_pred_total' == c or c.endswith('baseline_pred_total') for c in res['feature_columns']), 'baseline_pred_total not found in feature columns'

    # Predict with blending
    out_df = predict_distributional_totals(
        feats_path,
        out_dir,
        baseline_preds=baseline,
        blend_weight=0.3,
        global_shift=0.0,
        calibrate_to_baseline=False,
        calibration_max_ratio=2.0,
        sigma_cap=20.0,
    )
    assert {'pred_total_mu','pred_total_sigma'}.issubset(out_df.columns)
    # Check blended mu differs from raw mu values (via pred_total baseline influence) for at least one row
    assert (out_df['pred_total_mu'] != out_df['pred_total_mu_raw']).any(), 'Expected mu blending to modify at least one row'
