import pandas as pd
from pathlib import Path
from typer.testing import CliRunner
import numpy as np

from ncaab_model.cli import app as typer_app

runner = CliRunner()


def test_distributional_bankroll_sigma_scaling(tmp_path: Path):
    # Two totals rows: same mu-line difference, different sigma
    rows = [
        {
            'game_id': 'G1', 'event_id': 'E1', 'date': '2025-11-08', 'book': 'BookX', 'market': 'totals', 'period': 'full_game',
            'pred_total_mu': 152.0, 'pred_total_sigma': 6.0, 'total': 150.0, 'over_price': -110, 'under_price': -110
        },
        {
            'game_id': 'G2', 'event_id': 'E2', 'date': '2025-11-08', 'book': 'BookX', 'market': 'totals', 'period': 'full_game',
            'pred_total_mu': 152.0, 'pred_total_sigma': 18.0, 'total': 150.0, 'over_price': -110, 'under_price': -110
        }
    ]
    df = pd.DataFrame(rows)
    merged_csv = tmp_path / 'merged.csv'
    df.to_csv(merged_csv, index=False)
    out_csv = tmp_path / 'stakes.csv'
    result = runner.invoke(typer_app, [
        'bankroll-optimize', '--merged-csv', str(merged_csv), '--bankroll', '1000', '--kelly-fraction', '0.5', '--use-distributional', '--max-pct-per-bet', '0.10', '--out', str(out_csv)
    ])
    assert result.exit_code == 0, result.output
    stakes = pd.read_csv(out_csv)
    assert len(stakes) == 2, 'Expected two picks'
    # Lower sigma should produce larger scale => larger stake
    s_low = stakes.loc[stakes['game_id'] == 'G1', 'stake'].iloc[0]
    s_high = stakes.loc[stakes['game_id'] == 'G2', 'stake'].iloc[0]
    assert s_low > s_high, f"Expected stake for low sigma > high sigma, got {s_low} vs {s_high}"
    # Ensure uncertainty_scale column present for distributional picks
    assert 'uncertainty_scale' in stakes.columns
