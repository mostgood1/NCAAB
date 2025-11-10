import pandas as pd
from pathlib import Path
from typer.testing import CliRunner
import numpy as np

from ncaab_model.cli import app as typer_app

runner = CliRunner()


def test_bankroll_optimize_basic(tmp_path: Path):
    # Synthetic merged odds + predictions rows
    rows = []
    for i in range(10):
        rows.append({
            'game_id': f'G{i}',
            'event_id': f'E{i}',
            'date': '2025-11-08',
            'book': 'BookX',
            'market': 'totals',
            'period': 'full_game',
            'pred_total': 150 + i,  # increasing predicted totals
            'total': 148 + i,       # market 2 points lower
            'pred_margin': 5.0,
            'home_spread': -4.5,
            'home_spread_price': -110,
            'away_spread': 4.5,
            'away_spread_price': -110,
            'moneyline_home': -180,
            'moneyline_away': 160,
        })
    df = pd.DataFrame(rows)
    merged_csv = tmp_path / 'merged.csv'
    df.to_csv(merged_csv, index=False)

    out_csv = tmp_path / 'stakes.csv'
    # Do not pass '--use-distributional' flag (absence => False) to keep legacy edge path
    result = runner.invoke(typer_app, [
        'bankroll-optimize', '--merged-csv', str(merged_csv), '--bankroll', '500', '--kelly-fraction', '0.5', '--min-edge-total', '0.5', '--max-pct-per-bet', '0.05', '--out', str(out_csv)
    ])
    assert result.exit_code == 0, result.output
    assert out_csv.exists(), 'Stake sheet not created'
    stakes = pd.read_csv(out_csv)
    assert not stakes.empty, 'Stake sheet unexpectedly empty'
    # All selections should have stake <= max_pct_per_bet * bankroll (25)
    assert stakes['stake'].max() <= 25 + 1e-6
    # Totals picks chosen should be over (edge positive)
    assert (stakes['selection'] == 'over').all()
    # Stakes scaled by fractional Kelly so none should be zero
    assert (stakes['stake'] > 0).all()
