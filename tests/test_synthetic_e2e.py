import json, os, subprocess, sys, pathlib

# Invoke the synthetic harness via python -m ncaab_model.cli synthetic-e2e

def test_synthetic_e2e_artifacts(tmp_path):
    # Use today's date; harness writes into outputs/
    from datetime import datetime
    d_iso = datetime.utcnow().strftime('%Y-%m-%d')
    root = pathlib.Path(__file__).resolve().parent.parent
    outputs = root / 'outputs'
    cmd = [sys.executable, '-m', 'ncaab_model.cli', 'synthetic-e2e', '--date', d_iso, '--n-games', '5', '--seed', '7']
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    summary_path = outputs / f'synthetic_e2e_{d_iso}.json'
    assert summary_path.exists(), 'Missing summary JSON'
    data = json.loads(summary_path.read_text())
    for key in ['date','n_games','preds_path','enriched_path','cal_share_total','cal_share_margin']:
        assert key in data
    # Predictions file exists
    preds_path = pathlib.Path(data['preds_path'])
    assert preds_path.exists(), 'Predictions CSV missing'
    # Enriched path exists
    enriched_path = pathlib.Path(data['enriched_path'])
    assert enriched_path.exists(), 'Enriched CSV missing'
    # Stake sheet optional
    if data.get('stake_path'):
        assert pathlib.Path(data['stake_path']).exists(), 'Stake sheet missing'
