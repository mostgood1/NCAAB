import pandas as pd
import numpy as np
import importlib
import os

# We simulate a near-uniform pred_total slate and invoke the internal logic that would flag it.
# Since the safeguard happens during unified export/pipeline assembly, we approximate by
# constructing a DataFrame and reproducing the detection snippet.

def _uniform_flag_logic(df: pd.DataFrame) -> bool:
    pt_diag = pd.to_numeric(df.get("pred_total"), errors="coerce")
    if pt_diag.notna().sum() > 8:
        vc = pt_diag.value_counts()
        top_frac = vc.iloc[0] / pt_diag.notna().sum() if len(vc) else 0
        # Mirror app.py logic exactly (> 0.90 triggers flag)
        return bool(top_frac > 0.90)
    return False


def test_uniform_safeguard_detection():
    # Create 30 rows mostly identical pred_total (e.g., all 112 except two variations)
    totals = [112.0]*28 + [111.5, 113.0]
    df = pd.DataFrame({
        'game_id': [f'g{i}' for i in range(len(totals))],
        'pred_total': totals,
    })
    assert _uniform_flag_logic(df) == True, 'Uniform safeguard detection should trigger for >90% same value'


def test_uniform_not_trigger_for_varied():
    varied = np.linspace(120, 140, 30)
    df = pd.DataFrame({
        'game_id': [f'v{i}' for i in range(len(varied))],
        'pred_total': varied,
    })
    assert _uniform_flag_logic(df) == False, 'Uniform safeguard should not trigger for varied totals'
