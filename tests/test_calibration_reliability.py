from __future__ import annotations

import pandas as pd
import numpy as np

from src.modeling.metrics import expected_calibration_error, reliability_curve


def test_reliability_curve_monotonic_bins():
    # Synthetic probabilities roughly calibrated
    y_prob = np.linspace(0.05, 0.95, 50)
    y_true = (y_prob > 0.5).astype(int)
    curve = reliability_curve(y_true, y_prob, n_bins=10)
    bins = curve['bins']
    assert len(bins) == 10
    # Confidence within bin should be in range
    for b in bins:
        if b['n'] > 0:
            assert b['conf'] >= b['bin_lo'] - 1e-6 and b['conf'] <= b['bin_hi'] + 1e-6


def test_ece_reduces_after_perfect_calibration():
    # Simulate uncalibrated probabilities then calibrate perfectly by mapping to empirical outcome
    rng = np.random.default_rng(42)
    raw_p = rng.uniform(0, 1, 400)
    # True outcome: threshold at 0.6 creates miscalibration
    y_true = (raw_p > 0.6).astype(int)
    ece_before = expected_calibration_error(y_true, raw_p)
    # Perfect calibration mapping: assign probability = empirical rate in fine bins
    bins = np.linspace(0, 1, 41)
    calibrated = np.zeros_like(raw_p)
    for i in range(len(bins) - 1):
        mask = (raw_p >= bins[i]) & (raw_p < bins[i+1]) if i < len(bins)-2 else (raw_p >= bins[i]) & (raw_p <= bins[i+1])
        if np.any(mask):
            calibrated[mask] = y_true[mask].mean()
    ece_after = expected_calibration_error(y_true, calibrated)
    assert ece_after <= ece_before + 1e-9
