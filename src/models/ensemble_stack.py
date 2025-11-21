"""Ensemble stacking utilities.

Provides a simple linear/blended ensemble between calibrated model predictions
and derived feature estimates (tempo/off/def) plus market anchoring.

This is a lightweight placeholder; future expansion can incorporate out-of-fold
meta-learners (e.g. LightGBM) stored under model artifacts.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class EnsembleConfig:
    weight_calibrated: float = 0.55
    weight_raw_model: float = 0.25
    weight_derived: float = 0.15
    weight_market_anchor: float = 0.05  # small pull toward market total when present
    min_total: float = 110.0            # guardrail lower bound
    max_total: float = 190.0            # guardrail upper bound


def build_ensemble(df: pd.DataFrame, cfg: EnsembleConfig | None = None) -> pd.DataFrame:
    cfg = cfg or EnsembleConfig()
    work = df.copy()
    # Pull needed columns safely
    pt_cal = pd.to_numeric(work.get('pred_total_calibrated'), errors='coerce')
    pt_raw = pd.to_numeric(work.get('pred_total_model_raw') or work.get('pred_total_model'), errors='coerce')
    derived = pd.to_numeric(work.get('derived_total_est'), errors='coerce')
    market = pd.to_numeric(work.get('market_total'), errors='coerce')
    # Fallbacks
    base = pt_cal.where(pt_cal.notna(), pt_raw)
    comp_derived = derived.where(derived.notna(), base)
    comp_market = market.where(market.notna(), base)
    # Weighted blend
    blend = (
        cfg.weight_calibrated * base +
        cfg.weight_raw_model * pt_raw.where(pt_raw.notna(), base) +
        cfg.weight_derived * comp_derived +
        cfg.weight_market_anchor * comp_market
    )
    # Guardrails
    blend = blend.clip(lower=cfg.min_total, upper=cfg.max_total)
    work['pred_total_ensemble'] = blend
    # Margin ensemble: simple average of margin model + derived margin (if exists)
    pm_raw = pd.to_numeric(work.get('pred_margin_model'), errors='coerce')
    pm_der = pd.to_numeric(work.get('derived_margin_est'), errors='coerce')
    margin_blend = pm_raw.where(pm_raw.notna(), 0)
    if pm_der.notna().any():
        margin_blend = (0.7 * pm_raw.where(pm_raw.notna(), 0) + 0.3 * pm_der.where(pm_der.notna(), 0))
    work['pred_margin_ensemble'] = margin_blend
    return work

__all__ = ['EnsembleConfig','build_ensemble']
