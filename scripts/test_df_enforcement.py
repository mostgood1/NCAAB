import json
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path('outputs')
# Prefer today's unified enriched if exists
import datetime as dt
use_date = dt.datetime.utcnow().strftime('%Y-%m-%d')

candidates = [
    OUT / f'predictions_unified_enriched_{use_date}.csv',
    OUT / f'predictions_unified_{use_date}.csv',
]
src = None
for p in candidates:
    if p.exists():
        src = p
        break

if not src:
    print('no source file found')
    raise SystemExit(0)

df = pd.read_csv(src)
print(f'src={src} rows={len(df)} cols={list(df.columns)[:8]}..')

# Copy enforcement logic on a copy of df (df_tpl-like)
df_tpl = df.copy()

# 0) Neutralize exactly-zero calibrated margins
if 'pred_margin' in df_tpl.columns and 'pred_margin_basis' in df_tpl.columns:
    _pmv = pd.to_numeric(df_tpl['pred_margin'], errors='coerce')
    _pmb = df_tpl['pred_margin_basis'].astype(str)
    _mask_zero_cal = _pmb.eq('cal') & _pmv.fillna(0).abs().le(1e-6)
    if _mask_zero_cal.any():
        df_tpl.loc[_mask_zero_cal, 'pred_margin'] = np.nan
        df_tpl.loc[_mask_zero_cal, 'pred_margin_basis'] = None

pm = pd.to_numeric(df_tpl.get('pred_margin'), errors='coerce') if 'pred_margin' in df_tpl.columns else pd.Series([np.nan]*len(df_tpl))
basis_m = df_tpl.get('pred_margin_basis').astype(str) if 'pred_margin_basis' in df_tpl.columns else pd.Series(['']*len(df_tpl))
used = pm.notna() & (~pm.fillna(0).eq(0.0))

# 1) calibrated (non-zero only)
if 'pred_margin_calibrated' in df_tpl.columns:
    pm_cal = pd.to_numeric(df_tpl['pred_margin_calibrated'], errors='coerce')
    mask = (~used) & pm_cal.notna() & (~pm_cal.fillna(0).eq(0.0))
    pm[mask] = pm_cal[mask]
    basis_m = basis_m.where(~mask, 'cal')
    used = used | mask

# 2) model
if 'pred_margin_model' in df_tpl.columns:
    pm_mod = pd.to_numeric(df_tpl['pred_margin_model'], errors='coerce')
    mask = (~used) & pm_mod.notna()
    pm[mask] = pm_mod[mask]
    basis_m = basis_m.where(~mask, 'model')
    used = used | mask

# 3) blend
if 'pred_margin_blend' in df_tpl.columns:
    pm_blend = pd.to_numeric(df_tpl['pred_margin_blend'], errors='coerce')
    mask = (~used) & pm_blend.notna()
    pm[mask] = pm_blend[mask]
    basis_m = basis_m.where(~mask, 'blend')
    used = used | mask

# 4) segmentation
if 'pred_margin_seg' in df_tpl.columns:
    pm_seg = pd.to_numeric(df_tpl['pred_margin_seg'], errors='coerce')
    mask = (~used) & pm_seg.notna()
    pm[mask] = pm_seg[mask]
    basis_m = basis_m.where(~mask, 'seg')
    used = used | mask

# 5) reconstruct from edge + spread (prefer closing)
sh_series = None
if 'closing_spread_home' in df_tpl.columns:
    sh_series = pd.to_numeric(df_tpl['closing_spread_home'], errors='coerce')
elif 'spread_home' in df_tpl.columns:
    sh_series = pd.to_numeric(df_tpl['spread_home'], errors='coerce')
ea = pd.to_numeric(df_tpl['edge_ats'], errors='coerce') if 'edge_ats' in df_tpl.columns else None
if sh_series is not None and ea is not None:
    pm_rec = (-ea) - sh_series
    mask = ((~used) | pm.eq(0.0)) & pm_rec.notna()
    pm[mask] = pm_rec[mask]
    basis_m = basis_m.where(~mask, 'reconstructed_from_edge')
    used = used | mask

# 6) median/quantile fallback
cand = None
for k in ('pred_margin_p50','q50_margin','pred_margin_q50'):
    if k in df_tpl.columns:
        cand = pd.to_numeric(df_tpl[k], errors='coerce')
        break
if cand is not None:
    mask = ((~used) | pm.eq(0.0)) & cand.notna()
    pm[mask] = cand[mask]
    basis_m = basis_m.where(~mask, 'median_q50')
    used = used | mask

nz = int((pm.fillna(0)!=0).sum())
zeros = int((pm.fillna(0)==0).sum())
nan_ct = int(pm.isna().sum())

bc = pd.Series(basis_m).value_counts()
bc = {str(k): int(v) for k, v in bc.items()}

print(json.dumps({
    'rows': int(len(pm)),
    'non_zero': nz,
    'zeros': zeros,
    'nan': nan_ct,
    'basis_counts': bc,
}, indent=2))
