from __future__ import annotations
import pandas as pd
import numpy as np

# Empirical Bayes style conference-level shrinkage.
# For each metric m: shrunk = (n/(n+alpha))*team_mean + (alpha/(n+alpha))*conf_mean
# where n is count of historical games contributing to team metric (approx via availability of value).

def apply_conference_shrinkage(df: pd.DataFrame, conf_col: str, metrics: list[str], alpha: float = 5.0) -> pd.DataFrame:
    if conf_col not in df.columns:
        return df
    out = df.copy()
    conf_groups = out.groupby(conf_col)
    for m in metrics:
        if m not in out.columns:
            continue
        vals = pd.to_numeric(out[m], errors='coerce')
        conf_mean = conf_groups[m].transform(lambda s: pd.to_numeric(s, errors='coerce').mean())
        # Approximate n: number of non-null metric entries for team historically (fallback to global mean if missing)
        # If team-level history not present, treat n=0 so we rely on conference mean.
        # Here we don't have explicit team counts; use presence of value (1) vs missing (0) as proxy.
        n = vals.notna().astype(float)
        shrunk = (n/(n+alpha))*vals + (alpha/(n+alpha))*conf_mean
        out[m + '_shrunk'] = shrunk.where(vals.notna(), conf_mean)
    return out

__all__ = ['apply_conference_shrinkage']
