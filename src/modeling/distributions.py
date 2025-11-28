import math
import numpy as np
import pandas as pd

# Lightweight advanced distribution helpers (Cornish-Fisher skew adjustment)
# Avoid heavy SciPy dependency; approximate skew-normal tail impact.
# Mean-median relationship: mean - median ≈ gamma1 * sigma / 6 for moderate skew.
# We approximate median by midpoint of q25 and q75.

SQRT2 = math.sqrt(2.0)

def estimate_skew(mean: pd.Series, q25: pd.Series, q75: pd.Series, sigma: pd.Series) -> pd.Series:
    mean = pd.to_numeric(mean, errors='coerce')
    q25 = pd.to_numeric(q25, errors='coerce')
    q75 = pd.to_numeric(q75, errors='coerce')
    sigma = pd.to_numeric(sigma, errors='coerce')
    mid = (q25 + q75) / 2.0
    skew = 6.0 * (mean - mid) / sigma.replace(0, np.nan)
    # Clip extreme values to maintain stability
    return skew.clip(lower=-2.5, upper=2.5)

def cornish_fisher_adjust_z(z: pd.Series, skew: pd.Series, kurtosis: pd.Series | None = None) -> pd.Series:
    """Cornish-Fisher expansion (first + second order) for skew and optional excess kurtosis.

    z_cf = z + (skew/6)*(z**2 - 1) + (k/24)*(z**3 - 3*z) - (skew**2/36)*(2*z**3 - 5*z)
    We clip extreme adjustments for stability.
    """
    z = pd.to_numeric(z, errors='coerce')
    skew = pd.to_numeric(skew, errors='coerce')
    if kurtosis is not None:
        kurtosis = pd.to_numeric(kurtosis, errors='coerce')
    else:
        kurtosis = pd.Series([0]*len(z))
    term1 = (skew/6.0)*(z**2 - 1.0)
    term2 = (kurtosis/24.0)*(z**3 - 3.0*z)
    term3 = (skew**2/36.0)*(2.0*z**3 - 5.0*z)
    out = z + term1 + term2 - term3
    return out.clip(lower=-10, upper=10)

def estimate_kurtosis(q25: pd.Series, q75: pd.Series, sigma: pd.Series) -> pd.Series:
    """Rough excess kurtosis estimate using IQR ratio vs normal.

    For normal: IQR = 1.34898 * sigma. Heavier tails inflate IQR relative to sigma? Actually heavy tails widen outer quantiles but may or may not widen IQR.
    We treat ratio r = IQR / (1.34898*sigma). If r < 1 implies lighter tails (negative excess kurtosis), r > 1 heavier tails.
    Map r to excess kurtosis via linear scaling capped.
    """
    q25 = pd.to_numeric(q25, errors='coerce')
    q75 = pd.to_numeric(q75, errors='coerce')
    sigma = pd.to_numeric(sigma, errors='coerce').replace(0, np.nan)
    iqr = q75 - q25
    r = iqr / (1.34898 * sigma)
    exk = (r - 1.0) * 3.0  # scale factor heuristic
    return exk.clip(lower=-1.5, upper=4.0)

def survival_from_cf(mean: pd.Series, sigma: pd.Series, skew: pd.Series, line: pd.Series, kurtosis: pd.Series | None = None) -> pd.Series:
    mean = pd.to_numeric(mean, errors='coerce')
    sigma = pd.to_numeric(sigma, errors='coerce').replace(0, np.nan)
    line = pd.to_numeric(line, errors='coerce')
    z = (line - mean) / sigma
    z_adj = cornish_fisher_adjust_z(z, skew, kurtosis)
    # Normal survival using adjusted z
    return 0.5 * (1 - z_adj.map(lambda x: math.erf(x / SQRT2) if pd.notna(x) else np.nan))

def mixture_survival(mean: pd.Series, sigma: pd.Series, skew: pd.Series, line: pd.Series) -> pd.Series:
    """Approximate survival probability using a skew-driven two-component normal mixture.

    We convert skew into a weight and component mean shift:
    - weight w = sigmoid(skew / 2) bounds influence (0,1)
    - component means: m1 = mean - d, m2 = mean + d where d = |skew| * sigma / 4
    - shared sigma. If skew<0 heavier left component, else heavier right.
    Survival = w * S(line | m2, sigma) + (1-w) * S(line | m1, sigma)
    Falls back to standard normal survival when skew is NaN/zero.
    """
    mean = pd.to_numeric(mean, errors='coerce')
    sigma = pd.to_numeric(sigma, errors='coerce').replace(0, np.nan)
    line = pd.to_numeric(line, errors='coerce')
    skew = pd.to_numeric(skew, errors='coerce')
    # Sigmoid weight from skew
    w = 1.0 / (1.0 + np.exp(-skew/2.0))
    d = skew.abs() * sigma / 4.0
    m1 = mean - d
    m2 = mean + d
    # Survival of normal: S = 0.5 * (1 - erf(z/sqrt2)) where z = (x - mu)/sigma
    def _surv(x, mu, s):
        z = (x - mu) / s
        return 0.5 * (1 - z.map(lambda val: math.erf(val / SQRT2) if pd.notna(val) else np.nan))
    S1 = _surv(line, m1, sigma)
    S2 = _surv(line, m2, sigma)
    # Blend respecting skew sign (weight leans toward tail consistent with skew direction)
    # If skew positive, emphasize right tail component (m2). If negative, invert weight.
    w_eff = w.where(skew >= 0, 1 - w)
    return w_eff * S2 + (1 - w_eff) * S1

__all__ = [
    'estimate_skew','cornish_fisher_adjust_z','estimate_kurtosis','survival_from_cf','mixture_survival'
]
