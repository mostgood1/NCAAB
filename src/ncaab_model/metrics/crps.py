import math
from typing import Iterable, Sequence, Tuple

import numpy as np


def _phi(z: np.ndarray) -> np.ndarray:
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * np.square(z))


def _Phi(z: np.ndarray) -> np.ndarray:
    # Standard normal CDF via erf
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


def erf(x: np.ndarray) -> np.ndarray:
    # Vectorized math.erf
    vec = np.vectorize(math.erf)
    return vec(x)


def gaussian_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """CRPS for a Gaussian predictive distribution.

    CRPS(N(mu, sigma^2); y) = sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],
    where z = (y - mu) / sigma.

    For very small sigma, falls back to |y - mu| (degenerate distribution limit).
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    sigma_safe = np.maximum(sigma, eps)
    z = (y - mu) / sigma_safe
    term = z * (2.0 * _Phi(z) - 1.0) + 2.0 * _phi(z) - (1.0 / math.sqrt(math.pi))
    crps = sigma_safe * term

    # Where sigma is extremely small, approximate with absolute error
    tiny = sigma < eps
    if np.any(tiny):
        crps[tiny] = np.abs(y[tiny] - mu[tiny])
    return crps


def pinball_loss(y: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    """Quantile (pinball) loss ρ_α(y - q)."""
    u = y - q
    return np.where(u >= 0.0, alpha * u, (alpha - 1.0) * u)


def crps_from_quantiles(
    y: np.ndarray,
    q_preds: Sequence[np.ndarray],
    q_levels: Sequence[float],
) -> np.ndarray:
    """Approximate CRPS via integrated quantile loss.

    Uses discrete approximation: CRPS ≈ 2 * Σ w_i * ρ_{α_i}(y - q_i),
    where w_i are trapezoidal weights based on quantile levels.
    """
    if len(q_preds) != len(q_levels):
        raise ValueError("q_preds and q_levels must have same length")

    # Sort by quantile level
    order = np.argsort(q_levels)
    alphas = np.asarray(q_levels, dtype=float)[order]
    q_sorted = [np.asarray(q_preds[i], dtype=float) for i in order]

    # Trapezoidal weights over [0,1]
    # For interior points: w_i = 0.5*(α_{i+1} - α_{i-1}); endpoints get half-interval
    m = len(alphas)
    w = np.empty(m, dtype=float)
    for i in range(m):
        if i == 0:
            w[i] = 0.5 * (alphas[i + 1] - 0.0)
        elif i == m - 1:
            w[i] = 0.5 * (1.0 - alphas[i - 1])
        else:
            w[i] = 0.5 * (alphas[i + 1] - alphas[i - 1])

    y_arr = np.asarray(y, dtype=float)
    total = np.zeros_like(y_arr, dtype=float)
    for qi, ai, wi in zip(q_sorted, alphas, w):
        total += wi * pinball_loss(y_arr, qi, ai)
    return 2.0 * total


def choose_truth_column(df, candidates: Iterable[str]) -> Tuple[str, np.ndarray]:
    for c in candidates:
        if c in df.columns:
            return c, df[c].to_numpy()
    raise KeyError(f"None of the candidate columns found: {candidates}")
