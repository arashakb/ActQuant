"""
HSIC (Hilbert-Schmidt Independence Criterion) utilities.

Implements kernels and HSIC estimators following HBaR (arXiv:2106.02734).
"""

import numpy as np


def distmat(X: np.ndarray) -> np.ndarray:
    """Pairwise squared distance matrix."""
    sq_norms = np.sum(X ** 2, axis=1)
    D = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
    return np.maximum(D, 0.0)


def linear_kernel(X: np.ndarray) -> np.ndarray:
    """Linear kernel: K = X @ X.T (no centering, matches HBaR)."""
    return X @ X.T


def rbf_kernel(X: np.ndarray, sigma: float = None) -> np.ndarray:
    """RBF kernel with dimension-scaled sigma (HBaR style).

    When sigma is provided:
        variance = 2 * sigma^2 * d
        K = exp(-D / variance)
    When sigma is None:
        Falls back to median heuristic matching HBaR's sigma_estimation:
        sigma_est = median of pairwise squared distances,
        variance  = 2 * sigma_est^2  (sigma_est is already a squared distance).
    """
    D = distmat(X)
    d = X.shape[1]

    if sigma is not None:
        variance = 2.0 * sigma * sigma * d
        K = np.exp(-D / variance)
    else:
        # Median heuristic: match HBaR sigma_estimation.
        # D contains squared distances; sigma_est = median(D), variance = 2 * sigma_est^2.
        n = X.shape[0]
        triu_idx = np.triu_indices(n, k=1)
        sigma_est = float(np.median(D[triu_idx]))
        if sigma_est <= 0:
            sigma_est = float(np.mean(D[triu_idx]))
        if sigma_est < 1e-2:
            sigma_est = 1e-2
        K = np.exp(-D / (2.0 * sigma_est * sigma_est))

    return K


def kernelmat(X: np.ndarray, sigma: float = None, k_type: str = "gaussian") -> np.ndarray:
    """Compute centered kernel matrix Kc = K @ H (matches HBaR kernelmat).

    Args:
        X: (n, d) array
        sigma: bandwidth parameter (None for median heuristic)
        k_type: "gaussian" or "linear"

    Returns:
        Kc: (n, n) centered kernel matrix (K @ H)
    """
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n

    if k_type == "gaussian":
        K = rbf_kernel(X, sigma=sigma)
    elif k_type == "linear":
        K = linear_kernel(X)
    else:
        raise ValueError(f"Unknown kernel type: {k_type}")

    return K @ H


def hsic_normalized_cca(X: np.ndarray, Y: np.ndarray,
                        sigma: float = 5.0,
                        k_type_x: str = "gaussian",
                        k_type_y: str = "linear") -> float:
    """CCA-normalized HSIC (matches HBaR hsic_normalized_cca).

    Computes:
        Kxc = kernelmat(X)
        Kyc = kernelmat(Y)
        Rx = Kxc @ inv(Kxc + epsilon*m*I)
        Ry = Kyc @ inv(Kyc + epsilon*m*I)
        HSIC_cca = sum(Rx * Ry.T)

    This is scale-invariant — handles high-dimensional kernels gracefully.
    """
    n = X.shape[0]
    Kxc = kernelmat(X, sigma=sigma, k_type=k_type_x)
    Kyc = kernelmat(Y, sigma=sigma, k_type=k_type_y)

    epsilon = 1e-5
    K_I = np.eye(n)
    Kxc_i = np.linalg.solve(Kxc + epsilon * n * K_I, np.eye(n))
    Kyc_i = np.linalg.solve(Kyc + epsilon * n * K_I, np.eye(n))

    Rx = Kxc @ Kxc_i
    Ry = Kyc @ Kyc_i

    return float(np.sum(Rx * Ry.T))


# Keep old functions for backward compatibility
def centering_matrix(n: int) -> np.ndarray:
    """Centering matrix H = I - (1/n)·11^T."""
    return np.eye(n) - np.ones((n, n)) / n


def hsic(K: np.ndarray, L: np.ndarray, H: np.ndarray) -> float:
    """Standard biased HSIC estimator: tr(KHLH) / (n-1)^2."""
    n = K.shape[0]
    HK = H @ K
    HL = H @ L
    return float(np.sum(HK * HL.T)) / (n - 1) ** 2


# ─── Efficient linear-kernel HSIC (scales to large N) ────────────────────────

def center(X: np.ndarray) -> np.ndarray:
    """Mean-center X along axis 0."""
    return X - X.mean(axis=0)


def hsic_linear(X_c: np.ndarray, Z_c: np.ndarray) -> float:
    """Biased HSIC for linear kernels, O(N·d_X·d_Z) — no N×N matrix formed.

    Equivalent to tr(K_X H K_Z H) / (n-1)² for K_X = X@X.T, K_Z = Z@Z.T.

    Proof:  tr(K_X H K_Z H) = tr(X̃ X̃.T Z̃ Z̃.T)
                             = tr(X̃.T Z̃ Z̃.T X̃)   [cyclic trace]
                             = ‖X̃.T @ Z̃‖_F²

    Args:
        X_c: (n, d_X) mean-centered array  [pre-center with center()]
        Z_c: (n, d_Z) mean-centered array  [pre-center with center()]

    Returns:
        scalar HSIC value
    """
    n = X_c.shape[0]
    C = X_c.T @ Z_c          # (d_X, d_Z)
    return float(np.sum(C * C)) / (n - 1) ** 2
