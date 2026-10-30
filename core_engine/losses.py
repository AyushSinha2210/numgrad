# numgrad/losses.py
from typing import Tuple
import numpy as np


def _validate_shapes(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> int:
    """Enforces strict 2D column-vector geometry to prevent broadcasting bugs."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, D), got shape {X.shape}")
    N, D = X.shape
    if y.shape != (N, 1):
        raise ValueError(f"y must have shape ({N}, 1), got {y.shape}. Reshape with y.reshape(-1, 1).")
    if w.shape != (D, 1):
        raise ValueError(f"w must have shape ({D}, 1), got {w.shape}. Reshape with w.reshape(-1, 1).")
    return N


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid preventing exp() overflow on large negative values.
    For z >= 0: 1 / (1 + exp(-z))
    For z < 0:  exp(z) / (1 + exp(z))
    """
    out = np.empty_like(z, dtype=np.float64)
    pos_mask = (z >= 0)
    neg_mask = ~pos_mask

    out[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    exp_z = np.exp(z[neg_mask])
    out[neg_mask] = exp_z / (1.0 + exp_z)
    return out


def ridge_loss_and_grad(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, l2_reg: float = 0.1
) -> Tuple[float, np.ndarray]:
    """
    Computes Ridge Loss and Analytical Gradient:
    J(w) = (1 / 2N) * ||Xw - y||_2^2 + (lambda / 2) * ||w||_2^2
    grad = (1 / N) * X^T (Xw - y) + lambda * w
    """
    N = _validate_shapes(X, y, w)
    residuals = (X @ w) - y  # Shape: (N, 1)

    data_loss = 0.5 * np.mean(np.square(residuals))
    reg_loss = 0.5 * l2_reg * np.sum(np.square(w))
    total_loss = float(data_loss + reg_loss)

    grad = (1.0 / N) * (X.T @ residuals) + (l2_reg * w)
    return total_loss, grad


def bce_loss_and_grad(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, eps: float = 1e-15
) -> Tuple[float, np.ndarray]:
    """
    Computes Binary Cross-Entropy Loss and Analytical Gradient:
    J(w) = -(1 / N) * sum(y * log(p) + (1 - y) * log(1 - p))
    grad = (1 / N) * X^T (p - y)
    """
    N = _validate_shapes(X, y, w)
    logits = X @ w
    probs = stable_sigmoid(logits)

    # Clip probabilities slightly away from 0 and 1 to prevent log(0) -> -inf
    probs_clipped = np.clip(probs, eps, 1.0 - eps)

    loss = -float(np.mean(y * np.log(probs_clipped) + (1.0 - y) * np.log(1.0 - probs_clipped)))
    grad = (1.0 / N) * (X.T @ (probs - y))
    return loss, grad