# numgrad/verifier.py
from typing import Callable, Dict, Any
import numpy as np


def compute_numerical_gradient(
    loss_fn: Callable[[np.ndarray], float],
    w: np.ndarray,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Computes numerical gradient via Central Finite Differences: O(eps^2) error.
    df/dw_j ≈ (f(w + eps*e_j) - f(w - eps*e_j)) / (2 * eps)
    """
    w_copy = w.astype(np.float64, copy=True)
    num_grad = np.zeros_like(w_copy, dtype=np.float64)

    # Iterate over each parameter coordinate
    for i in range(w_copy.shape[0]):
        orig_val = w_copy[i, 0]

        # f(w + eps)
        w_copy[i, 0] = orig_val + eps
        loss_plus = loss_fn(w_copy)

        # f(w - eps)
        w_copy[i, 0] = orig_val - eps
        loss_minus = loss_fn(w_copy)

        # Central difference formula
        num_grad[i, 0] = (loss_plus - loss_minus) / (2.0 * eps)

        # Restore exact original parameter value
        w_copy[i, 0] = orig_val

    return num_grad


def verify_gradients(
    analytical_grad: np.ndarray,
    numerical_grad: np.ndarray,
    tolerance: float = 1e-7
) -> Dict[str, Any]:
    """
    Computes relative L2 error between analytical and numerical gradients.
    """
    diff_norm = float(np.linalg.norm(numerical_grad - analytical_grad))
    num_norm = float(np.linalg.norm(numerical_grad))
    ana_norm = float(np.linalg.norm(analytical_grad))

    denominator = max(num_norm, ana_norm, 1e-15)
    rel_error = diff_norm / denominator
    passed = rel_error <= tolerance

    return {
        "passed": passed,
        "relative_error": rel_error,
        "tolerance": tolerance,
        "max_component_diff": float(np.max(np.abs(numerical_grad - analytical_grad))),
    }