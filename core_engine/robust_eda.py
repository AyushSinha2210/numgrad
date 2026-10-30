# numgrad/robust_eda.py
from typing import Dict, Tuple
import numpy as np


def generate_telemetry_with_poisoning(
    n_normal: int = 1000,
    n_true_anomalies: int = 30,
    n_poison: int = 5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic network latency telemetry (ms):
    - Normal traffic: ~ N(50ms, 5ms)
    - True anomalies (e.g. data exfiltration spikes): ~ N(95ms, 4ms)
    - Poisoning attack (extreme spikes injected to blind standard deviation): ~ 10,000ms
    Returns:
        clean_stream (normal + anomalies), poisoned_stream (normal + anomalies + poison)
    """
    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=50.0, scale=5.0, size=n_normal)
    anomalies = rng.normal(loc=95.0, scale=4.0, size=n_true_anomalies)
    poison = np.full(shape=n_poison, fill_value=10000.0, dtype=np.float64)

    clean_stream = np.concatenate([normal, anomalies])
    poisoned_stream = np.concatenate([normal, anomalies, poison])
    return clean_stream, poisoned_stream


def standard_z_score(x: np.ndarray) -> np.ndarray:
    """Classical Z-score: vulnerable to mean/variance shifting."""
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    return np.abs((x - mean) / np.maximum(std, 1e-15))


def modified_z_score_mad(x: np.ndarray) -> np.ndarray:
    """
    Robust Modified Z-score using Median Absolute Deviation (MAD).
    M_i = 0.6745 * |x_i - median(x)| / MAD
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (0.6745 * np.abs(x - med)) / np.maximum(mad, 1e-15)


def evaluate_poisoning_impact(
    n_normal: int = 1000, n_anomalies: int = 30, threshold: float = 3.5
) -> Dict[str, Any]:
    """
    Quantifies how many true anomalies are detected before and after poisoning.
    """
    clean, poisoned = generate_telemetry_with_poisoning(n_normal, n_anomalies)

    # Slice indices corresponding strictly to the 30 real anomalies
    anomaly_slice = slice(n_normal, n_normal + n_anomalies)

    # Before poisoning
    z_clean = standard_z_score(clean)[anomaly_slice]
    mad_clean = modified_z_score_mad(clean)[anomaly_slice]

    # After poisoning
    z_poisoned = standard_z_score(poisoned)[anomaly_slice]
    mad_poisoned = modified_z_score_mad(poisoned)[anomaly_slice]

    return {
        "total_real_anomalies": n_anomalies,
        "z_detected_clean": int(np.sum(z_clean > threshold)),
        "z_detected_poisoned": int(np.sum(z_poisoned > threshold)),
        "mad_detected_clean": int(np.sum(mad_clean > threshold)),
        "mad_detected_poisoned": int(np.sum(mad_poisoned > threshold)),
        "poisoned_array": poisoned,
        "z_scores_all": standard_z_score(poisoned),
        "mad_scores_all": modified_z_score_mad(poisoned),
    }