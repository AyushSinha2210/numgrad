# run_pipeline.py
import numpy as np
from core_engine.memory import inspect_memory, benchmark_contiguity
from core_engine.losses import ridge_loss_and_grad, bce_loss_and_grad
from core_engine.verifier import compute_numerical_gradient, verify_gradients
from core_engine.robust_eda import evaluate_poisoning_impact
from core_engine.visualizer import render_diagnostic_dashboard


def main() -> None:
    print("=" * 65)
    print("1. MEMORY LAYOUT & STRIDE BENCHMARK")
    print("=" * 65)
    sample = np.zeros((100, 50), dtype=np.float64)
    meta = inspect_memory(sample)
    print(f"Sample Array Metadata: {meta}")

    c_time, s_time, shared = benchmark_contiguity(size=6000)
    print(f"Zero-Copy Shared Address : {shared}")
    print(f"C-Contiguous Row Sum     : {c_time * 1000:.2f} ms")
    print(f"Transposed View Row Sum  : {s_time * 1000:.2f} ms ({s_time / c_time:.2f}x slower due to cache misses)\n")

    print("=" * 65)
    print("2. ANALYTICAL VS NUMERICAL GRADIENT VERIFICATION")
    print("=" * 65)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 10))
    w = rng.normal(size=(10, 1))
    y_reg = rng.normal(size=(200, 1))
    y_cls = rng.integers(0, 2, size=(200, 1)).astype(np.float64)

    # Check Ridge
    _, ana_ridge = ridge_loss_and_grad(X, y_reg, w, l2_reg=0.5)
    num_ridge = compute_numerical_gradient(lambda w_var: ridge_loss_and_grad(X, y_reg, w_var, 0.5)[0], w)
    res_ridge = verify_gradients(ana_ridge, num_ridge)
    print(f"Ridge Gradient Check     : Passed={res_ridge['passed']} | RelError={res_ridge['relative_error']:.3e}")

    # Check Logistic BCE
    _, ana_bce = bce_loss_and_grad(X, y_cls, w)
    num_bce = compute_numerical_gradient(lambda w_var: bce_loss_and_grad(X, y_cls, w_var)[0], w)
    res_bce = verify_gradients(ana_bce, num_bce)
    print(f"Logistic BCE Grad Check  : Passed={res_bce['passed']} | RelError={res_bce['relative_error']:.3e}\n")

    print("=" * 65)
    print("3. ADVERSARIAL POISONING: Z-SCORE VS ROBUST MAD")
    print("=" * 65)
    eda_res = evaluate_poisoning_impact()
    print(f"Total Real Anomalies Injected        : {eda_res['total_real_anomalies']}")
    print(f"Standard Z-Score (Clean -> Poisoned) : {eda_res['z_detected_clean']}/30 -> {eda_res['z_detected_poisoned']}/30 detected!")
    print(f"Robust MAD Score (Clean -> Poisoned) : {eda_res['mad_detected_clean']}/30 -> {eda_res['mad_detected_poisoned']}/30 detected!\n")

    render_diagnostic_dashboard(eda_res, ana_bce, num_bce, "eda_diagnostics.png")
    print("[+] Saved 4-panel diagnostic plot to 'eda_diagnostics.png'.")


if __name__ == "__main__":
    main()