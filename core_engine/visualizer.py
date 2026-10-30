# numgrad/visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from core_engine.losses import ridge_loss_and_grad


def render_diagnostic_dashboard(
    eda_results: dict,
    ana_grad: np.ndarray,
    num_grad: np.ndarray,
    output_path: str = "eda_diagnostics.png"
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 2: numgrad Verification & Adversarial EDA Diagnostics", fontsize=14, fontweight="bold")

    # Panel 1: Clean + Anomaly Distribution (excluding 10,000ms poison for visual scale)
    ax1 = axes[0, 0]
    non_poison = eda_results["poisoned_array"][:-5]
    ax1.hist(non_poison[:1000], bins=30, alpha=0.7, label="Normal Traffic (~50ms)", color="steelblue")
    ax1.hist(non_poison[1000:], bins=10, alpha=0.9, label="True Anomalies (~95ms)", color="crimson")
    ax1.set_title("1. Telemetry Distribution (Pre-Poisoning Scale)")
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Standard Z-Score vs Modified MAD Z-Score on True Anomalies
    ax2 = axes[0, 1]
    idx = np.arange(1000, 1030)
    ax2.scatter(idx, eda_results["z_scores_all"][1000:1030], color="orange", label="Standard Z-Score (Blinded)", s=45)
    ax2.scatter(idx, eda_results["mad_scores_all"][1000:1030], color="seagreen", label="MAD Modified Z-Score (Robust)", s=45)
    ax2.axhline(3.5, color="red", linestyle="--", label="Detection Threshold (3.5)")
    ax2.set_title("2. Poisoned Stream: Anomaly Detection Breakdown")
    ax2.set_xlabel("Sample Index (True Anomalies)")
    ax2.set_ylabel("Score Magnitude (Log Scale)")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: 2D Ridge Loss Contour & Gradient Descent Trajectory
    ax3 = axes[1, 0]
    rng = np.random.default_rng(0)
    X_2d = rng.normal(size=(100, 2))
    true_w = np.array([[2.5], [-1.8]])
    y_2d = X_2d @ true_w + rng.normal(scale=0.2, size=(100, 1))

    w0_vals = np.linspace(-1.0, 4.0, 60)
    w1_vals = np.linspace(-4.0, 1.0, 60)
    W0, W1 = np.meshgrid(w0_vals, w1_vals)
    Z_loss = np.zeros_like(W0)

    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            w_test = np.array([[W0[i, j]], [W1[i, j]]])
            Z_loss[i, j], _ = ridge_loss_and_grad(X_2d, y_2d, w_test, l2_reg=0.1)

    ax3.contourf(W0, W1, Z_loss, levels=25, cmap="viridis", alpha=0.8)
    # Run 15 steps of Gradient Descent to plot trajectory
    w_curr = np.array([[-0.5], [-3.5]])
    path = [w_curr.flatten()]
    for _ in range(15):
        _, g = ridge_loss_and_grad(X_2d, y_2d, w_curr, l2_reg=0.1)
        w_curr = w_curr - 0.25 * g
        path.append(w_curr.flatten())
    path_arr = np.array(path)
    ax3.plot(path_arr[:, 0], path_arr[:, 1], "w-o", markersize=4, label="GD Trajectory")
    ax3.set_title("3. 2D Ridge Loss Surface & Gradient Descent")
    ax3.set_xlabel("Weight w_0")
    ax3.set_ylabel("Weight w_1")
    ax3.legend()

    # Panel 4: Analytical vs Numerical Gradient Discrepancy
    ax4 = axes[1, 1]
    dims = np.arange(ana_grad.shape[0])
    abs_err = np.abs(ana_grad.flatten() - num_grad.flatten())
    ax4.bar(dims, abs_err, color="darkslateblue")
    ax4.set_title("4. Per-Coordinate Gradient Error |Grad_ana - Grad_num|")
    ax4.set_xlabel("Parameter Dimension Index")
    ax4.set_ylabel("Absolute Discrepancy")
    ax4.set_yscale("log")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()