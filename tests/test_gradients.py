# tests/test_gradients.py
import unittest
import numpy as np
from core_engine.losses import ridge_loss_and_grad, bce_loss_and_grad
from core_engine.verifier import compute_numerical_gradient, verify_gradients


class TestGradients(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(100)
        self.X = self.rng.normal(size=(50, 8))
        self.w = self.rng.normal(size=(8, 1))

    def test_ridge_gradient_accuracy(self) -> None:
        y = self.rng.normal(size=(50, 1))
        _, ana = ridge_loss_and_grad(self.X, y, self.w, l2_reg=0.25)
        num = compute_numerical_gradient(lambda w_: ridge_loss_and_grad(self.X, y, w_, 0.25)[0], self.w)
        report = verify_gradients(ana, num, tolerance=1e-7)
        self.assertTrue(report["passed"], f"Ridge gradient check failed: {report}")

    def test_bce_gradient_accuracy(self) -> None:
        y = self.rng.integers(0, 2, size=(50, 1)).astype(np.float64)
        _, ana = bce_loss_and_grad(self.X, y, self.w)
        num = compute_numerical_gradient(lambda w_: bce_loss_and_grad(self.X, y, w_)[0], self.w)
        report = verify_gradients(ana, num, tolerance=1e-7)
        self.assertTrue(report["passed"], f"BCE gradient check failed: {report}")

    def test_broadcasting_shape_guard(self) -> None:
        # Passing a 1D target array y of shape (50,) instead of (50, 1) must raise ValueError
        y_bad = np.ones((50,))
        with self.assertRaises(ValueError):
            ridge_loss_and_grad(self.X, y_bad, self.w)


if __name__ == "__main__":
    unittest.main()