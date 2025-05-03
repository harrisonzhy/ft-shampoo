from fractions import Fraction

import torch
import unittest

from matrix_functions import (
    check_diagonal,
    matrix_eigendecomposition,
    matrix_inverse_root,
    compute_matrix_root_inverse_residuals,
    NewtonConvergenceFlag,
    _estimated_eigenvalues_criterion_below_or_equal_tolerance,
    CoupledHigherOrderConfig,
    CoupledNewtonConfig, 
    DefaultEigenConfig
)
from matrix_functions_types import EighEigendecompositionConfig, QREigendecompositionConfig
from commons import get_all_subclasses
import re

class CheckDiagonalTests(unittest.TestCase):
    def test_strictly_diagonal(self):
        A = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(check_diagonal(A))

    def test_not_diagonal(self):
        A = torch.tensor([[1.0, 0.0, 5.0], [0.0, 2.0, 0.0], [5.0, 0.0, 3.0]])
        self.assertFalse(check_diagonal(A))

    def test_not_two_dimensional(self):
        with self.assertRaisesRegex(ValueError, r"Matrix is not 2-dimensional"):
            check_diagonal(torch.randn(2, 2, 2))

    def test_not_square(self):
        with self.assertRaisesRegex(ValueError, r"Matrix is not square"):
            check_diagonal(torch.randn(2, 3))


class EstimatedEigenvaluesCriterionTests(unittest.TestCase):
    def test_exactly_zero_offdiag(self):
        B = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
        # off-diagonals are zero ⇒ criterion ≤ any non-negative tol
        self.assertTrue(_estimated_eigenvalues_criterion_below_or_equal_tolerance(B, 0.0))

    def test_offdiag_exceeds_tolerance(self):
        B = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        # off-diagonal norm > tol * total norm when tol small
        self.assertFalse(_estimated_eigenvalues_criterion_below_or_equal_tolerance(B, 0.1))


class MatrixEigendecompositionTests(unittest.TestCase):
    def test_scalar_input(self):
        x = torch.tensor(7.0)
        vals, vecs = matrix_eigendecomposition(x)
        self.assertEqual(vals, x)
        self.assertTrue(torch.equal(vecs, torch.ones_like(x)))

    def test_diagonal_flag(self):
        A = torch.diag(torch.tensor([3.0, 4.0, 5.0]))
        vals, vecs = matrix_eigendecomposition(A, is_diagonal=True)
        self.assertTrue(torch.equal(vals, torch.tensor([3.0, 4.0, 5.0])))
        self.assertTrue(torch.equal(vecs, torch.eye(3)))

    def test_standard_eigh_vs_torch(self):
        A = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
        vals, vecs = matrix_eigendecomposition(A, eigendecomposition_config=EighEigendecompositionConfig())
        tv, tq = torch.linalg.eigh(A)
        # eigenvalues
        self.assertTrue(torch.allclose(vals, tv))
        # eigenvectors up to sign
        self.assertTrue(torch.allclose(vecs.abs(), tq.abs()))

class MatrixInverseRootTests(unittest.TestCase):
    def test_scalar_root(self):
        A = torch.tensor(16.0)
        # 1/root = -1/2  => (A)**(-1/2) = 1/4
        X = matrix_inverse_root(A, Fraction(2, 1))
        self.assertAlmostEqual(X.item(), 1 / 4)

    def test_non_square_raises(self):
        with self.assertRaisesRegex(ValueError, r"Matrix is not square!"):
            matrix_inverse_root(torch.randn(2, 3), Fraction(2, 1))

    def test_diagonal_backend(self):
        A = torch.diag(torch.tensor([1.0, 4.0, 9.0]))
        X = matrix_inverse_root(A, Fraction(2, 1), is_diagonal=True)
        expected = torch.diag(torch.tensor([1.0, 0.5, 1/3]))
        self.assertTrue(torch.allclose(X, expected))

    def test_eigendecomposition_backend(self):
        A = torch.tensor([[5.0, 2.0], [2.0, 5.0]])
        X = matrix_inverse_root(A, Fraction(2, 1), root_inv_config=DefaultEigenConfig)
        # Check A @ X @ X ≈ I
        I_approx = A @ X @ X
        self.assertTrue(torch.allclose(I_approx, torch.eye(2), atol=1e-5))

    def test_coupled_newton_backend_converges_or_times_out(self):
        A = torch.tensor([[5.0, 2.0], [2.0, 5.0]])
        cfg = CoupledNewtonConfig(max_iterations=50, tolerance=1e-6)
        X = matrix_inverse_root(A, Fraction(2, 1), root_inv_config=cfg)
        # should still return a tensor
        self.assertIsInstance(X, torch.Tensor)


class ComputeResidualsTests(unittest.TestCase):
    def test_only_eigenconfig_supported(self):
        # CoupledNewtonConfig should trigger assertion in compute_matrix_root_inverse_residuals
        A = torch.eye(2)
        X_hat = torch.eye(2)
        with self.assertRaises(AssertionError):
            compute_matrix_root_inverse_residuals(A, X_hat, Fraction(2, 1), 0.0, root_inv_config=CoupledNewtonConfig())

    def test_shape_mismatch(self):
        A = torch.eye(2)
        X_hat = torch.eye(3)
        with self.assertRaisesRegex(ValueError, r"Matrix shapes do not match"):
            compute_matrix_root_inverse_residuals(A, X_hat, Fraction(2, 1), 0.0, root_inv_config=DefaultEigenConfig)

    def test_residuals_behavior(self):
        # For A=diag([4,9]) the relative_error is tiny for the exact inverse-root,
        # and the relative_residual is a finite Tensor > 0.
        A = torch.diag(torch.tensor([4.0, 9.0]))
        X = matrix_inverse_root(A, Fraction(2, 1), root_inv_config=DefaultEigenConfig)

        rel_err, rel_res = compute_matrix_root_inverse_residuals(
            A, X, Fraction(2, 1), 0.0, root_inv_config=DefaultEigenConfig
        )

        self.assertLess(rel_err, 1e-6)
        self.assertIsInstance(rel_res, torch.Tensor)
        self.assertGreater(rel_res, 0.0)


class NewtonFlagEnumTests(unittest.TestCase):
    def test_enum_members(self):
        members = {flag.name for flag in NewtonConvergenceFlag}
        self.assertEqual(members, {"REACHED_MAX_ITERS", "CONVERGED", "EARLY_STOP"})



class QREigendecompositionConfigSubclassesTest(unittest.TestCase):
    def test_illegal_tolerance(self) -> None:
        for cls in get_all_subclasses(QREigendecompositionConfig):
            # tolerance has to be in the interval [0.0, 1.0].
            for tolerance in [-1.0, 1.1]:
                with self.subTest(cls=cls):
                    self.assertRaisesRegex(
                        ValueError,
                        re.escape(
                            f"Invalid tolerance value: {tolerance}. Must be in the interval [0.0, 1.0]."
                        ),
                        cls,
                        tolerance=tolerance,
                    )

    def test_illegal_eigenvectors_estimate(self) -> None:
        for cls in get_all_subclasses(QREigendecompositionConfig):
            with self.subTest(cls=cls):
                self.assertRaisesRegex(
                    TypeError,
                    re.escape(
                        f"{cls.__name__}.__init__() got an unexpected keyword argument 'eigenvectors_estimate'"
                    ),
                    cls,
                    eigenvectors_estimate=torch.eye(3),
                )

class MatrixInverseRootErrorBranchTests(unittest.TestCase):
    def test_diagonal_negative_root_raises(self):
        A = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
        # root ≤ 0 should hit the ValueError in _matrix_inverse_root_diagonal
        with self.assertRaisesRegex(ValueError, r"Root -2 should be positive!"):
            matrix_inverse_root(A, Fraction(-2, 1), is_diagonal=True)

    def test_coupled_newton_noninteger_denominator(self):
        A = torch.eye(2)
        cfg = CoupledNewtonConfig()
        # denominator != 1 must raise the special ValueError
        with self.assertRaisesRegex(
            ValueError, r"2 must be equal to 1 to use coupled inverse Newton iteration!"
        ):
            matrix_inverse_root(A, Fraction(3, 2), root_inv_config=cfg)

    def test_unknown_root_inv_config_raises(self):
        A = torch.eye(2)
        class Dummy: pass
        with self.assertRaisesRegex(NotImplementedError, r"Root inverse config is not implemented"):
            matrix_inverse_root(A, Fraction(2, 1), root_inv_config=Dummy())


class CoupledHigherOrderBackendTest(unittest.TestCase):
    def test_higher_order_identity(self):
        # On the identity matrix, any inverse-root is itself
        A = torch.eye(3)
        cfg = CoupledHigherOrderConfig(
            rel_epsilon=0.0,
            max_iterations=5,
            tolerance=1e-6,
            order=2,
            disable_tf32=True,
        )
        X = matrix_inverse_root(A, Fraction(2, 1), root_inv_config=cfg)
        self.assertTrue(torch.allclose(X, torch.eye(3)))

class MatrixEigendecompositionErrorBranchTests(unittest.TestCase):
    def test_non_two_dimensional_raises(self):
        with self.assertRaisesRegex(ValueError, r"Matrix is not 2-dimensional"):
            matrix_eigendecomposition(torch.randn(2, 2, 2))

    def test_non_square_raises(self):
        with self.assertRaisesRegex(ValueError, r"Matrix is not square"):
            matrix_eigendecomposition(torch.randn(2, 3))

    def test_unknown_eig_config_raises(self):
        A = torch.eye(2)
        class DummyConfig: pass
        with self.assertRaisesRegex(NotImplementedError, r"Eigendecomposition config is not implemented"):
            matrix_eigendecomposition(A, eigendecomposition_config=DummyConfig())

    def test_qr_dtype_mismatch_assertion(self):
        A = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
        cfg = QREigendecompositionConfig()
        # give it a non-zero, mismatched-dtype estimate to force the assertion in _qr_algorithm
        cfg.eigenvectors_estimate = torch.eye(2, dtype=torch.float64)
        # The assertion message is actually: "Q and A must have the same dtype! torch.float64 torch.float32"
        with self.assertRaisesRegex(AssertionError, r"Q and A must have the same dtype!"):
            # the exact message is "Q and A must have the same dtype! Q.dtype=... A.dtype=..."
            matrix_eigendecomposition(A, eigendecomposition_config=cfg)


class ComputeResidualsAdditionalTests(unittest.TestCase):
    def test_compute_residuals_non_two_dimensional_A(self):
        # Should raise on a 3-D A
        X_hat = torch.eye(2)
        with self.assertRaisesRegex(ValueError, r"Matrix is not 2-dimensional"):
            compute_matrix_root_inverse_residuals(torch.randn(2, 2, 2), X_hat, Fraction(2, 1), 0.0, root_inv_config=DefaultEigenConfig)


class NewtonFlagAdditionalTests(unittest.TestCase):
    def test_enum_value_identity(self):
        # sanity check that the enum values are distinct and ordered
        vals = [flag.value for flag in NewtonConvergenceFlag]
        self.assertEqual(len(vals), len(set(vals)))
        self.assertCountEqual(
            vals,
            [NewtonConvergenceFlag.REACHED_MAX_ITERS.value,
             NewtonConvergenceFlag.CONVERGED.value,
             NewtonConvergenceFlag.EARLY_STOP.value]
        )
