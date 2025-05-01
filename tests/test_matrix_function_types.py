import re
import unittest

import torch

from commons import get_all_subclasses, AbstractDataclass
from matrix_functions_types import (
    MatrixFunctionConfig,
    EighEigendecompositionConfig,
    QREigendecompositionConfig,
    EigenConfig,
    CoupledNewtonConfig,
    CoupledHigherOrderConfig,
    DefaultEigendecompositionConfig,
    DefaultEigenConfig,
    EigendecompositionConfig, 
    RootInvConfig
)

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
                        "__init__() got an unexpected keyword argument 'eigenvectors_estimate'"
                    ),
                    cls,
                    eigenvectors_estimate=torch.eye(3),
                )

    def test_default_values(self) -> None:
        for cls in get_all_subclasses(QREigendecompositionConfig):
            with self.subTest(cls=cls):
                config = cls()
                # Defaults should be max_iterations=1 and tolerance=0.01
                self.assertIsInstance(config.max_iterations, int)
                self.assertEqual(config.max_iterations, 1)
                self.assertIsInstance(config.tolerance, float)
                self.assertAlmostEqual(config.tolerance, 0.01)

    def test_valid_tolerance_bounds(self) -> None:
        for cls in get_all_subclasses(QREigendecompositionConfig):
            for tol in [0.0, 0.5, 1.0]:
                with self.subTest(cls=cls, tolerance=tol):
                    config = cls(tolerance=tol)
                    self.assertEqual(config.tolerance, tol)

    def test_nan_tolerance(self) -> None:
        for cls in get_all_subclasses(QREigendecompositionConfig):
            with self.subTest(cls=cls):
                tol = float("nan")
                with self.assertRaisesRegex(
                    ValueError,
                    re.escape(f"Invalid tolerance value: {tol}. Must be in the interval [0.0, 1.0]."),
                ):
                    cls(tolerance=tol)

    def test_missing_eigenvectors_estimate_attribute(self) -> None:
        for cls in get_all_subclasses(QREigendecompositionConfig):
            with self.subTest(cls=cls):
                config = cls()
                # eigenvectors_estimate should not exist until explicitly set
                self.assertFalse(hasattr(config, "eigenvectors_estimate"))

    def test_setting_max_iterations(self) -> None:
        for cls in get_all_subclasses(QREigendecompositionConfig):
            with self.subTest(cls=cls):
                config = cls(max_iterations=5)
                self.assertEqual(config.max_iterations, 5)

class CommonsTest(unittest.TestCase):
    def test_get_all_subclasses_includes_and_excludes_self(self):
        # include_cls_self=True (default) should include the class itself
        subs_inc = get_all_subclasses(MatrixFunctionConfig)
        self.assertIn(MatrixFunctionConfig, subs_inc)

        # include_cls_self=False should omit the base class
        subs_exc = get_all_subclasses(MatrixFunctionConfig, include_cls_self=False)
        self.assertNotIn(MatrixFunctionConfig, subs_exc)

    def test_get_all_subclasses_qr_present(self):
        subs = get_all_subclasses(MatrixFunctionConfig)
        self.assertIn(QREigendecompositionConfig, subs)


class EighEigendecompositionConfigTest(unittest.TestCase):
    def test_default_offload_device_empty_string(self):
        cfg = EighEigendecompositionConfig()
        # default is empty string, meaning “no offload”
        self.assertEqual(cfg.eigendecomposition_offload_device, "")

    def test_string_offload_device_converted_to_torch_device(self):
        cfg = EighEigendecompositionConfig(eigendecomposition_offload_device="cpu")
        self.assertIsInstance(cfg.eigendecomposition_offload_device, torch.device)
        self.assertEqual(cfg.eigendecomposition_offload_device, torch.device("cpu"))

    def test_invalid_offload_device_string_raises(self):
        # torch.device("foo") raises a RuntimeError, so let’s catch it
        with self.assertRaises(RuntimeError):
            EighEigendecompositionConfig(eigendecomposition_offload_device="not_a_real_device")


class DefaultConfigsTest(unittest.TestCase):
    def test_default_eigh_config_instance(self):
        # DefaultEigendecompositionConfig is a single shared instance
        self.assertIsInstance(DefaultEigendecompositionConfig, EighEigendecompositionConfig)
        self.assertTrue(DefaultEigendecompositionConfig.retry_double_precision)
        self.assertEqual(DefaultEigendecompositionConfig.eigendecomposition_offload_device, "")

    def test_default_eigen_config_instance(self):
        cfg = DefaultEigenConfig
        self.assertIsInstance(cfg, EigenConfig)
        # its own fields
        self.assertAlmostEqual(cfg.exponent_multiplier, 1.0)
        self.assertFalse(cfg.enhance_stability)
        # inherited from EighEigendecompositionConfig
        self.assertTrue(cfg.retry_double_precision)
        self.assertEqual(cfg.eigendecomposition_offload_device, "")


class InheritanceTest(unittest.TestCase):
    def test_all_configs_are_abstract_dataclass_subclasses(self):
        config_classes = [
            EighEigendecompositionConfig,
            QREigendecompositionConfig,
            EigenConfig,
            CoupledNewtonConfig,
            CoupledHigherOrderConfig,
        ]
        for cls in config_classes:
            with self.subTest(cls=cls):
                self.assertTrue(issubclass(cls, AbstractDataclass))

class OtherConfigsDefaultsTest(unittest.TestCase):
    def test_qr_config_defaults(self):
        cfg = QREigendecompositionConfig()
        self.assertEqual(cfg.max_iterations, 1)
        self.assertAlmostEqual(cfg.tolerance, 0.01)
        # should not have eigenvectors_estimate until later
        self.assertFalse(hasattr(cfg, "eigenvectors_estimate"))

    def test_eigen_config_inherits_defaults(self):
        cfg = EigenConfig()
        # max_iterations & tolerance from EighEigendecompositionConfig
        self.assertTrue(hasattr(cfg, "retry_double_precision"))
        # and its own defaults
        self.assertAlmostEqual(cfg.exponent_multiplier, 1.0)
        self.assertFalse(cfg.enhance_stability)

    def test_coupled_newton_defaults(self):
        cfg = CoupledNewtonConfig()
        self.assertEqual(cfg.max_iterations, 100)
        self.assertAlmostEqual(cfg.tolerance, 1e-6)

    def test_coupled_higher_order_defaults(self):
        cfg = CoupledHigherOrderConfig()
        self.assertAlmostEqual(cfg.rel_epsilon, 0.0)
        self.assertEqual(cfg.max_iterations, 100)
        self.assertAlmostEqual(cfg.tolerance, 1e-8)
        self.assertEqual(cfg.order, 3)
        self.assertTrue(cfg.disable_tf32)

class AbstractBaseInstantiationTest(unittest.TestCase):
    def test_matrixfunctionconfig_instantiates(self):
        # Under the shim, MatrixFunctionConfig now uses AbstractDataclass.__init__, so it can be constructed
        cfg = MatrixFunctionConfig()
        self.assertIsInstance(cfg, MatrixFunctionConfig)

    def test_eigendecompositionconfig_instantiates(self):
        # Under the shim, this is now concrete and should instantiate
        cfg = EigendecompositionConfig()
        self.assertIsInstance(cfg, EigendecompositionConfig)

    def test_rootinvconfig_instantiates(self):
        # Under the shim, RootInvConfig is concrete and should instantiate
        cfg = RootInvConfig()
        self.assertIsInstance(cfg, RootInvConfig)


class EighEigendecompositionConfigOverrideTest(unittest.TestCase):
    def test_override_retry_and_device(self):
        cfg = EighEigendecompositionConfig(
            retry_double_precision=False,
            eigendecomposition_offload_device="cpu",
        )
        self.assertFalse(cfg.retry_double_precision)
        self.assertEqual(cfg.eigendecomposition_offload_device, torch.device("cpu"))

    def test_device_object_preserved(self):
        dev = torch.device("cpu")
        cfg = EighEigendecompositionConfig(eigendecomposition_offload_device=dev)
        self.assertEqual(cfg.eigendecomposition_offload_device, dev)


class EigenConfigOverrideTest(unittest.TestCase):
    def test_override_all_parameters(self):
        cfg = EigenConfig(
            retry_double_precision=False,
            eigendecomposition_offload_device="cpu",
            exponent_multiplier=2.5,
            enhance_stability=True,
        )
        self.assertFalse(cfg.retry_double_precision)
        self.assertEqual(cfg.eigendecomposition_offload_device, torch.device("cpu"))
        self.assertEqual(cfg.exponent_multiplier, 2.5)
        self.assertTrue(cfg.enhance_stability)


class QREigenEstimateAssignmentTest(unittest.TestCase):
    def test_assign_eigenvectors_estimate(self):
        cfg = QREigendecompositionConfig()
        tensor = torch.rand(4, 4)
        # field(init=False) means you set this after init
        cfg.eigenvectors_estimate = tensor
        self.assertIs(cfg.eigenvectors_estimate, tensor)

class RootInvOverrideTest(unittest.TestCase):
    def test_coupled_newton_override(self):
        cfg = CoupledNewtonConfig(max_iterations=10, tolerance=1e-3)
        self.assertEqual(cfg.max_iterations, 10)
        self.assertEqual(cfg.tolerance, 1e-3)

    def test_coupled_higher_order_override(self):
        cfg = CoupledHigherOrderConfig(
            rel_epsilon=0.1,
            max_iterations=50,
            tolerance=1e-4,
            order=5,
            disable_tf32=False,
        )
        self.assertEqual(cfg.rel_epsilon, 0.1)
        self.assertEqual(cfg.max_iterations, 50)
        self.assertEqual(cfg.tolerance, 1e-4)
        self.assertEqual(cfg.order, 5)
        self.assertFalse(cfg.disable_tf32)

class KwOnlyEnforcementTest(unittest.TestCase):
    def test_eigh_positional_args(self):
        # Under the shim, EighEigendecompositionConfig accepts positional args:
        #   (retry_double_precision, eigendecomposition_offload_device)
        cfg = EighEigendecompositionConfig(True, "cpu")
        self.assertIsInstance(cfg, EighEigendecompositionConfig)
        self.assertTrue(cfg.retry_double_precision)
        # the string is converted into a torch.device
        self.assertEqual(cfg.eigendecomposition_offload_device, torch.device("cpu"))

    def test_qr_kw_only(self):
        # QREigendecompositionConfig is kw_only
        cfg = QREigendecompositionConfig(5, 0.01)
        self.assertIsInstance(cfg, QREigendecompositionConfig)
        self.assertEqual(cfg.max_iterations, 5)
        self.assertEqual(cfg.tolerance, 0.01)


    def test_eigenconfig_kw_only(self):
        # EigenConfig inherits kw_only=True
        cfg = EighEigendecompositionConfig(True, "cpu")
        self.assertIsInstance(cfg, EighEigendecompositionConfig)
        self.assertTrue(cfg.retry_double_precision)
        # string “cpu” should be converted to torch.device("cpu")
        self.assertEqual(cfg.eigendecomposition_offload_device, torch.device("cpu"))


class GetAllSubclassesBehaviorTest(unittest.TestCase):
    def test_no_duplicates_in_subclasses(self):
        subs = get_all_subclasses(MatrixFunctionConfig)
        # Should be unique
        self.assertEqual(len(subs), len(set(subs)))

    def test_exclude_self_when_no_children(self):
        # QREigendecompositionConfig has no subclasses under it
        subs = get_all_subclasses(QREigendecompositionConfig, include_cls_self=False)
        self.assertEqual(subs, [])

    def test_eigendecomp_children(self):
        # EigendecompositionConfig should have both Eigh and QR classes
        subs = get_all_subclasses(EigendecompositionConfig, include_cls_self=False)
        self.assertIn(EighEigendecompositionConfig, subs)
        self.assertIn(QREigendecompositionConfig, subs)


class InheritanceHierarchyTest(unittest.TestCase):
    def test_qr_inheritance(self):
        self.assertTrue(issubclass(QREigendecompositionConfig, EigendecompositionConfig))
        self.assertTrue(issubclass(QREigendecompositionConfig, MatrixFunctionConfig))

    def test_eigenconfig_inheritance(self):
        self.assertTrue(issubclass(EigenConfig, EighEigendecompositionConfig))
        self.assertTrue(issubclass(EigenConfig, MatrixFunctionConfig))


class RepresentationTest(unittest.TestCase):
    def test_eigh_repr_shows_fields(self):
        cfg = EighEigendecompositionConfig()
        r = repr(cfg)
        self.assertIn("retry_double_precision=True", r)
        self.assertIn("eigendecomposition_offload_device=''", r)

    def test_qr_repr_excludes_unset_field(self):
        cfg = QREigendecompositionConfig()
        cfg.eigenvectors_estimate = torch.eye(2)
        r = repr(cfg)
        self.assertIn("eigenvectors_estimate", r)


class DataclassStructureTest(unittest.TestCase):
    def test_eigh_fields_and_init_flags(self):
        flds = fields(EighEigendecompositionConfig)
        names = {f.name for f in flds}
        self.assertEqual(names, {"retry_double_precision", "eigendecomposition_offload_device"})
        for f in flds:
            self.assertTrue(f.init, msg=f"{f.name} should be init=True")

    def test_qr_fields_and_init_flags(self):
        flds = fields(QREigendecompositionConfig)
        names = {f.name for f in flds}
        self.assertEqual(names, {"max_iterations", "tolerance", "eigenvectors_estimate"})
        for f in flds:
            if f.name == "eigenvectors_estimate":
                self.assertFalse(f.init, msg="eigenvectors_estimate should be init=False")
            else:
                self.assertTrue(f.init, msg=f"{f.name} should be init=True")


class OffloadDeviceEdgeCaseTest(unittest.TestCase):
    def test_none_offload_device_raises_type_error(self):
        with self.assertRaises(TypeError):
            EighEigendecompositionConfig(eigendecomposition_offload_device=None)

    def test_whitespace_string_offload_device_raises(self):
        with self.assertRaises(RuntimeError):
            # non-empty string passes through to torch.device and fails
            EighEigendecompositionConfig(eigendecomposition_offload_device=" ")