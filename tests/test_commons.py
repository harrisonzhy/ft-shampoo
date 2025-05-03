"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import re
import unittest
from dataclasses import dataclass

from commons import AbstractDataclass, get_all_subclasses

# --- Dummy classes for AbstractDataclass tests ---

@dataclass(init=False)
class DummyOptimizerConfig(AbstractDataclass):
    """Dummy abstract dataclass for testing. Instantiation should fail."""

# --- Dummy classes for get_all_subclasses tests ---

class DummyRootClass:
    """Dummy root class for GetAllSubclassesTest."""

class DummyFirstSubclass(DummyRootClass):
    """First dummy subclass for GetAllSubclassesTest."""

class DummySecondSubclass(DummyFirstSubclass):
    """Second dummy subclass for GetAllSubclassesTest."""

class DummySecondRootClass:
    """Second dummy root class for GetAllSubclassesTest."""

class DummyMixedSubclass(DummySecondRootClass, DummySecondSubclass):
    """Dummy subclass with mixed inheritance for GetAllSubclassesTest."""

class DummyLeafClass(DummyMixedSubclass):
    """Dummy leaf class for GetAllSubclassesTest."""

# --- Additional dummy hierarchies for expanded tests ---

class DiamondRoot: pass
class Left(DiamondRoot): pass
class Right(DiamondRoot): pass
class Bottom(Left, Right): pass

class Lone: pass

# --- TestCases ---

class InvalidAbstractDataclassInitTest(unittest.TestCase):
    def test_invalid_init(self) -> None:
        for abstract_cls in (
            AbstractDataclass,
            DummyOptimizerConfig,
        ):
            with self.subTest(abstract_cls=abstract_cls):
                self.assertRaisesRegex(
                    TypeError,
                    re.escape(
                        f"Can't instantiate abstract class {abstract_cls.__name__} "
                    ),
                    abstract_cls,
                )

@dataclass
class ConcreteDataclass(AbstractDataclass):
    x: int
    y: str

class ConcreteDataclassInitTest(unittest.TestCase):
    def test_manual_init_dataclass(self):
        @dataclass(init=False)
        class ManualInitDataclass(AbstractDataclass):
            x: int

            def __init__(self, x: int):
                self.x = x

        inst = ManualInitDataclass(42)
        self.assertEqual(inst.x, 42)

    def test_missing_init_subclass(self):
        @dataclass(init=False)
        class MissingInit(AbstractDataclass):
            a: int

        with self.assertRaisesRegex(
            TypeError,
            re.escape(f"Can't instantiate abstract class MissingInit ")
        ):
            MissingInit(5)

@dataclass(init=False)
class ManualInitDataclass(AbstractDataclass):
    x: int
    def __init__(self, x: int):
        self.x = x

class ManualInitDataclassTest(unittest.TestCase):
    def test_manual_init_dataclass(self):
        inst = ManualInitDataclass(42)
        self.assertEqual(inst.x, 42)

class GetAllSubclassesTest(unittest.TestCase):
    def test_get_all_subclasses(self) -> None:
        """Test with class hierarchy and multiple inheritance."""
        for include_cls_self in (True, False):
            with self.subTest(
                "Test with the root class", include_cls_self=include_cls_self
            ):
                subclasses = {
                    DummyFirstSubclass,
                    DummySecondSubclass,
                    DummyMixedSubclass,
                    DummyLeafClass,
                }
                expected = ({DummyRootClass} | subclasses) if include_cls_self else subclasses
                self.assertEqual(
                    set(get_all_subclasses(DummyRootClass, include_cls_self=include_cls_self)),
                    expected,
                )
            with self.subTest(
                "Test with the second subclass (parents should not be included)",
                include_cls_self=include_cls_self,
            ):
                subclasses = {DummyMixedSubclass, DummyLeafClass}
                expected = ({DummySecondSubclass} | subclasses) if include_cls_self else subclasses
                self.assertEqual(
                    set(get_all_subclasses(DummySecondSubclass, include_cls_self=include_cls_self)),
                    expected,
                )
            with self.subTest(
                "Test with leaf class (no subclasses)", include_cls_self=include_cls_self
            ):
                expected_list = [DummyLeafClass] if include_cls_self else []
                self.assertEqual(
                    get_all_subclasses(DummyLeafClass, include_cls_self=include_cls_self),
                    expected_list,
                )

    def test_diamond_inheritance(self):
        """Ensure diamond-shaped multiple inheritance yields unique subclasses only."""
        subs = set(get_all_subclasses(DiamondRoot, include_cls_self=False))
        self.assertEqual(subs, {Left, Right, Bottom})

    def test_dynamic_subclass(self):
        """Classes defined at runtime should be discovered."""
        class TempRoot: pass
        class TempSub(TempRoot): pass
        subs = get_all_subclasses(TempRoot, include_cls_self=False)
        self.assertIn(TempSub, subs)

    def test_no_subclasses_explicit(self):
        """Explicitly test behavior for a class with no subclasses."""
        self.assertEqual(get_all_subclasses(Lone, include_cls_self=False), [])
        self.assertEqual(get_all_subclasses(Lone, include_cls_self=True), [Lone])

    def test_order_independence(self):
        """Repeated calls should return the same set, regardless of internal order."""
        first = get_all_subclasses(DummyRootClass, include_cls_self=False)
        second = get_all_subclasses(DummyRootClass, include_cls_self=False)
        self.assertEqual(set(first), set(second))

    def test_no_duplicates(self):
        """Ensure the returned list has no duplicate classes."""
        result = get_all_subclasses(DummyRootClass, include_cls_self=False)
        # length of set should match length of list
        self.assertEqual(len(result), len(set(result)))

    def test_dynamic_multilevel(self):
        """Dynamically defined multi-level subclass chain should be discovered."""
        class A: pass
        class B(A): pass
        class C(B): pass

        subs = get_all_subclasses(A, include_cls_self=False)
        self.assertIn(B, subs)
        self.assertIn(C, subs)
        # no other types sneaking in
        self.assertEqual({B, C}, set(subs))

    def test_include_self_default(self):
        """By default include_cls_self=True, so the root appears in the results."""
        subs = get_all_subclasses(DummySecondRootClass)  # omit include_cls_self
        self.assertIn(DummySecondRootClass, subs)

    def test_invalid_input_type(self):
        """Passing a non‐class should raise an AttributeError (no __subclasses__)."""
        with self.assertRaises(AttributeError):
            get_all_subclasses(123) 

class GetAllSubclassesStressTests(unittest.TestCase):
    def test_large_dynamic_hierarchy(self):
        """Stress test: generate many subclasses at runtime."""
        Base = type("BaseDynamic", (), {})
        generated = []
        for i in range(50):
            cls = type(f"DynamicSub{i}", (Base,), {})
            generated.append(cls)

        subs = get_all_subclasses(Base, include_cls_self=False)
        # every generated class should appear, and no duplicates
        for cls in generated:
            self.assertIn(cls, subs)
        self.assertEqual(len(subs), len(set(subs)))
        self.assertEqual(len(subs), len(generated))

    def test_none_as_input(self):
        """Passing None instead of a class should raise AttributeError."""
        with self.assertRaises(AttributeError):
            get_all_subclasses(None)

    def test_repeated_calls_idempotence(self):
        """Multiple calls return the same set each time."""
        first = set(get_all_subclasses(DummyRootClass, include_cls_self=False))
        second = set(get_all_subclasses(DummyRootClass, include_cls_self=False))
        self.assertEqual(first, second)