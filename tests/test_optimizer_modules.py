import unittest
from dataclasses import dataclass

import torch
from optimizer_modules import OptimizerModule

from typing import Union, Optional, List, Tuple, Dict

@dataclass
class OptimizerTestModule(OptimizerModule):
    attribute: Union[torch.Tensor, int]
    list_of_values: Optional[List[Union[OptimizerModule, torch.Tensor]]] = None
    tuple_of_values: Optional[Tuple[float, ...]] = None
    dictionary_of_values: Optional[Dict[str, torch.Tensor]] = None
    other_module: Optional[OptimizerModule] = None


class OptimizerModulesTest(unittest.TestCase):
    def init_optimizer_module(self) -> OptimizerModule:
        other_module = OptimizerTestModule(
            attribute=42,
        )
        test_module = OptimizerTestModule(
            attribute=torch.tensor(42),
            list_of_values=[OptimizerModule(), torch.tensor(1.0)],
            tuple_of_values=(1.0, 2.0, 3.0),
            dictionary_of_values={"tensor": torch.tensor(2.0)},
            other_module=other_module,
        )
        return test_module

    def test_state_dict(self) -> None:
        test_module = self.init_optimizer_module()

        self.assertEqual(
            test_module.state_dict(store_non_tensors=True),
            {
                "attribute": torch.tensor(42),
                "list_of_values": {0: {}, 1: torch.tensor(1.0)},
                "tuple_of_values": {0: 1.0, 1: 2.0, 2: 3.0},
                "dictionary_of_values": {"tensor": torch.tensor(2.0)},
                "other_module": {
                    "attribute": 42,
                    "list_of_values": None,
                    "tuple_of_values": None,
                    "dictionary_of_values": None,
                    "other_module": None,
                },
            },
        )

    def test_state_dict_without_non_tensor_objects(self) -> None:
        test_module = self.init_optimizer_module()

        self.assertEqual(
            test_module.state_dict(store_non_tensors=False),
            {
                "attribute": torch.tensor(42),
                "list_of_values": {0: {}, 1: torch.tensor(1.0)},
                "tuple_of_values": {},
                "dictionary_of_values": {"tensor": torch.tensor(2.0)},
                "other_module": {},
            },
        )

    def test_load_state_dict_with_non_tensor_objects(self) -> None:
        test_module = self.init_optimizer_module()

        # state dict to load
        state_dict = {
            "attribute": torch.tensor(24),
            "list_of_values": {0: {}, 1: torch.tensor(3.0)},
            "tuple_of_values": {0: 4.0, 1: 5.0, 2: 6.0},
            "dictionary_of_values": {"tensor": torch.tensor(4.0)},
            "other_module": {
                "attribute": 24,
                "list_of_values": None,
                "tuple_of_values": None,
                "dictionary_of_values": None,
                "other_module": None,
            },
        }
        test_module.load_state_dict(state_dict=state_dict, store_non_tensors=True)

        self.assertEqual(test_module.state_dict(store_non_tensors=True), state_dict)

    def test_load_state_dict_without_non_tensor_objects(self) -> None:
        test_module = self.init_optimizer_module()

        # state dict to load
        state_dict = {
            "attribute": torch.tensor(24),
            "list_of_values": {0: {}, 1: torch.tensor(3.0)},
            "tuple_of_values": {},
            "dictionary_of_values": {"tensor": torch.tensor(4.0)},
            "other_module": {},
        }
        test_module.load_state_dict(state_dict=state_dict, store_non_tensors=False)

        self.assertEqual(test_module.state_dict(store_non_tensors=False), state_dict)

    def test_load_state_dict_with_non_matching_objects(self) -> None:
        test_module = self.init_optimizer_module()

        # state dict to load
        state_dict = {
            "attribute": 24,
            "list_of_values": {0: {}, 1: torch.tensor(3.0)},
            "tuple_of_values": {0: "hello", 1: 5.0, 2: 6.0},
            "dictionary_of_values": torch.tensor(4.0),
            "other_module": {
                "attribute": 24,
                "list_of_values": None,
                "tuple_of_values": None,
                "dictionary_of_values": None,
                "other_module": None,
            },
        }
        expected_state_dict = {
            "attribute": torch.tensor(42),
            "list_of_values": {0: {}, 1: torch.tensor(3.0)},
            "tuple_of_values": {0: 1.0, 1: 5.0, 2: 6.0},
            "dictionary_of_values": {"tensor": torch.tensor(2.0)},
            "other_module": {
                "attribute": 24,
                "list_of_values": None,
                "tuple_of_values": None,
                "dictionary_of_values": None,
                "other_module": None,
            },
        }
        test_module.load_state_dict(state_dict=state_dict, store_non_tensors=True)

        self.assertEqual(
            test_module.state_dict(store_non_tensors=True), expected_state_dict
        )

class KeepVarsAndDetachTests(unittest.TestCase):
    def test_detached_by_default_and_keep_vars(self):
        m = OptimizerTestModule(attribute=torch.tensor(1.0, requires_grad=True))
        sd_default = m.state_dict()
        # default should detach → requires_grad False
        self.assertFalse(sd_default["attribute"].requires_grad)
        sd_keep = m.state_dict(keep_vars=True)
        # keep_vars=True should preserve requires_grad
        self.assertTrue(sd_keep["attribute"].requires_grad)


class DestinationDictTests(unittest.TestCase):
    def test_destination_is_used_and_returned(self):
        m = OptimizerTestModule(attribute=torch.tensor(5.0))
        dest = {"already": 123}
        ret = m.state_dict(destination=dest)
        # returns exactly the same dict
        self.assertIs(ret, dest)
        # our new key was written alongside the old one
        self.assertIn("attribute", dest)
        self.assertEqual(dest["already"], 123)


class LoadStateCopySemanticsTests(unittest.TestCase):
    def test_loading_tensor_copies_not_rebinds(self):
        m = OptimizerTestModule(attribute=torch.tensor([1.0, 2.0]))
        original_tensor = m.attribute
        new_state = {"attribute": torch.tensor([9.0, 8.0])}

        m.load_state_dict(new_state)
        # same object, but its contents have changed
        self.assertIs(m.attribute, original_tensor)
        self.assertTrue(torch.allclose(m.attribute, torch.tensor([9.0, 8.0])))

        # mutating the source state afterwards does not affect the module's tensor
        new_state["attribute"][0] = -1.0
        self.assertNotAlmostEqual(m.attribute[0].item(), -1.0)


class EmptyModuleTests(unittest.TestCase):
    def test_empty_module_state_dict_is_empty(self):
        class Empty(OptimizerModule):
            pass

        e = Empty()
        sd = e.state_dict()
        self.assertEqual(sd, {})

    def test_empty_module_load_no_error(self):
        class Empty(OptimizerModule):
            pass

        e = Empty()
        # loading an empty dict should be a no-op
        try:
            e.load_state_dict({})
        except Exception as ex:
            self.fail(f"load_state_dict raised unexpectedly: {ex}")

class NestedCollectionsTests(unittest.TestCase):
    def test_mixed_dict_list_tuple_no_non_tensors(self):
        class M(OptimizerModule):
            def __init__(self):
                # nested dict containing a list and a tuple
                self.nested = {
                    "lst": [torch.tensor(1.0), 2, "three"],
                    "tup": (4, torch.tensor(5.0), 6.0),
                }

        m = M()

        # store_non_tensors=False should only keep tensors
        sd0 = m.state_dict(store_non_tensors=False)
        # 'lst' → only index 0; 'tup' → only index 1
        self.assertEqual(set(sd0["nested"]["lst"].keys()), {0})
        self.assertEqual(set(sd0["nested"]["tup"].keys()), {1})
        self.assertTrue(isinstance(sd0["nested"]["lst"][0], torch.Tensor))
        self.assertTrue(isinstance(sd0["nested"]["tup"][1], torch.Tensor))

    def test_mixed_dict_list_tuple_with_non_tensors(self):
        class M(OptimizerModule):
            def __init__(self):
                self.nested = {
                    "lst": [torch.tensor(1.0), 2, "three"],
                    "tup": (4, torch.tensor(5.0), 6.0),
                }

        m = M()
        # store_non_tensors=True should keep everything
        sd1 = m.state_dict(store_non_tensors=True)
        self.assertEqual(set(sd1["nested"]["lst"].keys()), {0, 1, 2})
        self.assertEqual(set(sd1["nested"]["tup"].keys()), {0, 1, 2})
        self.assertEqual(sd1["nested"]["lst"][1], 2)
        self.assertEqual(sd1["nested"]["lst"][2], "three")
        self.assertEqual(sd1["nested"]["tup"][0], 4)
        self.assertEqual(sd1["nested"]["tup"][2], 6.0)


class LoadStateWarningBranchTests(unittest.TestCase):
    def test_loading_non_tensor_into_tensor_logs_warning(self):
        m = OptimizerTestModule(attribute=torch.tensor(1.0))
        # attempt to overwrite tensor with a non-tensor
        bad = {"attribute": 123}
        with self.assertLogs("optimizer_modules", level="WARNING") as cm:
            m.load_state_dict(bad, store_non_tensors=False)
        # warning about “must be tensors”
        self.assertTrue(any("must be tensors" in msg for msg in cm.output))
        # attribute remains unchanged
        self.assertTrue(torch.equal(m.attribute, torch.tensor(1.0)))

    def test_loading_non_dict_into_dict_logs_warning(self):
        m = OptimizerTestModule(
            attribute=torch.tensor(1.0),
            list_of_values=None,
            tuple_of_values=None,
            dictionary_of_values={"x": torch.tensor(2.0)},
            other_module=None,
        )
        bad = {"dictionary_of_values": torch.tensor(9.0)}
        with self.assertLogs("optimizer_modules", level="WARNING") as cm:
            m.load_state_dict(bad, store_non_tensors=True)
        # warning about “must be dict”
        self.assertTrue(any("must be dict" in msg for msg in cm.output))
        # original mapping preserved
        sd = m.state_dict(store_non_tensors=True)
        self.assertTrue(torch.equal(sd["dictionary_of_values"]["x"], torch.tensor(2.0)))

    def test_loading_mismatched_list_raises_index_error(self):
        m = OptimizerTestModule(
            attribute=torch.tensor(0.0),
            list_of_values=[torch.tensor(1.0), torch.tensor(2.0)],
        )
        bad = {"list_of_values": torch.tensor(7.0)}
        with self.assertRaises(IndexError):
            m.load_state_dict(bad, store_non_tensors=True)

class NestedModuleStateTests(unittest.TestCase):
    def test_modules_inside_list(self):
        # child has a tensor attribute
        child = OptimizerTestModule(attribute=torch.tensor(10), list_of_values=None,
                                    tuple_of_values=None, dictionary_of_values=None,
                                    other_module=None)
        # parent wraps the child in a list
        parent = OptimizerTestModule(attribute=torch.tensor(0),
                                     list_of_values=[child],
                                     tuple_of_values=None,
                                     dictionary_of_values=None,
                                     other_module=None)

        sd = parent.state_dict(store_non_tensors=False)
        # list_of_values[0] should be the child's state-dict
        sub = sd["list_of_values"][0]
        self.assertIsInstance(sub, dict)
        # child's tensor attribute must appear
        self.assertTrue(torch.equal(sub["attribute"], torch.tensor(10)))

    def test_modules_inside_dict(self):
        child = OptimizerTestModule(attribute=torch.tensor(7), list_of_values=None,
                                    tuple_of_values=None, dictionary_of_values=None,
                                    other_module=None)
        class Holder(OptimizerModule):
            def __init__(self):
                self.mapping = {"c": child}

        h = Holder()
        sd = h.state_dict(store_non_tensors=False)
        sub = sd["mapping"]["c"]
        self.assertIsInstance(sub, dict)
        self.assertTrue(torch.equal(sub["attribute"], torch.tensor(7)))

    def test_modules_inside_tuple(self):
        m1 = OptimizerTestModule(attribute=torch.tensor(1), list_of_values=None,
                                 tuple_of_values=None, dictionary_of_values=None,
                                 other_module=None)
        m2 = OptimizerTestModule(attribute=torch.tensor(2), list_of_values=None,
                                 tuple_of_values=None, dictionary_of_values=None,
                                 other_module=None)
        class TupHolder(OptimizerModule):
            def __init__(self):
                self.mods = (m1, m2)

        th = TupHolder()
        sd = th.state_dict(store_non_tensors=False)
        a0 = sd["mods"][0]
        a1 = sd["mods"][1]
        self.assertTrue(torch.equal(a0["attribute"], torch.tensor(1)))
        self.assertTrue(torch.equal(a1["attribute"], torch.tensor(2)))


class NestedModuleLoadTests(unittest.TestCase):
    def test_load_nested_module_in_list(self):
        child = OptimizerTestModule(attribute=torch.tensor(5), list_of_values=None,
                                    tuple_of_values=None, dictionary_of_values=None,
                                    other_module=None)
        parent = OptimizerTestModule(attribute=torch.tensor(0),
                                     list_of_values=[child],
                                     tuple_of_values=None,
                                     dictionary_of_values=None,
                                     other_module=None)

        sd = parent.state_dict(store_non_tensors=True)
        # mutate the child's attribute in the saved dict
        sd["list_of_values"][0]["attribute"] = torch.tensor(15)
        parent.load_state_dict(sd, store_non_tensors=True)
        # ensure the actual child instance was updated
        self.assertTrue(torch.equal(parent.list_of_values[0].attribute, torch.tensor(15)))

    def test_load_nested_module_in_dict(self):
        child = OptimizerTestModule(attribute=torch.tensor(3), list_of_values=None,
                                    tuple_of_values=None, dictionary_of_values=None,
                                    other_module=None)
        class Holder(OptimizerModule):
            def __init__(self):
                self.d = {"child": child}

        h = Holder()
        sd = h.state_dict(store_non_tensors=True)
        sd["d"]["child"]["attribute"] = torch.tensor(99)
        h.load_state_dict(sd, store_non_tensors=True)
        self.assertTrue(torch.equal(h.d["child"].attribute, torch.tensor(99)))

    def test_load_nested_module_in_tuple(self):
        m1 = OptimizerTestModule(attribute=torch.tensor(11), list_of_values=None,
                                 tuple_of_values=None, dictionary_of_values=None,
                                 other_module=None)
        m2 = OptimizerTestModule(attribute=torch.tensor(22), list_of_values=None,
                                 tuple_of_values=None, dictionary_of_values=None,
                                 other_module=None)
        class TupHolder(OptimizerModule):
            def __init__(self):
                self.mods = (m1, m2)

        th = TupHolder()
        sd = th.state_dict(store_non_tensors=True)
        sd["mods"][1]["attribute"] = torch.tensor(222)
        th.load_state_dict(sd, store_non_tensors=True)
        self.assertTrue(torch.equal(th.mods[1].attribute, torch.tensor(222)))


class CopyIsolationTests(unittest.TestCase):
    def test_state_dict_isolation(self):
        # Ensures that modifying the returned dict doesn't alter module state
        child = OptimizerTestModule(attribute=torch.tensor(4), list_of_values=None,
                                    tuple_of_values=None, dictionary_of_values=None,
                                    other_module=None)
        parent = OptimizerTestModule(attribute=torch.tensor(8),
                                     list_of_values=[child],
                                     tuple_of_values=None,
                                     dictionary_of_values=None,
                                     other_module=None)
        sd1 = parent.state_dict(store_non_tensors=True)
        sd1["list_of_values"][0]["attribute"] = torch.tensor(999)
        # regenerate a fresh state_dict
        sd2 = parent.state_dict(store_non_tensors=True)
        # child attribute in sd2 remains original
        self.assertTrue(torch.equal(sd2["list_of_values"][0]["attribute"], torch.tensor(4)))