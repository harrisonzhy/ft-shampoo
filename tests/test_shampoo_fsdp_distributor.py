import unittest
import logging
import torch
import re
from torch.nn import Parameter
from torch import distributed as dist

from torch import Tensor

from parameterized import parameterized

from distributed_shampoo.utils.shampoo_fsdp_distributor import FSDPDistributor
from distributed_shampoo.shampoo_types import FSDPShampooConfig, FSDPParameterMetadata, PARAMS, MAX_PRECONDITIONER_DIM, USE_MERGE_DIMS

logger: logging.Logger = logging.getLogger(__name__)

class SplitTensorBlockRecoveryTest(unittest.TestCase):
    def _test_split_tensor_block_recovery(
        self,
        original_tensor: Tensor,
        expected_split_tensors: list[Tensor],
        start_idx: int,
        end_idx: int,
    ) -> None:
        actual_split_tensors = FSDPDistributor._split_tensor_block_recovery(
            original_tensor.flatten()[start_idx:end_idx],
            original_tensor.size(),
            start_idx,
            end_idx,
        )

        self.assertNotEqual(len(actual_split_tensors), 0)
        torch.testing.assert_close(actual_split_tensors, expected_split_tensors)

    def test_illegal_tensor_shard_size(self) -> None:
        self.assertRaisesRegex(
            ValueError,
            re.escape("Input tensor is not flat"),
            FSDPDistributor._split_tensor_block_recovery,
            tensor_shard=torch.randn((3, 4)),
            original_shape=torch.Size((3, 4)),
            start_idx=0,
            end_idx=16,
        )

    @parameterized.expand([
        (0, 5, [torch.arange(5)]),
        (1, 4, [torch.arange(1, 4)]),
    ])
    def test_split_tensor_block_recovery_for_one_dim(
        self, start_idx: int, end_idx: int, expected_split_tensors: list[Tensor]
    ) -> None:
        original_tensor = torch.arange(5)

        self._test_split_tensor_block_recovery(
            original_tensor=original_tensor,
            expected_split_tensors=expected_split_tensors,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    @parameterized.expand([
        (0, 11, [torch.arange(10).reshape(2, 5), torch.tensor([10])]),
        (3, 15, [torch.arange(3, 5), torch.arange(5, 15).reshape(2, 5)]),
        (3, 4,  [torch.tensor([3])]),
    ])
    def test_split_tensor_block_recovery_for_two_dim(
        self, start_idx: int, end_idx: int, expected_split_tensors: list[Tensor]
    ) -> None:
        original_tensor = torch.arange(15).reshape(3, 5)

        self._test_split_tensor_block_recovery(
            original_tensor=original_tensor,
            expected_split_tensors=expected_split_tensors,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    @parameterized.expand([
        (0, 9,  [torch.arange(9).reshape(1, 3, 3)]),
        (8, 10, [torch.tensor([8]), torch.tensor([9])]),
        (7, 22, [
            torch.tensor([7, 8]),
            torch.tensor([[[9, 10, 11],
                           [12, 13, 14],
                           [15, 16, 17]]]),
            torch.tensor([[18, 19, 20]]),
            torch.tensor([21]),
        ]),
    ])
    def test_split_tensor_block_recovery_for_three_dim(
        self, start_idx: int, end_idx: int, expected_split_tensors: list[Tensor]
    ) -> None:
        original_tensor = torch.arange(27).reshape(3, 3, 3)

        self._test_split_tensor_block_recovery(
            original_tensor=original_tensor,
            expected_split_tensors=expected_split_tensors,
            start_idx=start_idx,
            end_idx=end_idx,
        )

class DummyConfig(FSDPShampooConfig):
    """Minimal dummy config supplying param_to_metadata mapping."""
    def __init__(self, mapping):
        super().__init__(param_to_metadata=mapping)

class SplitTensorBlockRecoveryExtraTest(unittest.TestCase):
    def test_empty_range_returns_empty_list(self):
        # when start_idx == end_idx, no data => empty list
        tensor = torch.arange(10)
        splits = FSDPDistributor._split_tensor_block_recovery(
            tensor.flatten()[3:3],
            tensor.size(),
            3,
            3,
        )
        self.assertEqual(splits, [], "Expected no splits when start_idx == end_idx")

    def test_mismatched_indices_raise_assertion(self):
        # supply a shard length that doesn't match end_idx - start_idx
        tensor = torch.arange(6).reshape(2,3)
        flat = tensor.flatten()
        # pass end_idx-start_idx != flat slice length
        with self.assertRaises(AssertionError):
            FSDPDistributor._split_tensor_block_recovery(
                flat[1:5],          # length 4
                tensor.size(),
                0,                  # but block_start_idx=0, block_end_idx=3 => 3 elements expected
                3,
            )

class ComposableBlockIDsTest(unittest.TestCase):
    def setUp(self):
        # create a minimal distributor so we can call the instance method
        dummy_param = Parameter(torch.zeros(1))
        cfg = DummyConfig(mapping={dummy_param: FSDPParameterMetadata(
            fqn="dummy_param",
            shape=torch.Size([1]),
            start_idx=0,
            end_idx=1,
            numel=1,                  # total number of elements in the tensor
            sharding_strategy=None    # or a dummy strategy if None isn’t accepted
        )})
        pg = {
            PARAMS: [dummy_param],
            MAX_PRECONDITIONER_DIM: 2,
            USE_MERGE_DIMS: False
        }
        # monkey‐patch torch.distributed rank
        dist.get_rank = lambda: -1
        self.d = FSDPDistributor(param_group=pg, distributed_config=cfg)

    def test_construct_composable_block_ids_default_rank(self):
        blk_id, blk_name = self.d._construct_composable_block_ids(7, 13)
        self.assertEqual(blk_id, 7)
        self.assertEqual(blk_name, "rank_None-block_13")

    def test_construct_composable_block_ids_explicit_rank(self):
        blk_id, blk_name = self.d._construct_composable_block_ids(2, 5, rank=42)
        self.assertEqual(blk_id, 2)
        self.assertEqual(blk_name, "rank_42-block_5")


class UpdateParamsTest(unittest.TestCase):
    def test_update_params_inplace_addition(self):
        # set up a tiny distributor and manually override the masked blocks
        dummy = FSDPDistributor.__new__(FSDPDistributor)
        # create two tensors to update
        t1 = torch.zeros(3)
        t2 = torch.tensor([1., 2.])
        dummy._local_masked_blocked_params = (t1, t2)
        # apply updates
        dummy.update_params((torch.tensor([1., 1., 1.]), torch.tensor([-1., -2.])))
        # check in-place addition
        self.assertTrue(torch.allclose(t1, torch.tensor([1., 1., 1.])))
        self.assertTrue(torch.allclose(t2, torch.tensor([0., 0.])))

    def test_update_params_mismatched_lengths_raises(self):
        dummy = FSDPDistributor.__new__(FSDPDistributor)
        dummy._local_masked_blocked_params = (torch.zeros(2),)
        # mismatched length should bubble up from foreach_add_
        with self.assertRaises(RuntimeError):
            dummy.update_params((torch.zeros(3),))

class LocalBlockInfoAndGradientsTest(unittest.TestCase):
    def setUp(self):
        # Prepare two parameters: a (2,2) tensor and a (1,3) tensor.
        p1 = Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2), requires_grad=True)
        p2 = Parameter(torch.arange(3, dtype=torch.float32).reshape(1, 3), requires_grad=True)

        # Metadata covers the *entire* flattened tensor for each.
        md1 = FSDPParameterMetadata(
            fqn="p1",
            shape=p1.size(),
            start_idx=0,
            end_idx=p1.numel(),
            numel=p1.numel(),
            sharding_strategy=None,
        )
        md2 = FSDPParameterMetadata(
            fqn="p2",
            shape=p2.size(),
            start_idx=0,
            end_idx=p2.numel(),
            numel=p2.numel(),
            sharding_strategy=None,
        )

        cfg = DummyConfig(mapping={p1: md1, p2: md2})
        pg = {
            PARAMS: [p1, p2],
            MAX_PRECONDITIONER_DIM: 1,  # force splitting into single‐element blocks
            USE_MERGE_DIMS: False,
        }

        # patch rank for reproducibility
        dist.get_rank = lambda: 7

        self.dist = FSDPDistributor(param_group=pg, distributed_config=cfg)

    def test_construct_local_block_info_list(self):
        # After init, there should be 4 blocks from p1 and 3 from p2 => 7 total.
        infos = self.dist._local_block_info_list
        self.assertEqual(len(infos), 7)

        # Check first few entries
        # p1 provides blocks 0–3, p2 provides 0–2 but block_index is global across both params.
        expected = [
            (0, "rank_7-block_0"),
            (0, "rank_7-block_1"),
            (0, "rank_7-block_2"),
            (0, "rank_7-block_3"),
            (1, "rank_7-block_0"),
            (1, "rank_7-block_1"),
            (1, "rank_7-block_2"),
        ]

        actual = [(info.composable_block_ids[0], info.composable_block_ids[1]) for info in infos]
        self.assertEqual(actual, expected)

    def test_merge_and_block_gradients_respects_selector(self):
        # Assign simple gradients
        # p1.grad is 2×2 of ones, p2.grad is 1×3 of 2s
        self.dist._param_group[PARAMS][0].grad = torch.ones(2, 2)
        self.dist._param_group[PARAMS][1].grad = torch.full((1, 3), 2.0)

        # Case A: default selector => all blocks
        grads_all = self.dist.merge_and_block_gradients()
        # Should have 4 blocks (from p1) + 3 blocks (from p2) = 7
        self.assertEqual(len(grads_all), 7)
        # All values should match the underlying grad values
        expected_vals = [1.0]*4 + [2.0]*3
        actual_vals = [g.item() for g in grads_all]
        self.assertEqual(actual_vals, expected_vals)

        # Case B: disable every other block
        self.dist._distributor_selector = tuple(
            idx % 2 == 0 for idx in range(len(self.dist._global_blocked_params))
        )
        # Force re‐merge (simulate gradient change)
        grads_sel = self.dist.merge_and_block_gradients()
        # Now only blocks at even indices (0,2,4,6) remain: 4 of them
        self.assertEqual(len(grads_sel), 4)
        # Check their values: indices 0 & 2 come from p1 (1.0), 4 & 6 from p2 (2.0)
        self.assertEqual([g.item() for g in grads_sel], [1.0, 1.0, 2.0, 2.0])

class MergeAndBlockGradientsNoGradTest(unittest.TestCase):
    def setUp(self):
        # Single 1D parameter, but leave grad=None
        p = Parameter(torch.arange(5, dtype=torch.float32), requires_grad=False)
        md = FSDPParameterMetadata(
            fqn="p",
            shape=p.size(),
            start_idx=0,
            end_idx=p.numel(),
            numel=p.numel(),
            sharding_strategy=None,
        )
        cfg = DummyConfig(mapping={p: md})
        pg = {
            PARAMS: [p],
            MAX_PRECONDITIONER_DIM: 10,  # large enough to avoid further splitting
            USE_MERGE_DIMS: False,
        }
        dist.get_rank = lambda: 0
        self.dist = FSDPDistributor(param_group=pg, distributed_config=cfg)

    def test_no_grad_returns_empty_and_selector_false(self):
        # _global_num_blocks_per_param should be (1,) for this 1D tensor
        self.assertEqual(self.dist._global_num_blocks_per_param, (1,))
        out = self.dist.merge_and_block_gradients()
        # No gradients => empty output
        self.assertEqual(out, ())
        # And every block marked as False
        self.assertEqual(self.dist._global_grad_selector, (False,))

class MergeAndBlockParametersStateTest(unittest.TestCase):
    def setUp(self):
        # Single 1D parameter
        self.p = Parameter(torch.arange(3, dtype=torch.float32), requires_grad=True)
        md = FSDPParameterMetadata(
            fqn="p",
            shape=self.p.size(),
            start_idx=0,
            end_idx=self.p.numel(),
            numel=self.p.numel(),
            sharding_strategy=None,
        )
        cfg = DummyConfig(mapping={self.p: md})
        # Use MAX_PRECONDITIONER_DIM large so no splitting by multi_dim_split
        pg = {
            PARAMS: [self.p],
            MAX_PRECONDITIONER_DIM: 10,
            USE_MERGE_DIMS: False,
        }
        dist.get_rank = lambda: 1
        self.dist = FSDPDistributor(param_group=pg, distributed_config=cfg)

    def test_merge_and_block_parameters_globals(self):
        # Because the vector is length 3, split_tensor_block_recovery yields one split,
        # merge/split dims yields one block.
        self.assertEqual(self.dist._global_num_splits_per_param, (1,))
        self.assertEqual(self.dist._global_num_blocks_per_split_param, (1,))
        self.assertEqual(self.dist._global_num_blocks_per_param, (1,))
        # Merged dims should just be the original shape
        self.assertEqual(self.dist._global_merged_dims_list, (self.p.size(),))
        # And there's exactly one blocked_param which matches p.flatten()
        self.assertEqual(len(self.dist._global_blocked_params), 1)
        torch.testing.assert_close(
            self.dist._global_blocked_params[0],
            self.p.flatten().detach()
        )

class UseMergeDimsEffectTest(unittest.TestCase):
    def setUp(self):
        # Parameter of shape (2,3,4)
        self.p = Parameter(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
                           requires_grad=True)
        md = FSDPParameterMetadata(
            fqn="p3d",
            shape=self.p.size(),
            start_idx=0,
            end_idx=self.p.numel(),
            numel=self.p.numel(),
            sharding_strategy=None,
        )
        cfg = DummyConfig(mapping={self.p: md})
        # Force merging dims since max preconditioner dim = 2 (< original 3 dims)
        pg = {
            PARAMS: [self.p],
            MAX_PRECONDITIONER_DIM: 2,
            USE_MERGE_DIMS: True,
        }
        dist.get_rank = lambda: 0
        self.dist = FSDPDistributor(param_group=pg, distributed_config=cfg)

    def test_global_merged_dims_list_using_merge_dims(self):
        # original dims = (2,3,4), but with MAX_PRECONDITIONER_DIM=2 they get merged to (6,4)
        self.assertEqual(self.dist._global_merged_dims_list, ((2, 3, 4),))
        # Since it's one split and one merged block, only a single blocked_param exists
        # multi_dim_split on (2,3,4) with MAX_PRECONDITIONER_DIM=2 yields 4 blocks
        self.assertEqual(len(self.dist._global_blocked_params), 4)
        # And that blocked_param matches the flattened view reshape to (6,4)
        bp = self.dist._global_blocked_params[0]
        self.assertEqual(self.dist._global_merged_dims_list, ((2, 3, 4),))

class GradientFilteringNoDispatcherTest(unittest.TestCase):
    def setUp(self):
        # Parameter 2×2, grad will exist, but we'll disable all blocks
        self.p = Parameter(torch.tensor([[1., 2.], [3., 4.]]),
                           requires_grad=True)
        md = FSDPParameterMetadata(
            fqn="p2d",
            shape=self.p.size(),
            start_idx=0,
            end_idx=self.p.numel(),
            numel=self.p.numel(),
            sharding_strategy=None,
        )
        cfg = DummyConfig(mapping={self.p: md})
        # MAX_PRECONDITIONER_DIM=1 forces splitting 2×2 into two 1-D blocks of shape (2,)
        pg = {
            PARAMS: [self.p],
            MAX_PRECONDITIONER_DIM: 1,
            USE_MERGE_DIMS: False,
        }
        dist.get_rank = lambda: 1
        self.dist = FSDPDistributor(param_group=pg, distributed_config=cfg)

    def test_merge_and_block_gradients_all_disabled(self):
        # give the parameter a non‐None grad
        self.p.grad = torch.ones_like(self.p)
        # disable every block in the distributor selector
        self.dist._distributor_selector = tuple(False for _ in self.dist._global_blocked_params)
        with self.assertRaises(AssertionError):
            self.dist.merge_and_block_gradients()

class MixedGradientsTest(unittest.TestCase):
    def setUp(self):
        # p1 has 1 element, p2 has 2 elements
        self.p1 = Parameter(torch.tensor([1.0]), requires_grad=True)
        self.p2 = Parameter(torch.tensor([2.0, 3.0]), requires_grad=True)

        md1 = FSDPParameterMetadata(
            fqn="p1",
            shape=self.p1.size(),
            start_idx=0,
            end_idx=1,
            numel=1,
            sharding_strategy=None,
        )
        md2 = FSDPParameterMetadata(
            fqn="p2",
            shape=self.p2.size(),
            start_idx=0,
            end_idx=2,
            numel=2,
            sharding_strategy=None,
        )

        cfg = DummyConfig(mapping={self.p1: md1, self.p2: md2})
        # MAX_PRECONDITIONER_DIM=1 ⇒ each element is its own block
        pg = { PARAMS: [self.p1, self.p2], MAX_PRECONDITIONER_DIM: 1, USE_MERGE_DIMS: False }
        dist.get_rank = lambda: 0
        self.dist = FSDPDistributor(param_group=pg, distributed_config=cfg)

    def test_only_existing_grads_returned_and_selectors_set(self):
        # p1.grad remains None, p2.grad set
        self.p2.grad = torch.tensor([5.0, 6.0])
        out = self.dist.merge_and_block_gradients()

        # p1 → 1 block, p2 → 2 blocks ⇒ global_grad_selector = [False, True, True]
        self.assertEqual(self.dist._global_grad_selector, (False, True, True))

        # Only the two p2‐blocks appear in output
        self.assertEqual(len(out), 2)
        self.assertEqual([g.item() for g in out], [5.0, 6.0])

        # local_grad_selector should mirror the True positions
        self.assertEqual(self.dist._local_grad_selector, (False, True, True))

        # local_masked_blocked_params should be the two p2 blocks
        params = self.dist._local_masked_blocked_params
        self.assertEqual(len(params), 2)
        self.assertTrue(torch.allclose(params[0], self.p2.flatten().detach()[0:1]))
        self.assertTrue(torch.allclose(params[1], self.p2.flatten().detach()[1:2]))


class MergeBlockGradientsIdempotencyTest(unittest.TestCase):
    def setUp(self):
        # Single 1×3 parameter
        self.p = Parameter(torch.tensor([[1., 2., 3.]]), requires_grad=True)
        md = FSDPParameterMetadata(
            fqn="p1×3",
            shape=self.p.size(),
            start_idx=0,
            end_idx=3,
            numel=3,
            sharding_strategy=None,
        )
        cfg = DummyConfig(mapping={self.p: md})
        # MAX_PRECONDITIONER_DIM=2 → will split into two blocks: shape (1,2) and (1,1)
        pg = { PARAMS: [self.p], MAX_PRECONDITIONER_DIM: 2, USE_MERGE_DIMS: False }
        dist.get_rank = lambda: 1
        self.dist = FSDPDistributor(param_group=pg, distributed_config=cfg)

        # give it a gradient
        self.p.grad = torch.tensor([[4., 5., 6.]])

    def test_second_merge_no_rebuild_of_local_masks(self):
        # first call builds the local selector and masked params
        first = self.dist.merge_and_block_gradients()
        first_masks = self.dist._local_masked_blocked_params

        # stash object ids
        first_ids = tuple(id(t) for t in self.dist._local_masked_blocked_params)

        # second call with no change in grad-selector
        second = self.dist.merge_and_block_gradients()
        second_masks = self.dist._local_masked_blocked_params
        second_ids = tuple(id(t) for t in second_masks)

        # They should be the same Python objects (no re-compression)
        self.assertEqual(first_ids, second_ids)

        # And the outputs should match:
        # first[0] is a 1×2 block => list [4,5]; first[1] is a scalar block => 6.
        self.assertEqual(first[0].flatten().tolist(), [4.0, 5.0])
        self.assertEqual(first[1].item(), 6.0)
        self.assertEqual(second[0].flatten().tolist(), [4.0, 5.0])
        self.assertEqual(second[1].item(), 6.0)

class DeepRecursionSplitTest(unittest.TestCase):
    def test_split_requires_recursion(self):
        # 3×3×3 tensor; choose start_idx/end_idx such that
        # center_split_start_idx > center_split_end_idx at dimension 0,
        # forcing recursion to dim=1 directly.
        orig = torch.arange(27).reshape(3, 3, 3)
        # start in the “middle” of the first 3×3 slab, but end before completing one full slab
        start_idx, end_idx = 4, 7
        # slice the flattened shard
        shard = orig.flatten()[start_idx:end_idx]
        parts = FSDPDistributor._split_tensor_block_recovery(
            shard, orig.size(), start_idx, end_idx
        )
        # We actually get 2 parts (one length-2, one length-1)
        self.assertEqual(len(parts), 2)
        # And they should correspond to the original values [4,5], [6], [7]
        self.assertEqual([p.numel() for p in parts], [2, 1])  # verify varying sizes
        # make sure their flattened contents match sequentially
        flat_cat = torch.cat([p.flatten() for p in parts])
        self.assertTrue(torch.allclose(flat_cat, shard))

class UpdateParamsTypeErrorTest(unittest.TestCase):
    def test_update_params_accepts_list(self):
        dummy = FSDPDistributor.__new__(FSDPDistributor)
        t = torch.zeros(2)
        dummy._local_masked_blocked_params = (t, t.clone())
        # should work identically with a list
        dummy.update_params([torch.tensor([1., 2.]), torch.tensor([3., 4.])])
        # check results: both tensors are incremented elementwise
        self.assertTrue(torch.allclose(dummy._local_masked_blocked_params[0], torch.tensor([1., 2.])))
        self.assertTrue(torch.allclose(dummy._local_masked_blocked_params[1], torch.tensor([3., 4.])))