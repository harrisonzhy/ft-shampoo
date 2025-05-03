import unittest
from unittest.mock import patch
import torch

# Replace `shampoo_preconditioner` with the actual module path of your code under test.
from shampoo_preconditioner_list import (
    encode_shard,
    decode_shard,
    ShampooPreconditionerList,
    ShampooKroneckerFactorsList,
)

def encode_pickle(data):
    """Helper to compare pickled content if needed."""
    import pickle
    return pickle.dumps(data)

class DummyShampoo(ShampooPreconditionerList):
    """Minimal subclass to inject testable internal state."""
    def __init__(self):
        # Bypass the original __init__
        # Manually set up the masked lists
        self._masked_order_list = (1, 2, 3)
        self._masked_roots_list = ((0.5,), (0.3,), (0.2,))
        self._masked_preconditioned_dims_selector_list = (
            (True,),
            (False, True),
            (True, True, True),
        )
        self._masked_failed_amortized_computation_counter_list = [0, 1, 2]
        self._bias_correction2 = torch.tensor(1.234)
        # Build a fake kronecker list of length 3
        kfs = []
        for i in range(3):
            mats = (torch.eye(i + 1),)
            invs = (torch.eye(i + 1) * 2.0,)
            idxs = (f"{i}.0",)
            kfs.append(ShampooKroneckerFactorsList(
                factor_matrices=mats,
                inv_factor_matrices=invs,
                factor_matrix_indices=idxs,
            ))
        self._masked_kronecker_factors_list = tuple(kfs)


class TestShampooPreconditioner(unittest.TestCase):
    def test_encode_decode_shard_roundtrip(self):
        data = ("foo", 42, [1, 2, 3])
        frags = encode_shard(data, n_fragments=5, nsym=2)
        self.assertIsInstance(frags, tuple)
        self.assertEqual(len(frags), 5)
        out = decode_shard(list(frags), nsym=2)
        self.assertEqual(out, data)

    def test_encode_shard_fragment_sizes_and_remainder(self):
        data = tuple(range(100))
        frags = encode_shard(data, n_fragments=3, nsym=1)
        sizes = [len(f) for f in frags]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_decode_shard_too_many_corrupt_raises(self):
        data = ("bar", 99)
        frags = list(encode_shard(data, n_fragments=4, nsym=2))
        # corrupt 3 shards (nsym=2 → can only correct up to 2)
        for i in [0, 1, 2]:
            frags[i] = b"\x00" * len(frags[i])
        with self.assertRaises(Exception):
            decode_shard(frags, nsym=2)

    @patch('torch.distributed.get_world_size', return_value=2)
    def test_fragmented_and_load_state_dict_roundtrip(self, mock_get_ws):
        inst = DummyShampoo()

        # 1) Fragment the state
        shards = inst.fragmented_state_dict(n_parity=1)

        # 2) Decode every shard back into its original Python value
        decoded = {}
        for key, v in shards.items():
            if key != "masked_kronecker_factors_list":
                # v is a tuple of byte‐fragments → reconstruct the original tuple/number
                decoded[key] = decode_shard(list(v), nsym=1)
            else:
                # v is a List[Tuple[bytes,...]] → for each factor, decode into dict
                decoded_kfs = [decode_shard(list(frag), nsym=1) for frag in v]
                decoded[key] = decoded_kfs

        # 3) Load into a fresh instance
        inst2 = DummyShampoo()
        inst2.load_state_dict(decoded)

        # 4) Compare scalar fields
        self.assertEqual(inst2._masked_order_list, inst._masked_order_list)
        self.assertEqual(inst2._masked_roots_list, inst._masked_roots_list)
        self.assertEqual(
            inst2._masked_preconditioned_dims_selector_list,
            inst._masked_preconditioned_dims_selector_list
        )
        self.assertEqual(
            inst2._masked_failed_amortized_computation_counter_list,
            inst._masked_failed_amortized_computation_counter_list
        )
        self.assertTrue(torch.allclose(inst2._bias_correction2, inst._bias_correction2))

        # 5) Finally, make sure the kronecker factors round‐trip correctly
        for a, b in zip(
            inst2._masked_kronecker_factors_list,
            inst._masked_kronecker_factors_list
        ):
            for t1, t2 in zip(a.factor_matrices, b.factor_matrices):
                self.assertTrue(torch.equal(t1, t2))
            for t1, t2 in zip(a.inv_factor_matrices, b.inv_factor_matrices):
                self.assertTrue(torch.equal(t1, t2))

    def test_serialize_deserialize_kronecker_factors_roundtrip(self):
        mats = (torch.randn(2, 2), torch.randn(3, 3))
        invs = (torch.eye(2), torch.eye(3) * 0.5)
        idxs = ("p.0", "p.1")
        kf = ShampooKroneckerFactorsList(
            factor_matrices=mats,
            inv_factor_matrices=invs,
            factor_matrix_indices=idxs,
        )
        inst = DummyShampoo()
        state = inst._serialize_kronecker_factors(kf)
        self.assertSetEqual(set(state.keys()),
                            {"factor_matrices", "inv_factor_matrices", "factor_matrix_indices"})
        for t in state["factor_matrices"] + state["inv_factor_matrices"]:
            self.assertEqual(t.device, torch.device("cpu"))

        kf2 = inst._deserialize_kronecker_factors(state)
        self.assertEqual(kf2.factor_matrix_indices, idxs)
        for a, b in zip(kf2.factor_matrices, mats):
            self.assertTrue(torch.allclose(a, b.cpu()))
        for a, b in zip(kf2.inv_factor_matrices, invs):
            self.assertTrue(torch.allclose(a, b.cpu()))

    @patch('torch.distributed.get_world_size', return_value=3)
    @patch('torch.distributed.all_gather_object')
    def test_gather_and_store_others(self, mock_all_gather, mock_get_ws):
        inst = DummyShampoo()
        def fake_all_gather(target_list, local):
            for i in range(3):
                target_list[i] = local
        mock_all_gather.side_effect = fake_all_gather

        inst.broadcast_and_store_others()
        keys = set(inst._gathered_peer_preconditioner_states.keys())
        self.assertEqual(keys, {0, 1, 2})
        for st in inst._gathered_peer_preconditioner_states.values():
            self.assertIn("masked_order_list", st)

    @patch('torch.distributed.get_world_size', return_value=3)
    @patch('torch.distributed.all_gather_object')
    def test_gather_all_state_dicts(self, mock_all_gather, mock_get_ws):
        inst = DummyShampoo()
        # stub out fragmented_state_dict to return a known marker
        marker = {'foo': 'bar'}
        inst.fragmented_state_dict = lambda n_parity: marker

        # fake all_gather_object: fill target_list with local marker
        def fake_gather(target_list, local):
            for i in range(3):
                target_list[i] = local
        mock_all_gather.side_effect = fake_gather

        collected = inst.gather_all_state_dicts()
        # should return a list of length world_size, each element == marker
        self.assertIsInstance(collected, list)
        self.assertEqual(len(collected), 3)
        for entry in collected:
            self.assertIs(entry, marker)

    def test_precondition_simple_inv_factors(self):
        # Build a 1-element gradient and a 1×1 inverse-factor = [[2.0]]
        dummy = DummyShampoo()
        dummy._masked_preconditioned_dims_selector_list = ((True,),)
        # single ShampooKroneckerFactorsList with a trivial inv factor
        inv = torch.tensor([[2.0]])
        kf = ShampooKroneckerFactorsList(
            factor_matrices=(torch.eye(1),),
            inv_factor_matrices=(inv,),
            factor_matrix_indices=('0.0',),
        )
        dummy._masked_kronecker_factors_list = (kf,)

        grad = torch.tensor([5.0])
        out = dummy.precondition((grad,))
        # tensordot over dims=([0],[0]) : 5.0 * 2.0 = 10.0
        self.assertEqual(len(out), 1)
        self.assertTrue(torch.equal(out[0], torch.tensor([10.0])))

    @patch('torch.distributed.get_world_size', return_value=2)
    def test_fragmented_state_dict_keys(self, mock_get_ws):
        inst = DummyShampoo()
        shards = inst.fragmented_state_dict(n_parity=1)
        expected = {
            "masked_order_list",
            "masked_roots_list",
            "masked_preconditioned_dims_selector_list",
            "masked_failed_amortized_computation_counter_list",
            "bias_correction2",
            "masked_kronecker_factors_list",
        }
        self.assertSetEqual(set(shards.keys()), expected)

    def test_encode_shard_single_fragment_nsym_zero_raises(self):
        # With 1 fragment and 0 parity symbols, Reed–Solomon decode should fail.
        data = ("alpha", "beta")
        frags = encode_shard(data, n_fragments=1, nsym=0)
        self.assertEqual(len(frags), 1)
        self.assertTrue(isinstance(frags[0], (bytes, bytearray)))
        with self.assertRaises(Exception):
            # decoding with nsym=0 raises EOFError / RSCodecError
            decode_shard(list(frags), nsym=0)

    def test_encode_shard_single_fragment_nsym_one_roundtrip(self):
        # With 1 fragment and 1 parity symbol, encoding+decoding should succeed.
        data = ("alpha", "beta")
        frags = encode_shard(data, n_fragments=1, nsym=1)
        self.assertEqual(len(frags), 1)
        out = decode_shard(list(frags), nsym=1)
        self.assertEqual(out, data)

    def test_encode_decode_shard_complex_object(self):
        # ensure that nested mappings round-trip correctly
        data = (
            {"x": [1, 2, 3], "z": {"inner": True}},
            3.14159,
            ("tuple", {"nested": [0, 1]}),
        )
        frags = encode_shard(data, n_fragments=3, nsym=1)
        self.assertEqual(len(frags), 3)
        out = decode_shard(list(frags), nsym=1)
        self.assertEqual(out, data)

    def test_decode_shard_single_byte_corruption_success(self):
        # nsym=2 can correct up to 2 byte‐errors; we flip one byte in one fragment
        data = ("ok", 123, [9, 8, 7])
        frags = list(encode_shard(data, n_fragments=4, nsym=2))

        # Corrupt JUST the first byte of fragment 1
        b1 = bytearray(frags[1])
        b1[0] ^= 0xFF
        frags[1] = bytes(b1)

        # Should still decode correctly
        out = decode_shard(frags, nsym=2)
        self.assertEqual(out, data)

    def test_decode_shard_too_many_byte_errors_raises(self):
        # nsym=2 → can correct up to 2 errors, but here we introduce 3 single-byte errors
        data = ("fail", 456, [1, 2, 3])
        frags = list(encode_shard(data, n_fragments=4, nsym=2))

        # Flip one byte in each of three fragments
        for i in (0, 1, 2):
            b = bytearray(frags[i])
            b[0] ^= 0xFF
            frags[i] = bytes(b)

        # Now Reed–Solomon should consider this “too many errors”
        with self.assertRaises(Exception):
            decode_shard(frags, nsym=2)

    def test_kronecker_factors_list_bad_lengths(self):
        # factor_matrices, inv_factor_matrices, and indices must all have same length
        mats = (torch.eye(2),)
        invs = (torch.eye(2), torch.eye(2))
        idxs = ("0.0",)  # only one index
        with self.assertRaises(AssertionError):
            ShampooKroneckerFactorsList(
                factor_matrices=mats,
                inv_factor_matrices=invs,
                factor_matrix_indices=idxs,
            )

    @patch('torch.distributed.get_world_size', return_value=4)
    def test_fragmented_state_dict_default_parity(self, mock_get_ws):
        inst = DummyShampoo()
        # explicit parity == world_size
        shards_explicit = inst.fragmented_state_dict(n_parity=4)
        # default parity == -1 → should pick up world_size
        shards_default = inst.fragmented_state_dict(n_parity=-1)

        # same keys
        self.assertSetEqual(set(shards_explicit.keys()), set(shards_default.keys()))

        for key in shards_explicit:
            v_exp = shards_explicit[key]
            v_def = shards_default[key]
            if key != "masked_kronecker_factors_list":
                # should be 4-tuples and identical
                self.assertIsInstance(v_exp, tuple)
                self.assertEqual(len(v_exp), 4)
                self.assertEqual(v_exp, v_def)
            else:
                # should be a list of length >0, each a 4-tuple
                self.assertIsInstance(v_exp, list)
                self.assertEqual(len(v_exp), len(v_def))
                for frag in v_exp:
                    self.assertIsInstance(frag, tuple)
                    self.assertEqual(len(frag), 4)

    def test_encode_shard_invalid_n_fragments_raises(self):
        # n_fragments must be >= 1
        data = ("only_one",)
        with self.assertRaises(Exception):
            encode_shard(data, n_fragments=0, nsym=1)

    def test_decode_shard_empty_list_raises(self):
        # You cannot decode zero fragments
        with self.assertRaises(Exception):
            decode_shard([], nsym=1)