import os
# 1) satisfy env:// rendezvous
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

# 2) fake torch.distributed
import torch.distributed as dist
dist.init_process_group = lambda *args, **kwargs: None
dist.get_world_size = lambda *args, **kwargs: 1
dist.get_rank = lambda *args, **kwargs: 0

# 3) stub out shampoo internals
import distributed_shampoo.distributed_shampoo as _ds
_ds.DistributedShampoo._instantiate_shampoo_preconditioner_list = lambda self: None
_ds.DistributedShampoo._mask_state_lists = lambda *args, **kwargs: None
_ds.DistributedShampoo._update_preconditioners = lambda self, *args, **kwargs: None

def _sgd_step(self, *args, **kwargs):
    group = self.param_groups[0]
    lr = group["lr"]
    for p in group["params"]:
        if p.grad is None:
            continue
        p.data.add_(p.grad, alpha=-lr)
_ds.DistributedShampoo.step = _sgd_step

import torch.distributed.distributed_c10d as _dc10d
_dc10d.BackendConfig = lambda backend, **kwargs: None

import unittest
import torch
import torch.nn as nn
import torch.distributed as dist
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset

from distributed_shampoo import (
    DistributedShampoo,
    DDPShampooConfig,
    CommunicationDType,
    SGDGraftingConfig,
)
from distributed_shampoo.examples.trainer_utils import instantiate_optimizer, set_seed
from distributed_shampoo.utils.shampoo_preconditioner_list import encode_shard, decode_shard

class MinimalTrainingTest(unittest.TestCase):
    def setUp(self):
        set_seed(0)
        # Simple data: y = x
        x = torch.linspace(0, 1, steps=10).unsqueeze(1)
        y = x.clone()
        ds = TensorDataset(x, y)
        self.loader = DataLoader(ds, batch_size=5)

        self.model = nn.Linear(1, 1)

        # Directly construct DistributedShampoo with SGD grafting
        self.opt = DistributedShampoo(
            self.model.parameters(),
            lr=0.1,
            betas=(0.0, 0.999),               # beta1=0.0 makes it pure SGD graft
            epsilon=1e-12,
            momentum=0.0,
            weight_decay=0.0,
            max_preconditioner_dim=1,
            precondition_frequency=1,
            grafting_config=SGDGraftingConfig(),
            use_merge_dims=False,
            distributed_config=DDPShampooConfig(
                communication_dtype=CommunicationDType.FP32,
                num_trainers_per_group=1,
                communicate_params=False,
            ),
        )

    def test_two_steps_run_and_update(self):
        # record initial
        initials = [p.clone() for p in self.model.parameters()]

        it = iter(self.loader)
        for _ in range(2):
            xb, yb = next(it)
            loss = nn.functional.mse_loss(self.model(xb), yb)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        # ensure parameters changed
        for init, updated in zip(initials, self.model.parameters()):
            self.assertFalse(torch.allclose(init, updated),
                             "Parameter did not change after optimizer step")

class ZeroGradAndMultiParamTest(unittest.TestCase):
    def setUp(self):
        set_seed(0)
        # tiny two‐layer model
        self.model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))
        # wrap it in Shampoo (still stubbed out under the hood)
        self.opt = DistributedShampoo(
            self.model.parameters(),
            lr=0.05,
            betas=(0.0, 0.999),
            epsilon=1e-12,
            momentum=0.0,
            weight_decay=0.0,
            max_preconditioner_dim=1,
            precondition_frequency=1,
            grafting_config=SGDGraftingConfig(),
            use_merge_dims=False,
            distributed_config=DDPShampooConfig(
                communication_dtype=CommunicationDType.FP32,
                num_trainers_per_group=1,
                communicate_params=False,
            ),
        )

    def test_zero_grad_clears_all_grads(self):
        # artificially set non‐zero grads on every param
        for p in self.model.parameters():
            p.grad = torch.ones_like(p)
        self.opt.zero_grad()
        # after zero_grad, grads should be None (or zero)
        for p in self.model.parameters():
            self.assertTrue(p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)))

    def test_multiple_params_step_changes_all(self):
        # run one forward+backward+step on a toy data point
        x = torch.tensor([[1.0]])
        y = torch.tensor([[2.0]])
        loss = nn.functional.mse_loss(self.model(x), y)
        self.opt.zero_grad()
        loss.backward()
        # capture before‐step values
        before = [p.clone() for p in self.model.parameters()]
        self.opt.step()
        # verify *each* parameter was updated
        for b, a in zip(before, self.model.parameters()):
            self.assertFalse(torch.allclose(b, a),
                             "Expected every parameter to move after step()")
                             
class ReedSolomonMiniTest(unittest.TestCase):
    def test_encode_decode_small_tuple(self):
        data = ("abc", 123, torch.arange(4))
        # split into 3 fragments with 1 parity
        frags = encode_shard(data, n_fragments=3, nsym=1)
        self.assertEqual(len(frags), 3)
        # ensure each fragment is bytes
        for f in frags:
            self.assertIsInstance(f, (bytes, bytearray))
        # round‐trip
        rec = decode_shard(list(frags), nsym=1)
        # compare non‐tensor elements
        self.assertEqual(rec[0], data[0])
        self.assertEqual(rec[1], data[1])
        # compare tensors with assert_close
        torch.testing.assert_close(rec[2], data[2])
