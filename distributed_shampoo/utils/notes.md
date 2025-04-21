## Setup
```sh
source env.sh
conda create -n "shampoo_env" python==3.11.0
conda activate shampoo_env
conda install https://anaconda.org/pytorch/pytorch-cuda/11.8/download/linux-64/pytorch-cuda-11.8-h7e8668a_5.tar.bz2
```
```sh
x-hzhang23@login05.anvil.rcac:[ft-shampoo] $ sinteractive --account=cis240216-gpu --partition=gpu  --time=1:00:00 --gres=gpu:1 --mem-per-gpu=20g --gpus-per-node=1
```

## Look at `shampoo_fdsp_distributor.py`
`shampoo_fdsp_distributor.py` implements `function_()` functions used by `function()` functions in the same file
as well as base class in `shampoo_distributor.py`, which is `DistributorInterface`.
```py
    elif optimizer_type == OptimizerType.DISTRIBUTED_SHAMPOO:
        optimizer = DistributedShampoo(
            model.parameters(),
            lr=lr,
            betas=betas,
            beta3=beta3,
            epsilon=epsilon,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            max_preconditioner_dim=max_preconditioner_dim,
            precondition_frequency=precondition_frequency,
            start_preconditioning_step=start_preconditioning_step,
            use_nesterov=use_nesterov,
            use_bias_correction=use_bias_correction,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            grafting_config=instantiate_grafting_config(
                grafting_type, grafting_beta2, grafting_epsilon
            ),
            use_merge_dims=use_merge_dims,
            distributed_config=distributed_config,
            preconditioner_dtype=preconditioner_dtype.value,
            preconditioner_config=instantiate_preconditioner_config(
                preconditioner_computation_type=preconditioner_computation_type,
                exponent_multiplier=exponent_multiplier,
            ),
        )  # type: ignore[assignment]
```

## Look at `distributed_shampoo.py`:
```py
@torch.no_grad()
def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
```
This calls:
```py
self._per_group_step(
                state_lists,
                step,
                lr,
                beta1,
                beta3,
                weight_decay,
                momentum_param,
                dampening,
                grafting_config_not_none,
                perform_amortized_computation,
                use_decoupled_weight_decay,
                use_bias_correction,
                use_grafting_method,
                use_nesterov,
            )
```
which calls:
```py
@torch.no_grad()
def _per_group_step_impl(
    self,
    state_lists: dict[str, Any],
    step: torch.Tensor,
    lr: torch.Tensor,
    beta1: float,
    beta3: float,
    weight_decay: float,
    momentum_param: float,
    dampening: float,
    grafting_config_not_none: bool,
    perform_amortized_computation: bool,
    use_decoupled_weight_decay: bool,
    use_bias_correction: bool,
    use_grafting_method: bool,
    use_nesterov: bool,
) -> None:
...
# Update Shampoo and grafting preconditioners.
        # Example for AdaGrad accumulation:
        # 1. Update factor matrices/grafting preconditioners.
        #   L <- L + G * G^T
        #   R <- R + G^T * G
        #   V <- V + G^2    (element-wise)
        #   (and similar)
        # 2. Compute root inverse if necessary.
        #   L_inv <- L ** (-1/4)
        #   R_inv <- R ** (-1/4)
        #   (and similar);
self._update_preconditioners(
            state_lists=state_lists,
            step=step,
            perform_amortized_computation=perform_amortized_computation,
            grafting_config_not_none=grafting_config_not_none,
        )
...
# Precondition and graft filtered gradients.
# PT2 compile is currently disabled for preconditioning and grafting.
# NOTE: Preconditioning and grafting is not compatible with PT2 compile.
#
#   P_shampoo <- L_inv * G_bar * R_inv (and similar)
#   P_grafting <- G_bar / (sqrt(V) + epsilon)
#   P <- P_grafting                                     if step < start_preconditioning_step
#   P <- ||P_grafting|| / ||P_shampoo|| * P_shampoo     otherwise
masked_blocked_search_directions = self._precondition_and_grafting(
    state_lists,
    masked_filtered_grad_list,
    use_grafting_method,
    grafting_config_not_none,
)
```
which (depending on if we want eigval correction) calls either this `BaseShampooPreconditionerList` base class method:
```py
    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool,
    ) -> None:
        """
        Updates the preconditioners.

        Args:
            masked_grad_list (tuple[Tensor, ...]): A list of gradients with their corresponding masks.
            step (Tensor): The current step.
            perform_amortized_computation (bool): Whether to perform an amortized computation.

        Returns:
            None
        """
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_preconditioners.__name__} ##"
        ):
            # Update the Kronecker factor matrices.
            self._update_factor_matrices(masked_grad_list=masked_grad_list)

            # Update bias correction term based on step.
            if self._use_bias_correction and self._beta2 < 1.0:
                self._bias_correction2 = torch.tensor(1.0) - self._beta2**step

            # In Shampoo, this is equivalent to computing the inverse factor matrix.
            # In eigenvalue-corrected Shampoo, this is equivalent to computing the eigenvectors of the factor matrix.
            if perform_amortized_computation:
                self._amortized_computation()
```
or the `EigenvalueCorrectedShampooPreconditionerList`, which calls the base class method's `update_preconditioners(...)`, then calls **eigval correction** function (maybe important). For now, we might be able to only focus on updating the base class.

### Update weights
At the end of `_per_group_step_impl(...)`, it updates weights by calling:
```py
# Updates parameters in distributed fashion.
# If DDP, executes AllGather communication to ensure all parameters are updated after local updates.
torch._foreach_mul_(masked_blocked_search_directions, -lr)
state_lists[DISTRIBUTOR].update_params(
    masked_blocked_search_directions=masked_blocked_search_directions
)
```


## Merge and block gradients
Also, we have:
```py
def merge_and_block_gradients(
        self,
    ) -> tuple[Tensor, ...]:
        """Merge and block gradients.

        NOTE: This function MUST be called in the step function of the optimizer after the
        gradient has been updated.

        Returns:
            local_masked_blocked_grads (tuple[Tensor, ...]): Local blocked gradients masked with grad existence.

        """
```
calls:
```py
def _merge_and_block_gradients(
    self,
) -> tuple[Tensor, ...]:
    """Split, merge, and block gradients.

    Returns:
        local_masked_blocked_grads (tuple[Tensor, ...]): Local gradients with grad not None.

    """
    local_masked_blocked_grads: list[Tensor] = []
    global_grad_selector = []

    for (
        flattened_param,
        num_blocks,
        (block_index, next_block_index),
        (split_index, next_split_index),
    ) in zip(
        self._param_group[PARAMS],
        self._global_num_blocks_per_param,
        generate_pairwise_indices(self._global_num_blocks_per_param),
        generate_pairwise_indices(self._global_num_splits_per_param),
        strict=True,
    ):
        flattened_grad = flattened_param.grad
        param_distributor_selector = self._distributor_selector[
            block_index:next_block_index
        ]

        # Update the selector.
        global_grad_selector.extend([flattened_grad is not None] * num_blocks)

        if flattened_grad is None or not any(param_distributor_selector):
            # Skip split_tensor_block_recovery and multi_dim_split if this blocked grad will not be used locally.
            continue

        # Split flattened gradients into valid tensor blocks of the gradient.
        split_grads = FSDPDistributor._split_tensor_block_recovery(
            flattened_grad,
            self._param_to_metadata[flattened_param].shape,
            self._param_to_metadata[flattened_param].start_idx,
            self._param_to_metadata[flattened_param].end_idx,
        )

        # Get the merged dimensions and the number of blocks for each split gradient.
        merged_dims_within_flattened_param = self._global_merged_dims_list[
            split_index:next_split_index
        ]
        num_blocks_within_split_grads = self._global_num_blocks_per_split_param[
            split_index:next_split_index
        ]

        for (
            grad,
            merged_dims,
            (blocks_within_split_index, next_blocks_within_split_index),
        ) in zip(
            split_grads,
            merged_dims_within_flattened_param,
            generate_pairwise_indices(num_blocks_within_split_grads),
            strict=True,
        ):
            # Obtain blocks for each split gradient after merging.
            blocks_within_grad = multi_dim_split(
                grad.view(merged_dims), self._param_group[MAX_PRECONDITIONER_DIM]
            )
            # Generate block-to-parameter metadata and extend blocked parameters list.
            local_masked_blocked_grads.extend(
                compress_list(
                    blocks_within_grad,
                    param_distributor_selector[
                        blocks_within_split_index:next_blocks_within_split_index
                    ],
                )
            )

    # Set global grad selector as tuple.
    self._global_grad_selector = tuple(global_grad_selector)

    return tuple(local_masked_blocked_grads)
```


## Update preconditioner
In `distributed_shampoo.py`:
```py
@torch.no_grad()
    def _instantiate_per_group_step(
        self, shampoo_pt2_compile_config: ShampooPT2CompileConfig | None
    ) -> None:
        # Use PT2 to compile the step function for each parameter group.
        self._per_group_step: Callable[
            [
                dict[str, Any],
                torch.Tensor,
                torch.Tensor,
                float,
                float,
                float,
                float,
                float,
                bool,
                bool,
                bool,
                bool,
                bool,
                bool,
            ],
            None,
        ] = (
            torch.compile(
                self._per_group_step_impl,
                backend=shampoo_pt2_compile_config.pytorch_compile_backend,
                dynamic=shampoo_pt2_compile_config.enable_shampoo_pt2_dynamic_shape,
            )
            if shampoo_pt2_compile_config is not None
            else self._per_group_step_impl
        )
        if shampoo_pt2_compile_config is not None:
            logger.info(
                f"DistributedShampoo optimizer initialization is using {shampoo_pt2_compile_config.pytorch_compile_backend} backend and enable_shampoo_pt2_dynamic_shape={shampoo_pt2_compile_config.enable_shampoo_pt2_dynamic_shape}"
            )
```
which calls:
```py
@torch.no_grad()
def _per_group_step_impl(
    self,
    state_lists: dict[str, Any],
    step: torch.Tensor,
    lr: torch.Tensor,
    beta1: float,
    beta3: float,
    weight_decay: float,
    momentum_param: float,
    dampening: float,
    grafting_config_not_none: bool,
    perform_amortized_computation: bool,
    use_decoupled_weight_decay: bool,
    use_bias_correction: bool,
    use_grafting_method: bool,
    use_nesterov: bool,
) -> None:
```
which calls:
```py
@torch.no_grad()
def _update_preconditioners(
    self,
    state_lists: dict[str, Any],
    step: torch.Tensor,
    perform_amortized_computation: bool,
    grafting_config_not_none: bool,
) -> None:
    # Update Shampoo and grafting preconditioners.
    state_lists[SHAMPOO_PRECONDITIONER_LIST].update_preconditioners(
        masked_grad_list=state_lists[MASKED_BLOCKED_GRADS],
        step=step,
        perform_amortized_computation=perform_amortized_computation,
    )
    if grafting_config_not_none:
        state_lists[GRAFTING_PRECONDITIONER_LIST].update_preconditioners(
            masked_grad_list=state_lists[MASKED_BLOCKED_GRADS],
            step=step,
        )
```
```py
def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """
    Preconditions a list of gradients using the Shampoo preconditioner.

    Args:
        masked_grad_list (tuple[Tensor, ...]): A list of gradients with their corresponding masks.

    Returns:
        preconditioned_grads (tuple[Tensor, ...]): A list of preconditioned gradients.
    """
    with profiler.record_function(
        f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
    ):
        return tuple(
            self._precondition_grad(
                grad=masked_grad,
                preconditioned_dims_selector=preconditioned_dims_selector,
                preconditioner_list=kronecker_factors.inv_factor_matrices,
            )
            for masked_grad, preconditioned_dims_selector, kronecker_factors in zip(
                masked_grad_list,
                self._masked_preconditioned_dims_selector_list,
                self._masked_kronecker_factors_list,
                strict=True,
            )
        )
```

## Step

After `step()` is called, we call `merge_and_block_gradients()` in `shampoo_distributor.py`.
```py
def merge_and_block_gradients(
    self,
) -> tuple[Tensor, ...]:
    """Merge and block gradients.

    NOTE: This function MUST be called in the step function of the optimizer after the
    gradient has been updated.

    Returns:
        local_masked_blocked_grads (tuple[Tensor, ...]): Local blocked gradients masked with grad existence.

    """
    local_masked_blocked_grads = self._merge_and_block_gradients()

    if self._previous_global_grad_selector != self._global_grad_selector:
        self._previous_global_grad_selector = self._global_grad_selector

        # Update _local_grad_selector and _local_masked_blocked_params only when global_grad_selector is changed.
        self._local_grad_selector = compress_list(
            self._global_grad_selector,
            self._distributor_selector,
        )
        self._local_masked_blocked_params = compress_list(
            self._local_blocked_params, self._local_grad_selector
        )

    return local_masked_blocked_grads
```

## Implementation
From `_per_group_step_impl(...)`, return the preconditioner state and 


### Model training
```py

def train_fully_shard_model(
    model: nn.Module,
    world_size: int,
    loss_function: nn.Module,
    sampler: torch.utils.data.Sampler,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 1,
    window_size: int = 100,
    use_distributed_checkpoint: bool = False,
    checkpoint_dir: str | None = None,
) -> tuple[float, float, int]:
    """Constructs the main training loop.

    Assumes torch.distributed is initialized.

    """

    # initialize metrics
    metrics = LossMetrics(window_size=window_size, device=device, world_size=world_size)

    # main training loop
    for epoch in range(epochs):
        metrics._epoch = epoch
        sampler.set_epoch(epoch)  # type: ignore[attr-defined]

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_function(output, labels)
            loss.backward()

            optimizer.step()
            metrics.update(loss)
            metrics.log()
            metrics.update_global_metrics()
            if LOCAL_RANK == 0:
                metrics.log_global_metrics()

    # checkpoint optimizer and model using distributed checkpointing solution
    if use_distributed_checkpoint and isinstance(optimizer, DistributedShampoo):
        assert checkpoint_dir is not None
        state_dict = {
            "model": model.state_dict(),
            "optim": optimizer.distributed_state_dict(
                key_to_param=model.named_parameters()
            ),
        }
        dist_checkpoint.save_state_dict(
            state_dict=state_dict,
            storage_writer=dist_checkpoint.FileSystemWriter(checkpoint_dir),
        )

    return (
        metrics._lifetime_loss.item(),
        metrics._window_loss.item(),
        metrics._iteration,
    )
```
`optimizer.step()` calls are as above.
`update_global_metrics` updates the loss metrics using `dist.all_reduce`:
```py
    def update_global_metrics(self):
        if dist.is_initialized() and self._world_size > 1:
            self._global_window_loss = self._window_loss / self._world_size
            self._global_lifetime_loss = self._lifetime_loss / self._world_size
            dist.all_reduce(self._global_window_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._global_lifetime_loss, op=dist.ReduceOp.SUM)
        else:
            pass
```

When checkpointing, you’ll need to save all the components that comprise the internal state of your preconditioner. In this setup, the preconditioner’s state isn’t stored in one monolithic matrix but is distributed over several objects and parameters. Here’s what to capture:

_masked_kronecker_factors_list:
This list holds objects representing the curvature information for each parameter block. For each element in the list, you should save:
Factor Matrices: These are the raw accumulated curvature approximations.
Inverse Factor Matrices (inv_factor_matrices): These matrices are computed in the _amortized_computation step and are used to precondition the gradients.
Factor Matrix Indices: Any indexing or identification information that helps associate the factor matrices with their corresponding parameters.
_masked_roots_list:
This list stores the “root” values for each preconditioner component that dictate the order of the matrix root used during the inverse computation. Preserving these is essential to ensure that when you reload your checkpoint, you compute or use the inverse roots consistently with previous computations.
Bias Correction Term (e.g., _bias_correction2):
Since the bias correction is used to adjust the factor matrices before the inversion (and it’s updated on every step if enabled), you must save its current value.


Should save these states:
```
_masked_kronecker_factors_list
_masked_order_list
_masked_roots_list
_masked_preconditioned_dims_selector_list
_masked_failed_amortized_computation_counter_list
```