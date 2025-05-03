# ft-shampoo

Curvature state replication using erasure coding for FDSP Distributed Shampoo by Meta Research. This builds off the [`facebookresearch/optimizers`](https://github.com/facebookresearch/optimizers/tree/main) repository of PyTorch optimization algorithms.

## Setup
To configure the environment, set the required variables in the project root directory:
```sh
source env.sh
```
Then create a new Python virtual environment and install the required packages from `requirements.txt`.
```sh
python -m venv shampoo_env
source shampoo_env/bin/activate
python -m pip install -r requirements.txt
```
Finally, set up the `torch-shampoo` project:
```sh
python -m pip install .
python -m pip install ".[examples]"
```
The training loop can be run like so:
```
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS -m distributed_shampoo.examples.fsdp_cifar10_example --optimizer-type DISTRIBUTED_SHAMPOO --precondition-frequency 100 --grafting-type ADAM --num-trainers-per-group -1 --use-bias-correction --use-decoupled-weight-decay --use-merge-dims
```

