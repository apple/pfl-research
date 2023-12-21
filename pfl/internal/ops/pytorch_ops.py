# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch  # type: ignore

from pfl.internal.ops.framework_types import MLFramework
from pfl.internal.platform.selector import get_platform

from .common_ops import get_pytorch_major_version, is_pytest_running
from .distributed import (horovod_is_active, DistributedContext,
                          HorovodDistributedContext)

logger = logging.getLogger(name=__name__)

FRAMEWORK_TYPE = MLFramework.PYTORCH

try:
    # Comment this out to cause segfault. Must be used to make
    # DataLoader and distributed training work together.
    torch.multiprocessing.set_start_method('forkserver')
except RuntimeError:
    pass


def get_default_device():
    manual_device = os.environ.get('PFL_PYTORCH_DEVICE', None)
    if manual_device:
        # Default device can be overridden with env var.
        default_device = torch.device(manual_device)
    elif is_pytest_running():
        # Always use CPU when running tests.
        default_device = torch.device('cpu')
    elif torch.cuda.is_available():
        default_device = torch.device('cuda')
    elif (hasattr(torch.backends, 'mps')
          and torch.backends.mps.is_available()):
        default_device = torch.device('mps')
    else:
        default_device = torch.device('cpu')
    return default_device


class PyTorchDistributedContext(DistributedContext):
    """
    Distributed training operations for PyTorch tensors using
    `torch.distributed` backend.
    Initializing an instance of this class starts the PyTorch
    server on each worker and synchronizes.
    Only supports single process, single GPU, multi-worker training.
    """

    def __init__(self):
        worker_rank, worker_addresses = get_platform(
        ).get_distributed_addresses(verbose=True)
        self._global_rank = worker_rank
        self._world_size = 1 if worker_addresses is None else len(
            worker_addresses)

        if self._world_size > 1:
            if torch.cuda.is_available():
                backend = torch.distributed.dist_backend.NCCL
            else:
                backend = torch.distributed.dist_backend.GLOO
            # We assume that the master is always in the first position.
            init_method = f'tcp://{worker_addresses[0]}'
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=worker_rank,
                world_size=len(worker_addresses))

    @property
    def local_rank(self) -> int:
        # Only supports single process, single GPU, multi-worker training,
        # hence local rank will always be 0.
        return 0

    @property
    def global_rank(self) -> int:
        return self._global_rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def local_size(self) -> int:
        # Only supports single process, single GPU, multi-worker training,
        # hence local size will always be 1.
        return 1

    def all_reduce(self,
                   tensors: List[torch.Tensor],
                   average: bool = False) -> List[torch.Tensor]:
        if self._world_size <= 1:
            return tensors

        flat, _, _ = flatten(tensors)
        torch.distributed.all_reduce(flat, op=torch.distributed.reduce_op.SUM)
        if average:
            flat /= self.world_size
        return inmemory_reshape(flat, tensors)


class PyTorchHorovodDistributedContext(HorovodDistributedContext):
    """
    Distributed training operations for PyTorch tensors using a
    Horovod backend.
    Initializing an instance of this class performs the Horovod setup.
    """

    def __init__(self):
        import horovod.torch as hvd
        super().__init__(hvd)
        hvd.init()

        logger.info('local_rank=%i local_size=%i rank=%i size=%i',
                    hvd.local_rank(), hvd.local_size(), hvd.rank(), hvd.size())
        self._post_init_check(hvd)

        if torch.cuda.is_available():
            gpu_id = hvd.local_rank() % torch.cuda.device_count()
            torch.cuda.set_device(gpu_id)
            logger.info('local rank %i uses GPU: %i', hvd.local_rank(), gpu_id)

    def _flatten(self, tensors):
        vector, *reshape_context = flatten(tensors)
        # Tensors on "MPS" does not work with Horovod, put on CPU.
        if (hasattr(torch.backends, 'mps')
                and torch.backends.mps.is_available()):
            vector = vector.cpu()
        return (vector, *reshape_context)

    def _reshape(self, vector, shapes, dtypes):
        vector = vector.to(device=get_default_device())
        return reshape(vector, shapes, dtypes)


# Only initialize a distributed context if in the main process.
# Otherwise, this will try to re-initialize for every subprocess in DataLoader.
distributed: Optional[DistributedContext]
if torch.multiprocessing.get_context().current_process().name == 'MainProcess':
    distributed = PyTorchHorovodDistributedContext() if horovod_is_active(
    ) else PyTorchDistributedContext()
else:
    # This will only execute inside a DataLoader subprocess, which should not
    # use any distributed context anyway.
    distributed = None


def get_shape(variable):
    """
    Get the shape of a PyTorch variable.

    :param variable:
        A PyTorch tensor.
    :returns:
        A tuple representing the shape.
    """
    return tuple(variable.shape)


def is_tensor(variable):
    """
    Check whether the input is a PyTorch tensor.
    """
    return isinstance(variable, torch.Tensor)


def flatten(
    tensors: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[Tuple], List[torch.dtype]]:
    """
    Flatten a list of PyTorch tensors into a single vector.

    :param tensors:
        A list of tensors to flatten.
    :return:
        `(vector, shapes, dtypes)`, where `vector` is the flattened tensor,
        `shapes` is a list of shapes of the input arrays and `dtypes` is a
        list of types of the input arrays. `shapes` and `dtypes` can be used
        with the `reshape` function to recover the original list of weights.
    """

    shapes = [tuple(v.shape) for v in tensors]
    dtypes = [v.dtype for v in tensors]
    vector = torch.cat([t.reshape(-1).type(torch.float32) for t in tensors])

    return vector, shapes, dtypes


def inmemory_reshape(flat: torch.Tensor,
                     tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    offset = 0
    for t in tensors:
        t.data.copy_(flat[offset:offset + t.numel()].reshape(t.shape))
        offset += t.numel()
    return tensors


def reshape(vector: torch.Tensor,
            shapes: List[Tuple],
            dtypes: Optional[List[torch.dtype]] = None) -> List[torch.Tensor]:
    """
    Split and reshape a vector into a list of PyTorch tensors.

    :param vector:
        A 1-dimensional tensor to split and reshape.
    :param shapes:
        A list of tuples of integers, representing the shapes of multiple
        target weights to construct.
    :param dtypes:
        A list of types for the new weights.
    :return:
        A list of PyTorch tensors constructed from the inputs.
    """
    separating_index = 0
    weights = []
    for i, shape in enumerate(shapes):
        weight_len = np.prod(shape)
        new_flat_weight = vector[separating_index:separating_index +
                                 weight_len]
        if dtypes is not None:
            new_flat_weight = new_flat_weight.type(dtypes[i])
        weights.append(new_flat_weight.reshape(shape))
        separating_index += weight_len
    return weights


def simulate_bfloat16_transport(ndarray):
    """ Convert a numpy array to bfloat16 and then back to float32 """
    return torch.Tensor(ndarray).type(torch.bfloat16).type(
        torch.float32).numpy()


class PyTorchSeedScope:
    """
    Context manager for temporarily using another PyTorch random state
    from the given seed.

    :param seed:
        The seed for the temporary random state.
    """

    def __init__(self, seed=None):
        self._seed = seed

    def __enter__(self):
        if self._seed is not None:
            self._saved_state = torch.random.get_rng_state()
            torch.random.manual_seed(self._seed)

    def __exit__(self, *args):
        if self._seed is not None:
            torch.random.set_rng_state(self._saved_state)


_placeholder_cache = {}


def _get_ph(shape):
    if shape not in _placeholder_cache:
        _placeholder_cache[shape] = torch.zeros(shape,
                                                device=get_default_device())
    return _placeholder_cache[shape]


def add_gaussian_noise(tensors: List[np.ndarray], stddev: float,
                       seed: Optional[int]) -> List[torch.Tensor]:
    """
    Add zero mean Gaussian noise to tensors.
    Transferring data to GPU, adding noise, and back to NumPy is faster than
    `np.random.normal`.

    :param tensors:
        A list of tensors to add noise to.
    :param stddev:
        Standard deviation of noise to add.
    :param seed:
        An integer for seed.
    :return:
        Same as `tensors` but with noise added.
    """
    if (get_default_device() == torch.device('mps')
            and get_pytorch_major_version() < 2):
        raise RuntimeError("You are trying to use gaussian noise with MPS, "
                           "please upgrade to torch>=2.0.0")
    g = torch.Generator(
        device=get_default_device()).manual_seed(int(seed)) if seed else None
    # This is a very fast in-memory way of adding noise. Only supported
    # for Gaussian noise.
    return [
        torch.tensor(v, device=get_default_device()).add(
            _get_ph(v.shape).normal_(mean=0.0, std=stddev, generator=g))
        for v in tensors
    ]


def add_laplacian_noise(tensors: List[torch.Tensor], scale: float,
                        seed: Optional[int]) -> List[torch.Tensor]:
    """
    Add zero mean Laplacian noise to tensors.

    :param tensors:
        A list of tensors to add noise to.
    :param scale:
        Scaling factor of noise to add.
    :param seed:
        An integer for seed.
    :return:
        Same as `tensors` but with noise added.
    """
    with PyTorchSeedScope(seed):
        laplace = torch.distributions.Laplace(loc=0.0, scale=scale)
        noised_tensors = [
            v + laplace.sample(sample_shape=v.shape).to(v.dtype)
            for v in tensors
        ]
    return noised_tensors


def clone(tensor: torch.Tensor) -> torch.Tensor:
    """
    Make a copy of the input tensor.
    """
    return torch.clone(tensor)


def norm(tensor: torch.Tensor, order) -> torch.Tensor:
    """
    Calculate the norm of a PyTorch tensor.

    :param tensor:
        A tensor to calculate the norm for.
    :param order:
        The order of the distance metric (norm).
    :returns:
        The norm.
    """
    if order == 1 or order == np.inf:
        tensor = tensor.abs()

    norm_value = tensor.max() if order == np.inf else tensor.pow(
        order).sum().pow(1.0 / order)
    return norm_value.type(torch.float32)


def global_norm(tensors: List[torch.Tensor], order: float) -> torch.Tensor:
    """
    Calculate the norm of the concatenation of the arrays.

    :param tensors:
        A list of numpy arrays to calculate global norm for.
    :param order:
        The order of the distance metric.
    :returns:
        The global norm.
    """
    return norm(torch.stack([norm(t, order) for t in tensors]), order)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a numpy array.
    """
    detached_tensor = tensor.detach()
    if tensor.device.type != 'cpu':
        return detached_tensor.cpu().numpy()
    else:
        return detached_tensor.numpy()


def to_tensor(values: Union[List, np.ndarray]) -> torch.Tensor:
    """
    Convert a list of values or a numpy array to a float32 Torch tensor.
    """
    if isinstance(values, torch.Tensor):
        return values

    tensor = torch.as_tensor(values, dtype=torch.float32)
    tensor = tensor.to(device=get_default_device())
    return tensor


def clone_variable(variable: torch.Tensor, name) -> torch.Tensor:
    """
    Return a cloned copy of PyTorch tensor.

    :param variable:
        A ``torch.Tensor``.
    :param name:
        An unused argument to match the signature of TensorFlow internal ops.
    :return:
        A ``torch.Tensor`` that is a cloned copy of ``variable``.
    """
    return torch.clone(variable)


def assign_variable(reference: torch.Tensor, value: torch.Tensor) -> None:
    """
    Assign value to reference variable.

    :param reference:
        A ``torch.Tensor`` that will be assigned to ``value``.
    :param value:
        A ``torch.Tensor`` whose value is assigned to ``reference``.
    """
    reference.data.copy_(value.data)


def exponential_moving_average_update(variables: List[torch.Tensor],
                                      ema_variables: List[torch.Tensor],
                                      decay: float) -> None:
    """
    Perform one step of EMA update for a list of variables and a list of
    paired EMA variables. For each (variable, EMA variable) pair, the update is
    as following: ``ema_variable -= (1 - decay) * (ema_variable - variable)``.

    :param variables:
        A list of ``torch.Tensor`` representing the current values.
    :param ema_variables:
        A list of ``torch.Tensor`` representing the EMA values to be updated.
    :param decay:
        A ``float`` defining the EMA decay rate.
    """
    for ema_variable, variable in zip(ema_variables, variables):
        ema_variable.sub_((1 - decay) * (ema_variable - variable))


def one_hot(indices: torch.Tensor, depth: int) -> torch.Tensor:
    """
    One-hot encode indices to vector with depth dimension.

    :param indices:
        A vector of indices to be one-hot encoded.
    :param depth:
        The dimension of one-hot encoding.
    :return:
        One-hot encoded vectors.
    """
    #  Implemented in a CoreML compatible way as torch.nn.functional.one_hot
    #  is not supported in CoreML conversion.
    return torch.eq(indices.unsqueeze(-1).int(), torch.arange(depth)).float()


def concatenate(tensors: List[torch.Tensor], axis: int) -> torch.Tensor:
    """
    Join a list of tensors along an existing axis.

    :param tensors:
        List of tensors to be concatenated.
    :param axis:
        Axis to concatenate the tensors.
    :return:
        A concatenated tensor.
    """
    return torch.cat(tensors, dim=axis)
