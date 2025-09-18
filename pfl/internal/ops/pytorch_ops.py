# Copyright © 2023-2024 Apple Inc.

import contextlib
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch  # type: ignore

from pfl.internal.ops.framework_types import MLFramework
from pfl.internal.platform.selector import get_platform

from .common_ops import get_pytorch_major_version, is_pytest_running
from .distributed import DistributedContext

logger = logging.getLogger(name=__name__)

FRAMEWORK_TYPE = MLFramework.PYTORCH

# Without suppressing the exception, this out to cause segfault.
# Must be used to make DataLoader and distributed training work
# together.
with contextlib.suppress(RuntimeError):
    torch.multiprocessing.set_start_method('forkserver')


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


def setup_amp(
    amp_dtype: Optional[torch.dtype], grad_scaling: bool
) -> Tuple[Optional[torch.amp.autocast], Optional[torch.cuda.amp.GradScaler]]:
    """
    Setup `torch.amp.autocast` context and `torch.cuda.amp.GradScaler` for
    PyTorch native mixed precision training. Gradient scaling is only used
    when training on CUDA.

    :param amp_dtype:
        An optional `torch.dtype` indicating the precision level. If set to
        `None` then mix precision training is not enabled.
    :param grad_scaling:
        Whether to turn on gradient scaling when training on CUDA.
    :return:
        A tuple of `torch.amp.autocast` context and `torch.cuda.amp.GradScaler`.
    """

    amp_context, grad_scaler = None, None
    if amp_dtype is not None:
        if get_default_device() == torch.device("mps"):
            logger.warning("PyTorch AMP is disabled on MPS")
            amp_dtype = torch.float32
        if (get_default_device() == torch.device("cpu")
                and amp_dtype == torch.float16):
            logger.warning("PyTorch AMP cast to float16 is disabled on CPU")
            amp_dtype = torch.float32
        if torch.cuda.is_available():
            if (amp_dtype == torch.bfloat16
                    and not torch.cuda.is_bf16_supported()):
                logger.warning(
                    "PyTorch AMP cast to bfloat16 is not supported on this GPU."
                )
                amp_dtype = torch.float32

    if amp_dtype is not None and amp_dtype != torch.float32:
        amp_context = torch.amp.autocast(
            "cuda" if torch.cuda.is_available() else "cpu", amp_dtype)
        logger.info("Mixed precision training with PyTorch AMP float type: "
                    f"{amp_context.fast_dtype}")
        if grad_scaling and torch.cuda.is_available():
            grad_scaler = torch.cuda.amp.GradScaler()
            logger.info("Gradient scaling enabled.")
    return amp_context, grad_scaler


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
        if torch.cuda.is_available():
            backend = 'nccl'
        else:
            backend = 'gloo'
        if "TORCHELASTIC_RUN_ID" in os.environ:
            # Using torchrun.
            torch.distributed.init_process_group(backend=backend)
            self._global_rank = int(os.environ['RANK'])
            self._world_size = int(os.environ['WORLD_SIZE'])
        elif worker_addresses is not None:
            self._global_rank = worker_rank
            self._world_size = len(worker_addresses)
            # We assume that the master is always in the first position.
            init_method = f'tcp://{worker_addresses[0]}'
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=worker_rank,
                world_size=len(worker_addresses))
        else:
            self._global_rank = 0
            self._world_size = 1

    @property
    def local_rank(self) -> int:
        return int(os.environ.get('LOCAL_RANK', 0))

    @property
    def global_rank(self) -> int:
        return self._global_rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def local_size(self) -> int:
        return int(os.environ.get('LOCAL_WORLD_SIZE', 1))

    def _flatten(self, tensors):
        return flatten(tensors)

    def _reshape(self, vector, tensors, _shapes, _dtypes):
        return inmemory_reshape(vector, tensors)

    def all_reduce(self,
                   tensors: List[torch.Tensor],
                   average: bool = False) -> List[torch.Tensor]:
        if self._world_size <= 1:
            return tensors

        flat, *reshape_context = self._flatten(tensors)
        torch.distributed.all_reduce(flat, op=torch.distributed.reduce_op.SUM)
        if average:
            flat /= self._world_size
        return self._reshape(flat, tensors, *reshape_context)


# Only initialize a distributed context if in the main process.
# Otherwise, this will try to re-initialize for every subprocess in DataLoader.
distributed: Optional[DistributedContext]
if torch.multiprocessing.get_context().current_process().name == 'MainProcess':
    distributed = PyTorchDistributedContext()
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
            # Always save the random state for CPU
            # Save random state for CUDA/MPS if they are available
            self._saved_state = {"cpu": torch.random.get_rng_state()}
            if torch.cuda.is_available():
                self._saved_state["cuda"] = torch.cuda.get_rng_state_all()
            if (hasattr(torch.backends, 'mps')
                    and torch.backends.mps.is_available()):
                self._saved_state["mps"] = torch.mps.get_rng_state()
            torch.random.manual_seed(self._seed)

    def __exit__(self, *args):
        if self._seed is not None:
            torch.random.set_rng_state(self._saved_state["cpu"])
            if "cuda" in self._saved_state:
                torch.cuda.set_rng_state_all(self._saved_state["cuda"])
            if "mps" in self._saved_state:
                torch.mps.set_rng_state(self._saved_state["mps"])


class Barrier:
    """
    Context manager for barrier in distributed communication. Replicates the
    functionality of:
    https://pytorch.org/docs/stable/distributed.html#torch.distributed.barrier
    """

    def __enter__(self):
        if distributed.local_rank != 0:
            distributed.all_reduce(
                [torch.zeros(1, device=get_default_device())])

    def __exit__(self, *args):
        if distributed.local_rank == 0:
            distributed.all_reduce(
                [torch.zeros(1, device=get_default_device())])


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


def to_tensor(values: Union[List, np.ndarray],
              dtype: Optional[str] = 'float32') -> torch.Tensor:
    """
    Convert a list of values or a numpy array to a float32 Torch tensor.
    """
    torch_dtype = getattr(torch, dtype) if dtype is not None and isinstance(
        dtype, str) else dtype

    if isinstance(values, torch.Tensor):
        return values.to(device=get_default_device())

    tensor = torch.as_tensor(values, dtype=torch_dtype)
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


class GradAccumulationState:
    """ Track gradient accumulation during local training. """

    def __init__(self, num_steps: int, accumulation_steps: int):
        self._num_steps = num_steps
        self._accumulation_steps = accumulation_steps
        self._steps = 0

    @property
    def optimizer_should_update(self) -> bool:
        """ Update every `grad_accumulation_steps` or is the last step """
        return (self._steps % self._accumulation_steps == 0
                or self._steps == self._num_steps)

    @property
    def accumulation_steps(self):
        return self._accumulation_steps

    def increment(self):
        self._steps += 1


@dataclass(frozen=True)
class PyTorchTrainStepArgs:
    """
    Common args used by different local training algorithms in PyTorch.
    """
    amp_context: Union[torch.amp.autocast, contextlib.AbstractContextManager]
    grad_scaler: Optional[torch.cuda.amp.GradScaler]
    max_grad_norm: Optional[float]
    grad_accumulation_state: GradAccumulationState
