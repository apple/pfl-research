# Copyright © 2023-2024 Apple Inc.

import logging
import os
from typing import List, Optional, Tuple, Type

import mlx.core as mx
import numpy as np

from pfl.internal.ops.common_ops import is_mpi_running
from pfl.internal.ops.framework_types import MLFramework

from .distributed import DistributedContext, NotDistributedContext

logger = logging.getLogger(name=__name__)

FRAMEWORK_TYPE = MLFramework.MLX


class MLXDistributedContext(DistributedContext):
    """
    Distributed training operations for MLX tensors using
    ``mlx.core.distributed`` (MPI).
    """

    def __init__(self):
        self._world = mx.distributed.init()
        logger.info('rank=%i size=%i', self._world.rank(), self._world.size())

    @property
    def local_rank(self) -> int:
        return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))

    @property
    def global_rank(self) -> int:
        return self._world.rank()

    @property
    def world_size(self) -> int:
        return self._world.size()

    @property
    def local_size(self) -> int:
        return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', 1))

    def all_reduce(self,
                   tensors: List[mx.array],
                   average: bool = False) -> List[mx.array]:
        if self.world_size <= 1:
            # In the case of a single worker, just return identity instead
            # of doing unnecessary flatten and reshape.
            return tensors
        assert not average

        flat_vector, reshape_shapes, reshape_types = flatten(tensors)
        reduced_flat_vector = mx.distributed.all_sum(flat_vector)
        reduced_tensors = reshape(to_tensor(reduced_flat_vector),
                                  reshape_shapes, reshape_types)
        mx.eval(reduced_tensors)
        return reduced_tensors


distributed: DistributedContext
if is_mpi_running():
    logger.info("Multi-process, initializing MLX distributed context.")
    distributed = MLXDistributedContext()
else:
    logger.info("Single process, don't initialize distributed context.")
    distributed = NotDistributedContext()


def get_shape(variable):
    """
    Get the shape of a ``mlx.core.array``.

    :variable:
        A ``mlx.core.array``.
    :returns:
        A tuple representing the shape.
    """
    return tuple(variable.shape)


def is_tensor(variable):
    """
    Check whether the input is a mlx array.
    """
    return isinstance(variable, mx.array)


def add_laplacian_noise(tensors: List[mx.array], scale: float,
                        seed: Optional[int]) -> List[mx.array]:
    """
    Add zero mean Laplacian noise to MLX arrays.

    :param tensors:
        A list of MLX arrays to add noise to.
    :param scale:
        The noise scale `b` of laplacian noise.
    :param seed:
        An integer for seed.
    :return:
        Same as `tensors` but with noise added.
    """
    rngs = mx.random.split(mx.random.key(seed),
                           num=len(tensors)) if seed else [None] * len(tensors)
    noised = []
    for rng, t in zip(rngs, tensors):
        u = mx.random.uniform(low=-1.0,
                              high=1.0,
                              shape=t.shape,
                              dtype=t.dtype,
                              key=rng)
        # Use inverse CDF to generate Laplacian noise
        noised.append(t + scale * mx.sign(u) * mx.log1p(-mx.abs(u)))
    return noised


def add_gaussian_noise(tensors: List[mx.array], stddev: float,
                       seed: Optional[int]) -> List[mx.array]:
    """
    Add zero mean Gaussian noise to MLX arrays.

    :param tensors:
        A list of MLX arrays to add noise to.
    :param stddev:
        Standard deviation of noise to add.
    :param seed:
        An integer for seed.
    :return:
        Same as `tensors` but with noise added.
    """
    rngs = mx.random.split(mx.random.key(seed),
                           num=len(tensors)) if seed else [None] * len(tensors)
    return [
        v + mx.random.normal(
            loc=0.0, scale=stddev, shape=v.shape, dtype=v.dtype, key=rng)
        for rng, v in zip(rngs, tensors)
    ]


def norm(tensor: mx.array, order) -> float:
    """
    Calculate the norm of a MLX array.

    :param tensor:
        An MLX array to calculate the norm for.
    :param order:
        The order of the distance metric.
    :returns:
        The norm.
    """
    return mx.linalg.norm(tensor.reshape(-1), order).item()


def global_norm(tensors: List[mx.array], order: float) -> float:
    """
    Calculate the norm of the concatenation of the arrays.

    :param tensors:
        A list of MLX arrays to calculate global norm for.
    :param order:
        The order of the distance metric.
    :returns:
        The global norm.
    """
    return norm(mx.array([norm(t, order) for t in tensors]), order)


def flatten(
        tensors: List[mx.array]) -> Tuple[mx.array, List[Tuple], List[Type]]:
    """
    Flatten a list of MLX arrays into a single vector.

    :param tensors:
        A list of MLX arrays to flatten.
    :return:
        `(vector, shapes, dtypes)`, where `vector` is the flattened vector,
        `shapes` is a list of shapes of the input arrays and `dtypes` is a
        list of types of the input arrays. `shapes` and `dtypes` can be used
        with the `reshape` function to recover the original list of weights.
    """
    shapes = [v.shape for v in tensors]
    dtypes = [v.dtype for v in tensors]
    vector = mx.concatenate(
        [v.reshape(-1).astype(mx.float32) for v in tensors])
    return vector, shapes, dtypes


def reshape(vector: mx.array,
            shapes: List[Tuple],
            dtypes: Optional[List[Type]] = None) -> List[mx.array]:
    """
    Split and reshape a vector into a list of MLX arrays.

    :param vector:
        A 1-dimensional MLX array to split and reshape.
    :param shapes:
        A list of tuples of integers, representing the shapes of multiple
        target weights to construct.
    :param dtypes:
        A list of types for the new weights.
    :return:
        A list of MLX arrays constructed from the inputs.
    """
    separating_index = 0
    weights = []
    for i, shape in enumerate(shapes):
        weight_len = np.prod(shape).item()
        new_flat_weight = vector[separating_index:separating_index +
                                 weight_len]
        if dtypes is not None:
            new_flat_weight = new_flat_weight.astype(dtypes[i])
        weights.append(new_flat_weight.reshape(shape))
        separating_index += weight_len
    return weights


def to_tensor(tensor: mx.array, dtype: Optional[str] = 'float32') -> mx.array:
    """
    Convert a numpy array to a mlx array,
    """
    mlx_dtype = getattr(
        mx, dtype) if dtype is not None and isinstance(dtype, str) else dtype

    if isinstance(tensor, mx.array):
        return tensor

    return mx.array(tensor, dtype=mlx_dtype)


def to_numpy(tensor: mx.array) -> mx.array:
    # https://ml-explore.github.io/mlx/build/html/usage/numpy.html?highlight=numpy#conversion-to-numpy-and-other-frameworks
    return np.array(tensor, copy=False)


def clone(tensor: mx.array) -> mx.array:
    """
    Clone a mlx array.
    """
    return mx.array(tensor)


def clone_variable(variable: mx.array, name) -> mx.array:
    """
    Return a cloned copy of MLX Array.

    :param variable:
        A ``mlx.core.array``.
    :param name:
        An unused argument to match the signature of TensorFlow internal ops.
    :return:
        A ``mlx.core.array`` that is a cloned copy of ``variable``.
    """
    return mx.array(variable)


def assign_variable(reference: mx.array, value: mx.array) -> None:
    """
    Assign value to reference variable.

    :param reference:
        A ``mx.array`` that will be assigned to ``value``.
    :param value:
        A ``mx.array`` whose value is assigned to ``reference``.
    """
    assert reference.shape == value.shape, (
        "Value should have the same shape as reference for assigning, but "
        f"reference shape is {reference.shape} and "
        f"value shape is {value.shape}")
    reference[:] = value


def exponential_moving_average_update(variables: List[mx.array],
                                      ema_variables: List[mx.array],
                                      decay: float) -> None:
    """
    Perform one step of EMA update for a list of variables and a list of
    paired EMA variables. For each (variable, EMA variable) pair, the update is
    as following: ``ema_variable -= (1 - decay) * (ema_variable - variable)``.

    :param variables:
        A list of ``mx.array`` representing the current values.
    :param ema_variables:
        A list of ``mx.array`` representing the EMA values to be updated.
    :param decay:
        A ``float`` defining the EMA decay rate.
    """
    for variable, ema_variable in zip(variables, ema_variables):
        ema_variable -= ((1 - decay) * (ema_variable - variable))


def one_hot(indices: mx.array, depth: int) -> mx.array:
    """
    One-hot encode indices to vector with depth dimension.

    :param indices:
        A vector of indices to be one-hot encoded.
    :param depth:
        The dimension of one-hot encoding.
    :return:
        One-hot encoded vectors.
    """
    return mx.eye(depth, dtype=mx.float32)[indices.astype(mx.int32)]


def concatenate(tensors: List[mx.array], axis: int) -> mx.array:
    """
    Join a list of tensors along an existing axis.

    :param tensors:
        List of tensors to be concatenated.
    :param axis:
        Axis to concatenate the tensors.
    :return:
        A concatenated tensor.
    """
    return mx.concatenate(tensors, axis=axis)
