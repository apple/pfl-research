# Copyright © 2023-2024 Apple Inc.

import importlib
import logging
import os
from typing import List, Optional, Tuple, Type

import numpy as np

from pfl.internal.ops.framework_types import MLFramework

from .distributed import NUMPY_DISTRIBUTE_VAR_NAME, DistributedContext, HorovodDistributedContext, NotDistributedContext

logger = logging.getLogger(name=__name__)

FRAMEWORK_TYPE = MLFramework.NUMPY


class NumpyHorovodDistributedContext(HorovodDistributedContext):
    """
    Distributed training operations for NumPy tensors using a
    Horovod backend.
    Initializing an instance of this class performs the Horovod setup.

    :param module_name:
        The Horovod api to use. Most commonly 'tensorflow' or 'pytorch'.
    """

    def __init__(self, module_name: str):
        logger.info('Trying to use Horovod with %s.', module_name)
        hvd = importlib.import_module(f'horovod.{module_name}')
        super().__init__(hvd)
        hvd.init()
        logger.info('local_rank=%i local_size=%i rank=%i size=%i',
                    hvd.local_rank(), hvd.local_size(), hvd.rank(), hvd.size())

    def _flatten(self, tensors):
        return flatten(tensors)

    def _reshape(self, vector, shapes, dtypes):
        # to_numpy in this case is able to handle tensors of both frameworks.
        return reshape(to_numpy(vector), shapes, dtypes)


# Initialize distributed context.
distributed: DistributedContext
if os.environ.get(NUMPY_DISTRIBUTE_VAR_NAME, '').lower() == 'tensorflow':
    distributed = NumpyHorovodDistributedContext('tensorflow')
elif os.environ.get(NUMPY_DISTRIBUTE_VAR_NAME, '').lower() == 'pytorch':
    distributed = NumpyHorovodDistributedContext('torch')
elif os.environ.get(NUMPY_DISTRIBUTE_VAR_NAME, '').lower() == 'none':
    distributed = NotDistributedContext()
else:
    assert NUMPY_DISTRIBUTE_VAR_NAME not in os.environ, (
        f'{os.environ[NUMPY_DISTRIBUTE_VAR_NAME]} is not supported. '
        'Valid values are `tensorflow` and `pytorch`')
    # No distributed training supported without Horovod.
    distributed = NotDistributedContext()


class NumpySeedScope:
    """
    Context manager for temporarily using another NumPy random state
    from the given seed.

    :param seed:
        The seed for the temporary random state.
    """

    def __init__(self, seed=None):
        self._seed = seed

    def __enter__(self):
        if self._seed is not None:
            self._saved_state = np.random.get_state()
            np.random.seed(self._seed)

    def __exit__(self, *args):
        if self._seed is not None:
            np.random.set_state(self._saved_state)


def get_shape(variable):
    """
    Get the shape of a ``np.ndarray``.

    :variable:
        A ``np.ndarray``.
    :returns:
        A tuple representing the shape.
    """
    return tuple(variable.shape)


def is_tensor(variable):
    """
    Check whether the input is a Numpy array.
    """
    return isinstance(variable, np.ndarray)


def add_laplacian_noise(tensors: List[np.ndarray], scale: float,
                        seed: Optional[int]) -> List[np.ndarray]:
    """
    Add zero mean Laplacian noise to numpy arrays.

    :param tensors:
        A list of numpy arrays to add noise to.
    :param scale:
        The noise scale `b` of laplacian noise.
    :param seed:
        An integer for seed.
    :return:
        Same as `tensors` but with noise added.
    """

    with NumpySeedScope(seed):
        data_with_noise = [
            v + np.random.laplace(loc=0, scale=scale, size=v.shape)
            for v in tensors
        ]
    return data_with_noise


def add_gaussian_noise(tensors: List[np.ndarray], stddev: float,
                       seed: Optional[int]) -> List[np.ndarray]:
    """
    Add zero mean Gaussian noise to numpy arrays.

    :param tensors:
        A list of numpy arrays to add noise to.
    :param stddev:
        Standard deviation of noise to add.
    :param seed:
        An integer for seed.
    :return:
        Same as `tensors` but with noise added.
    """

    with NumpySeedScope(seed):
        data_with_noise = [
            v + np.random.normal(loc=0, scale=stddev, size=v.shape)
            for v in tensors
        ]
    return data_with_noise


def norm(tensor: np.ndarray, order) -> float:
    """
    Calculate the norm of a numpy array.

    :param tensor:
        A numpy array to calculate the norm for.
    :param order:
        The order of the distance metric.
    :returns:
        The norm.
    """
    assert order > 0 or order == np.inf, \
        'Only supports positive norms and infinity norm.'
    if order == 1:
        tensor = abs(tensor)

    if order == np.inf:
        return np.max(tensor)
    else:
        return np.sum(tensor**order)**(1. / order)


def global_norm(tensors: List[np.ndarray], order: float) -> float:
    """
    Calculate the norm of the concatenation of the arrays.

    :param tensors:
        A list of numpy arrays to calculate global norm for.
    :param order:
        The order of the distance metric.
    :returns:
        The global norm.
    """
    # Implemented as described here
    # https://www.tensorflow.org/api_docs/python/tf/clip_by_global_norm.
    return norm(np.array([norm(t, order) for t in tensors]), order)


def flatten(
        tensors: List[np.ndarray]
) -> Tuple[np.ndarray, List[Tuple], List[Type]]:
    """
    Flatten a list of numpy arrays into a single vector.

    :param tensors:
        A list of numpy arrays to flatten.
    :return:
        `(vector, shapes, dtypes)`, where `vector` is the flattened vector,
        `shapes` is a list of shapes of the input arrays and `dtypes` is a
        list of types of the input arrays. `shapes` and `dtypes` can be used
        with the `reshape` function to recover the original list of weights.
    """

    shapes = [v.shape for v in tensors]
    dtypes = [v.dtype for v in tensors]
    vector = np.concatenate(
        [v.reshape(-1).astype(np.float32) for v in tensors])
    return vector, shapes, dtypes


def reshape(vector: np.ndarray,
            shapes: List[Tuple],
            dtypes: Optional[List[Type]] = None) -> List[np.ndarray]:
    """
    Split and reshape a vector into a list of numpy arrays.

    :param vector:
        A 1-dimensional numpy array to split and reshape.
    :param shapes:
        A list of tuples of integers, representing the shapes of multiple
        target weights to construct.
    :param dtypes:
        A list of types for the new weights.
    :return:
        A list of numpy arrays constructed from the inputs.
    """
    separating_index = 0
    weights = []
    for i, shape in enumerate(shapes):
        weight_len = np.prod(shape)
        new_flat_weight = vector[separating_index:separating_index +
                                 weight_len]
        if dtypes is not None:
            new_flat_weight = new_flat_weight.astype(dtypes[i])
        weights.append(new_flat_weight.reshape(shape))
        separating_index += weight_len
    return weights


def to_tensor(tensor: np.ndarray,
              dtype: Optional[str] = 'float32') -> np.ndarray:
    """
    Convert a numpy array to numpy array,
    i.e. identity in this case.
    """
    return np.asarray(tensor, dtype=dtype)


to_numpy = to_tensor


def clone(tensor: np.ndarray) -> np.ndarray:
    """
    Clone a numpy array.
    """
    return np.array(tensor)


def clone_variable(variable: np.ndarray, name) -> np.ndarray:
    """
    Return a cloned copy of Numpy Array.

    :param variable:
        A ``np.ndarray``.
    :param name:
        An unused argument to match the signature of TensorFlow internal ops.
    :return:
        A ``np.ndarray`` that is a cloned copy of ``variable``.
    """
    return np.array(variable)


def assign_variable(reference: np.ndarray, value: np.ndarray) -> None:
    """
    Assign value to reference variable.

    :param reference:
        A ``np.ndarray`` that will be assigned to ``value``.
    :param value:
        A ``np.ndarray`` whose value is assigned to ``reference``.
    """
    assert reference.shape == value.shape, (
        "Value should have the same shape as reference for assigning, but "
        f"reference shape is {reference.shape} and "
        f"value shape is {value.shape}")
    reference[:] = value


def exponential_moving_average_update(variables: List[np.ndarray],
                                      ema_variables: List[np.ndarray],
                                      decay: float) -> None:
    """
    Perform one step of EMA update for a list of variables and a list of
    paired EMA variables. For each (variable, EMA variable) pair, the update is
    as following: ``ema_variable -= (1 - decay) * (ema_variable - variable)``.

    :param variables:
        A list of ``np.ndarray`` representing the current values.
    :param ema_variables:
        A list of ``np.ndarray`` representing the EMA values to be updated.
    :param decay:
        A ``float`` defining the EMA decay rate.
    """
    for variable, ema_variable in zip(variables, ema_variables):
        ema_variable -= ((1 - decay) * (ema_variable - variable))


def one_hot(indices: np.ndarray, depth: int) -> np.ndarray:
    """
    One-hot encode indices to vector with depth dimension.

    :param indices:
        A vector of indices to be one-hot encoded.
    :param depth:
        The dimension of one-hot encoding.
    :return:
        One-hot encoded vectors.
    """
    return np.eye(depth, dtype=np.float32)[indices.astype(np.int32)]


def concatenate(tensors: List[np.ndarray], axis: int) -> np.ndarray:
    """
    Join a list of tensors along an existing axis.

    :param tensors:
        List of tensors to be concatenated.
    :param axis:
        Axis to concatenate the tensors.
    :return:
        A concatenated tensor.
    """
    return np.concatenate(tensors, axis=axis)
