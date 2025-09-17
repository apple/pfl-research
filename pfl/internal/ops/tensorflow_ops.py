# Copyright © 2023-2024 Apple Inc.

import json
import logging
import os
from collections import defaultdict
from distutils.version import StrictVersion
from functools import partial
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf  # type: ignore
import tensorflow_probability as tfp

from pfl.internal.ops.framework_types import MLFramework
from pfl.internal.platform.selector import get_platform
from pfl.metrics import MetricValue

from .distributed import DistributedContext

logger = logging.getLogger(name=__name__)

FRAMEWORK_TYPE = MLFramework.TENSORFLOW

jit_compile = os.getenv("TF_FUNCTION_JIT_COMPILE_DEFAULT",
                        "false").lower() in ("true", "1")
if jit_compile and StrictVersion('2.8.0') >= StrictVersion(tf.__version__):
    # This environment variable
    # https://github.com/tensorflow/tensorflow/blob/v2.8.0/tensorflow/python/eager/def_function.py#L382
    # is not supported for <2.8. In this case, pfl implements it.
    tf_function = partial(tf.function, experimental_compile=True)
else:
    tf_function = tf.function

graph_cache = {}
_graph_cache_errors: defaultdict = defaultdict(int)


def try_cached_call(fn, key, *args, **kwargs):
    """
    Call the graph of `fn` which is a tf.Function.
    If the graph of `fn` exists in pfl's cache, use the cached graph.
    This will result in significant speedups in TF>2.3 because this is
    bypassing TensorFlow's graph cache in tf.function (which has become
    incredibly slow).

    This feature can be disabled with the environment variable
    PFL_GRAPH_CACHE=false and should be done if one recognizes that pfl's
    graph cache is regenerated too much (because it is not very sophisticated
    yet).

    :param fn:
        A function decorated with tf.Function.
    :param key:
        A key for caching the graph of `fn`.
    :param args:
        Arguments for calling `fn`.
    :param kwargs:
        Keyword arguments for calling `fn`.
    :return:
        The returned value from `fn`.
    """
    if (os.getenv('PFL_GRAPH_CACHE', 'true').lower()
            not in ('true', '1', 't')):
        # Don't use graph cache.
        return fn(*args, **kwargs)

    if key not in graph_cache:
        graph_cache[key] = fn.get_concrete_function(*args, **kwargs)
        logger.debug(f'Cached {key} graph')
    try:
        result = graph_cache[key](*args, **kwargs)
    except (TypeError, tf.errors.InvalidArgumentError,
            tf.errors.FailedPreconditionError, tf.errors.UnknownError) as e:
        graph_cache[key] = fn.get_concrete_function(*args, **kwargs)
        _graph_cache_errors[key] += 1
        logger.warning(f'Regenerated graph for {key} after cache miss')
        if _graph_cache_errors[key] >= 10:
            logger.warning(f'Graph for {key} has been regenerated '
                           f'{_graph_cache_errors[key]} times. '
                           'Something is wrong and simulation is slow. '
                           f'Latest error is {e}. '
                           'Disable this feature using environment variable '
                           '`PFL_GRAPH_CACHE=false`')
        result = graph_cache[key](*args, **kwargs)
    return result


class TFDistributedContext(DistributedContext):
    """
    Distributed training operations for TF tensors using
    `tensorflow.distribute` backend.

    Initializing an instance of this class starts the TF servers and a
    distribution strategy that waits for synchronisation. If using
    distributed simulations, initialize a
    ``MultiWorkerMirroredStrategy``, otherwise initialize a
    ``OneDeviceStrategy``.

    Only supports single process, single GPU, multi-worker training.
    """

    def __init__(self):
        worker_rank, worker_addresses = get_platform(
        ).get_distributed_addresses(verbose=True)
        self._global_rank = worker_rank
        self._world_size = 1 if worker_addresses is None else len(
            worker_addresses)

        if worker_addresses is None:
            # Not distributed simulations, return single device strategy.
            if tf.test.is_gpu_available(cuda_only=True):
                device = 'device:GPU:0'
                logger.info(
                    f'Cuda-supported GPU found, training on device {device}')
            else:
                device = 'device:CPU:0'
                logger.info(f'No GPU available, training on device {device}')

            self._distributed = tf.distribute.OneDeviceStrategy(device)
        else:
            # See this address for `TF_CONFIG` specification.
            # https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#a_cluster_with_jobs_and_tasks
            os.environ['TF_CONFIG'] = json.dumps({
                'cluster': {
                    'worker': worker_addresses
                },
                'task': {
                    'type': 'worker',
                    'index': worker_rank
                }
            })

            try:
                self._distributed = \
                        tf.distribute.experimental.MultiWorkerMirroredStrategy(
                    tf.distribute.experimental.CollectiveCommunication.RING)
            except RuntimeError as e:
                if str(e) == (
                        "Collective ops must be configured at program startup"
                ):
                    raise RuntimeError(
                        "You must import the trainer before "
                        "defining any TensorFlow/Keras operations") from e
                else:
                    raise

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

    def _flatten(self, tensors):
        return flatten(tensors)

    def _reshape(self, vector, shapes, dtypes):
        return reshape(vector, shapes, dtypes)

    @tf_function
    def _all_reduce_tensor(self, tensor, reduce_op):
        """ Helper function for doing the all-reduce across workers """

        def id_function(v):
            return tf.identity(v)

        # The input of `tf.distribute.Strategy.reduce` requires a PerReplica
        # tensor, see
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/distribute/Strategy#reduce
        # This is most easily constructed by using the method `run`
        # (from TF 2.2) or `experimental_run_v2` (up to TF 2.1).
        # Since we are only looking for converting the tensor to a PerReplica
        # and not do any additional operations, we will return the identity.
        if hasattr(self._distributed, 'experimental_run_v2'):
            run = self._distributed.experimental_run_v2
        else:
            run = self._distributed.run
        per_replica_tensor = run(id_function, args=[tensor])

        return self._distributed.reduce(reduce_op,
                                        per_replica_tensor,
                                        axis=None)

    def all_reduce(self,
                   tensors: List[tf.Tensor],
                   average: bool = False) -> List[tf.Tensor]:
        if self.world_size <= 1:
            # In the case of a single worker, just return identity instead
            # of doing unnecessary flatten and reshape.
            return tensors

        reduce_op = tf.distribute.ReduceOp('MEAN' if average else 'SUM')

        vector, *reshape_context = self._flatten(tensors)

        with self._distributed.scope():
            # Reduce across replicas.
            reduced_vector = self._all_reduce_tensor(vector, reduce_op)

        return self._reshape(reduced_vector, *reshape_context)


# The correct distribution strategy needs to be initialized before any
# TensorFlow operations are defined because that is what the underlying
# TensorFlow server requires.
# This is why the strategy is initialized in the module scope.
distributed: DistributedContext
distributed = TFDistributedContext()


def get_shape(variable):
    """
    Get the shape of a TensorFlow variable.

    :variable:
        A ``tf.Variable``.
    :returns:
        A tuple representing the shape.
    """
    return tuple(variable.shape)


def is_tensor(variable):
    """
    Check whether the input is a TensorFlow tensor or variable.
    """
    return tf.is_tensor(variable)


def simulate_bfloat16_transport(tensor):
    """ Convert a tensor to bfloat16 and then back to float32 """
    return tf.cast(tf.cast(tensor, dtype=tf.bfloat16), dtype=tf.float32)


_normal_dist = None
_laplace_dist = None


@tf_function(experimental_relax_shapes=True)
def _add_gaussian_noise(tensors: List[tf.Tensor], stddev: float,
                        seed: Optional[int]) -> List[tf.Tensor]:
    if seed is not None:
        weight_seeds = tfp.random.split_seed(tf.cast(seed, tf.int32),
                                             salt='add_gaussian_noise',
                                             n=len(tensors))
    else:
        weight_seeds = [None] * len(tensors)
    assert _normal_dist is not None
    return [
        tensor + tf.cast(_normal_dist.sample(tensor.shape, seed=weight_seed),
                         tensor.dtype) * tf.cast(stddev, tensor.dtype)
        for weight_seed, tensor in zip(weight_seeds, tensors)
    ]


def add_gaussian_noise(tensors: List[tf.Tensor], stddev: float,
                       seed: Optional[int]) -> List[tf.Tensor]:
    """
    Add zero mean Gaussian noise to tensors.

    :param tensors:
        A list of tensors to add noise to.
    :param stddev:
        Standard deviation of noise to add.
    :param seed:
        An integer for seed.
    :return:
        Same as `tensors` but with noise added.
    """
    global _normal_dist
    if _normal_dist is None:
        _normal_dist = tfp.distributions.Normal(0., 1.0)
    # If seed is not a tensor when calling graph, it will never find the
    # existing graph in the cache by tf.function.
    if seed is not None:
        seed = tf.constant(seed)
    return _add_gaussian_noise(tensors, stddev, seed)


@tf_function(experimental_relax_shapes=True)
def _add_laplacian_noise(tensors: List[tf.Tensor], scale: float,
                         seed: Optional[int]) -> List[tf.Tensor]:
    if seed is not None:
        weight_seeds = tfp.random.split_seed(tf.cast(seed, tf.int32),
                                             salt='add_laplacian_noise',
                                             n=len(tensors))
    else:
        weight_seeds = [None] * len(tensors)
    assert _laplace_dist is not None
    return [
        tensor + tf.cast(_laplace_dist.sample(tensor.shape, seed=weight_seed),
                         tensor.dtype) * tf.cast(scale, tensor.dtype)
        for weight_seed, tensor in zip(weight_seeds, tensors)
    ]


def add_laplacian_noise(tensors: List[tf.Tensor], scale: float,
                        seed: Optional[int]) -> List[tf.Tensor]:
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
    global _laplace_dist
    if _laplace_dist is None:
        _laplace_dist = tfp.distributions.Laplace(0., 1.0)
    # If seed is not a tensor when calling graph, it will never find the
    # existing graph in the cache by tf.function.
    if seed is not None:
        seed = tf.constant(seed)
    return _add_laplacian_noise(tensors, scale, seed)


@tf_function
def clone(tensor: tf.Tensor) -> tf.Tensor:
    """
    Make a copy of the input tensor.
    """
    return tf.identity(tensor)


@tf_function
def _flatten(tensors: List[tf.Tensor]) -> tf.Tensor:
    return tf.concat(
        [tf.cast(tf.reshape(v, [-1]), tf.float32) for v in tensors], axis=0)


def flatten(
        tensors: List[tf.Tensor]) -> Tuple[tf.Tensor, List[Tuple], List[Type]]:
    """
    Flatten a list of tensors into a single vector.

    :param tensors:
        A list of tensors to flatten.
    :return:
        `(vector, shapes, dtypes)`, where `vector` is the flattened tensor,
        `shapes` is a list of shapes of the input arrays and `dtypes` is a
        list of types of the input arrays. `shapes` and `dtypes` can be used
        with the `reshape` function to recover the original list of weights.
    """
    try:
        shapes = [v.shape.as_list() for v in tensors]
    except AttributeError:
        # be compatible with numpy arrays as well.
        shapes = [v.shape for v in tensors]
    dtypes = [v.dtype for v in tensors]
    vector = _flatten(tensors)
    return vector, shapes, dtypes


@tf_function
def reshape(vector: tf.Tensor,
            shapes: List[Tuple],
            dtypes: Optional[List[Type]] = None) -> List[tf.Tensor]:
    """
    Split and reshape a vector into a list of TF tensors.

    :param vector:
        A 1-dimensional tensor to split and reshape.
    :param shapes:
        A list of tuples of integers, representing the shapes of multiple
        target weights to construct.
    :param dtypes:
        A list of types for the new weights.
    :return:
        A list of TF tensors constructed from the inputs.
    """
    separating_index = 0
    weights = []
    for i, shape in enumerate(shapes):
        weight_len = np.prod(shape)
        new_flat_weight = vector[separating_index:separating_index +
                                 weight_len]
        if dtypes is not None:
            new_flat_weight = tf.cast(new_flat_weight, dtypes[i])
        weights.append(tf.reshape(new_flat_weight, shape))
        separating_index += weight_len
    return weights


@tf_function
def norm(tensor: tf.Tensor, order) -> tf.Tensor:
    """
    Calculate the norm of a tensor.

    :param tensor:
        A tensor to calculate the norm for.
    :param order:
        The order of the distance metric (norm).
    :returns:
        The norm.
    """
    if order == 1 or order == np.inf:
        tensor = tf.abs(tensor)

    norm_value = tf.reduce_max(tensor) if order == np.inf else tf.norm(
        tf.cast(tensor, tf.float32), ord=order)
    return tf.cast(norm_value, tf.float32)


@tf_function
def global_norm(tensors: List[np.ndarray], order: float) -> tf.Tensor:
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
    return norm(tf.stack([norm(t, order) for t in tensors]), order)


def to_numpy(tensor: tf.Tensor) -> np.ndarray:
    """
    Convert a tensor to a numpy array.
    """
    return tensor.numpy()


def to_tensor(values: Union[List, np.ndarray],
              dtype: Optional[str] = 'float32') -> tf.Tensor:
    """
    Convert a list of values or a numpy array to a TF tensor.
    """
    return tf.convert_to_tensor(values, dtype=dtype)


def _shared_name(variable: tf.Variable):
    """
    Get TensorFlow variable name without the suffix :0.
    """
    try:
        return variable.name[:variable.name.rindex(":")]
    except ValueError:
        # Its not there in tf>=2.15
        return variable.name


def clone_variable(variable: tf.Variable, name: str) -> tf.Variable:
    """
    Return a cloned copy of TensorFlow variable.

    :param variable:
        A ``tf.Variable``.
    :param name:
        A ``str`` name for the cloned variable.
    :return:
        A ``tf.Variable`` that is a cloned copy of ``variable``.
    """
    cloned_variable = tf.Variable(variable,
                                  name=f"{name}_{_shared_name(variable)}")
    return cloned_variable


def assign_variable(reference: tf.Variable, value: tf.Variable) -> None:
    """
    Assign value to reference variable.

    :param reference:
        A ``tf.Variable`` that will be assigned to ``value``.
    :param value:
        A ``tf.Variable`` whose value is assigned to ``reference``.
    """
    reference.assign(value)


def exponential_moving_average_update(variables: List[tf.Variable],
                                      ema_variables: List[tf.Variable],
                                      decay: float) -> None:
    """
    Perform one step of EMA update for a list of variables and a list of
    paired EMA variables. For each (variable, EMA variable) pair, the update is
    as following: ``ema_variable -= (1 - decay) * (ema_variable - variable)``.

    :param variables:
        A list of ``tf.Variable`` representing the current values.
    :param ema_variables:
        A list of ``tf.Variable`` representing the EMA values to be updated.
    :param decay:
        A ``float`` defining the EMA decay rate.
    """
    for variable, ema_variable in zip(variables, ema_variables):
        ema_variable.assign_sub((1 - decay) * (ema_variable - variable))


def one_hot(indices: tf.Tensor, depth: int) -> np.ndarray:
    """
    One-hot encode indices to vector with depth dimension.

    :param indices:
        A vector of indices to be one-hot encoded.
    :param depth:
        The dimension of one-hot encoding.
    :return:
        One-hot encoded vectors.
    """
    return tf.one_hot(tf.cast(indices, tf.int32), depth, dtype=tf.float32)


def concatenate(tensors: List[tf.Tensor], axis: int) -> tf.Tensor:
    """
    Join a list of tensors along an existing axis.

    :param tensors:
        List of tensors to be concatenated.
    :param axis:
        Axis to concatenate the tensors.
    :return:
        A concatenated tensor.
    """
    return tf.concat(tensors, axis=axis)


_update_state_graph_cache = {}
_result_graph_cache = {}


def _run_data_to_state_graph(metric: tf.keras.metrics.Metric, labels, preds):
    """
    Reset metric, call `metric.update_state` in graph mode with input data and
    return the new state of the metric. The graph of each Keras metric is
    cached.
    """
    key = (metric, labels.dtype, preds.dtype)
    if key not in _update_state_graph_cache:
        # When doing federated learning, the batch dimension can be very
        # different for different users, so we try to create 1 graph for all
        # cases by having a variable batch dimension.
        spec = [
            tf.nest.map_structure(
                lambda x: tf.TensorSpec([None] + list(x.shape)[1:], x.dtype),
                labels),
            tf.nest.map_structure(
                lambda x: tf.TensorSpec([None] + list(x.shape)[1:], x.dtype),
                preds),
        ]

        @tf.function(experimental_relax_shapes=True, input_signature=spec)
        def data_to_state_graph(labels, preds):
            try:
                # tf>=2.15
                metric.reset_state()
            except AttributeError:
                metric.reset_states()
            metric.update_state(labels, preds)
            return [tf.identity(v) for v in metric.variables]

        _update_state_graph_cache[key] = data_to_state_graph
    return _update_state_graph_cache[key](labels, preds)


def _run_result_graph(metric: tf.keras.metrics.Metric, state):
    """
    Apply ``state`` to the Keras metric and thereafter call `metric.result`
    in graph mode to get the postprocessed results. The graph of each Keras
    metric is cached.
    """
    key = (metric, *[s.dtype for s in state])
    if key not in _result_graph_cache:

        spec = [tf.TensorSpec(x.shape, x.dtype) for x in state]

        @tf.function(experimental_relax_shapes=True, input_signature=spec)
        def result_graph(*state):
            for v1, v2 in zip(metric.variables, state):
                v1.assign(v2)
            return metric.result()

        _result_graph_cache[key] = result_graph
    return _result_graph_cache[key](*state)


@tf.function(experimental_relax_shapes=True)
def _add_states(state1, state2):
    new_states = []
    for v1, v2 in zip(state1, state2):
        new_states.append(v1 + v2)
    return new_states


class KerasMetricValue(MetricValue):
    """
    Wrapper for representing a ``tf.keras.metrics.Metric`` as a
    :class:`~pfl.metrics.MetricValue` to be compatible with pfl framework.

    :param keras_metric:
        The Keras metric to use for accumulating measurements of a metric.
        Keras metrics are mutable, but ``KerasMetricValue`` is not.
    :param labels:
        Ground-truth labels that are used with ``predictions`` to set
        the state of the metric value.
    :param predictions:
        ``labels`` and ``predictions`` are used to set the state of the metric
        value. Unlike ``tf.keras.metrics.Metric``, the state doesn't change.
        You should instead accumulate a metric value with addition of
        two ``KerasMetricValue`` objects.
    :param state:
        Specify the state of `keras_metric` directly instead of generating it
        from ``labels`` and ``predictions``. Don't set ``labels`` and
        ``predictions`` if ``state`` is set.
    """

    def __init__(self,
                 keras_metric,
                 labels=None,
                 predictions=None,
                 state=None):
        self._keras_metric = keras_metric
        if labels is not None:
            assert predictions is not None
            assert state is None, (
                'Provide either (labels, predictions) or state')
            # Initial state of internal Keras metric is
            # generated from the datapoint.
            self._metric_state = _run_data_to_state_graph(
                self._keras_metric, labels, predictions)
        else:
            assert predictions is None, (
                'Provide either (labels, predictions) or state')
            # Initial state of internal Keras metric is already
            # specified in the constructor.
            self._metric_state = state

    def __add__(self, other):
        assert self._keras_metric is other._keras_metric, (
            'Should wrap the same underlying Keras metric')
        new_state = _add_states(self._metric_state, other._metric_state)
        return KerasMetricValue(self._keras_metric, state=new_state)

    def __eq__(self, other):
        # Two instances of this class are equal if they have the same
        # underlying Keras metric and the same state.
        assert isinstance(other, KerasMetricValue)
        assert self._keras_metric is other._keras_metric, (
            'Should wrap the same underlying Keras metric')
        return np.all([
            np.array_equal(v1.numpy(), v2.numpy())
            for v1, v2 in zip(self._metric_state, other._metric_state)
        ])

    @property
    def overall_value(self):
        return _run_result_graph(self._keras_metric,
                                 self._metric_state).numpy()

    def to_vector(self) -> np.ndarray:
        return np.array(
            [v.numpy().astype(np.float32) for v in self._metric_state])

    def from_vector(self, vector: np.ndarray) -> 'MetricValue':
        new_state = [
            tf.Variable(state_value, dtype=v.dtype)
            for state_value, v in zip(vector, self._metric_state)
        ]
        new_metric_value = KerasMetricValue(self._keras_metric,
                                            state=new_state)
        return new_metric_value
