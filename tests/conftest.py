# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
# pylint: disable=too-many-lines
import asyncio
import itertools
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from pfl.aggregate.base import Backend, get_total_weight_name
from pfl.algorithm.base import FederatedAlgorithm, NNAlgorithmParams
from pfl.callback import TrainingProcessCallback
from pfl.common_types import Population
from pfl.context import CentralContext, UserContext
from pfl.data.dataset import Dataset, TabularDataset
from pfl.data.federated_dataset import FederatedDatasetBase
from pfl.hyperparam.base import AlgorithmHyperParams, HyperParam, ModelHyperParams, NNTrainHyperParams
from pfl.internal.ops.common_ops import get_pytorch_major_version, get_tf_major_version
from pfl.internal.ops.selector import _internal_reset_framework_module, get_framework_module, set_framework_module
from pfl.internal.tree.node import Node
from pfl.metrics import (
    MetricName,
    Metrics,
    StringMetricName,
    TrainMetricName,
    Weighted,
    get_overall_value,
    user_average,
)
from pfl.model.base import StatefulModel
from pfl.privacy.privacy_mechanism import CentrallyApplicablePrivacyMechanism
from pfl.stats import MappedVectorStatistics


@pytest.fixture
def check_equal_stats():
    """Compares two statistics. Weights are not compared.
    """

    def _check_equal_stats(stats1, stats2, to_numpy_fn=lambda x: x):
        # Assert two dictionaries of model updates to be "almost" the same.
        # This does not consider weights for weighted statistics.
        stats1_names_set = set(stats1.keys())
        stats2_names_set = set(stats2.keys())

        assert stats1_names_set == stats2_names_set

        for var_name in stats1_names_set:
            # Even if the stats per worker is correct, this can still fail in the
            # case where training and evaluation runs are mixed (no consistent
            # order) across workers.
            np.testing.assert_array_almost_equal(to_numpy_fn(stats1[var_name]),
                                                 to_numpy_fn(stats2[var_name]),
                                                 decimal=4)

    return _check_equal_stats


@pytest.fixture
def check_equal_metrics():
    """Compares two metrics.
    """

    def _check_equal_metrics(metrics1, metrics2):
        if isinstance(metrics1, Metrics):
            metrics1 = metrics1.to_simple_dict()
        if isinstance(metrics2, Metrics):
            metrics2 = metrics2.to_simple_dict()
        # Assert two metrics object to be "almost" the same.
        metrics1_names_set = {k for k, v in metrics1.items()}
        metrics2_names_set = {k for k, v in metrics2.items()}

        assert metrics1_names_set == metrics2_names_set

        for metric_name in metrics1_names_set:
            metric1 = metrics1[metric_name]
            metric2 = metrics2[metric_name]
            if isinstance(metric1, Weighted):
                assert isinstance(metric2, Weighted)
                assert metric1.weight == pytest.approx(metric2.weight)
                assert metric1.weighted_value == pytest.approx(
                    metric2.weighted_value)

            np.testing.assert_array_almost_equal(get_overall_value(metric1),
                                                 get_overall_value(metric2),
                                                 decimal=5)

    return _check_equal_metrics


@pytest.fixture
def assert_dict_equal():

    def _assert_dict_equal(actual,
                           expected,
                           except_field: Optional[List[str]] = None):
        """
        Test if two dictionaries are equal.
        `except_field` is a collection of dot-separated strings,
        which can be used to ignore some fields when comparing.

        Usage: if except_field=['a.b.c'], then
        actual['a']['b']['c'] and expected['a']['b']['c'] is not compared
        for equality
        """

        def drop_field(dictionary, keys):
            if not isinstance(dictionary, dict):
                return
            if '.' in keys:
                key, _, left_keys = keys.partition('.')
                drop_field(dictionary.get(key), left_keys)
            else:
                dictionary.pop(keys)

        s = TestCase()
        s.maxDiff = None
        if not except_field:
            s.assertDictEqual(actual, expected)
        else:
            actual_copy = deepcopy(actual)
            expected_copy = deepcopy(expected)
            for fields in except_field:
                drop_field(actual_copy, fields)
                drop_field(expected_copy, fields)
            s.assertDictEqual(actual_copy, expected_copy)

    return _assert_dict_equal


def pytest_addoption(parser):
    parser.addoption('--macos',
                     action='store_true',
                     default=False,
                     help='run tests that require MacOS')
    parser.addoption('--disable_horovod',
                     action='store_true',
                     default=False,
                     help='run tests that require MacOS')
    parser.addoption('--disable_slow',
                     action='store_true',
                     default=False,
                     help=('Disable slow tests. We need to speed up these '
                           'tests for reasonable runtime on CircleCI'))


def pytest_collection_modifyitems(config, items):
    disable_horovod = config.getoption("--disable_horovod")
    skip_horovod_marker = pytest.mark.skip(
        reason="Test disabled with --disable_horovod")
    disable_slow = config.getoption("--disable_slow")
    skip_slow_marker = pytest.mark.skip(
        reason="Test disabled with --disable_slow")

    disable_macos = not config.getoption('--macos')
    skip_macos_marker = pytest.mark.skip(reason='need --macos option to run')

    for item in items:
        if "horovod" in item.keywords and disable_horovod:
            item.add_marker(skip_horovod_marker)
        if "is_slow" in item.keywords and disable_slow:
            item.add_marker(skip_slow_marker)
        if "macos" in item.keywords and disable_macos:
            item.add_marker(skip_macos_marker)


@pytest.fixture
def new_event_loop():
    # Prevents
    # "RuntimeError: There is no current event loop in thread 'MainThread'"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope='session')
def user_dataset():
    return Dataset((np.array([[1., 0.], [0., 2.]], dtype=np.float32),
                    np.array([[4., 6.], [8., 12.]], dtype=np.float32)),
                   user_id='user1')


@pytest.fixture
def user_context():
    return UserContext(num_datapoints=2,
                       seed=1,
                       user_id='BestUser',
                       metrics=Metrics())


@pytest.fixture(scope='session')
def federated_dataset(user_dataset):
    federated_dataset = MagicMock(spec=FederatedDatasetBase)
    seeds = itertools.count()
    federated_dataset.__iter__.return_value = ((user_dataset, seed)
                                               for seed in seeds)
    federated_dataset.get_cohort.side_effect = lambda cohort_size: [
        (user_dataset, next(seeds)) for _ in range(cohort_size)
    ]
    return federated_dataset


class FederatedAlgorithmWithTrainingProcessCallback(FederatedAlgorithm,
                                                    TrainingProcessCallback):
    """
    Class that derives from both FederatedAlgorithm and
    TrainingProcessCallback, for testing.
    """
    pass


@pytest.fixture(scope='module', params=[0, 1])
def weight_by_samples(request):
    return request.param


@pytest.fixture(scope='function')
def mock_algorithm(request, central_context):
    algorithm = MagicMock(FederatedAlgorithmWithTrainingProcessCallback)
    algorithm.get_next_central_contexts.side_effect = (
        lambda model, iteration: ((central_context, ), model, Metrics()))
    # Return identity, do nothing.
    algorithm.process_aggregated_statistics.side_effect = (
        lambda c, m, model, u: (model, Metrics()))
    algorithm.after_central_iteration.side_effect = lambda i, c, m, model: (
        False, m)

    def mock_simulate_one_user(model, user_dataset, central_context):
        # pytype: disable=wrong-arg-count
        metrics = Metrics([(MetricName('simulate_one_user_loss',
                                       central_context.population),
                            Weighted.from_unweighted(1))])
        # pytype: enable=wrong-arg-count

        if central_context.population == Population.TRAIN:
            model_update = MappedVectorStatistics({'var1': np.ones((2, 3))})
        else:
            model_update = None
        return model_update, metrics

    algorithm.simulate_one_user.side_effect = mock_simulate_one_user
    return algorithm


@pytest.fixture(scope='function')
def mock_backend():
    backend = MagicMock(spec=Backend)

    async def mock_async_gather_results(model, training_algorithm,
                                        central_context):
        population = central_context.population
        metrics = Metrics()
        # pytype: disable=duplicate-keyword-argument,wrong-arg-count
        if population == Population.TRAIN:
            metrics[TrainMetricName('loss', population,
                                    after_training=True)] = 0.1

        metrics[TrainMetricName('loss', population,
                                after_training=False)] = 1337
        metrics[get_total_weight_name(
            population)] = central_context.cohort_size
        # pytype: enable=duplicate-keyword-argument,wrong-arg-count

        if population == population.TRAIN:
            model_updates = MappedVectorStatistics(
                {'var1': np.ones((2, 3))}, weight=central_context.cohort_size)
        else:
            model_updates = None
        return model_updates, metrics

    backend.async_gather_results.side_effect = mock_async_gather_results
    return backend


@pytest.fixture(scope='function')
def mock_callback():
    callback = MagicMock(spec=TrainingProcessCallback)
    callback.on_train_begin.side_effect = \
        lambda model: Metrics([(StringMetricName('called_begin'), 1)])
    # Return identity, do nothing.
    callback.after_central_iteration.side_effect = \
            lambda m, model, central_iteration: (
        False, Metrics([(StringMetricName('called_callback'), 1)]))
    return callback


@pytest.fixture(scope='session')
def mock_train_partition():
    return MagicMock(spec=Dataset)


@pytest.fixture(scope='session')
def mock_val_partition():
    return MagicMock(spec=Dataset)


@pytest.fixture(scope='function')
def mock_dataset(mock_train_partition, mock_val_partition):
    mock_dataset = MagicMock(spec=Dataset)
    mock_dataset.raw_data = [
        np.array([[1., 0.], [0., 2.]]),
        np.array([[4., 6.], [8., 12.]])
    ]
    mock_dataset.__len__.return_value = 2
    mock_dataset.split.return_value = (mock_train_partition,
                                       mock_val_partition)
    return mock_dataset


def make_mock_model(request, tmp_path):
    mock_model = MagicMock(spec=StatefulModel)

    def new_model(statistics):
        # TODO Return copy.deepcopy(model) and do something clever with
        #   tests.
        return mock_model, Metrics([(StringMetricName('applied_stats'), 1)])

    mock_model.apply_model_update.side_effect = new_model
    mock_model.variable_map = {'w1': np.zeros((1, 1))}

    mock_model.evaluate.side_effect = (
        lambda data, name_formatting_fn, eval_params: Metrics([(
            name_formatting_fn('loss'), Weighted.from_unweighted(1))]))

    def mock_get_model_difference(initial_model_state):
        return MappedVectorStatistics({'w1': np.ones((1, 1))})

    mock_model.get_model_difference.side_effect = (mock_get_model_difference)
    mock_model.variable_map = {'w1': np.zeros((1, 1))}
    return mock_model


@pytest.fixture
def mock_ops(numpy_ops):
    mock_ops = MagicMock()
    mock_ops.get_shape = lambda v: tuple(v.shape)
    mock_ops.distributed.all_reduce = lambda ndarrays, op='SUM': ndarrays
    mock_ops.distributed.local_rank = 0
    mock_ops.distributed.global_rank = 0
    mock_ops.distributed.world_size = 1
    mock_ops.distributed.distribute_range = lambda v: range(v)
    mock_ops.distributed.distribute_value = lambda v: v
    mock_ops.to_tensor = lambda t: np.array(t)
    mock_ops.to_numpy = lambda t: np.array(t)
    mock_ops.clone = lambda t: np.array(t)
    mock_ops.reshape = numpy_ops.reshape
    return mock_ops


@pytest.fixture(scope='function')
def mock_model(request, tmp_path, mock_ops):
    """
    Return a StatefulModel.
    """
    _internal_reset_framework_module()
    set_framework_module(mock_ops)
    yield make_mock_model(request, tmp_path)
    _internal_reset_framework_module()


@pytest.fixture
def nn_train_params():
    return NNTrainHyperParams(local_num_epochs=6,
                              local_learning_rate=0.1,
                              local_batch_size=7,
                              local_max_grad_norm=0.5)


@pytest.fixture
def nn_algorithm_params(request):
    kwargs = request.param if hasattr(request, 'param') else {}

    return NNAlgorithmParams(
        **{
            'central_num_iterations': 3,
            'evaluation_frequency': 2,
            'train_cohort_size': 5,
            'val_cohort_size': 5,
            **kwargs
        })


@pytest.fixture(scope='function')
def central_context(request):
    kwargs = request.param.copy() if hasattr(request, 'param') else {}

    central_context = CentralContext(
        **{
            'current_central_iteration': 0,
            'do_evaluation': True,
            'cohort_size': 10,
            'population': Population.TRAIN,
            'algorithm_params': AlgorithmHyperParams(),
            'model_train_params': ModelHyperParams(),
            'model_eval_params': ModelHyperParams(),
            'seed': None,
            **kwargs
        })

    return central_context


@pytest.fixture
def mock_privacy_mechanism():
    privacy_mechanism = MagicMock(spec=CentrallyApplicablePrivacyMechanism)
    name_formatting_fn = lambda n: StringMetricName(n)

    def privatize(statistics,
                  name_formatting_fn=name_formatting_fn,
                  seed=None):
        return statistics, Metrics([(name_formatting_fn('privatize'), 1.0)])

    privacy_mechanism.privatize.side_effect = privatize

    def add_noise(statistics,
                  cohort_size,
                  name_formatting_fn=name_formatting_fn,
                  seed=None):
        return statistics, Metrics([(name_formatting_fn('add_noise'), 1.0)])

    privacy_mechanism.add_noise.side_effect = add_noise

    def constrain_sensitivity(statistics,
                              name_formatting_fn=name_formatting_fn,
                              seed=None):
        return statistics, Metrics([
            (name_formatting_fn('constrain_sensitivity'), 1.0)
        ])

    privacy_mechanism.constrain_sensitivity.side_effect = constrain_sensitivity
    return privacy_mechanism


# The *_ops fixtures below should be used when you want to manually use a
# specific deep learning framework ops module for tests.
# When using a ``pfl.model.Model``, the ops module is set automatically.


@pytest.fixture(scope='function')
def numpy_ops():
    from pfl.internal.ops import numpy_ops
    _internal_reset_framework_module()
    set_framework_module(numpy_ops)
    yield get_framework_module()
    _internal_reset_framework_module()


@pytest.fixture(scope='function')
def tensorflow_ops():
    from pfl.internal.ops import tensorflow_ops
    _internal_reset_framework_module()
    set_framework_module(tensorflow_ops)
    yield get_framework_module()
    _internal_reset_framework_module()


@pytest.fixture(scope='function')
def pytorch_ops():
    from pfl.internal.ops import pytorch_ops
    _internal_reset_framework_module()
    set_framework_module(pytorch_ops)
    yield get_framework_module()
    _internal_reset_framework_module()


@pytest.fixture(scope='function')
def fix_global_random_seeds():
    """
    Set the random seed for NumPy as well as any framework that is loaded.
    This makes tests deterministic.
    """
    np.random.seed(2022)
    if get_tf_major_version() != 0:
        import tensorflow as tf
        tf.random.set_seed(2022)
    if get_pytorch_major_version() != 0:
        import torch
        torch.manual_seed(2022)


@pytest.fixture(scope='session')
def gbdt_datapoints() -> np.ndarray:
    return np.array([[-1, -1, 0, 0], [-1, 2, 0, 0], [1, 0, -1, 0],
                     [1, 0, 3, 0]])


@pytest.fixture(scope='session')
def gbdt_user_dataset(gbdt_datapoints):
    targets = np.array([1, 1, 0, 0])
    return TabularDataset(features=gbdt_datapoints, labels=targets)


@pytest.fixture(scope='function')
def tree_fully_trained_3_layers():
    root_node = Node(feature=0, threshold=0)
    left_child = Node(feature=1, threshold=1)
    right_child = Node(feature=2, threshold=2)
    leaf_left_left = Node(value=0.3)
    leaf_left_right = Node(value=0.4)
    leaf_right_left = Node(value=-0.3)
    leaf_right_right = Node(value=-0.4)

    root_node.left_child = left_child
    root_node.right_child = right_child
    left_child.left_child = leaf_left_left
    left_child.right_child = leaf_left_right
    right_child.left_child = leaf_right_left
    right_child.right_child = leaf_right_right

    return root_node


@pytest.fixture(scope='function')
def set_trees():
    from pfl.tree.gbdt_model import GBDTModelClassifier, GBDTModelRegressor, NodeRecord

    def _set_trees(model: Union[GBDTModelClassifier, GBDTModelRegressor],
                   trees: List[Node]):
        for tree in trees:
            model._gbdt.add_tree(tree)  # pylint: disable=protected-access

        # add all incomplete nodes of tree to model.nodes_to_split
        if not model._gbdt.trees[-1].training_complete():  # pylint: disable=protected-access

            def return_incomplete_nodes(
                    node: Node, decision_path: List[List]) -> List[NodeRecord]:
                if node.is_leaf():
                    return []

                if not node.left_child and not node.right_child:
                    return [
                        NodeRecord(node, [
                            *decision_path.copy(),
                            [node.feature, node.threshold, True]
                        ], True, 0.3, False),
                        NodeRecord(node, [
                            *decision_path.copy(),
                            [node.feature, node.threshold, False]
                        ], False, 0.3, False)
                    ]

                if node.left_child and node.right_child:
                    return [
                        *return_incomplete_nodes(node.left_child, [
                            *decision_path.copy(),
                            [node.feature, node.threshold, True]
                        ]), *return_incomplete_nodes(node.right_child, [
                            *decision_path.copy(),
                            [node.feature, node.threshold, False]
                        ])
                    ]

                raise ValueError(
                    'Cannot have one child node set and not the ',
                    'other child node set, since GBDTs are trained ',
                    'layer-wise in PFL')

            # pylint: disable=protected-access
            model._nodes_to_split = return_incomplete_nodes(
                model._gbdt.trees[-1], [])
            # pylint: enable=protected-access

        return model

    return _set_trees


@pytest.fixture(scope='function')
def tree_incomplete_2_layers():
    root_node = Node(feature=0, threshold=0)
    left_child = Node(feature=1, threshold=1)
    right_child = Node(feature=2, threshold=2)
    root_node.left_child = left_child
    root_node.right_child = right_child

    return root_node


@pytest.fixture(scope='function')
def gbdt_algorithm_params(request):
    from pfl.tree.federated_gbdt import GBDTAlgorithmHyperParams
    params = {'cohort_size': 2, 'val_cohort_size': 1, 'num_trees': 3}

    if hasattr(request, 'param'):
        params.update(**dict(request.param))

    print('params', params)
    algorithm_params = GBDTAlgorithmHyperParams(**params)

    return algorithm_params


@pytest.fixture(scope='function')
def gbdt_internal_algorithm_params(gbdt_algorithm_params, request):
    from pfl.tree.federated_gbdt import _GBDTInternalAlgorithmHyperParams

    gbdt_questions = [{
        'decisionPath': [],
        'splits': {
            '0': [-1, 0],
            '1': [-1, 0]
        }
    }, {
        'decisionPath': [(0, 0, False)],
        'splits': {
            '0': [-1, 0],
            '1': [-1, 0]
        }
    }, {
        'decisionPath': [(0, 0, True)],
        'splits': {
            '0': [-1, 0],
            '1': [-1, 0]
        }
    }, {
        'decisionPath': [(0, 0, True), (1, 0, True)],
        'splits': {
            '0': [-1, 0],
            '1': [-1, 0]
        }
    }]

    if hasattr(request, 'param'):
        internal_algorithm_params = (
            _GBDTInternalAlgorithmHyperParams.from_GBDTAlgorithmHyperParams(
                gbdt_algorithm_params,
                gbdt_questions=gbdt_questions,
                **dict(request.param)))
    else:
        internal_algorithm_params = (
            _GBDTInternalAlgorithmHyperParams.from_GBDTAlgorithmHyperParams(
                gbdt_algorithm_params, gbdt_questions=gbdt_questions))
    return internal_algorithm_params


@pytest.fixture(scope='session')
def check_dictionaries_almost_equal():

    def _check_dictionaries_almost_equal(dict_1: Dict, dict_2: Dict):
        """
        Test that two dictionaries are equal,
        or that numeric values are almost equal.

        Must have equal keys and (almost, if
        numeric) equal values.
        """
        assert set(dict_1.keys()) == set(dict_2.keys())

        for key in dict_1:
            if isinstance(dict_1[key], (int, float)):
                np.testing.assert_almost_equal(dict_1[key], dict_2[key])
            else:
                assert dict_1[key] == dict_2[key]

    return _check_dictionaries_almost_equal


@pytest.fixture
def make_hyperparameter():

    class MyHyperParameter(HyperParam):

        def __init__(self, v):
            self._value = v

        def set_value(self, v):
            self._value = v

        def value(self):
            return self._value

    return MyHyperParameter


# Struct to return from the model fixtures.
class ModelSetup(NamedTuple):
    # The model object.
    model: StatefulModel
    # A helper function to convert a variable into a `numpy.ndarray`.
    variable_to_numpy_fn: Callable[[Any], np.ndarray]
    # A dataset for test cases.
    user_dataset: Dataset
    # A list of the variables of the model.
    variables: Optional[List] = None
    # A list of the names of `variables`.
    variable_names: Optional[List[str]] = None
    # The path to the model if `model.save(save_model_path)` was called.
    load_model_path: Optional[Path] = None
    save_model_path: Optional[Path] = None
    # (Boolean) whether or not the deep learning framework averages the loss
    # internally in the framework.
    reports_average_loss: bool = False


@pytest.fixture
def model_checkpoint_dir(tmp_path):
    checkpoint_dir = tmp_path / 'checkpoints'
    checkpoint_dir.mkdir()
    yield checkpoint_dir


@pytest.fixture(scope="function")
def get_keras_model_functional():
    import tensorflow as tf  # type: ignore

    # Deterministic weight to calculate expected statistics.
    static_initializer = tf.constant_initializer(np.array([[2., 4.], [3.,
                                                                      5.]]))

    # Creates a model using the functional API.
    def model_functional(freeze, use_softmax=False):
        inputs = tf.keras.layers.Input(shape=(2, ))
        predictions = tf.keras.layers.Dense(
            2,
            activation='softmax' if use_softmax else None,
            use_bias=False,
            trainable=not freeze,
            kernel_initializer=static_initializer)(inputs)
        return tf.keras.models.Model(inputs=inputs, outputs=predictions)

    return model_functional


@pytest.fixture(scope="function")
def get_keras_model_sequential():
    import tensorflow as tf  # type: ignore

    # Deterministic weight to calculate expected statistics.
    static_initializer = tf.constant_initializer(np.array([[2., 4.], [3.,
                                                                      5.]]))

    # Creates a sequential model
    def model_sequential(freeze, use_softmax=False):
        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                2,
                activation='softmax' if use_softmax else None,
                input_shape=(2, ),
                use_bias=False,
                trainable=not freeze,
                kernel_initializer=static_initializer)
        ])

    return model_sequential


@pytest.fixture(scope="function",
                params=[
                    pytest.param(lazy_fixture('get_keras_model_functional'),
                                 id='functional'),
                    pytest.param(lazy_fixture('get_keras_model_sequential'),
                                 id='sequential'),
                ])
def get_model(request):
    # This proxy is needed because you can't parametrize both a fixture and the
    # method using the fixture:
    # https://github.com/pytest-dev/pytest/issues/3960.
    return request.param


@pytest.fixture(scope="function")
def tensorflow_model_setup(request, tensorflow_ops, user_dataset, get_model,
                           model_checkpoint_dir):
    """
    TF with Keras model fixture. Also returns parameters needed for test
    cases. Since a fixture can only return one value, the objects returned
    are wrapped into a namedtuple.

    This fixture is parametrized by `get_model` to exhaustively test both
    a `tf.keras.models.Sequential` and a `tf.keras.models.Model`
    """
    import tensorflow as tf  # type: ignore
    import tensorflow.keras.backend as K
    from tensorflow.python.keras.metrics import MeanMetricWrapper

    from pfl.model.tensorflow import TFModel

    if hasattr(request, 'param') and 'get_central_optimizer' in request.param:
        central_optimizer = request.param['get_central_optimizer']()
    else:
        central_optimizer = tf.keras.optimizers.SGD(1.0)
    freeze = False
    use_softmax = False

    def loss2(y_true, y_pred):
        return K.sum(K.abs(y_pred - y_true), axis=-1)

    model = get_model(freeze, use_softmax)
    model.compile(
        # This will result in 2 metrics recorded in
        # pfl named "loss" and "loss2".
        tf.keras.optimizers.SGD(100.0),
        loss=loss2,
        metrics=[loss2])

    variables = []
    for l in model.layers:
        if isinstance(l, tf.keras.layers.Layer):
            variables.extend(l.weights)

    model_params = {'model': model, 'central_optimizer': central_optimizer}
    # Override any parameter of the model with parametrization of the fixture.
    if hasattr(request, 'param') and 'model_kwargs' in request.param:
        model_params.update(**request.param['model_kwargs'])

    # Make a keras metric class that is equivalent to `loss2` above.
    class Loss2(MeanMetricWrapper):  # pylint: disable=abstract-method

        def __init__(self, *args, **kwargs):
            super().__init__(loss2, *args, **kwargs)

    model_params['metrics'] = {
        'loss': Loss2(),
        'loss2': Loss2(),
        'user_avg_loss': (Loss2(), user_average)
    }

    model = TFModel(**model_params)

    yield ModelSetup(model=model,
                     variables=variables,
                     variable_to_numpy_fn=lambda v: v.numpy(),
                     variable_names=[v.name for v in variables],
                     load_model_path=model_checkpoint_dir,
                     save_model_path=model_checkpoint_dir,
                     user_dataset=user_dataset,
                     reports_average_loss=True)

    _internal_reset_framework_module()
    tf.keras.backend.clear_session()


@pytest.fixture(scope="function")
def pytorch_model_setup(request, pytorch_ops, user_dataset,
                        model_checkpoint_dir):
    """
    PyTorch model fixture. Also returns parameters needed for test cases.
    Since a fixture can only return one value, the objects returned
    are wrapped into a namedtuple.
    """
    import torch  # type: ignore

    from pfl.model.pytorch import PyTorchModel

    class TestModel(torch.nn.Module):

        def __init__(self):

            super().__init__()
            self.weight = torch.nn.parameter.Parameter(
                torch.tensor(np.array([[2., 4.], [3., 5.]],
                                      dtype=np.float32)).cpu())
            self._l1loss = torch.nn.L1Loss(reduction='none')

        def forward(self, x):  # pylint: disable=arguments-differ
            return torch.matmul(x, self.weight)

        def loss(self, x, y, eval=False):
            if eval:
                self.eval()
            else:
                self.train()
            # Sum loss across output dim, average across batches,
            # similar to what is built-in in Keras model training.
            unreduced_loss = self._l1loss(self(torch.FloatTensor(x)),
                                          torch.FloatTensor(y))
            return torch.mean(torch.sum(unreduced_loss, dim=1))

        def loss2(self, x, y, eval=True):
            self.eval()
            l1loss = torch.nn.L1Loss(reduction='none')
            # Sum loss across output dim, average across batches,
            # similar to what is built-in in Keras model training.
            unreduced_loss = l1loss(self(torch.FloatTensor(x)),
                                    torch.FloatTensor(y))
            loss_value = torch.mean(torch.sum(unreduced_loss, dim=1))

            num_samples = x.shape[0]
            return loss_value, num_samples

        def metrics(self, x, y):
            self.eval()
            pred = self(x)
            # Sum across batch dimension as well because the denominator
            # is saved in Weighted.
            loss = torch.sum(self._l1loss(pred, y))
            num_samples = len(y)
            return {
                'loss': Weighted(loss, num_samples),
                'loss2': Weighted(loss, num_samples),
                'user_avg_loss': (Weighted(loss, num_samples), user_average)
            }

    pytorch_model = TestModel()
    central_learning_rate = 1.0

    model_params = {
        'model':
        pytorch_model,
        'local_optimizer_create':
        torch.optim.SGD,
        'central_optimizer':
        torch.optim.SGD(pytorch_model.parameters(), lr=central_learning_rate),
    }

    # Overrides any parameter with values in kwargs.
    if hasattr(request, 'param') and 'model_kwargs' in request.param:
        model_params.update(**request.param['model_kwargs'])
    model = PyTorchModel(**model_params)

    variables = [
        v for _, v in pytorch_model.named_parameters() if v.requires_grad
    ]
    variable_names = [
        name for name, v in pytorch_model.named_parameters() if v.requires_grad
    ]

    yield ModelSetup(model=model,
                     variables=variables,
                     variable_to_numpy_fn=lambda v: v.detach().numpy().copy(),
                     variable_names=variable_names,
                     load_model_path=model_checkpoint_dir,
                     save_model_path=model_checkpoint_dir,
                     user_dataset=user_dataset,
                     reports_average_loss=True)
