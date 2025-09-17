# Copyright © 2023-2024 Apple Inc.
import itertools
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_lazy_fixtures import lf

from pfl.aggregate.simulate import SimulatedBackend
from pfl.common_types import Population
from pfl.context import CentralContext
from pfl.data.federated_dataset import FederatedDatasetBase
from pfl.internal.ops.common_ops import check_pfl_tree_installed
from pfl.metrics import Metrics
from pfl.stats import MappedVectorStatistics

if check_pfl_tree_installed():
    from pfl.tree.federated_gbdt import (
        STATISTIC_INFO_NAME,
        FederatedGBDT,
        GBDTAlgorithmHyperParams,
        _GBDTInternalAlgorithmHyperParams,
    )
    from pfl.tree.gbdt_model import (
        GBDTClassificationModelHyperParams,
        GBDTModelClassifier,
        GBDTModelHyperParams,
        NodeRecord,
    )


@pytest.fixture(scope='function')
def regression_first_order_gradients_left():
    return np.array(
        [-2, -2, -1, -1, 0, 0, 0, 0, -2, -2, -1, -1, -1, -1, -1, -1])


@pytest.fixture(scope='function')
def regression_first_order_gradients_right():
    return np.array([0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0])


@pytest.fixture(scope='function')
def regression_second_order_gradients_left():
    return np.array([2, 2, 1, 3, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1, 1])


@pytest.fixture(scope='function')
def regression_second_order_gradients_right():
    return np.array([2, 2, 3, 1, 2, 2, 2, 0, 0, 0, 1, 1, 0, 0, 0, 0])


@pytest.fixture(scope='function')
def classification_first_order_gradients_left():
    return np.array([
        -1.0, -1.0, -0.5, 0.5, 0, 0, 0, 1, -1.0, -1.0, -0.5, -0.5, -0.5, -0.5,
        -0.5, -0.5
    ])


@pytest.fixture(scope='function')
def classification_first_order_gradients_right():
    return np.array(
        [1, 1, 0.5, -0.5, 1, 1, 1, 0, 0, 0, -0.5, -0.5, 0, 0, 0, 0])


@pytest.fixture(scope='function')
def classification_second_order_gradients_left():
    return np.array([
        0.5, 0.5, 0.25, 0.75, 0, 0, 0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25
    ])


@pytest.fixture(scope='function')
def classification_second_order_gradients_right():
    return np.array(
        [0.5, 0.5, 0.75, 0.25, 0.5, 0.5, 0.5, 0, 0, 0, 0.25, 0.25, 0, 0, 0, 0])


@pytest.fixture(scope='function')
def regression_first_order_gradients(regression_first_order_gradients_left,
                                     regression_first_order_gradients_right):
    regression_first_order_gradients = []
    for i in range(4):
        regression_first_order_gradients.extend(
            regression_first_order_gradients_left[i * 4:i * 4 + 4])
        regression_first_order_gradients.extend(
            regression_first_order_gradients_right[i * 4:i * 4 + 4])
    return regression_first_order_gradients


@pytest.fixture(scope='function')
def regression_second_order_gradients(regression_second_order_gradients_left,
                                      regression_second_order_gradients_right):
    regression_second_order_gradients = []
    for i in range(4):
        regression_second_order_gradients.extend(
            regression_second_order_gradients_left[i * 4:i * 4 + 4])
        regression_second_order_gradients.extend(
            regression_second_order_gradients_right[i * 4:i * 4 + 4])
    return regression_second_order_gradients


@pytest.fixture(scope='function')
def regression_first_second_order_gradients(regression_first_order_gradients,
                                            regression_second_order_gradients):
    return np.hstack(
        (regression_first_order_gradients, regression_second_order_gradients))


@pytest.fixture(scope='function')
def classification_first_order_gradients(
        classification_first_order_gradients_left,
        classification_first_order_gradients_right):
    classification_first_order_gradients = []
    for i in range(4):
        classification_first_order_gradients.extend(
            classification_first_order_gradients_left[i * 4:i * 4 + 4])
        classification_first_order_gradients.extend(
            classification_first_order_gradients_right[i * 4:i * 4 + 4])
    return classification_first_order_gradients


@pytest.fixture(scope='function')
def classification_second_order_gradients(
        classification_second_order_gradients_left,
        classification_second_order_gradients_right):
    classification_second_order_gradients = []
    for i in range(4):
        classification_second_order_gradients.extend(
            classification_second_order_gradients_left[i * 4:i * 4 + 4])
        classification_second_order_gradients.extend(
            classification_second_order_gradients_right[i * 4:i * 4 + 4])
    return classification_second_order_gradients


@pytest.fixture(scope='function')
def classification_first_second_order_gradients(
        classification_first_order_gradients,
        classification_second_order_gradients):
    return np.hstack((classification_first_order_gradients,
                      classification_second_order_gradients))


@pytest.fixture(scope='function')
def regression_first_order_gradients_one_side_only_node_sum(
        regression_first_order_gradients_left,
        regression_first_order_gradients_right):
    regression_first_order_gradients_one_side_only_node_sum = (
        regression_first_order_gradients_left.tolist())
    for i in range(4):
        regression_first_order_gradients_one_side_only_node_sum.insert(
            4 * (i + 1) + i,
            regression_first_order_gradients_left[4 * (i + 1) - 1] +
            regression_first_order_gradients_right[4 * (i + 1) - 1])
    return regression_first_order_gradients_one_side_only_node_sum


@pytest.fixture(scope='function')
def regression_first_order_gradients_one_side_only_one_right_gradient(
        regression_first_order_gradients_left,
        regression_first_order_gradients_right):
    regression_first_order_gradients_one_side_only_one_right_gradient = (
        regression_first_order_gradients_left.tolist())
    for i in range(4):
        regression_first_order_gradients_one_side_only_one_right_gradient.insert(  # pylint: disable=line-too-long
            4 * (i + 1) + i,
            regression_first_order_gradients_right[4 * (i + 1) - 1])
    return regression_first_order_gradients_one_side_only_one_right_gradient


@pytest.fixture(scope='function')
def regression_first_order_gradients_feature_difference(
        regression_first_order_gradients):
    regression_first_order_gradients_feature_difference = np.copy(
        regression_first_order_gradients)
    regression_first_order_gradients_feature_difference[1:] = np.subtract(
        regression_first_order_gradients_feature_difference[1:],
        regression_first_order_gradients_feature_difference[0:-1])
    regression_first_order_gradients_feature_difference[
        0:-1:2] = regression_first_order_gradients[0:-1:2]
    return regression_first_order_gradients_feature_difference


@pytest.fixture(scope='function')
def regression_first_order_gradients_node_difference(
        regression_first_order_gradients):
    regression_first_order_gradients_node_difference = np.copy(
        regression_first_order_gradients)
    regression_first_order_gradients_node_difference[1:] = np.subtract(
        regression_first_order_gradients_node_difference[1:],
        regression_first_order_gradients_node_difference[0:-1])
    regression_first_order_gradients_node_difference[
        0:-1:4] = regression_first_order_gradients[0:-1:4]
    return regression_first_order_gradients_node_difference


# N.b. this vector should be the same shape as fixture
# regression_first_second_order_gradients. However, it cannot be a fixture
# as it has to be input indirectly to fixture gbdt_internal_algorithm_params,
# which cannot be done if weight_translate_vector is a fixture.
weight_translate_vector = np.ones(64, )


@pytest.fixture(scope='function')
def weighted_regression_first_second_order_gradients(
        regression_first_second_order_gradients):
    return np.multiply(weight_translate_vector,
                       regression_first_second_order_gradients)


@pytest.fixture
def gbdt_algorithm(four_features, new_event_loop):
    return FederatedGBDT(features=four_features)


@pytest.fixture(scope='function')
def gbdt_regression_model_hyper_params():
    return GBDTModelHyperParams()


@pytest.fixture(scope='session')
def gbdt_federated_dataset(gbdt_user_dataset):
    federated_dataset = MagicMock(spec=FederatedDatasetBase)
    seeds = itertools.count()
    federated_dataset.__iter__.return_value = ((gbdt_user_dataset, seed)
                                               for seed in seeds)
    federated_dataset.get_cohort.side_effect = lambda cohort_size: [
        (gbdt_user_dataset, next(seeds)) for _ in range(cohort_size)
    ]
    return federated_dataset


@pytest.fixture
def statistics_1():
    return MappedVectorStatistics({
        'questions':
        np.array([(0, -1.), (0, 0.), (1, -1.), (1, 0.)]),
        'first_order_grads_left':
        np.array([-2, -2, -1, -1]),
        'first_order_grads_right':
        np.array([0, 0, -1, -1]),
        'second_order_grads_left':
        np.array([2, 2, 1, 3]),
        'second_order_grads_right':
        np.array([2, 2, -3, -1])
    })


@pytest.fixture
def statistics_2():
    return MappedVectorStatistics({
        'questions':
        np.array([(0, -1.), (0, 0.), (1, -1.), (1, 0.)]),
        'first_order_grads_left':
        np.array([-3, -2, -1, 0]),
        'first_order_grads_right':
        np.array([0, 0, -1, -1]),
        'second_order_grads_left':
        np.array([]),
        'second_order_grads_right':
        np.array([])
    })


@pytest.fixture
def fail_statistics_1():
    return MappedVectorStatistics({'questions': np.array([])})


@pytest.fixture
def fail_statistics_2():
    return MappedVectorStatistics({
        'questions':
        np.array([('0', -1), ('0', 0), ('1', -1), ('1', 0)]),
        'first_order_grads_left':
        np.array([]),
        'first_order_grads_right':
        np.array([0, 0, -1, -1]),
        'second_order_grads_left':
        np.array([2, 2, 1, 3]),
        'second_order_grads_right':
        np.array([2, 2, 3, 1])
    })


@pytest.fixture(scope='function')
def gbdt_model_classifier_incomplete_leaf_nodes(num_features,
                                                tree_incomplete_2_layers,
                                                set_trees):
    model = GBDTModelClassifier(num_features=num_features, max_depth=2)
    return set_trees(model, [tree_incomplete_2_layers])


@pytest.fixture(scope='function')
def gbdt_model_classifier_leaf_node_to_split():
    model = GBDTModelClassifier(num_features=4, max_depth=1)
    model._nodes_to_split = [NodeRecord(None, [], True, None, True)]  # pylint: disable=protected-access
    return model


@pytest.fixture(scope='function')
def gbdt_classification_model_hyper_params():
    return GBDTClassificationModelHyperParams(evaluation_threshold=0.5)


@pytest.fixture
def train_population():
    return Population.TRAIN


@pytest.fixture
def val_population():
    return Population.VAL


class TestGBDTAlgorithmHyperParams:

    def test_gbdt_algorithm_hyper_params(self):
        public_params = {
            'cohort_size': 10,
            'val_cohort_size': 9,
            'num_trees': 3,
            'cohort_size_per_layer_modifier_fn': 'linear',
            'l2_regularization': 0.3,
            'leaf_nodes_reduction_factor': 6,
            'compute_second_order_gradients': True,
            'report_gradients_both_sides_split': False,
            'report_node_sum_gradients': True,
            'report_per_feature_result_difference': True,
            'report_per_node_result_difference': True
        }

        internal_params = {
            'gbdt_questions': [{
                "decisionPath": [],
                "splits": {
                    "56": [-17.25, -6.75, 3.74],
                    "89": [-11.87, 0.52, 12.9]
                }
            }],
            'weight_vector': [],
            'translate_vector': []
        }

        algorithm_params = GBDTAlgorithmHyperParams(**public_params)

        for key, val in public_params.items():
            assert getattr(algorithm_params, key) == val

        internal_algorithm_params = (
            _GBDTInternalAlgorithmHyperParams.from_GBDTAlgorithmHyperParams(
                algorithm_params, **internal_params))
        for key, val in {**public_params, **internal_params}.items():
            assert getattr(internal_algorithm_params, key) == val

    @pytest.mark.xfail(raises=(AssertionError), strict=True)
    @pytest.mark.parametrize('num_trees, l2_regularization', [(0, 1),
                                                              (1, -0.5)])
    def test_fail(self, num_trees, l2_regularization):
        GBDTAlgorithmHyperParams(cohort_size=1,
                                 val_cohort_size=2,
                                 num_trees=num_trees,
                                 l2_regularization=l2_regularization)

    def test_to_context_dict(self):
        d = {
            'cohort_size': 10,
            'val_cohort_size': 2,
            'num_trees': 4,
            'cohort_size_per_layer_modifier_fn': 'linear',
            'l2_regularization': 2,
            'leaf_nodes_reduction_factor': 2,
            'compute_second_order_gradients': False,
            'report_gradients_both_sides_split': True,
            'report_node_sum_gradients': False,
            'report_per_feature_result_difference': False,
            'report_per_node_result_difference': False
        }

        algorithm_params = GBDTAlgorithmHyperParams(**d)

        assert algorithm_params.to_context_dict() == d


@pytest.mark.skipif(not check_pfl_tree_installed(),
                    reason='pfl [tree] not installed')
class TestFederatedGBDT:

    @pytest.mark.parametrize(
        'gbdt_internal_algorithm_params, aggregate_statistics',
        [
            ({
                'compute_first_order_gradients': True,
                'compute_second_order_gradients': True
            }, lf('regression_first_second_order_gradients')),
            ({
                'compute_first_order_gradients': True,
                'compute_second_order_gradients': False
            }, lf('regression_first_order_gradients')),
            ({
                'compute_first_order_gradients': False,
                'compute_second_order_gradients': True
            }, lf('regression_second_order_gradients')),
            ({
                'compute_first_order_gradients': True,
                'compute_second_order_gradients': False,
                'report_gradients_both_sides_split': False,
                'report_node_sum_gradients': True,
            },
             lf(
                 'regression_first_order_gradients_one_side_only_node_sum')),
            (
                {
                    'compute_first_order_gradients': True,
                    'compute_second_order_gradients': False,
                    'report_gradients_both_sides_split': False,
                    'report_node_sum_gradients': False,
                },
                lf(
                    'regression_first_order_gradients_one_side_only_one_right_gradient'  # pylint: disable=line-too-long
                )),
            ({
                'compute_first_order_gradients': True,
                'compute_second_order_gradients': False,
                'report_per_feature_result_difference': True,
            },
             lf(
                 'regression_first_order_gradients_feature_difference')),
            ({
                'compute_first_order_gradients': True,
                'compute_second_order_gradients': False,
                'report_per_node_result_difference': True,
            },
             lf('regression_first_order_gradients_node_difference'))
        ],
        indirect=['gbdt_internal_algorithm_params'])
    def test_decode_training_statistics(
            self, regression_first_order_gradients_left,
            regression_first_order_gradients_right,
            regression_second_order_gradients_left,
            regression_second_order_gradients_right,
            gbdt_internal_algorithm_params, aggregate_statistics):
        output = FederatedGBDT._decode_training_statistics(  # pylint: disable=protected-access
            MappedVectorStatistics({STATISTIC_INFO_NAME:
                                    aggregate_statistics}),
            gbdt_internal_algorithm_params)

        questions = gbdt_internal_algorithm_params.gbdt_questions
        num_nodes = len(questions)

        assert len(output) == num_nodes

        for i, (node_output,
                expected_questions) in enumerate(zip(output, questions)):
            q = node_output['questions']
            expected_q = expected_questions['splits']
            expected_q = [(int(k), float(v)) for k in sorted(expected_q.keys())
                          for v in expected_q[k]]
            np.testing.assert_array_equal(q, expected_q)
            num_node_q = len(expected_q)
            i_start = i * num_node_q
            i_end = i_start + num_node_q

            if gbdt_internal_algorithm_params.compute_first_order_gradients:
                assert len(node_output['first_order_grads_left']) == num_node_q
                assert len(
                    node_output['first_order_grads_right']) == num_node_q
                np.testing.assert_array_equal(
                    node_output['first_order_grads_left'],
                    regression_first_order_gradients_left[i_start:i_end])
                np.testing.assert_array_equal(
                    node_output['first_order_grads_right'],
                    regression_first_order_gradients_right[i_start:i_end])

            if gbdt_internal_algorithm_params.compute_second_order_gradients:
                assert len(
                    node_output['second_order_grads_left']) == num_node_q
                assert len(
                    node_output['second_order_grads_right']) == num_node_q
                np.testing.assert_array_equal(
                    node_output['second_order_grads_left'],
                    regression_second_order_gradients_left[i_start:i_end])
                np.testing.assert_array_equal(
                    node_output['second_order_grads_right'],
                    regression_second_order_gradients_right[i_start:i_end])

    @pytest.mark.parametrize('statistics, l2_regularization, expected',
                             [(lf('statistics_1'), 1,
                               (1, -1, 1.5, 1.0, 0.5, 1.0)),
                              (lf('statistics_2'), 1,
                               (0, -1, 9.0, 3.0, 3.0, 0.0))])
    def test_postprocess_training_statistics(self, statistics,
                                             l2_regularization, expected):
        output = FederatedGBDT.postprocess_training_statistics(
            statistics, l2_regularization)

        assert len(output) == 6
        np.testing.assert_array_equal(output, expected)

    @pytest.mark.xfail(raises=(AssertionError, ValueError), strict=True)
    @pytest.mark.parametrize('statistics, l2_regularization',
                             [(lf('fail_statistics_1'), 1),
                              (lf('fail_statistics_2'), 1),
                              (lf('statistics_1'), -1)])
    def test_postprocess_training_statistics_fail(self, statistics,
                                                  l2_regularization):
        FederatedGBDT.postprocess_training_statistics(statistics,
                                                      l2_regularization)

    @pytest.mark.parametrize(
        'model, gbdt_internal_algorithm_params, expected_statistics, population, model_params',  # pylint: disable=line-too-long
        [
            (lf('gbdt_model_classifier_empty'), {
                'compute_first_order_gradients': True,
                'compute_second_order_gradients': True
            }, lf('classification_first_second_order_gradients'),
             lf('train_population'),
             lf('gbdt_classification_model_hyper_params')),
            (lf('gbdt_model_classifier_empty'), {
                'compute_first_order_gradients': True,
                'compute_second_order_gradients': True
            }, None, lf('val_population'),
             lf('gbdt_classification_model_hyper_params')),
            (lf('gbdt_model_regressor_empty'), {
                'compute_first_order_gradients': True,
                'compute_second_order_gradients': True
            }, lf('regression_first_second_order_gradients'),
             lf('train_population'),
             lf('gbdt_regression_model_hyper_params')),
            (lf('gbdt_model_regressor_empty'), {
                'compute_first_order_gradients': True,
                'compute_second_order_gradients': False
            }, lf('regression_first_order_gradients'),
             lf('train_population'),
             lf('gbdt_regression_model_hyper_params')),
            (lf('gbdt_model_regressor_empty'), {
                'compute_first_order_gradients': False,
                'compute_second_order_gradients': True
            }, lf('regression_second_order_gradients'),
             lf('train_population'),
             lf('gbdt_regression_model_hyper_params')),
            (lf('gbdt_model_regressor_empty'), {
                'compute_first_order_gradients': True,
                'compute_second_order_gradients': False,
                'report_gradients_both_sides_split': False,
                'report_node_sum_gradients': True,
            },
             lf(
                 'regression_first_order_gradients_one_side_only_node_sum'),
             lf('train_population'),
             lf('gbdt_regression_model_hyper_params')),
            (
                lf('gbdt_model_regressor_empty'),
                {
                    'compute_first_order_gradients': True,
                    'compute_second_order_gradients': False,
                    'report_gradients_both_sides_split': False,
                    'report_node_sum_gradients': False,
                },
                lf(
                    'regression_first_order_gradients_one_side_only_one_right_gradient'  # pylint: disable=line-too-long
                ),
                lf('train_population'),
                lf('gbdt_regression_model_hyper_params')),
            (lf('gbdt_model_regressor_empty'), {
                'compute_first_order_gradients': True,
                'compute_second_order_gradients': False,
                'report_per_feature_result_difference': True,
            },
             lf('regression_first_order_gradients_feature_difference'
                          ), lf('train_population'),
             lf('gbdt_regression_model_hyper_params')),
            (lf('gbdt_model_regressor_empty'), {
                'compute_first_order_gradients': True,
                'compute_second_order_gradients': False,
                'report_per_node_result_difference': True,
            },
             lf('regression_first_order_gradients_node_difference'),
             lf('train_population'),
             lf('gbdt_regression_model_hyper_params')),
            (lf('gbdt_model_regressor_empty'), {
                'compute_first_order_gradients': True,
                'compute_second_order_gradients': True,
                'weight_vector': weight_translate_vector
            },
             lf('weighted_regression_first_second_order_gradients'),
             lf('train_population'),
             lf('gbdt_regression_model_hyper_params')),
        ],
        indirect=['gbdt_internal_algorithm_params'])
    def test_simulate_one_user(self, gbdt_algorithm, model, gbdt_user_dataset,
                               gbdt_internal_algorithm_params, population,
                               expected_statistics, model_params):
        central_context = CentralContext(
            current_central_iteration=0,
            do_evaluation=True,
            cohort_size=5,
            population=population,
            model_train_params=model_params,
            model_eval_params=model_params,
            algorithm_params=gbdt_internal_algorithm_params,
            seed=1)

        statistics, metrics = gbdt_algorithm.simulate_one_user(
            model, gbdt_user_dataset, central_context)

        if expected_statistics is not None:
            assert statistics
            assert STATISTIC_INFO_NAME in statistics
            np.testing.assert_almost_equal(statistics[STATISTIC_INFO_NAME],
                                           expected_statistics,
                                           decimal=2)
        else:
            assert statistics is None
        assert metrics and isinstance(metrics, Metrics)

    @pytest.mark.parametrize(
        'model, statistics',
        [(lf('gbdt_model_regressor_empty'),
          lf('regression_first_second_order_gradients'))])
    def test_process_aggregated_statistics(self, gbdt_algorithm,
                                           gbdt_internal_algorithm_params,
                                           gbdt_regression_model_hyper_params,
                                           train_population, model,
                                           statistics):

        central_context = CentralContext(
            current_central_iteration=0,
            do_evaluation=True,
            cohort_size=5,
            population=train_population,
            model_train_params=gbdt_regression_model_hyper_params,
            model_eval_params=gbdt_regression_model_hyper_params,
            algorithm_params=gbdt_internal_algorithm_params,
            seed=1)

        assert len(model._gbdt.trees) == 0

        model, metrics = gbdt_algorithm.process_aggregated_statistics(
            central_context, Metrics(), model,
            MappedVectorStatistics({STATISTIC_INFO_NAME: statistics}))

        assert model
        assert len(model._gbdt.trees) == 1
        assert model._gbdt.trees[0].num_nodes() == 1
        assert model._gbdt.trees[0].max_depth() == 1
        assert model._gbdt.trees[0].feature == 0

        assert not metrics

    @pytest.mark.parametrize('gbdt_algorithm_params, model',
                             [({
                                 'num_trees': 1
                             }, lf('gbdt_model_regressor_empty')),
                              ({
                                  'num_trees': 5
                              }, lf('gbdt_model_regressor_empty'))],
                             indirect=['gbdt_algorithm_params'])
    def test_run(self, gbdt_algorithm, gbdt_algorithm_params, model, tmp_path,
                 gbdt_federated_dataset, gbdt_regression_model_hyper_params):

        training_federated_dataset = gbdt_federated_dataset
        val_federated_dataset = gbdt_federated_dataset

        backend = SimulatedBackend(training_data=training_federated_dataset,
                                   val_data=val_federated_dataset,
                                   postprocessors=None)

        model = gbdt_algorithm.run(gbdt_algorithm_params,
                                   backend,
                                   model,
                                   gbdt_regression_model_hyper_params,
                                   gbdt_regression_model_hyper_params,
                                   callbacks=[])

        # pylint: disable=protected-access
        assert model.current_tree == gbdt_algorithm_params.num_trees + 1

    @pytest.mark.parametrize('model', [
        lf('gbdt_model_regressor_empty'),
        lf('gbdt_model_regressor_empty')
    ])
    def test_get_next_central_contexts(self, gbdt_algorithm, model,
                                       gbdt_algorithm_params,
                                       gbdt_regression_model_hyper_params):
        iteration = 1
        (configs, output_model,
         metrics) = (gbdt_algorithm.get_next_central_contexts(
             model, iteration, gbdt_algorithm_params,
             gbdt_regression_model_hyper_params,
             gbdt_regression_model_hyper_params))

        if model.current_depth == 0:
            assert len(configs) == 2
            assert configs[0].population == Population.TRAIN
            assert configs[1].population == Population.VAL
        else:
            assert len(configs) == 1

        if model.current_depth == model.max_depth - 1:
            for config in configs:
                assert config.compute_second_order_gradients

        assert model == output_model
        assert metrics.to_simple_dict() == {}
