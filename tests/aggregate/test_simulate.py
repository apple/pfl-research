# Copyright © 2023-2024 Apple Inc.
'''
Test simulated_federated_averaging.py.
'''
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pfl.aggregate.base import get_total_weight_name
from pfl.aggregate.simulate import SimulatedBackend
from pfl.common_types import Population
from pfl.data.federated_dataset import FederatedDatasetBase
from pfl.internal.ops.selector import _internal_reset_framework_module, set_framework_module
from pfl.metrics import MetricName, Metrics, StringMetricName, Weighted
from pfl.postprocessor.base import Postprocessor


@pytest.fixture(scope='function')
def use_ops(request, mock_ops):
    _internal_reset_framework_module()
    if hasattr(request, 'param') and request.param:
        # If the parameter sent to the use_ops fixture is `True`,
        # set a mock ops module.
        # This means that reducing across workers in
        # ``SimulatedBackend`` will be tested.
        set_framework_module(mock_ops)
    yield
    _internal_reset_framework_module()


@pytest.fixture(scope='function')
def simulated_backend(request, new_event_loop,
                      federated_dataset: FederatedDatasetBase,
                      simple_postprocessor, use_ops):

    # The mock federated dataset is not stateful, so we can use it multiple
    # times here.
    params = {
        'training_data': federated_dataset,
        'val_data': federated_dataset,
        'postprocessors': [Postprocessor(), simple_postprocessor]
    }
    if hasattr(request, 'param'):
        # update any parametrized optional arguments.
        params.update(**request.param)
    backend = SimulatedBackend(**params)  # type: ignore[arg-type]
    return backend


class TestSimulatedBackend:

    @pytest.mark.parametrize('use_ops', (False, True), indirect=True)
    @pytest.mark.parametrize('central_context', ({
        'cohort_size': 1,
        'population': Population.TRAIN
    }, {
        'cohort_size': 3,
        'population': Population.TRAIN
    }, {
        'cohort_size': 1,
        'population': Population.VAL
    }, {
        'cohort_size': 3,
        'population': Population.VAL
    }),
                             indirect=True)
    def test_gather_results(self, simulated_backend, mock_algorithm,
                            mock_model, central_context):
        population = central_context.population
        cohort_size = central_context.cohort_size

        model_update, metrics = simulated_backend.gather_results(
            model=mock_model,
            training_algorithm=mock_algorithm,
            central_context=central_context)

        # pytype: disable=wrong-arg-count
        assert metrics[MetricName('number of devices',
                                  population)] == cohort_size
        assert metrics[MetricName('number of data points',
                                  population)] == Weighted(
                                      2 * cohort_size, cohort_size)
        assert metrics[MetricName('number of data points',
                                  population)].overall_value == 2
        # pytype: enable=wrong-arg-count
        assert mock_algorithm.simulate_one_user.call_count == cohort_size

        if population == Population.TRAIN:
            assert len(model_update) == 1
            np.testing.assert_array_equal(model_update['var1'],
                                          np.ones((2, 3)) * cohort_size)

            assert metrics[get_total_weight_name(
                Population.TRAIN)] == Weighted(cohort_size, cohort_size)

            assert mock_algorithm.simulate_one_user.call_count == cohort_size

            # Expect metrics from postprocessor
            assert metrics[StringMetricName(
                'postprocess_user.num_datapoints')] == 2 * cohort_size
            assert metrics[StringMetricName(
                'postprocess_server.cohort_size')] == cohort_size
        else:
            assert model_update is None

    @pytest.mark.parametrize('central_context', ({
        'cohort_size': 110
    }, ),
                             indirect=True)
    @pytest.mark.parametrize('simulated_backend',
                             ({
                                 'max_overshoot_fraction': 0.1
                             }, ),
                             indirect=True)
    def test_overshoot(self, simulated_backend, central_context,
                       mock_algorithm, mock_model):

        with patch('numpy.random.uniform', lambda a, b: b):
            _, metrics = simulated_backend.gather_results(
                model=mock_model,
                training_algorithm=mock_algorithm,
                central_context=central_context)

        assert metrics[MetricName('number of devices',
                                  Population.TRAIN)] == 121  # pytype: disable=wrong-arg-count # pylint: disable=line-too-long


class TestPostprocessorChaining:
    """
    A postprocessor summarising what an earlier one emitted, which it can only
    do if `SimulatedBackend` re-binds the frozen `UserContext` as it accumulates
    metrics down the chain.
    """

    @pytest.fixture
    def summarising_postprocessor(self):
        """A postprocessor that reports whatever an earlier one emitted."""
        p = MagicMock(spec=Postprocessor)

        def postprocess_one_user(stats, user_context):
            source = StringMetricName('postprocess_user.num_datapoints')
            if source not in user_context.metrics:
                return stats, Metrics()
            return stats, Metrics([(StringMetricName('summary.seen'),
                                    user_context.metrics[source])])

        p.postprocess_one_user.side_effect = postprocess_one_user
        p.postprocess_server.side_effect = (
            lambda stats, central_context, aggregate_metrics:
            (stats, Metrics()))
        return p

    @pytest.mark.parametrize('central_context', ({
        'cohort_size': 2,
        'population': Population.TRAIN
    }, ),
                             indirect=True)
    def test_a_later_postprocessor_sees_an_earlier_ones_metric(
            self, new_event_loop, federated_dataset, simple_postprocessor,
            summarising_postprocessor, use_ops, mock_algorithm, mock_model,
            central_context):
        backend = SimulatedBackend(
            training_data=federated_dataset,
            val_data=federated_dataset,
            postprocessors=[simple_postprocessor, summarising_postprocessor])

        _, metrics = backend.gather_results(model=mock_model,
                                            training_algorithm=mock_algorithm,
                                            central_context=central_context)

        assert metrics[StringMetricName('summary.seen')] == 4

    @pytest.mark.parametrize('central_context', ({
        'cohort_size': 2,
        'population': Population.TRAIN
    }, ),
                             indirect=True)
    def test_order_matters_so_a_summariser_placed_first_sees_nothing(
            self, new_event_loop, federated_dataset, simple_postprocessor,
            summarising_postprocessor, use_ops, mock_algorithm, mock_model,
            central_context):
        """The negative control: the metric is threaded forwards, not backwards."""
        backend = SimulatedBackend(
            training_data=federated_dataset,
            val_data=federated_dataset,
            postprocessors=[summarising_postprocessor, simple_postprocessor])

        _, metrics = backend.gather_results(model=mock_model,
                                            training_algorithm=mock_algorithm,
                                            central_context=central_context)

        assert StringMetricName('summary.seen') not in metrics
