# Copyright © 2023-2024 Apple Inc.
'''
Test simulated_federated_averaging.py.
'''
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pfl.aggregate.base import Aggregator, SumAggregator, get_num_params_weight_name, get_total_weight_name
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


class RecordingPostprocessor(Postprocessor):
    """
    Records every hook call into a log shared across postprocessors, and
    emits one metric, named after itself, from each metrics-only hook.
    """

    def __init__(self, name, log):
        self.name = name
        self.log = log
        self.metrics_user_contexts = []
        self.stats_user_contexts = []
        self.metrics_aggregate_metrics = []
        self.stats_aggregate_metrics = []

    def user_metric_name(self):
        return StringMetricName(f'{self.name}.user_metrics')

    def server_metric_name(self):
        return StringMetricName(f'{self.name}.server_metrics')

    def postprocess_one_user_metrics(self, *, user_context):
        self.log.append((self.name, 'postprocess_one_user_metrics'))
        self.metrics_user_contexts.append(user_context)
        return Metrics([(self.user_metric_name(), user_context.num_datapoints)
                        ])

    def postprocess_one_user(self, *, stats, user_context):
        self.log.append((self.name, 'postprocess_one_user'))
        self.stats_user_contexts.append(user_context)
        return stats, Metrics()

    def postprocess_server_metrics(self, *, central_context,
                                   aggregate_metrics):
        self.log.append((self.name, 'postprocess_server_metrics'))
        self.metrics_aggregate_metrics.append(aggregate_metrics)
        return Metrics([(self.server_metric_name(),
                         central_context.cohort_size)])

    def postprocess_server(self, *, stats, central_context, aggregate_metrics):
        self.log.append((self.name, 'postprocess_server'))
        self.stats_aggregate_metrics.append(aggregate_metrics)
        return stats, Metrics()


class TestMetricsOnlyHooks:
    """
    `postprocess_one_user_metrics` and `postprocess_server_metrics` run
    whether or not there are statistics, so that a postprocessor which only
    touches metrics also runs on an evaluation round.
    """

    @pytest.fixture
    def log(self):
        return []

    @pytest.fixture
    def recorders(self, log):
        return (RecordingPostprocessor('a',
                                       log), RecordingPostprocessor('b', log))

    @pytest.fixture
    def backend(self, new_event_loop, federated_dataset, use_ops, recorders):
        return SimulatedBackend(training_data=federated_dataset,
                                val_data=federated_dataset,
                                postprocessors=list(recorders))

    @pytest.mark.parametrize('central_context', ({
        'cohort_size': 3,
        'population': Population.TRAIN
    }, {
        'cohort_size': 3,
        'population': Population.VAL
    }),
                             indirect=True)
    def test_one_user_metrics_hook_runs_for_every_user(self, backend,
                                                       recorders,
                                                       mock_algorithm,
                                                       mock_model,
                                                       central_context):
        _, metrics = backend.gather_results(model=mock_model,
                                            training_algorithm=mock_algorithm,
                                            central_context=central_context)

        for p in recorders:
            assert len(p.metrics_user_contexts) == 3
            for user_context in p.metrics_user_contexts:
                assert user_context.num_datapoints == 2
                assert user_context.user_id == 'user1'
                # pytype: disable=wrong-arg-count
                assert MetricName(
                    'simulate_one_user_loss',
                    central_context.population) in user_context.metrics
                # pytype: enable=wrong-arg-count
            # Summed over the cohort like any other user metric.
            assert metrics[p.user_metric_name()] == 2 * 3
        # Each user got its own seed.
        assert len({c.seed for c in recorders[0].metrics_user_contexts}) == 3

    @pytest.mark.parametrize('central_context', ({
        'cohort_size': 3,
        'population': Population.TRAIN
    }, {
        'cohort_size': 3,
        'population': Population.VAL
    }),
                             indirect=True)
    def test_server_metrics_hook_runs_once_on_the_aggregate(
            self, backend, recorders, mock_algorithm, mock_model,
            central_context):
        _, metrics = backend.gather_results(model=mock_model,
                                            training_algorithm=mock_algorithm,
                                            central_context=central_context)

        for p in recorders:
            assert len(p.metrics_aggregate_metrics) == 1
            aggregate_metrics = p.metrics_aggregate_metrics[0]
            # pytype: disable=wrong-arg-count
            assert aggregate_metrics[MetricName(
                'number of devices', central_context.population)] == 3
            # pytype: enable=wrong-arg-count
            for q in recorders:
                assert aggregate_metrics[q.user_metric_name()] == 2 * 3
            assert metrics[p.server_metric_name()] == 3

    @pytest.mark.parametrize('central_context', ({
        'cohort_size': 3,
        'population': Population.VAL
    }, ),
                             indirect=True)
    def test_eval_round_runs_only_the_metrics_hooks(self, backend, log,
                                                    mock_algorithm, mock_model,
                                                    central_context):
        """
        The reason the hooks exist: with no statistics, the statistics hooks
        are skipped, and the metrics ones are not.
        """
        model_update, metrics = backend.gather_results(
            model=mock_model,
            training_algorithm=mock_algorithm,
            central_context=central_context)

        assert model_update is None
        assert {hook
                for _, hook in log} == {
                    'postprocess_one_user_metrics',
                    'postprocess_server_metrics'
                }
        assert get_num_params_weight_name() not in metrics

    @pytest.mark.parametrize('central_context', ({
        'cohort_size': 1,
        'population': Population.TRAIN
    }, ),
                             indirect=True)
    def test_call_order(self, backend, log, mock_algorithm, mock_model,
                        central_context):
        backend.gather_results(model=mock_model,
                               training_algorithm=mock_algorithm,
                               central_context=central_context)

        assert log == [
            ('a', 'postprocess_one_user_metrics'),
            ('b', 'postprocess_one_user_metrics'),
            ('a', 'postprocess_one_user'),
            ('b', 'postprocess_one_user'),
            ('b', 'postprocess_server_metrics'),
            ('a', 'postprocess_server_metrics'),
            ('b', 'postprocess_server'),
            ('a', 'postprocess_server'),
        ]

    @pytest.mark.parametrize('central_context', ({
        'cohort_size': 2,
        'population': Population.TRAIN
    }, ),
                             indirect=True)
    def test_user_metrics_are_threaded_down_the_chain(self, backend, recorders,
                                                      mock_algorithm,
                                                      mock_model,
                                                      central_context):
        backend.gather_results(model=mock_model,
                               training_algorithm=mock_algorithm,
                               central_context=central_context)
        a, b = recorders

        for user_context in a.metrics_user_contexts:
            assert b.user_metric_name() not in user_context.metrics
        for user_context in b.metrics_user_contexts:
            assert user_context.metrics[a.user_metric_name()] == 2
        # Every metrics hook has run before the first statistics hook.
        for p in recorders:
            for user_context in p.stats_user_contexts:
                assert user_context.metrics[a.user_metric_name()] == 2
                assert user_context.metrics[b.user_metric_name()] == 2

    @pytest.mark.parametrize('central_context', ({
        'cohort_size': 2,
        'population': Population.TRAIN
    }, ),
                             indirect=True)
    def test_server_metrics_are_threaded_down_the_chain(
            self, backend, recorders, mock_algorithm, mock_model,
            central_context):
        backend.gather_results(model=mock_model,
                               training_algorithm=mock_algorithm,
                               central_context=central_context)
        a, b = recorders

        # Reversed on the server, so `b` runs first.
        assert a.server_metric_name() not in b.metrics_aggregate_metrics[0]
        assert a.metrics_aggregate_metrics[0][b.server_metric_name()] == 2
        for p in recorders:
            assert p.stats_aggregate_metrics[0][a.server_metric_name()] == 2
            assert p.stats_aggregate_metrics[0][b.server_metric_name()] == 2

    @pytest.mark.parametrize('central_context', ({
        'cohort_size': 2,
        'population': Population.TRAIN
    }, {
        'cohort_size': 2,
        'population': Population.VAL
    }),
                             indirect=True)
    def test_server_metrics_hook_sees_metrics_reduced_across_workers(
            self, new_event_loop, federated_dataset, recorders, mock_ops,
            mock_algorithm, mock_model, central_context):
        """
        The server hook runs after the all-reduce, on both the statistics and
        the metrics-only reduce paths, so it noises the global aggregate
        rather than each worker's share.
        """
        mock_ops.distributed.world_size = 2
        reduced = StringMetricName('reduced')
        aggregator = MagicMock(spec=Aggregator)
        aggregator.accumulate.side_effect = SumAggregator().accumulate
        aggregator.worker_reduce.side_effect = (
            lambda aggregated_worker_stats, central_context,
            aggregated_worker_metrics:
            (aggregated_worker_stats, aggregated_worker_metrics | Metrics([(
                reduced, 1)])))
        aggregator.worker_reduce_metrics_only.side_effect = (
            lambda central_context, aggregated_worker_metrics:
            aggregated_worker_metrics | Metrics([(reduced, 1)]))
        backend = SimulatedBackend(training_data=federated_dataset,
                                   val_data=federated_dataset,
                                   postprocessors=list(recorders),
                                   aggregator=aggregator)

        backend.gather_results(model=mock_model,
                               training_algorithm=mock_algorithm,
                               central_context=central_context)

        for p in recorders:
            assert reduced in p.metrics_aggregate_metrics[0]
