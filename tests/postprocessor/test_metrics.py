# Copyright © 2023-2024 Apple Inc.
from unittest.mock import MagicMock

import numpy as np
import pytest

from pfl.common_types import Population
from pfl.context import CentralContext, UserContext
from pfl.metrics import Histogram, MetricName, MetricNamePostfix, Metrics
from pfl.postprocessor.metrics import SummaryMetrics
from pfl.stats import MappedVectorStatistics


class TestSummaryMetrics:

    @pytest.fixture
    def metric_name(self):
        return MetricName('loss', Population.TRAIN)

    @pytest.fixture
    def postprocessor(self, metric_name):
        return SummaryMetrics(metric_name=metric_name,
                              min_bound=0.8,
                              max_bound=1.1,
                              num_bins=3,
                              quantiles=[0.4, 0.6],
                              frequency=2,
                              stddev=True)

    @pytest.fixture
    def central_context(self):
        return CentralContext(current_central_iteration=0,
                              do_evaluation=True,
                              cohort_size=1,
                              population=Population.TRAIN,
                              algorithm_params=MagicMock(),
                              model_train_params=MagicMock(),
                              model_eval_params=MagicMock())

    def test_postprocess_one_user(self, metric_name, postprocessor,
                                  check_equal_stats, check_equal_metrics):

        def check_postprocess(metric_val, expected_bin_counts, num_datapoints):
            stats = MappedVectorStatistics({'var1': np.arange(10)})
            metrics_before = Metrics([(metric_name, metric_val)])
            check_equal_metrics(metrics_before,
                                Metrics([(metric_name, metric_val)]))

            context = UserContext(num_datapoints=num_datapoints,
                                  seed=None,
                                  metrics=metrics_before)

            processed_stats, metrics_after = postprocessor.postprocess_one_user(
                stats=stats, user_context=context)

            check_equal_stats(stats, processed_stats)

            hist_name, hist_metric = next(iter(metrics_after))
            assert str(hist_name) == 'train population | loss | histogram'
            np.testing.assert_array_equal(hist_metric.bins,
                                          [0.8, 0.9, 1.0, 1.1])
            np.testing.assert_array_equal(hist_metric.bin_counts,
                                          expected_bin_counts)

        check_postprocess(1.0, [0, 0, 1], 1)
        check_postprocess(0.9, [0, 1, 0], 1)
        check_postprocess(1.11, [0, 0, 0], 1)

    def test_postprocess_one_user_skip(self, postprocessor, check_equal_stats,
                                       check_equal_metrics):
        stats = MappedVectorStatistics({'var1': np.arange(10)})

        metrics_before = Metrics()
        context = UserContext(num_datapoints=1,
                              seed=None,
                              metrics=metrics_before)

        processed_stats, metrics_after = postprocessor.postprocess_one_user(
            stats=stats, user_context=context)

        check_equal_stats(stats, processed_stats)
        check_equal_metrics(metrics_before, metrics_after)

    def test_postprocess_server(self, metric_name, postprocessor,
                                central_context, check_equal_stats):
        stats = MappedVectorStatistics({'var1': np.arange(10)})

        bins = [0.8, 0.9, 1.0, 1.1]
        counts = [4, 8, 12]
        histogram_metric = Histogram(counts, bins)
        metrics_before = Metrics([(MetricNamePostfix(metric_name, 'histogram'),
                                   histogram_metric)])

        processed_stats, metrics_after = postprocessor.postprocess_server(
            stats=stats,
            central_context=central_context,
            aggregate_metrics=metrics_before)

        check_equal_stats(stats, processed_stats)
        assert len(metrics_before) == 1

        assert len(metrics_after) == 3
        assert metrics_after.to_simple_dict() == {
            'Train population | loss | quantile=0.4': 0.87,
            'Train population | loss | quantile=0.6': 0.92,
            'Train population | loss | stddev': pytest.approx(0.0745356),
        }

    def test_postprocess_server_skip_metric_not_present(
            self, postprocessor, central_context, check_equal_stats,
            check_equal_metrics):
        stats = MappedVectorStatistics({'var1': np.arange(10)})

        metrics_before = Metrics([('useless_metric', 1.0)])

        processed_stats, metrics_after = postprocessor.postprocess_server(
            stats=stats,
            central_context=central_context,
            aggregate_metrics=metrics_before)

        check_equal_stats(stats, processed_stats)
        check_equal_metrics(metrics_after, Metrics())

    def test_postprocess_server_skip_wrong_iteration(self, postprocessor,
                                                     metric_name,
                                                     check_equal_stats,
                                                     check_equal_metrics):
        # skip due to frequency % current_central_iteration != 0
        central_context = CentralContext(current_central_iteration=1,
                                         do_evaluation=True,
                                         cohort_size=1,
                                         population=Population.TRAIN,
                                         algorithm_params=MagicMock(),
                                         model_train_params=MagicMock(),
                                         model_eval_params=MagicMock())

        stats = MappedVectorStatistics({'var1': np.arange(10)})

        bins = [0.8, 0.9, 1.0, 1.1]
        counts = [4, 8, 12]
        histogram_metric = Histogram(counts, bins)
        metrics_before = Metrics([(MetricNamePostfix(metric_name, 'histogram'),
                                   histogram_metric)])

        processed_stats, metrics_after = postprocessor.postprocess_server(
            stats=stats,
            central_context=central_context,
            aggregate_metrics=metrics_before)

        check_equal_stats(stats, processed_stats)
        check_equal_metrics(metrics_after, Metrics())
