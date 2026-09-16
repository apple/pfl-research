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
        # 40% of 24 users is the 9.6th, which sits in bin 1 (users 5-12), 70% of
        # the way through it: 0.9 + 0.7 * 0.1. Likewise the 14.4th for 0.6.
        assert metrics_after.to_simple_dict() == {
            'Train population | loss | quantile=0.4': pytest.approx(0.97),
            'Train population | loss | quantile=0.6': pytest.approx(1.02),
            'Train population | loss | stddev': pytest.approx(0.0745356),
        }

    @pytest.mark.parametrize('quantile', [0.01, 0.1, 0.25])
    def test_quantile_inside_the_first_bin_stays_in_range(
            self, postprocessor, quantile):
        # A right-skewed distribution puts its low quantiles inside bin 0, where
        # the bin below and the mass below both have to come from outside `cdf`.
        # Reading them as `cdf[-1]` and `bins[-1]` instead wrapped to the far end
        # of the histogram and extrapolated backwards past `min_bound`, so a
        # gradient-norm summary reported negative norms.
        bins = np.linspace(0.0, 1.5, 3001)
        counts = np.zeros(3000)
        counts[0] = 40
        counts[40] = 34
        counts[400] = 30
        counts[900] = 24

        value = postprocessor._quantile(counts, bins, quantile)
        assert bins[0] <= value <= bins[1]

    def test_quantiles_agree_with_numpy_on_the_underlying_sample(
            self, postprocessor):
        # The independent check that pins both halves of the interpolation: which
        # bin the target falls in, and where inside that bin it lands. Bins are
        # fine enough that a histogram quantile and a sample quantile may differ
        # by at most one bin width.
        rng = np.random.default_rng(0)
        sample = rng.lognormal(mean=0.0, sigma=0.5, size=200000)
        bins = np.linspace(0.0, 20.0, 20001)
        counts, _ = np.histogram(sample, bins=bins)

        for quantile in (0.01, 0.1, 0.5, 0.9, 0.99):
            assert postprocessor._quantile(counts, bins,
                                           quantile) == pytest.approx(
                                               np.quantile(sample, quantile),
                                               abs=bins[1] - bins[0])

    def test_a_missing_source_metric_warns_once(self, postprocessor, caplog):
        """
        A missing source produces no output at all, so a whole run completes
        looking healthy and carrying nothing. Once, not once per user per round.
        """
        stats = MappedVectorStatistics({'var1': np.arange(10)})
        context = UserContext(num_datapoints=1,
                              seed=None,
                              metrics=Metrics([('other', 1.0)]))

        with caplog.at_level('WARNING', logger='pfl.postprocessor.metrics'):
            for _ in range(3):
                _, metrics = postprocessor.postprocess_one_user(
                    stats=stats, user_context=context)

        assert len(metrics) == 0
        assert len(caplog.records) == 1
        assert 'no metric' in caplog.records[0].getMessage()

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
