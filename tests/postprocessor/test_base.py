# Copyright © 2023-2024 Apple Inc.
import pytest

from pfl.aggregate.data_transport import Float32DataTransport
from pfl.metrics import Metrics, StringMetricName
from pfl.postprocessor.base import Postprocessor
from pfl.postprocessor.metrics import SummaryMetrics


class TestMetricsOnlyHooks:
    """
    The defaults of the metrics-only hooks, which every existing
    postprocessor relies on to stay unaffected by them.
    """

    @pytest.mark.parametrize('postprocessor', [
        Postprocessor(),
        Float32DataTransport(),
        SummaryMetrics(StringMetricName('loss'), [0.5], 0., 1.),
    ])
    def test_defaults_add_no_metrics(self, postprocessor, user_context,
                                     central_context):
        user_metrics = postprocessor.postprocess_one_user_metrics(
            user_context=user_context)
        server_metrics = postprocessor.postprocess_server_metrics(
            central_context=central_context, aggregate_metrics=Metrics())

        assert isinstance(user_metrics, Metrics)
        assert len(user_metrics) == 0
        assert isinstance(server_metrics, Metrics)
        assert len(server_metrics) == 0
