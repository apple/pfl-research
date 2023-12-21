# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
from unittest.mock import MagicMock, patch

import pytest

from pfl.common_types import Population
from pfl.internal.ops.selector import _internal_reset_framework_module, set_framework_module
from pfl.metrics import Metrics, StringMetricName
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


@pytest.fixture(autouse=True)
def disable_sleep():
    with patch('time.sleep', autospec=True):
        yield


@pytest.fixture()
def simple_postprocessor(request):
    p = MagicMock(spec=Postprocessor)
    p.postprocess_one_user.side_effect = lambda stats, user_context: (
        stats,
        Metrics([(StringMetricName('postprocess_user.num_datapoints'),
                  user_context.num_datapoints)]))

    p.postprocess_server.side_effect = (
        lambda stats, central_context, aggregate_metrics:
        (stats,
         Metrics([(StringMetricName('postprocess_server.cohort_size'),
                   central_context.cohort_size)])))

    p.postprocess_server_live.side_effect = (
        lambda stats, central_context, aggregate_metrics:
        (stats,
         Metrics([(StringMetricName('postprocess_server_live.cohort_size'),
                   central_context.cohort_size)])))

    return p


@pytest.fixture(scope='module', params=[Population.TRAIN, Population.VAL])
def population(request):
    return request.param
