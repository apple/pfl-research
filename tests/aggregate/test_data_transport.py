# Copyright © 2023-2024 Apple Inc.
import typing
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from pfl.aggregate.data_transport import BFloat16DataTransport, Float32DataTransport
from pfl.internal.ops.common_ops import get_pytorch_major_version, get_tf_major_version
from pfl.stats import MappedVectorStatistics


class TestFloat32DataTransport:

    def test_simulate_transport(self):
        data_transport = Float32DataTransport()

        model_update = MappedVectorStatistics({'var1': np.arange(10)})
        transported_model_update, _ = data_transport.postprocess_one_user(
            stats=model_update, user_context=MagicMock())

        transported_model_update = typing.cast(MappedVectorStatistics,
                                               transported_model_update)
        assert len(transported_model_update) == 1
        np.testing.assert_array_equal(transported_model_update['var1'],
                                      model_update['var1'])


# These fixtures sets the internal framework module.
@pytest.mark.parametrize('ops_module', [
    pytest.param(lazy_fixture('tensorflow_ops'),
                 marks=[
                     pytest.mark.skipif(get_tf_major_version() < 2,
                                        reason='not tf>=2')
                 ]),
    pytest.param(lazy_fixture('pytorch_ops'),
                 marks=[
                     pytest.mark.skipif(not get_pytorch_major_version(),
                                        reason='PyTorch not installed')
                 ]),
])
class TestBFloat16DataTransport:

    def test_simulate_transport(self, ops_module):

        data_transport = BFloat16DataTransport()

        # This value is can be represented in bfloat16 and float32, but not
        # float16
        bfloatable_value = 2.56541e38
        # This value can be represented in float32, but not bfloat16 or float16
        non_bfloatable_value = 3.0156252
        non_bfloatable_rounded_value = 3.015625
        model_update = MappedVectorStatistics({
            'var1':
            np.array([bfloatable_value, non_bfloatable_value],
                     dtype=np.float32)
        })
        transported_model_update, _ = data_transport.postprocess_one_user(
            stats=model_update, user_context=MagicMock())
        transported_model_update = typing.cast(MappedVectorStatistics,
                                               transported_model_update)

        assert len(transported_model_update) == 1
        expected_array = np.array(
            [bfloatable_value,
             non_bfloatable_rounded_value]).astype(np.float32)
        np.testing.assert_array_equal(transported_model_update['var1'],
                                      expected_array)
