# Copyright © 2023-2024 Apple Inc.

import numpy as np
import pytest

from pfl.internal.ops.common_ops import check_mlx_installed

if check_mlx_installed():
    import mlx.core as mx

    from pfl.internal.ops import mlx_ops


@pytest.mark.skipif(not check_mlx_installed(), reason='MLXS not installed')
class TestMLXOps:

    def test_flatten_reshape(self):

        tensors = [mx.zeros((2, 2)), mx.ones((3, ), dtype=mx.int64)]
        vector, shapes, dtypes = mlx_ops.flatten(tensors)
        np.testing.assert_array_equal(np.array(vector), [0, 0, 0, 0, 1, 1, 1])
        assert shapes == [(2, 2), (3, )]
        assert dtypes == [mx.float32, mx.int64]

        reshaped = mlx_ops.reshape(vector, shapes, dtypes)

        np.testing.assert_array_equal(np.array(reshaped[0]), np.zeros((2, 2)))
        np.testing.assert_array_equal(np.array(reshaped[1]), np.ones((3, )))
