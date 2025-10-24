# Copyright © 2023-2024 Apple Inc.

import contextlib
import os
from unittest.mock import PropertyMock, patch

import pytest
from pytest_lazy_fixtures import lf

from pfl.internal.ops import get_pytorch_major_version, get_tf_major_version

# Fixtures for distributed contexts to test.
# World size is overridden to 3 and global_rank is overridden to 1
# to be able to better test methods that depend on these properties.


@pytest.fixture
def numpy_legacy_dist():
    from pfl.internal.ops.numpy_ops import NotDistributedContext
    with patch(
            'pfl.internal.ops.distributed.NotDistributedContext.world_size',
            new_callable=PropertyMock,
            return_value=3), \
         patch(
            'pfl.internal.ops.distributed.NotDistributedContext'
            '.global_rank',
            new_callable=PropertyMock,
            return_value=1):
        yield NotDistributedContext()


@pytest.fixture
def numpy_tf_dist():
    with patch.dict(os.environ, {'PFL_NUMPY_DISTRIBUTE_METHOD': 'tensorflow'}), \
         patch('pfl.internal.ops.tensorflow_ops.TFDistributedContext.world_size',
               new_callable=PropertyMock, return_value=3), \
         patch('pfl.internal.ops.tensorflow_ops.TFDistributedContext.global_rank',
               new_callable=PropertyMock, return_value=1):
        from pfl.internal.ops.numpy_ops import _create_tf_based_numpy_distributed_context
        yield _create_tf_based_numpy_distributed_context()


@pytest.fixture
def numpy_pytorch_dist():
    with patch.dict(os.environ, {'PFL_NUMPY_DISTRIBUTE_METHOD': 'pytorch'}), \
         patch('pfl.internal.ops.pytorch_ops.PyTorchDistributedContext.world_size',
               new_callable=PropertyMock, return_value=3), \
         patch('pfl.internal.ops.pytorch_ops.PyTorchDistributedContext.global_rank',
               new_callable=PropertyMock, return_value=1):
        from pfl.internal.ops.numpy_ops import _create_pytorch_based_numpy_distributed_context
        yield _create_pytorch_based_numpy_distributed_context()


@pytest.fixture
def pytorch_legacy_dist():
    from pfl.internal.ops.pytorch_ops import PyTorchDistributedContext

    # Mock these properties directly instead to avoid starting PyTorch server.
    with patch(
            'pfl.internal.ops.pytorch_ops.PyTorchDistributedContext'
            '.world_size',
            new_callable=PropertyMock,
            return_value=3), \
         patch(
            'pfl.internal.ops.pytorch_ops.PyTorchDistributedContext'
            '.global_rank',
            new_callable=PropertyMock,
            return_value=1):
        yield PyTorchDistributedContext()


@pytest.fixture
def pytorch_native_dist():
    with patch('pfl.internal.ops.pytorch_ops.PyTorchDistributedContext.world_size',
               new_callable=PropertyMock, return_value=3), \
         patch('pfl.internal.ops.pytorch_ops.PyTorchDistributedContext.global_rank',
               new_callable=PropertyMock, return_value=1):
        from pfl.internal.ops.pytorch_ops import PyTorchDistributedContext
        yield PyTorchDistributedContext()


@pytest.fixture
def tensorflow_legacy_dist():
    from pfl.internal.ops.tensorflow_ops import TFDistributedContext

    # Mock these properties directly instead to avoid starting TF server.
    with patch(
            'pfl.internal.ops.tensorflow_ops.TFDistributedContext'
            '.world_size',
            new_callable=PropertyMock,
            return_value=3), \
         patch(
            'pfl.internal.ops.tensorflow_ops.TFDistributedContext'
            '.global_rank',
            new_callable=PropertyMock,
            return_value=1):
        yield TFDistributedContext()


@pytest.fixture
def tensorflow_native_dist():
    with patch('pfl.internal.ops.tensorflow_ops.TFDistributedContext.world_size',
               new_callable=PropertyMock, return_value=3), \
         patch('pfl.internal.ops.tensorflow_ops.TFDistributedContext.global_rank',
               new_callable=PropertyMock, return_value=1):
        from pfl.internal.ops.tensorflow_ops import TFDistributedContext
        yield TFDistributedContext()


pytorch_mark = pytest.mark.skipif(not get_pytorch_major_version(),
                                  reason='PyTorch not installed')

tf_mark = pytest.mark.skipif(get_tf_major_version() != 2, reason='tf!=2')


@pytest.mark.parametrize('distributed,ops_setup', [
    pytest.param(
        lf('numpy_legacy_dist'), lf('numpy_ops_setup'), id='numpy_legacy'),
    pytest.param(lf('numpy_tf_dist'),
                 lf('numpy_ops_setup'),
                 id='numpy_tf_dist',
                 marks=[tf_mark]),
    pytest.param(lf('numpy_pytorch_dist'),
                 lf('numpy_ops_setup'),
                 id='numpy_pytorch_dist',
                 marks=[pytorch_mark]),
    pytest.param(lf('pytorch_legacy_dist'),
                 lf('pytorch_ops_setup'),
                 id='pytorch_legacy',
                 marks=[pytorch_mark]),
    pytest.param(lf('pytorch_native_dist'),
                 lf('pytorch_ops_setup'),
                 id='pytorch_native',
                 marks=[pytorch_mark]),
    pytest.param(lf('tensorflow_legacy_dist'),
                 lf('tensorflow_ops_setup'),
                 id='tensorflow_legacy',
                 marks=[tf_mark]),
    pytest.param(lf('tensorflow_native_dist'),
                 lf('tensorflow_ops_setup'),
                 id='tensorflow_native',
                 marks=[tf_mark]),
])
class TestDistributedContext:

    def test_properties(self, distributed, ops_setup):
        assert distributed.local_rank == 0
        assert distributed.global_rank == 1
        assert distributed.world_size == 3

    def test_all_reduce_single_device(self, distributed, ops_setup, numpy_vars,
                                      check_equal_tensors):
        # Patch for purpose of PyTorchDistributedContext
        distributed._world_size = 1  # pylint: disable=protected-access

        for average in [True, False]:
            reduced_tensors = ops_setup.ops.distributed.all_reduce(
                ops_setup.ops_variables, average=average)
            # Should be the identity since there is only one worker.
            check_equal_tensors(numpy_vars, reduced_tensors, ops_setup)

    def test_distribute_range(self, distributed, ops_setup):
        assert distributed.distribute_range(10) == range(3, 6)

    def test_distribute_value(self, distributed, ops_setup):
        assert distributed.distribute_value(10) == 3
