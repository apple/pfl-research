# Copyright © 2023-2024 Apple Inc.

import contextlib
import os
from unittest.mock import PropertyMock, patch

import pytest
from pytest_lazyfixture import lazy_fixture

from pfl.internal.ops import get_pytorch_major_version, get_tf_major_version
from pfl.internal.ops.distributed import horovod_is_active

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
def mock_hvd_properties():
    patches = []
    if get_tf_major_version():
        patches.append(patch('horovod.tensorflow.size', return_value=3))
        patches.append(patch('horovod.tensorflow.rank', return_value=1))
    if get_pytorch_major_version():
        patches.append(patch('horovod.torch.size', return_value=3))
        patches.append(patch('horovod.torch.rank', return_value=1))
    with contextlib.ExitStack() as stack:
        [stack.enter_context(p) for p in patches]
        yield


@pytest.fixture
def numpy_tf_dist(mock_hvd_properties):
    from pfl.internal.ops.numpy_ops import NumpyHorovodDistributedContext
    return NumpyHorovodDistributedContext('tensorflow')


@pytest.fixture
def numpy_pytorch_dist(mock_hvd_properties):
    from pfl.internal.ops.numpy_ops import NumpyHorovodDistributedContext
    return NumpyHorovodDistributedContext('torch')


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
def pytorch_hvd_dist(mock_hvd_properties):
    from pfl.internal.ops.pytorch_ops import PyTorchHorovodDistributedContext
    return PyTorchHorovodDistributedContext()


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
def tensorflow_hvd_dist(mock_hvd_properties):
    from pfl.internal.ops.tensorflow_ops import TFHorovodDistributedContext
    yield TFHorovodDistributedContext()


@pytest.mark.horovod
def test_horovod_is_active():
    with patch.object(os, 'environ', {}):
        assert not horovod_is_active()
    with patch.dict(os.environ, {
            'OMPI_COMMAND': 'python',
            'OMPI_ARGV': 'main.py'
    }):
        assert horovod_is_active()
    with patch.dict(os.environ, {
            'HOROVOD_HOSTNAME': 'localhost',
            'HOROVOD_RANK': '0',
            'HOROVOD_SIZE': '1'
    }):
        assert horovod_is_active()


pytorch_mark = pytest.mark.skipif(not get_pytorch_major_version(),
                                  reason='PyTorch not installed')

tf_mark = pytest.mark.skipif(get_tf_major_version() != 2, reason='tf!=2')


@pytest.mark.parametrize('distributed,ops_setup', [
    pytest.param(lazy_fixture('numpy_legacy_dist'),
                 lazy_fixture('numpy_ops_setup'),
                 id='numpy_legacy'),
    pytest.param(lazy_fixture('numpy_tf_dist'),
                 lazy_fixture('numpy_ops_setup'),
                 id='numpy_hvd_tf',
                 marks=[tf_mark, pytest.mark.horovod]),
    pytest.param(lazy_fixture('numpy_pytorch_dist'),
                 lazy_fixture('numpy_ops_setup'),
                 id='numpy_hvd_pytorch',
                 marks=[pytorch_mark, pytest.mark.horovod]),
    pytest.param(lazy_fixture('pytorch_legacy_dist'),
                 lazy_fixture('pytorch_ops_setup'),
                 id='pytorch_legacy',
                 marks=[pytorch_mark]),
    pytest.param(lazy_fixture('pytorch_hvd_dist'),
                 lazy_fixture('pytorch_ops_setup'),
                 id='pytorch_hvd',
                 marks=[pytorch_mark, pytest.mark.horovod]),
    pytest.param(lazy_fixture('tensorflow_legacy_dist'),
                 lazy_fixture('tensorflow_ops_setup'),
                 id='tensorflow_legacy',
                 marks=[tf_mark]),
    pytest.param(lazy_fixture('tensorflow_hvd_dist'),
                 lazy_fixture('tensorflow_ops_setup'),
                 id='tensorflow_hvd',
                 marks=[tf_mark, pytest.mark.horovod]),
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
