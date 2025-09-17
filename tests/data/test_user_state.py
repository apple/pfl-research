# Copyright © 2023-2024 Apple Inc.

import numpy as np
import pytest
from pytest_lazy_fixtures import lf

from pfl.data.user_state import DiskUserStateStorage, InMemoryUserStateStorage
from pfl.exception import UserNotFoundError
from pfl.internal.ops.common_ops import get_pytorch_major_version, get_tf_major_version


@pytest.fixture(scope='function')
def disk_user_state_storage(tmp_path):
    return DiskUserStateStorage(tmp_path)


@pytest.fixture(scope='function')
def in_memory_state_storage(tmp_path):
    return InMemoryUserStateStorage()


@pytest.mark.parametrize('ops_module', [
    pytest.param(lf('numpy_ops')),
    pytest.param(lf('tensorflow_ops'),
                 marks=[
                     pytest.mark.skipif(get_tf_major_version() < 2,
                                        reason='not tf>=2')
                 ]),
    pytest.param(lf('pytorch_ops'),
                 marks=[
                     pytest.mark.skipif(not get_pytorch_major_version(),
                                        reason='PyTorch not installed')
                 ]),
])
@pytest.mark.parametrize('storage', [
    lf('in_memory_state_storage'),
    lf('disk_user_state_storage')
])
class TestUserState:

    def test_save_load(self, storage, ops_module):
        array = [1., 2., 3.]
        number = 1
        state = {'number': number, 'tensor': ops_module.to_tensor(array)}
        storage.save_state(state, 'user1')
        loaded_state = storage.load_state('user1')
        assert len(loaded_state) == 2
        assert loaded_state['number'] == number

        # Assert it is back in tensor type.
        assert type(loaded_state['tensor']) == type(
            ops_module.to_tensor(loaded_state['tensor']))
        np.testing.assert_array_equal(
            ops_module.to_numpy(loaded_state['tensor']), array)

    def test_overwrite(self, storage, ops_module):
        state1 = {'number': 1}
        state2 = {'number': 2}
        storage.save_state(state1, 'user1')
        storage.save_state(state1, 'user1', 'unique-key')
        storage.save_state(state2, 'user1')
        assert storage.load_state('user1') == state2
        assert storage.load_state('user1', 'unique-key') == state1

    def test_not_found(self, storage, ops_module):
        state1 = {'number': 1}
        storage.save_state(state1, 'user1')
        storage.load_state('user1')
        with pytest.raises(UserNotFoundError):
            storage.load_state('user2')

    def test_not_found_keyed(self, storage, ops_module):
        state1 = {'number': 1}
        state3 = {'number': 1}
        storage.save_state(state1, 'user1', 'unique-key')
        storage.save_state(state3, 'user3', 'unique-key')
        storage.load_state('user1', 'unique-key')
        with pytest.raises(UserNotFoundError):
            storage.load_state('user2', 'unique-key')
        with pytest.raises(UserNotFoundError):
            storage.load_state('user1', None)
        with pytest.raises(UserNotFoundError):
            storage.load_state('user3')

    def test_clear(self, storage, ops_module):
        state1 = {'number': 1}
        state2 = {'number': 2}
        storage.save_state(state1, 'user1')
        storage.save_state(state2, 'user2')
        assert storage.load_state('user1') == state1
        assert storage.load_state('user2') == state2
        storage.clear_states()
        with pytest.raises(UserNotFoundError):
            storage.load_state('user1')
        with pytest.raises(UserNotFoundError):
            storage.load_state('user2')
