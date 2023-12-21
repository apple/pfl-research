# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
"""
Sometimes, Federated Learning algorithms require users to be stateful,
e.g. each user might have unique parameter values that are used and updated
only locally. This module implements various methods of saving and
retrieving such user state.
"""

import os
import shutil
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Optional

from pfl.exception import UserNotFoundError
from pfl.internal.bridge import FrameworkBridgeFactory as bridges


class AbstractUserStateStorage(ABC):
    """
    Abstract base class for storage to save and load
    user state.

    Incorporating user state in a custom algorithm might look something like
    this:

    :example:
      .. code-block:: python

        storage = InMemoryUserStateStorage()

        class MyAlgorithm(FederatedNNAlgorithm):

          def train_one_user(self, initial_model_state, model, user_dataset,
                             central_context):
            user_state = storage.load_state(user_dataset.user_id, 'my-algo')
            # ... do local algorithm here, which also generates a new state
            storage.save_state(new_user_state, user_dataset.user_id, 'my-algo')
            return model.get_model_difference(initial_model_state), Metrics()

    If the user state is particularly big, e.g. a full set of model weights,
    loading the state in the data loading stage can be beneficial if the data
    loading stage is already parallelized.

    :example:
      .. code-block:: python

        storage = InMemoryUserStateStorage()

        def make_dataset_fn(user_id):
            raw_data = load_raw_data(user_id)
            return Dataset(
                raw_data=[inputs, targets],
                local_state=storage.load_state(user_id, 'my-algo'))

        # fed_data is input to SimulatedAggregator
        fed_data = FederatedDataset(
            make_dataset_fn, sampler, user_id_to_weight=user_num_images)

        class MyAlgorithm(FederatedNNAlgorithm):

          def train_one_user(self, initial_model_state, model, user_dataset,
                             central_context):
            user_state = user_dataset.local_state
            # ... do local algorithm here, which also generate a new state
            storage.save_state(new_user_state, user_dataset.user_id, 'my-algo')
            return model.get_model_difference(initial_model_state), Metrics()
    """

    @abstractmethod
    def clear_states(self):
        """
        Remove all existing stored user states.
        """

    @abstractmethod
    def save_state(self,
                   state: Dict[str, Any],
                   user_id: str,
                   key: Optional[str] = None):
        """
        :param state:
            A dictionary with generic values to store as state.
        :param user_id:
            ID of the user of this state.
        :param key:
            Multiple states of same user can be stored simultaneously if unique
            keys are specified for each different state object.
            Not specifying a key will overwrite the default saved user state.
        """

    @abstractmethod
    def load_state(self,
                   user_id: str,
                   key: Optional[str] = None) -> Dict[str, Any]:
        """
        :param user_id:
            Restore the state of the user with this ID.
        :param key:
            Optional key to load a specific user state that was previously
            saved with this key. Not specifying a key will load the default
            saved user state.
        :returns:
            The user state.
        """


class InMemoryUserStateStorage(AbstractUserStateStorage):
    """
    Save and load user state for a given user ID.
    Keeps states of all users in memory such that loading
    and saving state is very fast.

    .. warning::

        Saving user state in memory is neither compatible out of the box with
        distributed simulations on multiple machines nor on multiple processes
        on same machine. If your large simulations require multiple processes,
        you can use a cross-silo user sampler such that a unique user is pinned
        to being sampled in the same worker process each time, where the state
        of that user is cached.
    """

    def __init__(self):
        self._cache = {}

    def clear_states(self):
        self._cache = {}

    def save_state(self,
                   state: Dict[str, Any],
                   user_id: str,
                   key: Optional[str] = None):
        self._cache[(user_id, key)] = state

    def load_state(self,
                   user_id: str,
                   key: Optional[str] = None) -> Dict[str, Any]:
        if (user_id, key) not in self._cache:
            raise UserNotFoundError(user_id)
        return self._cache[(user_id, key)]


class DiskUserStateStorage(AbstractUserStateStorage):
    """
    Save and load user state for a given user ID.
    Keeps states of all users on disk.
    This is slower than InMemoryUserStateStorage, but
    necessary if the user states are too large to fit
    into memory.

    .. warning::

        Saving user state on disk is not compatible out of the
        box with distributed simulations on multiple machines.
        Try to use a single machine with enough GPUs and
        a common disk space. If your large simulations
        require multiple machines, you can use a cross-silo
        user sampler such that a unique user is pinned
        to being sampled on the same machine each time.
    """

    def __init__(self, dir_path: str):
        self._dir_path = dir_path

    def clear_states(self):
        shutil.rmtree(self._dir_path)

    @contextmanager
    def acquire_lock(self, path):
        """
        Acquire lock on a specific file.
        The lock state is written to disk and thereby works
        across processes that share disk space.

        :param path:
            Acquire lock for this file.
        """
        lock_path = path + '.lock'
        sleep_count = 0
        while True:
            try:
                # Atomic create file or throw error if exist.
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL)
            except FileExistsError as e:
                time.sleep(0.1)
                sleep_count += 1
                if sleep_count > 100:
                    raise RuntimeError(f"Didn't acquire lock {lock_path} "
                                       "after waiting 10 seconds") from e
            else:
                break

        try:
            yield
        finally:
            os.close(fd)
            os.remove(lock_path)

    def save_state(self,
                   state: Dict[str, Any],
                   user_id: str,
                   key: Optional[str] = None):
        dir_path = os.path.join(self._dir_path,
                                key) if key is not None else self._dir_path
        os.makedirs(dir_path, exist_ok=True)
        state_path = os.path.join(dir_path, user_id + '.state')
        with self.acquire_lock(state_path):
            bridges.common_bridge().save_state(state, state_path)

    def load_state(self,
                   user_id: str,
                   key: Optional[str] = None) -> Dict[str, Any]:
        dir_path = os.path.join(self._dir_path,
                                key) if key is not None else self._dir_path
        os.makedirs(dir_path, exist_ok=True)
        state_path = os.path.join(dir_path, user_id + '.state')
        with self.acquire_lock(state_path):
            try:
                return bridges.common_bridge().load_state(
                    os.path.join(dir_path, user_id + '.state'))
            except (FileNotFoundError, RuntimeError) as exc:
                raise UserNotFoundError(user_id) from exc
