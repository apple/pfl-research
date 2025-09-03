# Copyright © 2023-2024 Apple Inc.
import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from pfl.callback.base import TrainingProcessCallback
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.metrics import Metrics

_WORKER_RANK_STR = 'PFL_WORKER_RANK'
_WORKER_ADDRESSES_STR = 'PFL_WORKER_ADDRESSES'


class Platform(ABC):
    """
    Collection of properties and methods related to compute platforms
    """

    @abstractmethod
    def consume_metrics(self,
                        metrics: Metrics,
                        iteration: int,
                        context: Optional[Dict] = None):
        """
        Process task status and metrics.
        """
        pass

    @abstractmethod
    def create_checkpoint_directories(self, subdir_names):
        """
        Given a list of directories, creates the directories either in the
        current working directory if executed on a local computer or on a
        remote compute platform.
        """
        pass

    @abstractmethod
    def get_distributed_addresses(self, verbose=False):
        """
        Get all IP addresses of compute nodes when running
        in a distributed context.
        """
        pass

    @abstractmethod
    def get_platform_context(self) -> Dict[str, str]:
        """
        Get a string-keyed dict of platform context information.
        """
        pass

    @abstractmethod
    def get_default_callbacks(self) -> List[TrainingProcessCallback]:
        """
        Get a list of callbacks to add to algorithms using this platform.
        """


class GenericPlatform(Platform):

    def __init__(self):
        self.logger = logging.getLogger(name=__name__)

    def consume_metrics(self,
                        metrics: Metrics,
                        iteration: int,
                        context: Optional[Dict] = None):
        """
        Send metrics to platform, which may e.g. output them on the terminal
        or display on a dashboard.

        :param metrics:
            Metrics to output.
        :param iteration:
            Current iteration.
        :param context:
            An optional dict defining the context, e.g. hyperparameters used.
        """
        if context is None:
            context = {}
        if get_ops().distributed.local_rank == 0:
            context_string = ', '.join(f'{key}: {value}'
                                       for (key, value) in context.items())
            sys.stdout.write(
                f'Metrics at iteration {iteration} ({context_string}):\n')
            for key, value in metrics.to_simple_dict().items():
                sys.stdout.write(f'    {key:<50}: {value}\n')

    def create_checkpoint_directories(self, subdir_names) -> List:
        """
        Given a list of directories, creates the directories in appropriate
        location on the current platform.

        :param subdir_names:
            A list of directory names to create.
        :returns:
            A list of the paths for every new directory.
        """

        return self.create_checkpoint_directories_in(".", subdir_names)

    def get_distributed_addresses(self, verbose=False) -> Tuple:
        if _WORKER_ADDRESSES_STR in os.environ:
            assert _WORKER_RANK_STR in os.environ, (
                f"If you set {_WORKER_ADDRESSES_STR}, you must also "
                f"set {_WORKER_RANK_STR} for each worker")
            # For debugging purposes on local computer you can do
            # # In terminal 1:
            # PFL_WORKER_ADDRESSES = localhost:8081,localhost:8082
            # PFL_WORKER_RANK = 0 python my_program.py
            # # In terminal 2:
            # PFL_WORKER_ADDRESSES = localhost:8081,localhost:8082
            # PFL_WORKER_RANK = 1 python my_program.py
            addresses = os.environ[_WORKER_ADDRESSES_STR].split(',')
            return int(os.environ[_WORKER_RANK_STR]), addresses
        else:
            return 0, None

    def get_platform_context(self) -> Dict[str, str]:
        return {}

    def create_checkpoint_directories_in(self, base_directory,
                                         subdir_names) -> List:
        return_subdir_paths = []

        # Create subdirectories.
        for subdir in subdir_names:
            subdir_path = os.path.join(base_directory, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)
            return_subdir_paths.append(subdir_path)
        return return_subdir_paths

    def get_default_callbacks(self) -> List[TrainingProcessCallback]:
        return []
