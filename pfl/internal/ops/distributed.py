# Copyright © 2023-2024 Apple Inc.
import logging
import os
from abc import ABC, abstractmethod
from typing import Generic, Iterable, List, Tuple, Type, TypeVar

logger = logging.getLogger(name=__name__)
Tensor = TypeVar('Tensor')

NUMPY_DISTRIBUTE_VAR_NAME = 'PFL_NUMPY_DISTRIBUTE_METHOD'


class DistributedContext(ABC, Generic[Tensor]):
    """
    Collection of properties and methods related to distributed training.
    """

    @property
    @abstractmethod
    def local_rank(self) -> int:
        """
        The rank of the current process over all processes on current machine.
        """
        pass

    @property
    @abstractmethod
    def global_rank(self) -> int:
        """
        The rank of the current process over all processes on all machines.
        """
        pass

    @property
    @abstractmethod
    def world_size(self) -> int:
        """
        The total number of processes over all machines.
        """
        pass

    @property
    @abstractmethod
    def local_size(self) -> int:
        """
        The total number of processes on 1 machines.
        """
        pass

    @abstractmethod
    def all_reduce(self,
                   tensors: List[Tensor],
                   average: bool = False) -> List[Tensor]:
        """
        Performs all reduce between processes on a list of tensors.
        When one process calls this method, it will block until `all_reduce`
        has been called on all other processes as well. Processes may be
        scattered across multiple workers.

        :param tensors:
            A list of tensors to reduce between processes.
        :param average:
            If `False` return sum, if `True` return the average.
        :returns:
            A list of tensors, representing the reduced versions of the
            input parameter `tensors`.
        """
        pass

    def distribute_range(self, value: int) -> Iterable:
        """
        Split `range(value)` among workers so that each workers gets
        a slice of approximately same length.

        Example:
            An input of ``5`` when using 2 workers will return ``range(0,3)``
            for worker 1 and ``range(3,5)`` for worker 2.

        :param value:
            The integer value to split.
        :returns:
            The split value for the current worker.
        """
        if self.world_size <= 1:
            return range(value)

        start_index = (value * self.global_rank) // self.world_size
        end_index = (value * (self.global_rank + 1)) // self.world_size
        return range(start_index, end_index)

    def distribute_value(self, value: int) -> int:
        """
        Split an integer ``value`` among workers. Parameter ``value`` is
        interpreted as the number of units of work. Each worker gets its
        own integer. The integers assigned to all workers sum to ``value``
        and are approximately equal.

        Example:
            An input of ``5`` when using 2 workers will return ``3`` for worker
            1 and ``2`` for worker 2.

        :param value:
            The integer value to split.
        :returns:
            The split value for the current worker.
        """
        if self.world_size <= 1:
            return value

        start_index = (value * self.global_rank) // self.world_size
        end_index = (value * (self.global_rank + 1)) // self.world_size
        return end_index - start_index


class NotDistributedContext(DistributedContext):
    """
    Single-process "distributed" context. Can be used to not do
    distributed training.
    """

    def __init__(self):
        logger.info('Use NotDistributedContext, which means no '
                    'distributed training is enabled')

    @property
    def local_rank(self) -> int:
        return 0

    @property
    def global_rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    @property
    def local_size(self) -> int:
        return 1

    def all_reduce(self,
                   tensors: List[Tensor],
                   average: bool = False) -> List[Tensor]:
        return tensors
