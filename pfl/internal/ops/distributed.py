# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import logging
import os
from abc import ABC, abstractmethod
from typing import Generic, Iterable, List, Tuple, Type, TypeVar

logger = logging.getLogger(name=__name__)
Tensor = TypeVar('Tensor')

NUMPY_DISTRIBUTE_VAR_NAME = 'PFL_NUMPY_DISTRIBUTE_METHOD'


def horovod_is_active() -> bool:
    """
    :return:
        `True` if program was called with `horovodrun`.
    """
    # MPI
    is_using_mpi_horovod = os.environ.get('OMPI_COMMAND') and os.environ.get(
        'OMPI_ARGV')
    # Gloo
    is_using_gloo_horovod = os.environ.get(
        'HOROVOD_HOSTNAME') and os.environ.get(
            'HOROVOD_RANK') and os.environ.get('HOROVOD_SIZE')
    return bool(is_using_mpi_horovod or is_using_gloo_horovod)


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


class HorovodDistributedContext(DistributedContext):
    """
    Base class for distributed training operations with Horovod.
    """

    def __init__(self, hvd):
        self._hvd = hvd

    def _post_init_check(self, hvd):
        if hvd.size() == 1:
            raise RuntimeError(
                'You are running Horovod backend but number of '
                'worker and processes are 1. If you intend to only run on a '
                'single worker and process, don\'t use Horovod.'
                'If you intend to run multiple processes/workers, then your '
                'run command was incorrect')

    @property
    def hvd(self):
        return self._hvd

    @property
    def local_rank(self) -> int:
        return self.hvd.local_rank()

    @property
    def global_rank(self) -> int:
        return self.hvd.rank()

    @property
    def world_size(self) -> int:
        return self.hvd.size()

    @property
    def local_size(self) -> int:
        return self.hvd.local_size()

    @abstractmethod
    def _flatten(
            self,
            tensors: List[Tensor]) -> Tuple[Tensor, List[Tuple], List[Type]]:
        pass

    @abstractmethod
    def _reshape(self, vector: Tensor, shapes: List[Tuple],
                 dtypes: List[Type]) -> List[Tensor]:
        pass

    def all_reduce(self,
                   tensors: List[Tensor],
                   average: bool = False) -> List[Tensor]:
        if self.world_size <= 1:
            # In the case of a single worker, just return identity instead
            # of doing unnecessary flatten and reshape.
            return tensors

        flat_vector, reshape_shapes, reshape_types = self._flatten(tensors)
        reduce_op = (self.hvd.mpi_ops.Average
                     if average else self.hvd.mpi_ops.Sum)
        reduced_flat_vector = self.hvd.allreduce(flat_vector, op=reduce_op)
        return self._reshape(reduced_flat_vector, reshape_shapes,
                             reshape_types)


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
