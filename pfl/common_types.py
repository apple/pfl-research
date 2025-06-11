# Copyright © 2023-2024 Apple Inc.
from abc import ABC, abstractmethod
from enum import Enum


class Population(Enum):
    """
    Enum representing the different pools the devices are divided up into.
    """
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class Checkpointer(ABC):
    """
    Mediator object for saving checkpoints.
    Creator decides the path to save to,
    caller decides when to save.
    """

    @abstractmethod
    def invoke_save(self, saveable: 'Saveable'):
        pass


class LocalDiskCheckpointer(Checkpointer):

    def __init__(self, dir_path: str):
        self._dir_path = dir_path

    def invoke_save(self, saveable: 'Saveable'):
        saveable.save(self._dir_path)


class Saveable(ABC):
    """
    Interface to allow save and load the state of an object to/from disk.

    This is useful to e.g. add fault tolerance to your components if you
    want to be able to resume training after a crash.
    """

    @abstractmethod
    def save(self, dir_path: str) -> None:
        """
        Save state of object to disk. Should be able
        to interpret saved state as a checkpoint that
        can be restored with ``load``.

        :param dir_path:
            Directory on disk to store state.
        """

    @abstractmethod
    def load(self, dir_path: str) -> None:
        """
        Load checkpoint from disk, which is the state previously
        saved with ``save``.

        :param dir_path:
            Path to root directory where checkpoint can be loaded from.
            Should be same path as used with ``save``.
        """

    @abstractmethod
    def set_checkpointer(self, checkpointer: Checkpointer) -> None:
        """
        Can optionally be implemented to let the component invoke a call
        of ``save`` to save intermediate checkpoints on-demand instead of
        only during scheduled calls by other components, usually
        "after central iteration" by callbacks.

        :param checkpointer:
            Can be called to invoke a ``save`` call on-demand.
        """
