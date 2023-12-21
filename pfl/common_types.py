# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
from abc import ABC
from enum import Enum


class Population(Enum):
    """
    Enum representing the different pools the devices are divided up into.
    """
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class Saveable(ABC):
    """
    Interface to allow save and load the state of an object to/from disk.

    This is useful to e.g. add fault tolerance to your components if you
    want to be able to resume training after a crash.
    """

    def save(self, dir_path: str) -> None:
        """
        Save state of object to disk. Should be able
        to interpret saved state as a checkpoint that
        can be restored with ``load``.

        :param dir_path:
            Directory on disk to store state.
        """

    def load(self, dir_path: str) -> None:
        """
        Load checkpoint from disk, which is the state previously
        saved with ``save``.

        :param dir_path:
            Path to root directory where checkpoint can be loaded from.
            Should be same path as used with ``save``.
        """
