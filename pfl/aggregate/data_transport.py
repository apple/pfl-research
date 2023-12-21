# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
from abc import abstractmethod, abstractproperty
from typing import Tuple

from pfl.context import UserContext
from pfl.internal.ops.selector import get_framework_module
from pfl.metrics import Metrics
from pfl.postprocessor.base import Postprocessor
from pfl.stats import TrainingStatistics


class DataTransport(Postprocessor):
    """
    Data transport base class. Specifies in what format model updates are sent.
    """

    @property
    @abstractmethod
    def transport_format(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def postprocess_one_user(
            self, *, stats: TrainingStatistics,
            user_context: UserContext) -> Tuple[TrainingStatistics, Metrics]:
        """
        Simulate the transport of model updates in the format specified by this
        class.
        This is simulated by converting to the transport format and then converting
        back to the original format, which will result in the same eventual loss
        of information.

        :param stats:
            The model update to simulate transport with.
        :param user_context:
            Information about the user.
        :returns:
            The same (transported) model update and any new metrics generated.
        """
        raise NotImplementedError


class Float32DataTransport(DataTransport):
    """
    Use the float32 (legacy) format for data transport. Has no effect in
    simulations.
    """

    @property
    def transport_format(self) -> str:
        return 'float32'

    def postprocess_one_user(
            self, *, stats: TrainingStatistics,
            user_context: UserContext) -> Tuple[TrainingStatistics, Metrics]:
        return stats, Metrics()


class BFloat16DataTransport(DataTransport):
    """
    Use the bfloat16 format for data transport. Simulates this behaviour by
    converting model updates
    ``original_format -> bfloat16 -> original_format``.
    """

    @property
    def transport_format(self) -> str:
        return 'bfloat16'

    def postprocess_one_user(
            self, *, stats: TrainingStatistics,
            user_context: UserContext) -> Tuple[TrainingStatistics, Metrics]:
        # Convert to bfloat16 and back to original type.
        # Only popular Deep Learning frameworks seem to have an implementation
        # of this type.
        return stats.apply_elementwise(
            get_framework_module().simulate_bfloat16_transport), Metrics()
