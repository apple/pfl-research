# Copyright © 2023-2024 Apple Inc.
import logging
from typing import Generic, Tuple

from pfl.metrics import Metrics
from pfl.model.base import ModelType

logger = logging.getLogger(name=__name__)

# pylint: disable=too-many-lines


class TrainingProcessCallback(Generic[ModelType]):
    """
    Base class for callbacks.
    """

    def on_train_begin(self, *, model: ModelType) -> Metrics:
        """
        Called before the first central iteration.
        """
        return Metrics()

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        """
        Finalize any computations after each central iteration.

        :param aggregate_metrics:
            A :class:`~pfl.metrics.Metrics` object with aggregated metrics
            accumulated from local training on users and central updates
            of the model.
        :param model:
            A reference to the `Model` that is trained.
        :param central_iteration:
            The current central iteration number.
        :returns:
            A tuple.
            The first value returned is a boolean, signaling that training
            should be interrupted if ``True``.
            Can be useful for implementing features with early stopping or
            convergence criteria.
            The second value returned is new metrics.
            Do not include any of the aggregate_metrics!
        """
        return False, Metrics()

    def on_train_end(self, *, model: ModelType) -> None:
        """
        Called at the end of training.
        """
        pass
