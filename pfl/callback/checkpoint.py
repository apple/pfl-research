# Copyright © 2023-2024 Apple Inc.
"""
Callbacks to save the intermediate model state during training.
"""
import logging
import operator
from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple

from pfl.callback.base import TrainingProcessCallback
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.metrics import MetricName, Metrics, get_overall_value
from pfl.model.base import StatefulModel

logger = logging.getLogger(name=__name__)


class CheckpointPolicy(ABC):
    """
    Controls when `PolicyBasedModelCheckpointingCallback` should checkpoint.
    """

    @abstractmethod
    def should_checkpoint_now(self, aggregate_metrics: Metrics,
                              central_iteration: int) -> bool:
        """
        Invoked at the end of each central iteration to decide whether
        a checkpoint should be made.
        """
        raise NotImplementedError

    @abstractmethod
    def should_checkpoint_at_end(self) -> bool:
        """
        Invoked at the end of training to decide whether a checkpoint should
        be made.
        """
        raise NotImplementedError


class IterationFrequencyCheckpointPolicy:
    """
    Checkpoint policy for `PolicyBasedModelCheckpointingCallback` that
    saves a checkpoint after every `checkpoint_frequency` iterations if the
    value is positive or at the end of training if it is zero.
    """

    def __init__(self, checkpoint_frequency: int):
        self.checkpoint_frequency = checkpoint_frequency

    def should_checkpoint_now(self, aggregate_metrics: Metrics,
                              central_iteration: int) -> bool:
        """
        Return true when the number of `central_iteration`s that have
        completed is a non-zero multiple of `self.checkpoint_frequency`.
        """
        return (self.checkpoint_frequency > 0
                and central_iteration % self.checkpoint_frequency
                == (self.checkpoint_frequency - 1))

    def should_checkpoint_at_end(self) -> bool:
        return self.checkpoint_frequency == 0


class MetricImprovementCheckpointPolicy(CheckpointPolicy):
    """
    Stateful checkpoint policy for `PolicyBasedModelCheckpointingCallback`
    to save a checkpoint after any iteration where the value of `metric_name`
    has improved versus the prior best value.

    :param metric_name:
        The metrics whose value to track.

    :param threshold_value:
        If present, only save a checkpoint if the metric value is better than
        this value.

    :param performance_is_better:
        A binary predicate indicating that `lhs` is better `rhs`.

        For metrics where higher values are better, like precision,
        you would want to use `operator.gt`, and for metrics like
        loss, you would want to use `operator.lt` (the default).
    """

    metric_name: MetricName
    best_value: float | None
    performance_is_better: Callable[[Any, Any], bool]

    def __init__(self,
                 metric_name: MetricName,
                 *,
                 threshold_value: float | None = None,
                 performance_is_better: Callable[[Any, Any],
                                                 bool] = operator.lt):
        self.metric_name = metric_name
        self.best_value = threshold_value
        self.performance_is_better = performance_is_better

    def should_checkpoint_now(self, aggregate_metrics: Metrics,
                              central_iteration: int):
        cur_value = get_overall_value(aggregate_metrics[self.metric_name])
        if (self.best_value is None
                or self.performance_is_better(cur_value, self.best_value)):
            self.best_value = cur_value
            return True
        return False

    def should_checkpoint_at_end(self):
        return False


class PolicyBasedModelCheckpointingCallback(TrainingProcessCallback):
    """
    Callback to save model checkpoints after iterations and after
    training, when indicated by `policy`.

    :param model_checkpoint_dir:
        A path to disk for saving the trained model.
        If running on Bolt, this will be a path relative to
        ``ARTIFACT_DIR``.
    :param policy:
        An instance of a `CheckpointPolicy` subclass.

    :param numbered: If true, include the iteration number in each
        checkpoint's path to save all the checkpoints without
        overwriting.
    """

    def __init__(self,
                 model_checkpoint_dir: str,
                 *,
                 checkpoint_policy: CheckpointPolicy,
                 numbered: bool = False):
        if get_ops().distributed.local_rank == 0:
            self.numbered = numbered
            self.checkpoint_policy = checkpoint_policy
            from pfl.internal.platform.selector import get_platform
            self.model_checkpoint_dir_name = model_checkpoint_dir
            if not numbered:
                self.model_checkpoint_dir = get_platform(
                ).create_checkpoint_directories([model_checkpoint_dir])[0]

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: StatefulModel, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        if get_ops().distributed.local_rank == 0:
            if self.checkpoint_policy.should_checkpoint_now(
                    aggregate_metrics, central_iteration):
                if self.numbered:
                    from pfl.internal.platform.selector import get_platform
                    self.model_checkpoint_dir = get_platform(
                    ).create_checkpoint_directories([
                        f'{self.model_checkpoint_dir_name}/'
                        f'{central_iteration:05}'
                    ])[0]
                model.save(self.model_checkpoint_dir)
        return False, Metrics()

    def on_train_end(self, *, model: StatefulModel) -> None:
        if get_ops().distributed.local_rank == 0 and (
                self.checkpoint_policy.should_checkpoint_at_end()):
            if self.numbered:
                from pfl.internal.platform.selector import get_platform
                self.model_checkpoint_dir = get_platform(
                ).create_checkpoint_directories(
                    [f'{self.model_checkpoint_dir_name}/final'])[0]
            model.save(self.model_checkpoint_dir)


class ModelCheckpointingCallback(PolicyBasedModelCheckpointingCallback):
    """
    Callback to save model checkpoints. Note that the model checkpoints
    can also be saved as part of ``RestoreTrainingCallback`` as long as
    the model is ``Saveable`` and provided in the list of saveeables in
    the initialization of the callback.

    :param model_checkpoint_dir:
        A path to disk for saving the trained model. Location
        will be relative to root dir on current platform.
    :param checkpoint_frequency:
        The number of central iterations after which to save a model.
        When zero (the default), the model is saved once after
        training is complete.
    :param numbered: If true, append the iteration number to each
        checkpoint path to save all the checkpoints without
        overwriting.
    """

    def __init__(self,
                 model_checkpoint_dir: str,
                 *,
                 checkpoint_frequency: int = 0,
                 numbered: bool = False):
        super().__init__(model_checkpoint_dir,
                         checkpoint_policy=IterationFrequencyCheckpointPolicy(
                             checkpoint_frequency),
                         numbered=numbered)
