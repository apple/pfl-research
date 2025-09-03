# Copyright © 2023-2024 Apple Inc.
import logging
from typing import Callable, List, Tuple, Union

from pfl.callback.base import TrainingProcessCallback
from pfl.common_types import Checkpointer, LocalDiskCheckpointer, Saveable
from pfl.exception import CheckpointNotFoundError
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.metrics import Metrics, StringMetricName
from pfl.model.base import ModelType

logger = logging.getLogger(name=__name__)


class RestoreTrainingCallback(TrainingProcessCallback):
    """
    Add fault-tolerance to your training. If the training run fails
    and you restart it, this callback will restore all recent
    checkpoints of the ``saveables`` before starting training again.
    Be careful if you've implemented any stateful component,
    these will only be restored if you've properly implemented the
    :class:`~pfl.common_types.Saveable` interface on the component
    and input it to this callback. For restoring a checkpoint, it
    is assumed that all ``saveables`` were successfully stored
    in the last attempt.

    :param saveables:
        The objects that need to save their states so that they can be
        loaded if training is interrupted and then resumed.
    :param checkpoint_dir:
        Root dir for where to store the saveables' states.
        Let this be a list of directory paths to specify a unique
        checkpoint directory for each saveable.
        Location will be relative to root dir on current platform.
    :param checkpoint_frequency:
        Save checkpoints of ``saveables`` every this many iterations.
    :param init_checkpointer_fn:
        When the ``Saveable`` wants to invoke checkpointing itself,
        it is called through instance of this class.
    """

    def __init__(
        self,
        saveables: List[Saveable],
        checkpoint_dir: Union[str, List[str]],
        checkpoint_frequency: int = 1,
        init_checkpointer_fn: Callable[[str],
                                       Checkpointer] = LocalDiskCheckpointer):
        self._saveables = saveables

        from pfl.internal.platform.selector import get_platform
        self._checkpoint_dirs: List[str]
        if isinstance(checkpoint_dir, list):
            assert len(saveables) == len(checkpoint_dir)
            self._checkpoint_dirs = get_platform(
            ).create_checkpoint_directories(checkpoint_dir)
        else:
            self._checkpoint_dirs = get_platform(
            ).create_checkpoint_directories([checkpoint_dir]) * len(saveables)
        for s, d in zip(saveables, self._checkpoint_dirs):
            checkpointer = init_checkpointer_fn(d)
            s.set_checkpointer(checkpointer)
        self._checkpoint_frequency = checkpoint_frequency

    def on_train_begin(self, *, model: ModelType) -> Metrics:
        """
        Restore from previous run's checkpoints if exists.
        """
        # Restore saveables.
        num_components_restored = 0
        for saveable, checkpoint_dir in zip(self._saveables,
                                            self._checkpoint_dirs):
            try:
                saveable.load(checkpoint_dir)
            except CheckpointNotFoundError as e:
                logger.info('RestoreTrainingRunCallback - %s for %s', e,
                            saveable)
            else:
                logger.info(
                    'RestoreTrainingRunCallback - Restored checkpoint for %s',
                    saveable)
                num_components_restored += 1
        return Metrics([(StringMetricName('restored components'),
                         num_components_restored)])

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        if (central_iteration % self._checkpoint_frequency == 0
                and get_ops().distributed.local_rank == 0):
            for saveable, checkpoint_dir in zip(self._saveables,
                                                self._checkpoint_dirs):
                saveable.save(checkpoint_dir)
                logger.info(
                    'RestoreTrainingRunCallback - Saved checkpoint for %s',
                    saveable)
        return False, Metrics()
