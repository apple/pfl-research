import os
from typing import Tuple

from pfl.callback.base import TrainingProcessCallback
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.metrics import Metrics
from pfl.model.base import StatefulModel


class ModelCheckpointingIterationCallback(TrainingProcessCallback):
    """
    Callback to save model checkpoints.

    :param model_checkpoint_dir:
        A path to disk for saving the trained model.
    :param checkpoint_frequency:
        The number of central iterations after which to save a model.
        When zero (the default), the model is saved once after
        training is complete.
    """

    def __init__(self,
                 model_checkpoint_dir: str,
                 *,
                 checkpoint_frequency: int = 0):
        if get_ops().distributed.local_rank == 0:
            self.checkpoint_frequency = checkpoint_frequency
            from pfl.internal.platform.selector import get_platform
            self.model_checkpoint_dir = get_platform(
            ).create_checkpoint_directories([model_checkpoint_dir])[0]

    def _should_checkpoint_now(self, central_iteration: int) -> bool:
        """
        Return true when the number of `central_iteration`s that have
        completed is a non-zero multiple of `self.checkpoint_frequency`.
        """
        return (self.checkpoint_frequency > 0
                and central_iteration % self.checkpoint_frequency
                == (self.checkpoint_frequency - 1))

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: StatefulModel, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        if get_ops().distributed.local_rank == 0:
            if self._should_checkpoint_now(central_iteration):
                model.save(
                    os.path.join(self.model_checkpoint_dir,
                                 f'{central_iteration}'))
        return False, Metrics()

    def on_train_end(self, *, model: StatefulModel) -> None:
        if get_ops().distributed.local_rank == 0:
            if self.checkpoint_frequency == 0:
                model.save(self.model_checkpoint_dir + '_end')
