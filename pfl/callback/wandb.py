# Copyright © 2023-2024 Apple Inc.
import logging
from typing import Optional, Tuple

from pfl.callback.training_process import TrainingProcessCallback
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.metrics import Metrics
from pfl.model.base import ModelType

logger = logging.getLogger(name=__name__)

# pylint: disable=too-many-lines


class WandbCallback(TrainingProcessCallback):
    """
    Callback for reporting metrics to Weights&Biases dashboard for comparing
    different PFL runs.
    This callback has basic support for logging metrics. If you seek more
    advanced features from the Wandb API, you should make your own callback.

    See https://wandb.ai/ and https://docs.wandb.ai/ for more information on
    Weights&Biases.

    :param wandb_project_id:
        The name of the project where you're sending the new run. If the
        project is not specified, the run is put in an "Uncategorized" project.
    :param wandb_experiment_name:
        A short display name for this run. Generates a random two-word name
        by default.
    :param wandb_config:
         Optional dictionary (or argparse) of parameters (e.g. hyperparameter
         choices) that are used to tag this run in the Wandb dashboard.
    :param wandb_kwargs:
        Additional keyword args other than ``project``, ``name`` and ``config``
        that you can input to ``wandb.init``, see
        https://docs.wandb.ai/ref/python/init for reference.
    """

    def __init__(self,
                 wandb_project_id: str,
                 wandb_experiment_name: Optional[str] = None,
                 wandb_config=None,
                 **wandb_kwargs):
        self._wandb_kwargs = {
            'project': wandb_project_id,
            'name': wandb_experiment_name,
            'config': wandb_config
        }
        self._wandb_kwargs.update(wandb_kwargs)

    @property
    def wandb(self):
        # Not necessarily installed by default.
        import wandb
        return wandb

    def on_train_begin(self, *, model: ModelType) -> Metrics:
        if get_ops().distributed.global_rank == 0:
            self.wandb.init(**self._wandb_kwargs)
        return Metrics()

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        """
        Submits metrics of this central iteration to Wandb experiment.
        """
        if get_ops().distributed.global_rank == 0:
            # Wandb package already uses a multithreaded solution
            # to submit log requests to server, such that this
            # call will not be blocking until server responds.
            self.wandb.log(aggregate_metrics.to_simple_dict(),
                           step=central_iteration)
        return False, Metrics()

    def on_train_end(self, *, model: ModelType) -> None:
        if get_ops().distributed.global_rank == 0:
            self.wandb.finish()
