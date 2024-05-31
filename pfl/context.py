# Copyright © 2023-2024 Apple Inc.
from dataclasses import dataclass, field
from typing import Generic, Optional

from pfl.common_types import Population
from pfl.hyperparam.base import AlgorithmHyperParamsType, ModelHyperParamsType
from pfl.metrics import Metrics


@dataclass(frozen=True)
class LocalResultMetaData:
    """
    Data that is typically returned by a model's local optimization procedure,
    e.g. ``PyTorchModel.do_multiple_epochs_of``. Can have useful information
    needed by the algorithm.

    :param num_steps:
        The number of local steps taken during the local optimization procedure.
    """
    num_steps: int


@dataclass(frozen=True)
class UserContext:
    """
    Provides read-only information about the user. This is exposed to
    postprocessors' local statistics postprocessing procedure in
    simulation, see
    :class:`~pfl.postprocessor.base.Postprocessor.postprocess_one_user`.

    :param num_datapoints:
        The number of datapoints of the user.
    :param seed:
        A seed to use for any stochastic postprocessing operations.
    :param user_id:
        ID of user. Can be ``None`` if user didn't specify any user IDs,
        which is fine if there are no algorithms in use that require user
        IDs.
    :param metrics:
        The metrics collected in the current central iteration for the
        user this context belongs to.
    """
    num_datapoints: int
    seed: Optional[int]
    user_id: Optional[str] = None
    metrics: Metrics = field(default_factory=Metrics)


@dataclass(frozen=True)
class CentralContext(Generic[AlgorithmHyperParamsType, ModelHyperParamsType]):
    # pylint: disable=line-too-long
    """
    Provides read-only information about server-side parameters. This is
    exposed to:

    * postprocessors' aggregated statistics postprocessing procedure,
      see :class:`~pfl.postprocessor.base.Postprocessor.postprocess_server`.
    * local and central procedures of an algorithm, see
      :class:`pfl.algorithm.base.FederatedAlgorithm.process_aggregated_statistics`
      and :class:`pfl.algorithm.base.FederatedAlgorithm.simulate_one_user`.

    :param current_central_iteration:
        The current central iteration number, starting at 0.
    :param do_evaluation:
         Whether the local evaluations need to be done or not.
         This can speed up the simulation considerably if the multiple local
         evaluations are expensive compared to the local training and the
         adopters need the metrics only sporadically (typically, every nth
         round).
    :param cohort_size:
        The requested cohort size.
    :param population:
        The population to target with this context.
    :param algorithm_params:
        The algorithm context for this central iteration.
    :param model_train_params:
        Hyper-parameter configuration for local training of model.
    :param model_eval_params:
        Hyper-parameter configuration for evaluation of model.
    :param seed:
        A seed to use for any stochastic postprocessing operations.
    """
    # pylint: enable=line-too-long
    current_central_iteration: int
    do_evaluation: bool
    cohort_size: int
    population: Population
    algorithm_params: AlgorithmHyperParamsType
    model_train_params: ModelHyperParamsType
    model_eval_params: Optional[ModelHyperParamsType] = None
    seed: Optional[int] = None

    @property
    def state_description(self):
        """
        Description of current state of training.
        """
        return f'iteration {self.current_central_iteration}'
