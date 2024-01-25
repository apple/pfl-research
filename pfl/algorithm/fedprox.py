# Copyright © 2023-2024 Apple Inc.
import sys
from dataclasses import dataclass, field
from typing import Tuple, Union

from pfl.algorithm.base import NNAlgorithmParams
from pfl.algorithm.federated_averaging import FederatedAveraging
from pfl.callback import TrainingProcessCallback
from pfl.context import CentralContext
from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import HyperParam, HyperParamClsOrFloat
from pfl.internal.bridge import FrameworkBridgeFactory as bridges
from pfl.metrics import Metrics, StringMetricName, get_overall_value
from pfl.model.base import ModelType, StatefulModel
from pfl.stats import WeightedStatistics


@dataclass(frozen=True)
class FedProxParams(NNAlgorithmParams):
    """
    Parameters for FedProx algorithm.

    :param mu:
        Scales the additional loss term added by FedProx.
        Only values ``[0,1]`` make sense. A value of ``0``
        means the additional loss term has no effect and this
        algorithm regresses back to Federated Averaging.
    """
    mu: HyperParamClsOrFloat


@dataclass
class AdaptMuOnMetricCallback(HyperParam[float], TrainingProcessCallback):
    """
    Adaptive scalar for proximal term (``mu``) in FedProx algorithm,
    described in Appendix C.3.3 in
    T. Li. et al. - Federated Optimization in Heterogeneous Networks
    (https://arxiv.org/pdf/1812.06127.pdf).

    Set an instance of this as ``mu`` in
    :class:`~pfl.algorithm.fedprox.FedProxParams` and add it to
    the algorithm's list of callbacks to make ``mu`` adaptive.

    :param metric_name:
        The metric name to use for adapting ``mu``. Lower should be better.
    :param adapt_frequency:
        Adapt ``mu`` according to the rules every this number of iterations.

        .. note::

          If you adapt on a metric that is not reported every round,
          e.g. central iteration, make sure that adapting ``mu`` is done
          at the same frequency such that the metric is available in the
          aggregated metrics.

    :param initial_value:
        Initial value for ``mu``. Appendix C.3.3 suggest ``0.0`` for
        homogeneous federated datasets and ``1.0`` for heterogeneous
        federated datasets.
    :param step_size:
        How much to increase or decrease ``mu`` each step it is calibrated.
        T. Li. et al. suggests ``0.1``.
    :param decrease_mu_after_consecutive_improvements:
        Decreaase ``mu`` if ``metric_value`` is lower this many times in a row.
    """
    metric_name: Union[str, StringMetricName]
    adapt_frequency: int
    initial_value: float = 0.0
    step_size: float = 0.1
    decrease_mu_after_consecutive_improvements: int = 1
    loss_decrease: int = field(default=0, init=False)
    prev_loss: float = field(default=sys.float_info.max, init=False)

    def __post_init__(self):
        self._mu = self.initial_value

    def value(self) -> float:
        return self._mu

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        """
        Extend adaptive mu as a callback such that it can be calibrated
        after each central iteration.
        """
        if (central_iteration % self.adapt_frequency == 0):
            new_loss = get_overall_value(aggregate_metrics[self.metric_name])
            if new_loss < self.prev_loss:
                self.loss_decrease += 1
                if (self.loss_decrease
                        >= self.decrease_mu_after_consecutive_improvements):
                    self._mu = max(0.0, self._mu - self.step_size)
            else:
                self.loss_decrease = 0
                self._mu = min(1.0, self._mu + self.step_size)
            self.prev_loss = new_loss

        return False, Metrics([('mu', self._mu)])


class FedProx(FederatedAveraging):
    """
    FedProx algorithm, introduced by
    T. Li. et al. - Federated Optimization in Heterogeneous Networks
    (https://arxiv.org/pdf/1812.06127.pdf).

    Adds a proximal term to loss during local training which is a soft
    constraint on the local model not diverging too far from current global
    model.
    """

    def train_one_user(
            self, initial_model_state: WeightedStatistics,
            model: StatefulModel, user_dataset: AbstractDatasetType,
            central_context: CentralContext
    ) -> Tuple[WeightedStatistics, Metrics]:
        assert isinstance(central_context.algorithm_params, FedProxParams)
        # Local training loop
        bridges.fedprox_bridge().do_proximal_sgd(
            model,
            user_dataset,
            central_context.model_train_params,
            mu=central_context.algorithm_params.mu)
        training_statistics = model.get_model_difference(initial_model_state)
        # Don't reset model, will be used for evaluation after local training.

        return training_statistics, Metrics()
