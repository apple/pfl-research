# Copyright © 2023-2024 Apple Inc.
from typing import Tuple

from pfl.algorithm.base import FederatedNNAlgorithm, NNAlgorithmParamsType
from pfl.context import CentralContext
from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import ModelHyperParamsType
from pfl.internal.bridge import FrameworkBridgeFactory as bridges
from pfl.metrics import Metrics
from pfl.model.base import StatefulModelType
from pfl.stats import WeightedStatistics

FedAvgCentralContextType = CentralContext[NNAlgorithmParamsType,
                                          ModelHyperParamsType]


class FederatedAveraging(FederatedNNAlgorithm[NNAlgorithmParamsType,
                                              ModelHyperParamsType,
                                              StatefulModelType,
                                              WeightedStatistics,
                                              AbstractDatasetType]):
    """
    Defines the `Federated Averaging <https://arxiv.org/abs/1602.05629>`_
    algorithm by providing the implementation as hooks into the training
    process.
    """

    def process_aggregated_statistics(
            self, central_context: FedAvgCentralContextType,
            aggregate_metrics: Metrics, model: StatefulModelType,
            statistics: WeightedStatistics
    ) -> Tuple[StatefulModelType, Metrics]:
        """
        Average the statistics and update the model.
        """
        statistics.average()
        return model.apply_model_update(statistics)

    def train_one_user(
        self, initial_model_state: WeightedStatistics,
        model: StatefulModelType, user_dataset: AbstractDatasetType,
        central_context: FedAvgCentralContextType
    ) -> Tuple[WeightedStatistics, Metrics]:
        # Local training loop
        bridges.sgd_bridge().do_sgd(model, user_dataset,
                                    central_context.model_train_params)
        training_statistics = model.get_model_difference(initial_model_state)
        # Don't reset model, will be used for evaluation after local training.

        return training_statistics, Metrics()
