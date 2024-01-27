# Copyright © 2023-2024 Apple Inc.
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from pfl.algorithm.base import FederatedNNAlgorithm, NNAlgorithmParams
from pfl.context import CentralContext
from pfl.data.dataset import AbstractDatasetType
from pfl.data.user_state import AbstractUserStateStorage
from pfl.exception import UserNotFoundError
from pfl.hyperparam.base import ModelHyperParamsType
from pfl.internal.bridge import FrameworkBridgeFactory as bridges
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.model.base import StatefulModel, StatefulModelType
from pfl.stats import MappedVectorStatistics, WeightedStatistics


@dataclass(frozen=True)
class SCAFFOLDParams(NNAlgorithmParams):
    """
    Parameters for SCAFFOLD algorithm, introduced by
    S. P. Karimireddy et al. - SCAFFOLD: Stochastic Controlled Averaging
    for Federated Learning,
    (https://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf).

    :param population:
        The total estimated population. $N$ in Algorithm 1.
    :param use_gradient_as_control_variate:
        If ``True``, use Option I (gradient of server model) for calculating
        $c_i$. If ``False``, use Option II for calculating $c_i$.
        Option II is faster and the paper claims it is stable enough to be
        used in all their experiments, but it can be too unstable for other
        datasets (e.g. CIFAR10!).
    :param user_state_storage:
        A storage instance to load and save user states.
        Will be used by SCAFFOLD to save and retrieve $c_i$ in Algorithm 1.
    """
    population: int
    use_gradient_as_control_variate: bool
    user_state_storage: AbstractUserStateStorage


SCAFFOLDCentralContextType = CentralContext[SCAFFOLDParams,
                                            ModelHyperParamsType]


class SCAFFOLD(FederatedNNAlgorithm[SCAFFOLDParams, ModelHyperParamsType,
                                    StatefulModelType, MappedVectorStatistics,
                                    AbstractDatasetType]):
    """
    SCAFFOLD algorithm, introduced by
    S. P. Karimireddy et al. - SCAFFOLD: Stochastic Controlled Averaging
    for Federated Learning,
    (https://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf).

    """

    def get_next_central_contexts(
        self,
        model: StatefulModelType,
        iteration: int,
        algorithm_params: SCAFFOLDParams,
        model_train_params: ModelHyperParamsType,
        model_eval_params: Optional[ModelHyperParamsType] = None,
    ) -> Tuple[Optional[Tuple[SCAFFOLDCentralContextType, ...]],
               StatefulModelType, Metrics]:
        if iteration == 0:
            # initial state all zeros.
            self._server_c = model.get_parameters().apply_elementwise(
                lambda w: w * 0)

        return super().get_next_central_contexts(
            model=model,
            iteration=iteration,
            algorithm_params=algorithm_params,
            model_train_params=model_train_params,
            model_eval_params=model_eval_params)

    def process_aggregated_statistics(
        self, central_context: SCAFFOLDCentralContextType,
        aggregate_metrics: Metrics, model: StatefulModelType,
        statistics: MappedVectorStatistics
    ) -> Tuple[StatefulModelType, Metrics]:
        """
        Average the statistics and update the model.
        """
        statistics.average()
        # Extract aggregated c's from stats.
        self._server_c += MappedVectorStatistics({
            k.split('/')[1]: (v * central_context.cohort_size /
                              central_context.algorithm_params.population)
            for k, v in statistics.items() if k.startswith('c/')
        })

        model_update = MappedVectorStatistics({
            k.split('/')[1]: v
            for k, v in statistics.items() if k.startswith('model_update/')
        })
        new_model, metrics = model.apply_model_update(model_update)

        server_c_norm = get_ops().global_norm(self._server_c.get_weights()[1],
                                              order=2)
        metrics[StringMetricName('server_c norm')] = Weighted.from_unweighted(
            server_c_norm)
        return new_model, metrics

    def train_one_user(
        self, initial_model_state: WeightedStatistics, model: StatefulModel,
        user_dataset: AbstractDatasetType, central_context: CentralContext
    ) -> Tuple[MappedVectorStatistics, Metrics]:
        assert isinstance(central_context.algorithm_params, SCAFFOLDParams)
        assert user_dataset.user_id is not None, (
            'SCAFFOLD requires datasets to have the `user_id` property set')
        # Local training loop
        model_params = central_context.model_train_params
        assert model_params.local_num_steps is None, (
            'Specifying local_num_steps is not supported yet for SCAFFOLD')
        storage = central_context.algorithm_params.user_state_storage
        try:
            local_c = storage.load_state(str(user_dataset.user_id), 'local_c')
        except UserNotFoundError:
            # initial state all zeros.
            local_c = model.get_parameters().apply_elementwise(lambda w: w * 0)

        if central_context.algorithm_params.use_gradient_as_control_variate:
            bridges.sgd_bridge().do_sgd(
                model, user_dataset,
                model_params.static_clone(local_learning_rate=1.0,
                                          local_batch_size=None,
                                          local_num_epochs=None,
                                          local_num_steps=1))
            local_c_plus = model.get_model_difference(initial_model_state)
            model.set_parameters(initial_model_state)

        bridges.scaffold_bridge().do_control_variate_sgd(
            model,
            user_dataset,
            model_params,
            local_c=local_c,
            server_c=self._server_c)
        # Make sure this model diff is a clone if method
        # already used previously for local_c_plus.
        model_diff = model.get_model_difference(
            initial_model_state,
            clone=central_context.algorithm_params.
            use_gradient_as_control_variate)

        if not central_context.algorithm_params.use_gradient_as_control_variate:
            local_c_plus = local_c
            local_c_plus += self._server_c.apply_elementwise(lambda w: -1 * w)
            data_size = len(user_dataset)
            batch_size = (data_size if model_params.local_batch_size is None
                          else model_params.local_batch_size)
            K = model_params.local_num_epochs * math.ceil(
                data_size / batch_size)
            local_c_plus += model_diff.apply_elementwise(
                lambda w: w / (K * model_params.local_learning_rate))

        storage.save_state(local_c_plus, str(user_dataset.user_id), 'local_c')

        # Combine model update and local_c diff into one payload.
        user_payload = MappedVectorStatistics({
            f'model_update/{k}': v
            for k, v in model_diff.items()
        })
        for name, local_c_weight in local_c.items():
            user_payload[f'c/{name}'] = local_c_plus[name] - local_c_weight

        # Don't reset model, will be used for evaluation after local training.
        metrics = Metrics([('scaffold', 1.0)])
        return user_payload, metrics
