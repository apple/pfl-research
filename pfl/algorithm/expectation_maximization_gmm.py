# Copyright © 2023-2024 Apple Inc.
import logging
import math
import typing
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from pfl.algorithm.base import FederatedAlgorithm
from pfl.common_types import Population
from pfl.context import CentralContext
from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam import AlgorithmHyperParams
from pfl.metrics import MetricName, Metrics, StringMetricName, Weighted
from pfl.model.gaussian_mixture_model import GaussianMixtureModel, GMMHyperParams
from pfl.stats import MappedVectorStatistics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EMGMMHyperParams(AlgorithmHyperParams):
    """
    Parameters for EM GMM algorithms.

    :param central_num_iterations:
        Total number of central iterations.
    :param evaluation_frequency:
        Frequency with which the model will be evaluated (in terms
        of central iterations).
    :param val_cohort_size:
        Cohort size for evaluation on validation users.
    :param compute_cohort_size:
        Function ``(int, int) -> int`` that receives
        ``(iteration, num_components)`` and returns the desired cohort size for
        that iteration.
    :param compute_new_num_components:
        Function ``(int, int, int) -> int`` that computes the desired number
        of components.
        This is passed
        ``(iteration, num_iterations_since_last_mix_up, num_components)``.
        It is called after the iteration.
        The number of components that it returns must be at least
        ``num_components``.
        If the return value is greater, components with the greatest weight
        will be split, so that they can differentiate themselves in the next
        round of training.
        If this parameter is set to ``None``, then the number of components will
        stay constant.
        It is often easy to generate this function using
        :func:`make_compute_new_num_components`.
    """
    central_num_iterations: int
    evaluation_frequency: int
    val_cohort_size: Optional[int]
    compute_cohort_size: Callable[[int, int], int]
    compute_new_num_components: Callable[[int, int, int], int]


def make_compute_new_num_components(
        num_initial_iterations: int,
        mix_up_interval: int,
        max_num_components: Optional[int] = None,
        step_components: int = 0,
        fraction_new_components: float = 0.) -> Callable[[int, int, int], int]:
    """
    Make a function to compute the desired number of components to generate.
    This can be passed to :class:`GMMHyperParams` as
    ``compute_new_num_components``.

    It is often useful in training GMMs to increase the number of components
    incrementally during training ("mixing up").
    This is done by splitting up components every few iterations.
    During the iterations in between, the components can settle.

    :param num_initial_iterations:
        The number of iterations to wait until mixing up starts.
    :param mix_up_interval:
        The number of iterations to wait between increasing the number of
        components.
    :param max_num_components:
        The number of components after which no more components should be added.
        If `None`, then there is no limit.
    :param step_components:
        The number of components to add when mixing up.
        If both ``fraction_new_components`` and this are given, the max of the
        two results is used.
    :param fraction_new_components:
        The fraction of current components to add when mixing up.
        To compute the actual number, this value is rounded down.
        It is usually useful to supply non-zero ``step_components`` to, so that
        even when the current number is small, mixing up happens.
    """

    assert mix_up_interval >= 1

    def compute_new_num_components(iteration, num_iterations_since_last_mix_up,
                                   current_num_components):
        if (iteration >= num_initial_iterations
                and num_iterations_since_last_mix_up >= mix_up_interval):
            proposed_num_components = max(
                # Increase by "step_components".
                current_num_components + step_components,
                # Increase by fraction of "fraction_new_components".
                math.floor(current_num_components *
                           (1 + fraction_new_components)))
            if max_num_components is None:
                return proposed_num_components
            else:
                return min(max_num_components, proposed_num_components)

        return current_num_components

    return compute_new_num_components


class ExpectationMaximizationGMM(FederatedAlgorithm[EMGMMHyperParams,
                                                    GMMHyperParams,
                                                    GaussianMixtureModel,
                                                    MappedVectorStatistics,
                                                    AbstractDatasetType]):
    """
    Train a :class:`~pfl.model.gaussian_mixture_model.GaussianMixtureModel`
    with expectation--maximization.
    for private federated learning, this uses sufficient statistic perturbation.
    """

    def __init__(self):
        super().__init__()
        self._num_iterations_since_last_mix_up = 0

    def get_next_central_contexts(
        self,
        model: GaussianMixtureModel,
        iteration: int,
        algorithm_params: EMGMMHyperParams,
        model_train_params: GMMHyperParams,
        model_eval_params: Optional[GMMHyperParams] = None,
    ) -> Tuple[Optional[Tuple[CentralContext, ...]], GaussianMixtureModel,
               Metrics]:
        if iteration == algorithm_params.central_num_iterations:
            return None, model, Metrics()

        if iteration == 0:
            self._num_iterations_since_last_mix_up = 0

        do_evaluation = iteration % algorithm_params.evaluation_frequency == 0
        num_components = len(model.components)

        # Mix up if requested.
        new_num_components = algorithm_params.compute_new_num_components(
            iteration, self._num_iterations_since_last_mix_up, num_components)
        if new_num_components != num_components:
            assert new_num_components > num_components
            model = model.mix_up(new_num_components - num_components)
            self._num_iterations_since_last_mix_up = 0
            # Note that the number of components can be less than
            # new_num_components, e.g. if there are not enough components to
            # split.
            num_components = len(model.components)

        self._num_iterations_since_last_mix_up += 1

        # Get aggregated model updates and metrics from training and
        # evaluation.
        assert model_eval_params is not None
        train_cohort_size = algorithm_params.compute_cohort_size(
            iteration, num_components)
        static_model_train_params = model_train_params.static_clone()
        static_model_eval_params = model_eval_params.static_clone()

        configs = [
            CentralContext(current_central_iteration=iteration,
                           do_evaluation=do_evaluation,
                           cohort_size=train_cohort_size,
                           population=Population.TRAIN,
                           model_train_params=static_model_train_params,
                           model_eval_params=static_model_eval_params,
                           algorithm_params=algorithm_params.static_clone(),
                           seed=self._get_seed())
        ]
        if do_evaluation and algorithm_params.val_cohort_size is not None:
            configs.append(
                CentralContext(
                    current_central_iteration=iteration,
                    do_evaluation=do_evaluation,
                    cohort_size=algorithm_params.val_cohort_size,
                    population=Population.VAL,
                    algorithm_params=algorithm_params.static_clone(),
                    model_train_params=static_model_train_params,
                    model_eval_params=static_model_eval_params,
                    seed=self._get_seed()))
        return tuple(configs), model, Metrics([
            (StringMetricName('Number of components'), num_components)
        ])

    def process_aggregated_statistics(
        self, central_context: CentralContext, aggregate_metrics: Metrics,
        model: GaussianMixtureModel, statistics: MappedVectorStatistics
    ) -> Tuple[GaussianMixtureModel, Metrics]:
        return model.apply_model_update(statistics)

    def simulate_one_user(
        self, model: GaussianMixtureModel, user_dataset: AbstractDatasetType,
        central_context: CentralContext
    ) -> Tuple[Optional[MappedVectorStatistics], Metrics]:
        model_train_params = typing.cast(GMMHyperParams,
                                         central_context.model_train_params)

        def name_formatting_fn(s: str):
            # pytype: disable=duplicate-keyword-argument,wrong-arg-count
            return MetricName(s, central_context.population)
            # pytype: enable=duplicate-keyword-argument,wrong-arg-count

        if central_context.population == Population.TRAIN:
            likelihood, statistics = model.get_mixture_statistics(
                user_dataset.raw_data, model_train_params.variance_scale,
                model_train_params.responsibility_scale)
            metrics = Metrics([
                (name_formatting_fn('log-likelihood before training'),
                 Weighted(likelihood.log_value, len(user_dataset.raw_data)))
            ])

            return statistics, metrics
        else:
            metrics = model.evaluate(user_dataset, name_formatting_fn,
                                     central_context.model_eval_params)
            return None, metrics
