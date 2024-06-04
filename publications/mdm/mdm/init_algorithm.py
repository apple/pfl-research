from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, TypeVar, Union

import numpy as np
import torch

from pfl.algorithm.base import AlgorithmHyperParams, FederatedAlgorithm
from pfl.common_types import Population
from pfl.context import CentralContext
from pfl.data.dataset import AbstractDataset, AbstractDatasetType
from pfl.hyperparam import get_param_value
from pfl.internal.ops import get_ops
from pfl.metrics import Metrics
from pfl.stats import MappedVectorStatistics
from publications.mdm.mdm.bridge.factory import FrameworkBridgeFactory as bridges
from publications.mdm.mdm.model import MDMModelHyperParamsType, MDMModelType


@dataclass(frozen=True)
class MDMInitializationAlgorithmParams(AlgorithmHyperParams):
    """
    Parameters for initialization algorithm of Polya Mixture.

    :param strategy:
        Strategy for a user to decide which component to contribute
        to for initialization. Currently only 'random' implemented.
    :param central_num_iterations:
        Number of iterations to perform in algorithm.
    :param cohort_size:
        Number of users over which to aggregate statistics in each
        iteration.
    """
    cohort_size: int
    num_samples_mixture_bins: np.ndarray
    strategy: str = 'random'
    central_num_iterations: int = 1
    extract_categories_fn: Callable[[AbstractDataset], Union[
        np.ndarray,
        torch.Tensor]] = lambda user_dataset: user_dataset.raw_data[1]

    def __post_init__(self):
        assert self.strategy in [
            'random'
        ], (f'strategy {self.strategy} is not supported')
        assert self.central_num_iterations >= 1
        assert self.cohort_size > 0
        assert np.all(self.num_samples_mixture_bins > 0)


MDMInitializationAlgorithmParamsType = TypeVar(
    'MDMInitializationAlgorithmParamsType',
    bound=MDMInitializationAlgorithmParams)


class MDMInitializationAlgorithm(
        FederatedAlgorithm[MDMInitializationAlgorithmParamsType,
                           MDMModelHyperParamsType, MDMModelType,
                           MappedVectorStatistics, AbstractDatasetType]):
    """
    Federated algorithm class for learning initialization of mixture of Polya
    (Dirichlet-Multinomial) distribution.
    """

    def __init__(self, statistics_dir: Optional[str] = None):
        super().__init__()
        self._running_sums = defaultdict(int)

    def get_next_central_contexts(
        self,
        model: MDMModelType,
        iteration: int,
        algorithm_params: MDMInitializationAlgorithmParamsType,
        model_train_params: MDMModelHyperParamsType,
        model_eval_params: Optional[MDMModelHyperParamsType] = None,
    ) -> Tuple[Optional[Tuple[CentralContext[
            MDMInitializationAlgorithmParamsType, MDMModelHyperParamsType],
                              ...]], MDMModelType, Metrics]:

        if iteration == algorithm_params.central_num_iterations:
            return None, model, Metrics()

        configs = [
            CentralContext(
                current_central_iteration=iteration,
                do_evaluation=False,
                cohort_size=get_param_value(algorithm_params.cohort_size),
                population=Population.TRAIN,
                model_train_params=model_train_params.static_clone(),
                model_eval_params=None,
                algorithm_params=algorithm_params.static_clone(),
                seed=self._get_seed())
        ]
        return tuple(configs), model, Metrics()

    def simulate_one_user(
        self,
        model: MDMModelType,
        user_dataset: AbstractDataset,
        central_context: CentralContext[MDMInitializationAlgorithmParamsType,
                                        MDMModelHyperParamsType],
    ) -> Tuple[Optional[MappedVectorStatistics], Metrics]:
        """
        Encode user's dataset into statistics with a `MDMModel`.
        """
        algorithm_params = central_context.algorithm_params

        if algorithm_params.strategy == 'random':
            # Randomly assign user to a mixture component
            # TODO this approach might not work when num mixture > 1 and the
            # cohort size is large, as the p and q values will likely be very
            # similar for all clusters and this symmetry could hurt convergence.
            component = np.random.choice(
                range(central_context.model_train_params.num_components))
        else:
            raise ValueError(
                f'Strategy {algorithm_params.strategy} not recognized.'
                'Only "random" strategy is implemented.')

        # Compute counts of each class and normalize to probability vectors
        # User contributes only to estimate for their mixture component
        categories = central_context.algorithm_params.extract_categories_fn(
            user_dataset)
        p = bridges.polya_mixture_bridge(
        ).category_probabilities_polya_mixture_initialization(
            central_context.model_train_params.num_components,
            central_context.model_train_params.num_categories, component,
            categories)
        q = p**2

        # Record user mixture component
        # component sizes are needed for initialization computation
        e = torch.zeros(
            (central_context.model_train_params.num_components,
             central_context.algorithm_params.num_samples_mixture_bins.shape[1]
             ))

        num_samples = len(categories)

        selected_bin = -1
        for i, bin_edge in enumerate(central_context.algorithm_params.
                                     num_samples_mixture_bins[component]):
            bin_edge = int(bin_edge)
            if num_samples <= bin_edge:
                selected_bin = i
                break

        e[component, selected_bin] = 1

        statistics = MappedVectorStatistics()
        statistics['p'] = p.to('cpu')
        statistics['q'] = q.to('cpu')
        statistics['e'] = e.to('cpu')
        return statistics, Metrics()

    def process_aggregated_statistics(
            self, central_context: CentralContext[
                MDMInitializationAlgorithmParamsType,
                MDMModelHyperParamsType], aggregate_metrics: Metrics,
            model: MDMModelType, statistics: MappedVectorStatistics
    ) -> Tuple[MDMModelType, Metrics]:

        # Directly aggregate running sum of statistics
        # only relevant if num_central_iterations > 1
        for key, val in statistics.items():
            self._running_sums[key] += val

        if (central_context.current_central_iteration ==
                central_context.algorithm_params.central_num_iterations - 1):

            num_components = central_context.model_train_params.num_components

            p = self._running_sums['p']
            q = self._running_sums['q']
            e = self._running_sums['e']

            print('init aggregated statistics')
            print('p', p)
            print('q', q)
            print('e', e)

            # Set all elements < 0 to 0
            # This may arise due to differential privacy noise
            p = torch.clamp(p, min=0)
            q = torch.clamp(q, min=0)
            e = torch.clamp(e, min=0)

            # Need to prevent p, q, and subsequently alpha, and num_samples_distribution from having no non-zero values.
            # Fix issue of alpha = 0 by assigning some small prob to all categories
            # Cannot have alpha = 0 for any of the categories.
            # In practice, we might get alpha = 0 if the cohort size used for the initialisation step was too small,
            # such that we did not see any instances of this category occuring in the population.
            # In practice, one should go over the entire population to find the probability of each category.
            # Fix this issue by apportioning a small amount of the probability to that value of alpha.

            # Check if any element is equal to zero
            if (p == 0).any():
                num_categories_leq_zero = torch.sum(p <= 0, dim=1)
                extra_mass = torch.sum(p,
                                       dim=1) * 0.01 / num_categories_leq_zero

                p = torch.where(p > 0, p, extra_mass.unsqueeze(1).expand_as(p))
                # Note fix zero issue in p and q separately because q >= p^2

            # TODO Consider approximating p as p/(cohort size/num_components),
            # since users are randomly assigned to components
            # Similarly can approximate q as q/(cohort_size/num_components).
            # This might make the results more accurate, since DP noise will
            # be added to e, component_sums will be noisy.
            #component_sums = torch.sum(e, dim=1, keepdim=True)
            num_users_component = central_context.algorithm_params.cohort_size / num_components
            p = torch.divide(p, num_users_component)

            if (q == 0).any():
                q[q == 0] = torch.pow(p[q == 0], 2) * 1.1
            q = torch.divide(q, num_users_component)

            if (e == 0).any():
                num_zero = torch.sum(e == 0, dim=1, keepdim=True)
                extra_mass = torch.sum(e, dim=1,
                                       keepdim=True) * 0.01 / num_zero
                e = torch.where(e > 0, e, extra_mass.expand_as(e))

            num_samples_distribution = torch.divide(e, num_users_component)

            # Compute alpha that matches the first two moments
            # of the empirical distribution

            # Use either category with max probability or else average/median over all categories
            init_coefficient = (p - q) / (q - torch.pow(p, 2))

            k_pmax = torch.argmax(p, dim=1)
            k_pmax_coefficient = init_coefficient[
                torch.arange(init_coefficient.size(0)), k_pmax]

            moment_matching_alpha_k_pmax = p * k_pmax_coefficient.reshape(
                -1, 1)

            alphas = moment_matching_alpha_k_pmax

            # if only two categories and one component
            #alpha_0 = 0.5 * (1 - p[0,1]) / (1 - p[0,0] - p[0,1])
            #alpha_1 = 0.5 * (1 - p[0,0]) / (1 - p[0,0] - p[0,1])
            #alphas = [torch.Tensor([alpha_0, alpha_1])]

            phi = 1 / num_components * torch.ones(num_components)

            model, metrics = model.apply_model_update(
                MappedVectorStatistics({
                    'alphas':
                    alphas,
                    'phi':
                    phi,
                    'num_samples_distribution':
                    num_samples_distribution
                }))
            metrics['alphas'] = alphas
            metrics['phi'] = phi
            metrics['num_samples_distribution'] = num_samples_distribution
            return model, metrics
        else:
            return model, Metrics()
