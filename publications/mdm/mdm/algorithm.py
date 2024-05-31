# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Tuple, Optional, TypeVar, Callable, Union

import numpy as np
import torch

from pfl.common_types import Population
from pfl.data.dataset import AbstractDataset
from pfl.hyperparam import get_param_value
from pfl.metrics import Metrics
from pfl.context import CentralContext
from pfl.stats import MappedVectorStatistics
from pfl.algorithm.base import FederatedAlgorithm, AlgorithmHyperParams
from pfl.data.dataset import AbstractDatasetType

from mdm.model import MDMModelType, MDMModelHyperParamsType
from mdm.bridge.factory import FrameworkBridgeFactory as bridges


@dataclass(frozen=True)
class MDMAlgorithmParams(AlgorithmHyperParams):
    """
    Parameters for initialization algorithm of Polya Mixture.

    :param central_num_iterations:
        Number of iterations of training
    :param extract_categories_fn:
        Function to extract categories from user dataset. By default return
        labels of user dataset.
    """
    cohort_size: int
    num_samples_mixture_bins: np.ndarray
    central_num_iterations: int = 1
    extract_categories_fn: Callable[[AbstractDataset], Union[
        np.ndarray,
        torch.Tensor]] = lambda user_dataset: user_dataset.raw_data[1]


MDMAlgorithmParamsType = TypeVar('MDMAlgorithmParamsType',
                                 bound=MDMAlgorithmParams)


class MDMAlgorithm(FederatedAlgorithm[MDMAlgorithmParamsType,
                                      MDMModelHyperParamsType, MDMModelType,
                                      MappedVectorStatistics, AbstractDatasetType]):
    """
    Federated algorithm class for learning mixture of Polya
    (Dirichlet-Multinomial) distribution using MLE algorithm.
    """

    def get_next_central_contexts(
        self,
        model: MDMModelType,
        iteration: int,
        algorithm_params: MDMAlgorithmParamsType,
        model_train_params: MDMModelHyperParamsType,
        model_eval_params: Optional[MDMModelHyperParamsType] = None,
    ) -> Tuple[Optional[Tuple[CentralContext[MDMAlgorithmParamsType,
                                             MDMModelHyperParamsType], ...]],
               MDMModelType, Metrics]:

        if (model.alphas <= 0).any():
            raise AssertionError(
                f'Cannot have zero elements in alphas: {model.alphas}')
        if (model.num_samples_distribution <= 0).any():
            raise AssertionError(
                f'Cannot have zero elements in num_samples_distribution: {model.num_samples_distribution}'
            )
        if (model.phi <= 0).any():
            raise AssertionError(
                f'Cannot have zero elements in phi: {model.phi}')

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
        central_context: CentralContext[MDMAlgorithmParamsType,
                                        MDMModelHyperParamsType],
    ) -> Tuple[Optional[MappedVectorStatistics], Metrics]:
        """
        Encode user's dataset into statistics with a `MDMModel`.

        central_context.algorithm_params.extract_categories_fn is a callable
        used to extract the categories tracked with the Polya-Mixture model.
        """

        if (model.alphas <= 0).any():
            raise AssertionError(
                '> 0 alpha params have zero values, which would cause algorithm to fail. Cannot proceed.'
            )

        if (model.phi <= 0).any():
            raise AssertionError(
                '> 0 phi params have zero values, which would cause algorithm to fail. Cannot proceed.'
            )

        if (model.num_samples_distribution <= 0).any():
            raise AssertionError(
                '> 0 num_samples_distribution have zero values, which would cause algorithm to fail. Cannot proceed.'
            )

        # Get counts
        categories = central_context.algorithm_params.extract_categories_fn(
            user_dataset)
        #num_user_samples = len(categories)
        category_counts = bridges.polya_mixture_bridge(
        ).category_counts_polya_mixture(
            categories, central_context.model_train_params.num_categories)

        e = torch.zeros(
            (central_context.model_train_params.num_components,
             central_context.algorithm_params.num_samples_mixture_bins.shape[1]
             ))

        num_samples = len(categories)

        # TODO make more flexible so it supports if each mixture has different bin edges.
        # TODO here I just assume that all components have the same bin edges, which is why I index 0 in central_context.algorithm_params.num_samples_mixture_bins[0]
        selected_bin = -1
        for i, bin_edge in enumerate(
                central_context.algorithm_params.num_samples_mixture_bins[0]):
            bin_edge = int(bin_edge)
            if num_samples <= bin_edge:
                selected_bin = i
                break

        user_num_samples_distribution = model.num_samples_distribution[:,
                                                                       selected_bin]

        # E Step - compute posterior probability of each component
        posterior_probabilities = bridges.polya_mixture_bridge(
        ).expectation_step(model.phi, model.alphas,
                           user_num_samples_distribution, category_counts)

        # M Step - compute client update to alphas for fixed point update
        # which will be applied by the model in process_aggregated_statistics.
        # Note the numerator and denominator are both weighted by w (the
        # probability vector giving the client belonging to each component).
        (numerator,
         denominator) = bridges.polya_mixture_bridge().maximization_step(
             posterior_probabilities, category_counts, model.alphas)

        e[:, selected_bin] = posterior_probabilities.view(-1)

        statistics = MappedVectorStatistics()
        statistics['posterior_probabilities'] = posterior_probabilities.to('cpu')
        statistics['numerator'] = numerator.to('cpu')
        statistics['denominator'] = denominator.to('cpu')
        statistics['num_samples_distribution'] = e.to('cpu')

        return statistics, Metrics()

    def process_aggregated_statistics(
            self, central_context: CentralContext[MDMAlgorithmParamsType,
                                                  MDMModelHyperParamsType],
            aggregate_metrics: Metrics, model: MDMModelType,
            statistics: MappedVectorStatistics
    ) -> Tuple[MDMModelType, Metrics]:

        # The new weight of a mixture component is the mean client weight of
        # that component

        # TODO prevent any <= 0 values in posterior_probabilities, numerator and denominator and num_samples_distribution
        posterior_probabilities = statistics['posterior_probabilities']
        numerator = statistics['numerator']
        denominator = statistics['denominator']
        num_samples_distribution = statistics['num_samples_distribution']

        posterior_probabilities = torch.clamp(posterior_probabilities, min=0)
        numerator = torch.clamp(numerator, min=0)
        denominator = torch.clamp(denominator, min=0)
        num_samples_distribution = torch.clamp(num_samples_distribution, min=0)

        print('\n\nProcess aggregated statistics')
        print('numerator', numerator.shape, numerator)
        print('denominator', denominator.shape, denominator)
        print('num_samples_distribution', num_samples_distribution.shape,
              num_samples_distribution)

        def prevent_zero(tensor, min_val=0, mass_reallocation_percentage=0.01):
            if not (tensor == 0).any():
                print('no zeros in tensor')
                return tensor
            num_zero = torch.sum(tensor <= min_val, dim=1, keepdim=True)
            total_mass = torch.sum(tensor, dim=1, keepdim=True)
            extra_mass = total_mass * mass_reallocation_percentage / num_zero
            tensor = torch.where(tensor > min_val, tensor,
                                 extra_mass.expand_as(tensor))
            tensor = tensor / torch.sum(tensor, dim=1,
                                        keepdim=True) * total_mass
            return tensor

        if (posterior_probabilities == 0).any():
            num_zero = torch.sum(posterior_probabilities == 0)
            total_mass = torch.sum(posterior_probabilities)
            extra_mass = total_mass * 0.01 / num_zero
            posterior_probabilities = torch.where(posterior_probabilities > 0,
                                                  posterior_probabilities,
                                                  extra_mass)
            posterior_probabilities = posterior_probabilities / torch.sum(
                posterior_probabilities) * total_mass
            assert total_mass == torch.sum(posterior_probabilities)

        if (numerator == 0).any():
            numerator = prevent_zero(numerator, min_val=1)
            if (numerator == 0).any():
                raise AssertionError('prevent zero did not work on numerator')

        if (denominator == 0).any():
            denominator = prevent_zero(denominator)
            if (denominator == 0).any():
                raise AssertionError(
                    'prevent zero did not work on denominator')

        if (num_samples_distribution == 0).any():
            modified_num_samples_distribution = prevent_zero(
                num_samples_distribution, min_val=0.001)
            mass_reallocation_percentage = 0.01
            while (modified_num_samples_distribution == 0).any():
                mass_reallocation_percentage *= 2
                if mass_reallocation_percentage >= 1:
                    raise AssertionError(
                        f'prevent zero did not work on num_samples_distribution: {num_samples_distribution}'
                    )
                modified_num_samples_distribution = prevent_zero(
                    num_samples_distribution,
                    min_val=0.001,
                    mass_reallocation_percentage=mass_reallocation_percentage)
            num_samples_distribution = modified_num_samples_distribution

        phi = posterior_probabilities / central_context.algorithm_params.cohort_size

        # Each alpha is updated using the fixed point update, note that the
        # numerator and denominator are weighted by each client before being
        # aggregated, so this is a weighted update.

        if (model.alphas == 0).any():
            raise AssertionError('model.alphas had zeros before update')
        alphas = bridges.polya_mixture_bridge().update_alpha(
            model.alphas, numerator, denominator)
        if (alphas == 0).any():
            raise AssertionError('alphas has zeros after update')

        num_samples_distribution = num_samples_distribution / posterior_probabilities.reshape(
            -1, 1).expand_as(num_samples_distribution)

        # renormalise num_samples_distribution again on server, since DP might mean that statistics don't sum to 1 per mixture component
        num_samples_distribution = num_samples_distribution / num_samples_distribution.sum(
            dim=1, keepdim=True)
        if (num_samples_distribution == 0).any():
            num_samples_distribution = prevent_zero(num_samples_distribution,
                                                    min_val=0.001)

        # renormalise phi
        phi = phi / phi.sum()

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
