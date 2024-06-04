# -*- coding: utf-8 -*-

from typing import Tuple
import torch

from ..base import PolyaMixtureFrameworkBridge


class PyTorchPolyaMixtureBridge(PolyaMixtureFrameworkBridge[torch.Tensor]):
    """
    Interface for Polya-Mixture algorithm in PyTorch.
    """

    @staticmethod
    def category_counts_polya_mixture(categories: torch.Tensor,
                                      num_categories: int) -> torch.Tensor:
        categories = categories.to('cpu')
        uniques, counts = torch.unique(categories, return_counts=True)
        category_counts = torch.zeros(num_categories).int()
        category_counts[uniques.int().to(torch.int64)] = counts.int()
        return category_counts

    @staticmethod
    def category_probabilities_polya_mixture_initialization(
            num_components, num_categories, component,
            categories) -> torch.Tensor:
        # Compute counts of each class and normalize to probability vectors
        uniques, counts = torch.unique(categories, return_counts=True)
        counts = counts.to('cpu')
        uniques = uniques.to('cpu')
        p = torch.zeros((num_components, num_categories))
        p = p.to('cpu')
        p[component][uniques.int().to(torch.int64)] = counts.float(
        )  # User contributes only to estimate for their mixture component
        p[component] /= torch.sum(p[component])

        return p

    @staticmethod
    def expectation_step(phi, alphas, num_samples_distribution,
                         category_counts) -> torch.Tensor:
        if (num_samples_distribution == 0).any():
            raise AssertionError('num_samples_distribution contains zero values, which cannot work with expectation step on clients')

        # E Step - compute posterior probability of each component
        # Compute log prior + log likelihood
        # TODO log_v might be missing + torch.lgamma(torch.sum(counts)+1) - torch.sum(torch.lgamma(category_counts+1), dim=1, keepdim=False) 
        phi = torch.Tensor(phi).to('cpu')
        alphas = torch.Tensor(alphas).to('cpu')
        category_counts = category_counts.to('cpu')
        num_samples_distribution = num_samples_distribution.to('cpu')
        log_v = torch.log(phi) + (
            torch.lgamma(torch.sum(alphas, dim=1, keepdim=False)) -
            torch.lgamma(
                torch.sum(category_counts + alphas, dim=1, keepdim=False)) +
            torch.sum(
                torch.lgamma(category_counts + alphas) - torch.lgamma(alphas),
                dim=1,
                keepdim=False)) + torch.log(num_samples_distribution) 

        # TODO Ignore this as log(0) => NaN
        # TODO fix this equation so that it works with num_samples_distribution = 0
        # + torch.log(num_samples_distribution[:, num_user_samples])

        # Compute log probability of the data, computed like this for numerical stability
        # computes sum(v)
        log_normalization_constant = log_v[0] + torch.log(
            torch.sum(torch.exp(log_v - log_v[0])))
        #print('log_normalization_constant', log_normalization_constant)

        # Compute posterior probability
        w = torch.exp(log_v - log_normalization_constant)

        return w

    @staticmethod
    def maximization_step(posterior_probabilities, category_counts,
                             alphas) -> torch.Tensor:
        # M Step - compute client update to alphas for fixed point update
        # which will be applied by the model in process_aggregated_statistics.
        # Note the numerator and denominator are both weighted by w (the
        # probability vector giving the client belonging to each component).
        posterior_probabilities = torch.Tensor(posterior_probabilities).to('cpu')
        category_counts = torch.Tensor(category_counts).to('cpu')
        alphas = torch.Tensor(alphas).to('cpu')
        numerator = posterior_probabilities.reshape(
            (-1, 1)) * (torch.digamma(category_counts + alphas) -
                        torch.digamma(alphas))
        # Paper currently says something different, where alphas should all be summed first.
        denominator = posterior_probabilities.reshape(
            (-1, 1)) * (torch.digamma(
                torch.sum(category_counts + alphas, dim=1, keepdim=True)) -
                        torch.digamma(torch.sum(alphas, dim=1, keepdim=True)))

        return numerator, denominator

    @staticmethod
    def update_alpha(alphas, numerator, denominator) -> torch.Tensor:
        alphas = alphas.to('cpu')
        numerator = numerator.to('cpu')
        denominator = denominator.to('cpu')
        return torch.Tensor(alphas) * numerator / denominator
