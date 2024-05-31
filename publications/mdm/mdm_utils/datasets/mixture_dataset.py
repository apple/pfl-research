from collections import defaultdict
from typing import Callable, Iterable, List, Tuple

import numpy as np
import joblib

from pfl.data import ArtificialFederatedDataset, FederatedDatasetBase
from pfl.data.dataset import AbstractDataset
from pfl.internal.ops.selector import (get_default_framework_module as get_ops)


class ArtificialFederatedDatasetMixture(FederatedDatasetBase):
    """
    A type of federated dataset that is a mixture of multiple
    ArtificialFederatedDataset. To sample a new users we randomly
    sample a component with corresponding probability vector phi,
    we then sample a user from the corresponding
    ArtificialFederatedDataset.
    :param phi:
        A np.ndarray of shape len(mixture_component_datasets),
        probability vector that gives the probability of
        each mixture component
    :param mixture_component_datasets:
        List of type ArtificialFederatedDataset
    """

    def __init__(self, phi, mixture_component_datasets):
        super().__init__()
        self.phi = phi
        self.mixture_component_datasets = mixture_component_datasets

    def __next__(self):
        mixture_component = np.random.choice(range(len(self.phi)), p=self.phi)
        return next(self.mixture_component_datasets[mixture_component])

    def get_cohort(self,
                   cohort_size: int) -> Iterable[Tuple[AbstractDataset, int]]:
        for i in range(cohort_size):
            if (i % get_ops().distributed.world_size
                ) == get_ops().distributed.global_rank:
                yield next(self)

    @classmethod
    def from_slices(cls, phi, data, data_samplers, dataset_len_samplers):
        """
        Construct a mixture of simulated federated datasets from a single
        regular dataset where there is no such thing as a user identifier.
        Each mixture samples a user from the same data but using its
        own data_sampler and dataset_len_sampler.
        :param phi:
            A np.ndarray probability vector
        :param data:
            A list of ``np.ndarray``, i.e. the same format as a ``Dataset``
            accepts.
        :param data_samplers:
            List of callables of length len(phi), each callable is a data
            sampler for an ArtificialFederatedDataset
        :param dataset_len_samplers:
            List of callables of length len(phi), each callable is a data
            length sampler for an ArtificialFederatedDataset
        :returns:
            An instance of `ArtificialFederatedDatasetMixture`.
        """
        mixture_component_datasets = []
        for data_sampler, dataset_len_sampler in zip(data_samplers,
                                                     dataset_len_samplers):
            mixture_component_datasets.append(
                ArtificialFederatedDataset.from_slices(data, data_sampler,
                                                       dataset_len_sampler))
        return cls(phi, mixture_component_datasets)


def partition_by_dirichlet_mixture_class_distribution(
    labels: np.ndarray,
    phi: np.ndarray,
    alphas: np.ndarray,
    user_dataset_len_samplers: List[Callable],
    spread_distribution_after_num_fails: int = 20,
    spread_distribution_after_fails_percentage: float = 0.02
) -> List[List[int]]:
    """
    Partitions central data using a mixture of dirichlet distributions. Works
    the same as partition_by_dirichlet_class_distribution except that it first
    randomly samples a mixture component using probability vector phi, and then
    selects the corresponding alpha and user_dataset_len_sampler.
    """
    num_components = len(phi)
    num_classes = len(np.unique(labels))
    indices_per_class = [
        list(np.where(labels == i)[0]) for i in range(num_classes)
    ]
    users_to_indices = defaultdict(list)

    user_id = 0
    while True:
        component = np.random.choice(num_components, p=phi)
        alpha = alphas[component]
        user_dataset_len_sampler = user_dataset_len_samplers[component]
        class_priors = np.random.dirichlet(alpha=alpha)
        class_prior_cdf = np.cumsum(class_priors)
        user_num_datapoints = user_dataset_len_sampler()
        if user_num_datapoints > sum(
            [len(cidxs) for cidxs in indices_per_class]):
            # Not enough datapoints left.
            break

        i = 1
        while True:
            if len(users_to_indices[user_id]) >= user_num_datapoints:
                user_id += 1
                break
            # Sample class from user's class distribution (Dirichlet)
            sampled_class = np.argmax(np.random.uniform() <= class_prior_cdf)
            if len(indices_per_class[sampled_class]):
                # Add datapoint to user if there are still datapoints
                # available of that class.
                users_to_indices[user_id].append(
                    indices_per_class[sampled_class].pop())
            if i % (user_num_datapoints *
                    spread_distribution_after_num_fails) == 0:
                # Every this number of failed samples,
                # even out the class distribution a tiny bit (at least 2%
                # chance for every class) such that
                # sampling classes with datapoints remaining to be allocated
                # are more probable. This will typically only be an issue for
                # the final few 1-5 users.
                class_priors += spread_distribution_after_fails_percentage
                class_priors /= sum(class_priors)
                class_prior_cdf = np.cumsum(class_priors)
            i += 1
    return list(users_to_indices.values())


def get_user_counts(training_federated_dataset, num_classes,
                    num_central_iterations, cohort_size, save_path):
    """
    Helper function to check the label counts of a cohort of users.
    Can be used to visualize the users generated from different experiments
    over a number of central iterations in train.py.
    """
    print('get_user_counts')
    all_counts = dict()
    for r in range(num_central_iterations):
        all_counts[r + 1] = []
        l = list(training_federated_dataset.get_cohort(cohort_size))
        for d, _ in l:
            _, y = d.raw_data
            y = y.cpu().numpy()
            uniques, counts = np.unique(y, return_counts=True)
            full_y = np.zeros(num_classes)
            full_y[uniques.astype(int)] = counts
            all_counts[r + 1].append(full_y.tolist())

    joblib.dump(all_counts, save_path)
