# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np


def partition_by_dirichlet_class_distribution(
    labels: np.ndarray,
    alpha: float,
    user_dataset_len_sampler: Callable,
    spread_distribution_after_num_fails: int = 20,
    spread_distribution_after_fails_percentage: float = 0.02
) -> List[List[int]]:
    """
    Given an array of labels, create a partitioning representing artificial user
    dataset splits where each user's class distribution is sampled ~Dir(alpha).

    It is common to use `alpha=0.1` (majority of samples for each user will be
    from 1 class). See S.J. Reddi et al. https://arxiv.org/pdf/2003.00295.pdf,
    J. Wang. et al https://arxiv.org/pdf/2007.07481.pdf

    :param labels:
        A one-dimensional array of all labels (integers). Classes should be
        consecutive non-negative integers starting at 0, i.e. from
        {0, 1, ..., num_classes-1}.
    :param alpha:
        The alpha parameter for Dirichlet distribution.
    :param user_dataset_len_sampler:
        A function which samples the dataset length of the user to construct
        next. E.g., use ``lambda: 25`` to sample 25 data points for all users.
    :param spread_distribution_after_num_fails:
        When there are few datapoints left to sample, there might not be any
        datapoints left to sample for particular classes.
        Each (user_num_datapoints * spread_distribution_after_num_fails)
        iterations of sampling each user, even out the class distribution by
        adding `spread_distribution_after_fails_percentage` to each class
        probability and normalize (this will make the distribution less
        heterogeneous and move it closer to the uniform one).
        This will normally only start occurring when there are only a few
        datapoints left to sample from.
    :param spread_distribution_after_fails_percentage:
        See above how this works with ``spread_distribution_after_num_fails``.
        E.g., the default value 0.02 increases each class probability by
        2% and then renormalizes to ensure that the probabilities sum to 1.
        This moves the distribution closer to the uniform one. E.g., for 2
        classes with probabilities [0.1, 0.9] we end up adjusting to
        [0.12, 0.92] first and then normalize to [0.11538462, 0.88461538].
    :returns:
        A list, where each element is a list of ``label`` indices that
        represents one sampled user.
    """
    num_classes = len(np.unique(labels))
    indices_per_class = [
        list(np.where(labels == i)[0]) for i in range(num_classes)
    ]
    users_to_indices: Dict = defaultdict(list)

    user_id = 0
    while True:
        class_priors = np.random.dirichlet(alpha=[alpha] * num_classes)
        class_prior_cdf = np.cumsum(class_priors)
        class_prior_cdf[-1] = 1.0  # force to be 1.0 in case of imprecision
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
            sampled_class = np.argmax(np.random.uniform() < class_prior_cdf)
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
                class_priors[-1] = 1.0
            i += 1
    return list(users_to_indices.values())
