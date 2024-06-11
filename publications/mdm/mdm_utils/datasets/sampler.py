import itertools

import numpy as np


class DirichletDataSampler:
    """
    Data sampling mechanism that samples user class proportions from a
    Dirichlet distribution with a given alpha parameter.
    Sampling is done by first drawing a vector of class proportions
    p ~ Dir(alpha), then sampling a class from a categorical
    distribution with parameter p and uniformly at random choosing
    (with replacement) an index with the corresponding class.

    :param alpha:
        Parameter of the Dirichlet distribution. Must be array_like and
        have length equal to the number of unique classes present in labels.
    :param labels:
        A one-dimensional array of all labels (integers). This should have
        length equal to the size of the corresponding dataset.
    """

    def __init__(self, alpha: np.ndarray, labels: np.ndarray):
        self.unique_labels = np.unique(labels)
        assert len(alpha) == len(
            self.unique_labels
        ), "Number of classes doesn't equal dirichlet parameter dimension."
        self.indices_per_class = {
            i: np.nonzero(labels == i)[0]
            for i in self.unique_labels
        }
        self.alpha = alpha

    def __call__(self, n: int):
        """
        Sample a list of datapoint indices.
        :param n:
            Number of samples to draw.
        :returns:
            Sampled indices in range '[0, len(labels)]'
        """
        class_priors = np.random.dirichlet(alpha=self.alpha)
        sampled_class_counts = np.random.multinomial(n, pvals=class_priors)
        sampled_indices = [
            list(
                np.random.choice(self.indices_per_class[i],
                                 size=class_count,
                                 replace=True))
            for i, class_count in zip(self.unique_labels, sampled_class_counts)
        ]
        return sum(sampled_indices, [])


class DirichletMixtureDataSampler:

    def __init__(self, phi: np.ndarray, alphas: np.ndarray,
                 labels: np.ndarray):
        self.phi = phi
        self.dirichlet_samplers = [
            DirichletDataSampler(alpha, labels) for alpha in alphas
        ]

    def __call__(self, n: int):
        j = np.random.choice(range(len(self.phi)), p=self.phi)
        return self.dirichlet_samplers[j](n)


class MinimizeReuseDataSampler:
    """
    Data sampling mechanism that maximises the time between instances of reuse.
    This is done by simply iterating through the sample space in linear fashion
    and starting over once the end is reached.
    Every data sampling mechanism returns a list of indices when called.
    The indices can be used to construct an artificial user dataset.
    :param max_bound:
        Maximum bound for sampling space.
        Will sample in the range `[0, max_bound)`.
    """

    def __init__(self, max_bound):
        self._index_iter = itertools.cycle(range(max_bound))

    def __call__(self, n):
        """
        Sample a list of data point indices.
        :param n:
            Number of samples to draw.
        :returns:
            Sampled indices in range `[0, max_bound)`.
        """
        return list(itertools.islice(self._index_iter, n))


def get_data_sampler(sample_type,
                     max_bound=None,
                     phi=None,
                     alphas=None,
                     alpha=None,
                     labels=None):
    """
    Factory for data sampling mechanisms.
    These samplers can be used when sampling data points for an artificial
    user dataset in `ArtificialFederatedDataset`, by providing it as the
    `sampler` argument.
    Implemented samplers:
    * random - Randomly sample from the range `[0, max_bound)`.
    * minimize_reuse - Sample while minimizing the number of times a number is
    sampled again.
    * dirichlet - Sample class proportions from a Dirichlet with given alpha
    parameter and sample classes according to these proportions.
    """
    if sample_type == 'random':
        return lambda n: np.random.randint(0, max_bound, size=n)
    elif sample_type == 'minimize_reuse':
        return MinimizeReuseDataSampler(max_bound)
    elif sample_type == 'dirichlet':
        return DirichletDataSampler(alpha, labels)
    elif sample_type == 'dirichlet_mixture':
        return DirichletMixtureDataSampler(phi, alphas, labels)
    else:
        raise NotImplementedError
