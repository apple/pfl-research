# Copyright © 2023-2024 Apple Inc.

import itertools
from collections import defaultdict

import numpy as np

from pfl.internal.ops import get_ops


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


def get_data_sampler(sample_type, max_bound=None, alpha=None, labels=None):
    """
    Factory for data sampling mechanisms.

    These samplers can be used when sampling data points for an artificial
    user dataset in `ArtificialFederatedDataset`, by providing it as the
    `sampler` argument.

    Implemented samplers:
    * random - Randomly sample from the range `[0, max_bound)`, max_bound
    must be specified.
    * minimize_reuse - Sample while minimizing the number of times a number is
    sampled again, max_bound must be specified.
    * dirichlet - Sample class proportions from a Dirichlet with given alpha
    parameter and sample classes according to these proportions. Must specify
    values for alpha and labels.
    """
    if sample_type == 'random':
        return lambda n: np.random.randint(0, max_bound, size=n)
    elif sample_type == 'minimize_reuse':
        return MinimizeReuseDataSampler(max_bound)
    elif sample_type == 'dirichlet':
        return DirichletDataSampler(alpha, labels)
    else:
        raise NotImplementedError


class MinimizeReuseUserSampler:
    """
    User sampling mechanism that maximizes the time between instances of reuse,
    similar to `MinimizeReuseDataSampler` but for sampling user ids.

    In live PFL training, it is common that a user can not participate more
    than one time per day for a particular use case.
    In other words, the distance of a user being selected again during the day
    is infinite.
    This is what the sampling strategy is trying to mimic.

    :param user_ids:
        A list of user ids as the sampling space.
    """

    def __init__(self, user_ids):
        self._user_iter = itertools.cycle(user_ids)

    def __call__(self):
        """
        Sample a single user id, which is just the next in line for this
        sampler.

        :returns:
            The sampled user id.
        """
        return next(self._user_iter)


class CrossSiloUserSampler:
    """
    User sampling mechanism that assumes the users are partitioned into
    disjoint subsets belonging to different silos. In each round, all silos
    contribute to the cohort and each silo may sample one or more users.

    In the distributed setting, silos will be split into different nodes.
    In each sampling call, a silo is picked according to a fixed order and then
    a user is sampled from the subset in that silo.

    :example:
        .. code-block::

            silo_to_user_ids = {
                0: [0, 1],
                1: [2, 3],
                2: [4, 5],
                3: [6, 7]
            }
            num_silos = 4
            # Assume 4 processes on 2 nodes
            # Node 1 will hold silo 0 and 1
            # Node 2 will hold silos 2 and 3
            sampler = CrossSiloUserSampler(
                sampling_type='minimize_reuse',
                silo_to_user_ids=silo_to_user_ids)
            [sampler() for _ in range(4)] = [0, 4, 2, 6]

    :param sampling_type:
        The sampling method used to choose a user from a silo.
    :param user_ids:
        A list of user ids as the sampling space.
    :param silo_to_user_ids:
        An optional dictionary where the keys are the silos and the values are
        the corresponding subset of user ids.
        If not provided, the `user_ids` will be evenly split into `n` silos
        where `n` is the number of workers in the distributed setting.
    :param num_silos:
        An optional int for indicating number of silos to partition the users
        if `silo_to_user_ids` is not provided.
    """

    def __init__(self,
                 sampling_type='random',
                 user_ids=None,
                 silo_to_user_ids=None,
                 num_silos=None):
        num_global_processes = get_ops().distributed.world_size
        num_local_processes = get_ops().distributed.local_size
        assert num_global_processes % num_local_processes == 0, (
            "Each node needs have equal number of processes")
        num_nodes = num_global_processes // num_local_processes

        if silo_to_user_ids is None:
            assert user_ids is not None, (
                "When `silo_to_user_ids` is `None`, `user_ids` must be "
                "provided to partition users into silos.")
            num_silos = num_silos or num_global_processes
            # Evenly split users into silos.
            silo_to_user_ids = defaultdict(list)
            for i, user_id in enumerate(user_ids):
                silo_to_user_ids[i % num_silos].append(user_id)
        else:
            num_silos = len(silo_to_user_ids)
        self._silo_user_sampler = {
            k: get_user_sampler(sampling_type, v)
            for k, v in silo_to_user_ids.items()
        }
        self._silos = list(self._silo_user_sampler.keys())
        process_to_silos = self.assign_silo_to_process(
            num_silos, num_nodes,
            get_ops().distributed.local_size)

        def silo_iter():
            # Match with iterating processes in
            # `pfl.data.federated_dataset._distributed_sampler`
            process_iter = itertools.cycle(range(num_global_processes))
            process_to_silo_iter = {
                k: itertools.cycle(v)
                for k, v in process_to_silos.items()
            }
            while True:
                yield next(process_to_silo_iter[next(process_iter)])

        self._silo_iter = iter(silo_iter())

    @staticmethod
    def assign_silo_to_process(num_silos, num_nodes, num_local_processes):
        process_to_silos = defaultdict(list)
        assert num_silos >= num_nodes
        # Each node holds one or multiple silos
        node_to_silos = defaultdict(list)
        # Assign silo to node
        for node, silo in zip(
                itertools.islice(itertools.cycle(range(num_nodes)), num_silos),
                range(num_silos)):
            node_to_silos[node].append(silo)
        # For each node and the silos, assign silo to local process
        for node in range(num_nodes):
            node_silos = node_to_silos[node]
            n = max(len(node_silos), num_local_processes)
            for silo, local_rank in zip(
                    itertools.islice(itertools.cycle(node_silos), n),
                    itertools.islice(
                        itertools.cycle(range(num_local_processes)), n)):
                global_rank = node * num_local_processes + local_rank
                process_to_silos[global_rank].append(silo)
        return process_to_silos

    def __call__(self):
        # Each process always picks a fixed silo
        idx = next(self._silo_iter)
        # Get the silo first and sample a user from the subset of that silo.
        sampled = self._silo_user_sampler[self._silos[idx]]()
        return sampled


def get_user_sampler(sample_type,
                     user_ids,
                     silo_to_user_ids=None,
                     num_silos=None):
    """
    Factory for user sampling mechanisms.

    These can be used when sampling the next user for `FederatedDataset`
    by providing it as the `user_sampler` argument.

    Implemented samplers:
    * random - Randomly sample a user id.
    * minimize_reuse - Sample a user while minimizing the number of times the
    user is sampled again.
    """
    if silo_to_user_ids is not None or num_silos is not None:
        return CrossSiloUserSampler(sample_type, user_ids, silo_to_user_ids,
                                    num_silos)

    if sample_type == 'random':
        return lambda: user_ids[np.random.randint(0, len(user_ids))]
    elif sample_type == 'minimize_reuse':
        return MinimizeReuseUserSampler(user_ids)
    else:
        raise NotImplementedError
