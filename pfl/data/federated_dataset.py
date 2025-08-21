# Copyright © 2023-2024 Apple Inc.
"""
A dataset is simply an iterator for iterating through datasets of individual
users.
How the user's dataset is constructed is defined by inherited classes of
``FederatedDatasetBase``.

Every time ``__next__`` is called, a tuple ``(user_data, seed)`` is returned,
where ``user_data`` is a :class:`~pfl.data.dataset.Dataset` and ``seed`` is a
random integer.
The user for every call is chosen by a sampling strategy provided as argument to
the inherited subclasses of ``FederatedDatasetBase``.
The random integer ``seed`` is different for each call, and different on each
worker, and should be used as seed for local DP to break the otherwise unwanted
behaviour of each worker generating identical noise patterns.
"""

import atexit
import contextlib
import queue
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import multiprocess as mp
import numpy as np

from pfl.data.dataset import AbstractDataset, Dataset
from pfl.data.partition import partition_by_dirichlet_class_distribution
from pfl.data.sampling import CrossSiloUserSampler, get_user_sampler
from pfl.internal.ops.numpy_ops import NumpySeedScope
from pfl.internal.ops.selector import get_default_framework_module as get_ops

_INITIAL_USER_SAMPLES_CACHE_SIZE = 10000
_NUM_SAMPLES_BATCHED_IN_SUBPROCESS = 25
_COHORT_TO_CACHE_MULTIPLIER = 5


def _distributed_sampler(sampler, get_seed, rank, world_size):
    while True:
        users_all_workers, seeds_all_workers = [], []
        # Sample users, one for each worker.
        for _ in range(world_size):
            seed = get_seed()
            seeds_all_workers.append(seed)
            # Use dataset seed for sampling the user, which will make the
            # sampled user consistent irrespectible of the number of workers.
            with NumpySeedScope(seed):
                users_all_workers.append(sampler())
        yield users_all_workers[rank], seeds_all_workers[rank]


def _sorted_cohort_subprocess(q_send, q_request, q_send_num, sampler,
                              seed_sampler, rank, world_size,
                              user_id_to_weight):
    cache = []
    cohort_size = None
    cache_size = _INITIAL_USER_SAMPLES_CACHE_SIZE
    while True:
        try:
            # Try get new cohort request if does not already have one.
            if cohort_size is None:
                blocking = len(cache) >= cache_size
                cohort_size = q_request.get(blocking)
                cache_size = max(cache_size,
                                 _COHORT_TO_CACHE_MULTIPLIER * cohort_size)
        except queue.Empty:
            pass

        if cohort_size is not None and len(cache) > cohort_size:
            # Perform request
            cohort_samples = cache[:cohort_size]
            cohort_samples = sorted(cohort_samples,
                                    key=lambda tup: user_id_to_weight[tup[0]],
                                    reverse=True)

            # This optimization procedure should be deterministic
            # across all worker processes.
            worker_samples = [[] for _ in range(world_size)]
            worker_total_weight = np.zeros(world_size)
            for sample in cohort_samples:
                # Add user to worker with the minimum load.
                min_worker_index = np.argmin(worker_total_weight)
                worker_samples[min_worker_index].append(sample)
                worker_total_weight[min_worker_index] += user_id_to_weight[
                    sample[0]]

            q_send_num.put(len(worker_samples[rank]))
            q_send.put(worker_samples[rank])
            cache = cache[cohort_size:]
            cohort_size = None

        # Check if the queue is full
        if len(cache) < cache_size:
            # Generate multiple samples at a time
            for _ in range(_NUM_SAMPLES_BATCHED_IN_SUBPROCESS):
                seed = seed_sampler()
                with NumpySeedScope(seed):
                    cache.append((sampler(), seed))


class _SortedCohortSampler:
    """
    Convert `sampler` to sample specified cohorts and return samples
    in sorted order depending on the weight given by `user_id_to_weight`.

    :param sampler:
        Sampling function to enhance.
    :param seed_sampler:
        Sampling function for getting seeds.
    :param rank:
        Rank of current process.
    :param world_size:
        Total number of processes across all workers.
    :param user_id_to_weight:
        A dictionary mapping user id to a weight which acts as a proxy
        for compute time to train this user. In most cases, when model
        training time scales with data, number of user
        datapoints/tokens/batches should be a good estimate.
        This is solely used for minimizing straggling processes in distributed
        simulations. Leaving this ``None`` will have same performance result
        but simulations will be slower if users have varying dataset sizes.
    """

    def __init__(self, sampler, seed_sampler, rank, world_size,
                 user_id_to_weight):
        self._sampler = sampler
        self._seed_sampler = seed_sampler
        self._world_size = world_size
        self._rank = rank
        self._user_id_to_weight = user_id_to_weight

        self._samples_q = mp.Queue()
        self._cohort_request_q = mp.Queue()
        self._cohort_num_response_q = mp.Queue()
        self._sample_process = mp.Process(
            target=_sorted_cohort_subprocess,
            args=(self._samples_q, self._cohort_request_q,
                  self._cohort_num_response_q, sampler, seed_sampler, rank,
                  world_size, user_id_to_weight))
        self._sample_process.start()
        atexit.register(self.__del__)

    def __iter__(self):
        while True:
            yield from self._samples_q.get()

    def set_cohort_size(self, cohort_size):
        self._cohort_request_q.put(cohort_size)
        return self._cohort_num_response_q.get()

    def __del__(self):
        self._samples_q.close()
        self._cohort_request_q.close()
        self._sample_process.terminate()
        self._sample_process.join()


class FederatedDatasetBase(ABC):
    """
    Base class for federated datasets used by the simulator.
    A federated dataset contains many smaller subsets of data, representing a
    user's data.
    """

    def __init__(self):
        # This random state will have the same sequence on all workers
        # if the same global seed is set on each worker available.
        self._random_state = np.random.RandomState(
            np.random.randint(0, 2**32, dtype=np.uint32))

    def _get_ops(self):
        # Easier to mock in tests when stored this way.
        return get_ops()

    def __iter__(self):
        return self

    def next(self):  # noqa: A003
        # Python 2 support.
        return self.__next__()

    @abstractmethod
    def __next__(self) -> Tuple[AbstractDataset, int]:
        """
        Sample a user dataset from this federated dataset.

        :returns:
            A user dataset, and local seed.
        """
        pass

    @abstractmethod
    def get_cohort(self,
                   cohort_size: int) -> Iterable[Tuple[AbstractDataset, int]]:
        """
        Fetch an entire cohort of users. In the context of multi worker
        training, only the users assigned to the current worker should be
        returned.

        :param cohort_size:
            The number of users to fetch.
        :return:
            An iterable to iterate through each user dataset from the cohort.
            Each step of the iterator returns a tuple `(user_dataset, seed)`.
            Lazy loading of the users' datasets is performed when stepping
            through iterator.
        """
        pass

    def _get_seed(self):
        return self._random_state.randint(0, 2**32, dtype=np.uint32)


class ArtificialFederatedDataset(FederatedDatasetBase):
    """
    Simulates a federated dataset by automatically grouping data points into
    simulated users.
    This class is useful when there is no such thing as an associated user
    identifier for each sample, but will of course work if you happen to have
    user identifiers as well.
    How a user dataset is generated is implemented by the sampling mechanism
    provided.

    :param make_dataset_fn:
        A function that takes as input a list of sample indices and returns the
        generated user dataset.
        You are expected to implement this function yourself in any way you
        want, but the interface should be
        `func(dataset_indices) -> user_dataset`.
        You can find example implementations in the class methods.
        Note that the function to implement here differs from the function to
        provide when constructing a `FederatedDataset`:

        * `ArtificialFederatedDataset` -
          `func(dataset_indices) -> user_dataset`.

        * `FederatedDataset` - `func(user_id) -> user_dataset`.

    :param data_sampler:
        Sampling mechanism that samples datapoint indices to later use for
        constructing a new user dataset.
        The definition of a sampling mechanism is simply a callable
        `callable(dataset_length) -> dataset_indices`.
        The purpose of the sampling mechanism is for you to be able to have
        control over the distribution of the generated user dataset, which
        should ideally be as close to how a real federated dataset would look
        like for your use case.
        The factory called `get_data_sampler` provides some examples of how this
        callable might look like.
    :param sample_dataset_len:
        A callable that should sample the dataset length of a user, i.e.
        `callable() -> dataset_length`.

        Example:
            `data_sampler = lambda: max(1, np.random.poisson(5)`.
            `data_sampler` is a callable (function) that draws a user dataset
            size from a poisson distribution of mean 5, and should be a minimum
            of size 1.
    """

    def __init__(self, make_dataset_fn, data_sampler, sample_dataset_len):
        super().__init__()
        self.make_dataset_fn = make_dataset_fn
        dataindex_sampler = lambda: data_sampler(sample_dataset_len())
        self.sampler = iter(
            _distributed_sampler(dataindex_sampler, self._get_seed,
                                 get_ops().distributed.global_rank,
                                 get_ops().distributed.world_size))

    def __next__(self) -> Tuple[AbstractDataset, int]:
        # Each worker will make a dataset with its own set of indices.
        dataset_indices, seed = next(self.sampler)
        return self.make_dataset_fn(dataset_indices), seed

    @classmethod
    def from_slices(cls,
                    data,
                    data_sampler,
                    sample_dataset_len,
                    create_dataset_fn: Callable = lambda data: Dataset(data)):
        """
        Construct a simulated federated dataset from a regular dataset where
        there is no such thing as a user identifier.

        :param data:
            A list of ``np.ndarray``, i.e. the same format as a ``Dataset``
            accepts.
        :param create_dataset_fn:
            A lambda function to create an instance of Dataset, or a subclass
            of Dataset.
        :returns:
            An instance of `ArtificialFederatedDataset`.
        """
        assert isinstance(data, Iterable)

        def make_dataset_fn(indices):
            user_data = tuple([v[indices] for v in data])
            return create_dataset_fn(user_data)

        return cls(make_dataset_fn, data_sampler, sample_dataset_len)

    def get_cohort(self,
                   cohort_size: int) -> Iterable[Tuple[AbstractDataset, int]]:
        # No optimization supported for this class in terms of fetching
        # an entire cohort at once.
        for i in range(cohort_size):
            if (i % get_ops().distributed.world_size
                ) == get_ops().distributed.global_rank:
                yield next(self)


class FederatedDataset(FederatedDatasetBase):
    """
    A federated dataset is a collection of smaller datasets that are each
    associated to a unique user.
    Iterating through an instance of this class will each time return the
    dataset for a specific user.

    :param make_dataset_fn:
        A function that takes as input a user identifier and returns a user
        dataset.
        You are expected to implement this function yourself in any way you
        want, but the interface should be `func(user_id) -> user_dataset`.
        You can find example implementations in the class methods.
    :param user_sampler:
        Sampling mechanism that samples a user id.
        The interface of the user sampling mechanism should be
        `callable() -> user_id`.
        In most cases you want to use `MinimizeReuseUserSampler` because its
        behaviour mimics what usually happens in live federated learning with
        user devices.
        See `pfl.data.sampling.MinimizeReuseUserSampler` for explanation why.

        The factory called `get_user_sampler` provides some examples of how this
        callable might look like.
    :param user_id_to_weight:
        A dictionary mapping user id to a weight which acts as a proxy
        for compute time to train this user. In most cases, when model
        training time scales with data, number of user
        datapoints/tokens/batches should be a good estimate.
        This is solely used for minimizing straggling processes in distributed
        simulations. Leaving this ``None`` will have same performance result
        but simulations will be slower if users have varying dataset sizes.
    """

    def __init__(self,
                 make_dataset_fn,
                 user_sampler,
                 user_id_to_weight: Optional[Dict[Any, int]] = None):
        super().__init__()
        self.make_dataset_fn = make_dataset_fn
        self.user_sampler = user_sampler
        if user_id_to_weight is None or isinstance(
                user_sampler,
                CrossSiloUserSampler) or get_ops().distributed.world_size == 1:
            self._sample_fn = _distributed_sampler(
                user_sampler, self._get_seed,
                get_ops().distributed.global_rank,
                get_ops().distributed.world_size)
        else:
            self._sample_fn = _SortedCohortSampler(
                user_sampler, self._get_seed,
                get_ops().distributed.global_rank,
                get_ops().distributed.world_size, user_id_to_weight)
        self.sampler = iter(self._sample_fn)

    def _try_set_cohort_size(self, cohort_size: int) -> Optional[int]:
        # Doesn't need a cohort to be set, can continue.
        worker_cohort_size = None
        with contextlib.suppress(AttributeError):
            # pytype: disable=attribute-error
            worker_cohort_size = self._sample_fn.set_cohort_size(cohort_size)
            # pytype: enable=attribute-error
        return worker_cohort_size

    def __next__(self) -> Tuple[AbstractDataset, int]:
        # Each worker will make a dataset with its own sampled user.
        self._try_set_cohort_size(get_ops().distributed.world_size)
        user_id, seed = next(self.sampler)
        return self.make_dataset_fn(user_id), seed

    @classmethod
    def from_slices(cls, data, user_sampler):
        """
        Construct a federated dataset from a dictionary of user to dataset
        mappings.

        :param data:
            A dictionary `user_id:dataset`, where `user_id` is the unique user
            identifier and `dataset` is the dataset of the user (represented as
            a list of ``np.ndarray`` like in ``Dataset``).
        :returns:
            An instance of `FederatedDataset`.
        """

        def make_dataset_fn(user_id):
            return Dataset(data[user_id], user_id=user_id)

        return cls(make_dataset_fn, user_sampler)

    @classmethod
    def from_slices_with_dirichlet_class_distribution(
            cls,
            data: Tuple,
            labels: np.ndarray,
            alpha: float,
            user_dataset_len_sampler: Callable,
            spread_distribution_after_num_fails: int = 20,
            spread_distribution_after_fails_percentage: float = 0.02):
        """
        Create a federated dataset by partitioning ``data`` into artificial
        users generated by
        :func:`~pfl.data.partition.partition_by_dirichlet_class_distribution`.
        See the above partition function for more information and references.
        The user partitions are constructed once using all data, and unlike
        a :class:`~pfl.data.federated_dataset.ArtificialFederatedDataset`
        the user partitioning remains the same throughout the training.
        Sampling the generated users is done uniformly at random.

        :param data:
            A tuple of tensors, representing the data to be partitioned
            (including labels). Each user will have a dataset consisting of
            the same number of tensors as in ``data``, but they are slices of
            ``data``.
        :param labels:
            A one-dimensional array of all labels (integers). Must have same
            length as the first dimension of every tensor in ``data``.
            If ``labels`` should also be included as one of the tensors in
            the dataset, be sure to also include it in ``data``.

        See
        :func:`~pfl.data.partition.partition_by_dirichlet_class_distribution`
        for further description of the parameters.
        """
        users_to_indices = partition_by_dirichlet_class_distribution(
            labels, alpha, user_dataset_len_sampler,
            spread_distribution_after_num_fails,
            spread_distribution_after_fails_percentage)
        users_to_data = [
            tuple(t[indices] for t in data) for indices in users_to_indices
        ]
        user_sampler = get_user_sampler('random', range(len(users_to_data)))
        return FederatedDataset.from_slices(users_to_data, user_sampler)

    def get_cohort(self,
                   cohort_size: int) -> Iterable[Tuple[AbstractDataset, int]]:
        # Set next cohort size for sampler if possible
        worker_cohort_size = self._try_set_cohort_size(cohort_size)
        if worker_cohort_size is None:
            for i in range(cohort_size):
                if (i % get_ops().distributed.world_size
                    ) == get_ops().distributed.global_rank:
                    user_id, seed = next(self.sampler)
                    yield self.make_dataset_fn(user_id), seed
        else:
            for _ in range(worker_cohort_size):
                user_id, seed = next(self.sampler)
                yield self.make_dataset_fn(user_id), seed


class FederatedDatasetMixture(FederatedDatasetBase):
    """
    Simulates a mixture of federated datasets and/or artificial federated
    datasets.

    To sample new users we sample from the component datasets using the
    probability vector mixture_weights. We then sample a user from the sampled
    dataset.

    The mixture of federated datasets is useful for modelling clusters or
    different modes of users in a federated datasets. In particular, this class
    is used for modelling a Mixture-of-Polya (Dirichlet-Multinomial)
    distribution.

    :example:

        .. code-block:: python

            # First component dataset is an artificial federated dataset
            # created from data inputs X and data targets Y.
            component_dataset_0 = ArtificialFederatedDataset.from_slices(
                data=[X, Y],
                data_sampler=lambda n: list(range(n)),
                sample_dataset_len=lambda: np.random.sample(4))

            # Second component dataset is a federated dataset, created
            # from a dictionary mapping user IDs to datapoints
            data = {0: [X_0, y_0], 1: [X_1, y_1], 2: [X_2, y_2]}
            user_sampler = lambda: MinimizeReuseUserSampler(
                list(data.keys()))
            make_user_dataset = lambda user_id: Dataset(data[user_id])
            component_dataset_1 = FederatedDataset(make_user_dataset,
                user_sampler)

            mixture_weights = [0.5, 0.5]

            FederatedDatasetMixture(mixture_weights,
                                   [component_dataset_0, component_dataset_1])

    :param mixture_weights:
        A list or np.ndarray containing the weights for each of the component
        datasets in the mixture. The mixture weights give the probability of
        occurrence of each component dataset. Ideally the mixture weights sum
        to 1, but if not, this class will normalise the mixture weights to sum
        to 1.
    :param mixture_component_datasets:
        Individual federated datasets that are the components of the mixture
        of federated datasets. Each federated dataset combined form a mixture.
        List of type ArtificialFederatedDataset
    """

    def __init__(self, mixture_weights: Union[List[float], np.ndarray],
                 mixture_component_datasets: List[FederatedDatasetBase]):
        super().__init__()
        assert len(mixture_weights) == len(mixture_component_datasets), (
            'Must have the same number of mixture weights as mixture ',
            'component datasets in a FederatedDatasetMixture.')
        self._mixture_weights = mixture_weights / np.sum(mixture_weights)
        self._mixture_component_datasets = mixture_component_datasets

    def __next__(self):
        mixture_component = np.random.choice(range(len(self._mixture_weights)),
                                             p=self._mixture_weights)
        return next(self._mixture_component_datasets[mixture_component])

    def get_cohort(self,
                   cohort_size: int) -> Iterable[Tuple[AbstractDataset, int]]:
        for i in range(cohort_size):
            if (i % get_ops().distributed.world_size
                ) == get_ops().distributed.global_rank:
                yield next(self)
