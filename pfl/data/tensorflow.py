# Copyright © 2023-2024 Apple Inc.

import itertools
from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import tensorflow as tf

from pfl.data.dataset import AbstractDataset, Dataset
from pfl.data.federated_dataset import FederatedDataset
from pfl.internal.ops.selector import get_framework_module as get_ops


class TFTensorDataset(Dataset):
    """
    In-memory TensorFlow tensors representing a user dataset.
    See :class:`~pfl.data.dataset.Dataset` for more information about
    the class and its parameters.

    :param features:
        A single ``tf.Tensor`` or a tuple of ``tf.Tensor``. Should be valid
        input to your Tensorflow model's ``call`` method.
    :param labels:
        A single ``tf.Tensor`` or a tuple of ``tf.Tensor``. Should be valid
        input to ``tf.keras.losses.Loss``
    :param eval_kwargs:
        A dictionary of any additional evaluation parameters for user.
        These will not be input to loss or metrics for TF, only stored in
        this dataset for other custom usage.

    :example:
        ``features`` and ``labels`` should be in a format compatible with
        this:

        .. code-block:: python

            preds = model(features, **train_kwargs)
            tf.keras.losses.Loss()(labels, preds)
    """

    def __init__(self,
                 features: Union[tf.Tensor, Tuple[tf.Tensor, ...]],
                 labels: Union[tf.Tensor, Tuple[tf.Tensor, ...]],
                 user_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 val_indices: Optional[List[int]] = None,
                 train_kwargs: Optional[Dict[str, Any]] = None,
                 eval_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__(raw_data=(features, labels),
                         user_id=user_id,
                         metadata=metadata,
                         val_indices=val_indices,
                         train_kwargs=train_kwargs,
                         eval_kwargs=eval_kwargs)

    @property
    def features(self):
        return self._raw_data[0]

    @property
    def labels(self):
        return self._raw_data[1]

    def iter(self, batch_size: Optional[int]):  # noqa: A003
        # Alternative batch splitting that is faster than the generic
        # implementation in `Dataset`.
        if batch_size is None:
            yield self.raw_data
            return

        if batch_size not in self._batches:

            datalen = len(self)
            full_splits = datalen // batch_size
            # batches is a list of the size for each slice.
            batches = [batch_size] * full_splits
            partial = datalen % batch_size
            # slices must add up to length of the tensor.
            if partial:
                batches.append(partial)

            tensors_batches = tf.nest.map_structure(
                lambda t: tf.split(t, batches, 0), self.raw_data)

            def get_nested_batch(nested_data, batch_idx):
                assert isinstance(nested_data, Sequence)
                if tf.is_tensor(nested_data[0]):
                    return nested_data[batch_idx]
                return [
                    get_nested_batch(data, batch_idx) for data in nested_data
                ]

            self._batches[batch_size] = [
                get_nested_batch(tensors_batches, i)
                for i in range(len(batches))
            ]

        yield from self._batches[batch_size]


class TFDataDataset(AbstractDataset):
    """
    Dataset for representing a user using ``tf.data.Dataset``
    as source. Use this if the (user) dataset is big and can't fit
    into memory, e.g. a big central eval dataset or cross-silo FL.
    Otherwise, because the overhead to initialize a ``tf.data.Dataset``
    is big, consider using faster in-memory dataset
    :class:`~pfl.data.tensorflow.TFTensorDataset` along with
    :class:`~pfl.data.tensorflow.TFFederatedDataset` for parallel
    preprocessing.

    :example:

        .. code-block:: python

            data = tf.data.Dataset.zip(
                (tf.data.Dataset.from_tensor_slices(features),
                 tf.data.Dataset.from_tensor_slices(labels)))
            TFDataDataset(data, prefetch=5)

    :param raw_data:
        The TF dataset to represent a user.
    :param prefetch:
        How many batches to prefetch. ``0`` for no prefetching.
    :param train_kwargs:
        Additional keyword arguments to propagate through pfl for the
        user. Will be unpacked in the tensorflow model's call to loss function.
    :param eval_kwargs:
        Additional keyword arguments to propagate through pfl for the
        user. Will be unpacked in the tensorflow model's call to ``metrics``
        function.
    """

    def __init__(self,
                 raw_data: tf.data.Dataset,
                 prefetch: int,
                 user_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 train_kwargs: Optional[Dict[str, Any]] = None,
                 eval_kwargs: Optional[Dict[str, Any]] = None):
        assert prefetch >= 0
        self._raw_data = raw_data
        self._prefetch = prefetch
        self._user_id = user_id
        self._metadata = metadata or {}
        self._train_kwargs = train_kwargs or {}
        self._eval_kwargs = eval_kwargs or {}
        self._length: Optional[int] = None

    @property
    def raw_data(self) -> tf.data.Dataset:
        return self._raw_data

    @property
    def user_id(self) -> Optional[str]:
        return self._user_id

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def train_kwargs(self) -> Dict[str, Any]:
        return self._train_kwargs

    @property
    def eval_kwargs(self) -> Dict[str, Any]:
        return self._eval_kwargs

    def __len__(self):
        """
        Length of the inner ``tf.data.Dataset``. You must have iterated
        through this dataset once before ``len`` is available.
        """
        if self._length is None:
            raise ValueError(
                "You must have iterated through the dataset at least "
                "once using `iter` before the dataset length is known.")
        return self._length

    def iter(self, batch_size: Optional[int]):  # noqa: A003
        length = 0
        data = self._raw_data.batch(batch_size)
        if self._prefetch > 0:
            data = data.prefetch(self._prefetch)
        for tensors in data:
            if self._length is None:
                length += self._first_tensor_length(tensors)
            yield tensors
        if self._length is None:
            self._length = length

    def split(
            self,
            fraction: Optional[float] = None,
            min_train_size: Optional[int] = None,
            min_val_size: Optional[int] = None) -> Tuple['Dataset', 'Dataset']:
        raise NotImplementedError("Not implemented for this class, use "
                                  "`pfl.data.Dataset` instead")

    def get_worker_partition(self) -> 'TFDataDataset':
        return TFDataDataset(self.raw_data.shard(
            num_shards=get_ops().distributed.world_size,
            index=get_ops().distributed.global_rank),
                             prefetch=self._prefetch,
                             train_kwargs=self.train_kwargs,
                             eval_kwargs=self.eval_kwargs)


# pytype: disable=name-error
class TFFederatedDataset(FederatedDataset):
    """
    Create a federated dataset from a TF dataset and user sampler.

    .. note::

        We highly encourage to use this class instead of
        :class:`~pfl.data.federated_dataset.FederatedDataset` when data
        is loaded from disk in `make_dataset_fn` because using
        ``tf.data.Dataset.prefetch`` in your code will allow users to be
        loaded asynchronously into memory.

    :param make_dataset_fn:
        To make your `tf.data.Dataset` compatible with a user sampler,
        this argument specifies a function that takes a ``tf.data.Dataset``
        as input and returns another ``tf.data.Dataset``. The input dataset
        will return a new user id each time a step in its iterator is
        taken. The output dataset should use this user id to load and
        preprocess the data for an entire user, and generate a structure
        that is expected by ``dataset_cls``, see e.g.
        :class:`~pfl.data.dataset.Dataset`.
        To load users asynchronously, you need to have called `prefetch`
        on the dataset that you return from ``make_dataset_fn`` (see the
        code snippet at the end of this docstring for an example).
    :param user_sampler:
        A callable with no parameters that return one sampled user ID when
        called.
    :param user_id_dtype:
        Tensorflow data type of user ids. ``tf.int32`` by default.
    :param dataset_cls:
        The dataset class to wrap around tensors returned from
        ``tf.data.Dataset``. This is
        :class:`~pfl.data.dataset.Dataset` by default and doesn't
        need to be changed in most cases.
    :param dataset_kwargs:
        A dictionary for keyword arguments when constructing the pfl dataset.
        If the `dataset_cls` is :class:`~pfl.data.dataset.Dataset` then the
        valid keyword arguments are `val_indices`, `train_kwargs` and
        `eval_kwargs`.
    :param user_id_to_weight:
        A dictionary mapping user id to a weight which acts as a proxy
        for compute time to train this user. In most cases, when model
        training time scales with data, number of user
        datapoints/tokens/batches should be a good estimate.
        This is solely used for minimizing straggling processes in distributed
        simulations. Leaving this ``None`` will have same performance result
        but simulations will be slower if users have varying dataset sizes.
    :return:
        A federated dataset where user datasets are generated from a
        ``tf.data.Dataset``.

    :Example:

    .. code-block:: python

        # Assume files with data exists.
        sample_it = itertools.cycle(['user1.txt', 'user2.txt'])
        # Sampler needs to be a callable.
        sampler = lambda: next(sample_it)

        @tf.function
        def pipeline(data):
            # Load user dataset and do any kind of preprocessing here.
            data = data.map(lambda user_id: tf.io.read_file(user_id))
            # Shuffle dataset of 1 user.
            data = data.shuffle(100)
            # Prefetch 10 users.
            data = data.prefetch(10)
            return data

        fed = FederatedDataset.from_tf_dataset(pipeline, sampler,
                                               user_id_dtype=tf.string)
    """

    def __init__(self,
                 make_dataset_fn: Callable[[tf.data.Dataset], tf.data.Dataset],
                 user_sampler: Callable[[], Any],
                 user_id_dtype: Optional[tf.DType] = None,
                 dataset_cls: Optional[Type[TFTensorDataset]] = None,
                 dataset_kwargs: Optional[Dict] = None,
                 user_id_to_weight: Optional[Dict[Any, int]] = None):
        # pytype: enable=name-error
        super().__init__(make_dataset_fn, user_sampler, user_id_to_weight)
        self._make_tfdata_fn = make_dataset_fn
        self._dataset_cls = (TFTensorDataset
                             if dataset_cls is None else dataset_cls)
        self._dataset_kwargs = dataset_kwargs or {}
        self._user_id_dtype = (tf.int32
                               if user_id_dtype is None else user_id_dtype)

        # This is a hack, but input to `self.make_dataset_fn` is now the
        # tensors already loaded by tf.data instead of the user ID.
        self.make_dataset_fn = self._tensors_to_pfl_dataset
        self.sampler = self._get_tf_sampler(self.sampler)

    def _tensors_to_pfl_dataset(self, tensors):
        features, labels = tensors
        return self._dataset_cls(features=features,
                                 labels=labels,
                                 **self._dataset_kwargs)

    def _get_tf_sampler(self, sampler):

        # We need to iterate through the data and the seeds in order, but
        # separately.
        sampler_1, sampler_2 = itertools.tee(sampler)
        underlying_data_iterator = (data for (data, _seed) in sampler_1)
        seed_iterator = (seed for (_data, seed) in sampler_2)

        data = tf.data.Dataset.from_generator(
            # requires a callable.
            lambda: underlying_data_iterator,
            output_types=self._user_id_dtype)
        data = self._make_tfdata_fn(data)
        data_iterator = iter(data)

        return zip(data_iterator, seed_iterator)
