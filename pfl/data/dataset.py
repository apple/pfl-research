# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar

import numpy as np

from pfl.internal.ops.selector import get_framework_module as get_ops
from pfl.internal.ops.selector import has_framework_module


def _slice_raw_data(data, slice_):
    if isinstance(data, (list, tuple)):
        datas = []
        for d in data:
            data = _slice_raw_data(d, slice_)
            datas.append(data)
        return datas

    elif isinstance(data, dict):
        datas = {}
        for k, d in data.items():
            data = _slice_raw_data(d, slice_)
            datas[k] = data
        return datas

    else:
        return data[slice_.start:slice_.stop]


AbstractDatasetType = TypeVar('AbstractDatasetType', bound='AbstractDataset')


class AbstractDataset(ABC):
    """
    Base class for user dataset representations.
    """

    @property
    @abstractmethod
    def raw_data(self) -> Any:
        pass

    @property
    @abstractmethod
    def user_id(self) -> Optional[str]:
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def train_kwargs(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def eval_kwargs(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def iter(self, batch_size: Optional[int]) -> Iterable:
        pass

    @abstractmethod
    def split(
        self,
        fraction: Optional[float] = None,
        min_train_size: Optional[int] = None,
        min_val_size: Optional[int] = None
    ) -> Tuple['AbstractDataset', 'AbstractDataset']:
        """
        Split the dataset into two smaller disjoint datasets. Used by
        algorithms that require both a train and val dataset for each user.

        :param fraction:
            The fraction of data to split at. Defaults to ``0.8``
        :param min_train_size:
            The minimum number of samples for the train partition after
            the split. If the dataset size is ``4`` and split is done with the
            parameters ``fraction=0.1`` and ``min_train_size=1``, the split
            would result in a train set of size ``0``, but ``min_train_size``
            overrides that to ``1``.
            Defaults to ``1``.
        :param min_val_size:
            Same as ``min_train_size``, but for the val partition. An error
            is thrown if dataset length before split is less than
            ``min_train_size+min_val_size``.
            Defaults to ``1``.
        :returns:
            A tuple ``(left_dataset, right_dataset)``, where ``left_dataset``
            is a ``Dataset`` with a fraction of ``fraction`` of the data
            and ``right_dataset`` is a ``Dataset`` with a fraction of
            ``1-fraction`` of the data.
        """
        pass

    @abstractmethod
    def get_worker_partition(self) -> 'AbstractDataset':
        """
        Partition the dataset among active workers in multi-worker training and
        return a unique partition for the current worker.
        If multi-worker training is not in use, this method just returns the
        identity because the partitioning is 1 set with the full data.

        :returns:
            A subset of this dataset, unique to the current worker. All worker
            partitions add up to the original dataset.
        """
        pass

    def _first_tensor_length(self, data) -> int:
        if isinstance(data, (list, tuple)):
            return self._first_tensor_length(data[0])
        elif isinstance(data, dict):
            return self._first_tensor_length(next(iter(data.values())))
        elif has_framework_module() and get_ops().is_tensor(data):
            return get_ops().get_shape(data)[0]
        else:
            assert isinstance(data, np.ndarray), (
                "The data must be of type NumPy array. "
                "If you want to use tensors of a particular framework you "
                "must import its corresponding Model class from "
                "`pfl.model` at the top of your script.")
            return data.shape[0]


class Dataset(AbstractDataset):
    """
    A representation of a flat (user) dataset.
    Use only for smaller (user) datasets that fit in memory.

    If using PyTorch or TF tensors, use datasets in
    :mod:`pfl.data.tensorflow` and :mod:`pfl.data.pytorch`
    instead.

    :param raw_data:
        A tuple of data inputs in the order they are specified by the model.
        The "order" for some common models is given by:
        * Tensorflow - First entry is a tuple of features (input to model),
        second entry is a tuple of labels (input to loss/metrics).
        * PyTorch - a tuple of tensors, unrolled to ``model.loss`` and
        ``model.metrics``.
    :param user_id:
        (Optional) String user identifier. Only needed if you are using
        algorithms or other components that explicitly make use of user IDs.
        You will notice if that is the case from errors that say user IDs
        must be defined.
    :param metadata:
        (Optional) Store additional data about the user. Can be retrieved
        later by the algorithm.
    :param train_kwargs:
        A dictionary of any additional training parameters to unpack in the
        training call of the deep learning framework.
    :param eval_kwargs:
        A dictionary of any additional evaluation parameters to unpack in the
        evaluation call of the deep learning framework.
    :param val_indices:
        TODO: rdar://115345691 (enable val_indices feature in user
          Dataset). this parameter is not actually used right now. Meanwhile,
          use parameters of ``split`` instead.

        A list of datapoint indices specifying which datapoints in
        ``raw_data`` to split into the val partition when calling
        ``split`` on this user dataset. This parameter is optional but useful
        when you have an explicit set of val datapoints for this user. As
        noted above, this parameter is currently not used.

        .. note::

            Splitting a user's dataset is only done by algorithms
            that require a local val dataset for measuring the generalized
            performance after training, e.g. meta-learning or personalization
            via fine-tuning the model locally on each device.

    """

    def __init__(self,
                 raw_data: Tuple[Any, ...],
                 user_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 train_kwargs: Optional[Dict[str, Any]] = None,
                 eval_kwargs: Optional[Dict[str, Any]] = None,
                 val_indices: Optional[List[int]] = None):
        self._raw_data = raw_data
        self._user_id = user_id
        self._metadata = metadata or {}
        self._train_kwargs = train_kwargs or {}
        self._eval_kwargs = eval_kwargs or {}
        self._val_indices = val_indices
        # Cache batch splits because it is time consuming.
        self._batches: Dict = {}

    @property
    def raw_data(self) -> Tuple[Any, ...]:
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
        Recursively search for an `np.array` in `raw_data` and return its
        first dimension, which should be the number of data points.
        """
        return self._first_tensor_length(self.raw_data)

    def iter(self, batch_size: Optional[int]):
        if batch_size is None:
            yield self.raw_data
            return

        def get_slice(data, start_ix, end_ix):
            if isinstance(data, (list, tuple)):
                # Input is a list of tensors.
                sliced = [
                    get_slice(tensor, start_ix, end_ix) for tensor in data
                ]
            else:
                assert get_ops().is_tensor(data) or isinstance(
                    data, np.ndarray)
                # No list of multiple inputs.
                sliced = data[start_ix:end_ix]
            return sliced

        if batch_size not in self._batches:
            # Only keep a cache size of 1 to limit memory growth.
            # Only 1 is needed anyway if program uses static batch size.
            self._batches = {
                batch_size: [
                    get_slice(self.raw_data, start_index,
                              start_index + batch_size)
                    for start_index in range(0, len(self), batch_size)
                ]
            }
        yield from self._batches[batch_size]

    def split(
            self,
            fraction: Optional[float] = None,
            min_train_size: Optional[int] = None,
            min_val_size: Optional[int] = None) -> Tuple['Dataset', 'Dataset']:
        if fraction is None:
            fraction = 0.8
        if min_train_size is None:
            min_train_size = 1
        if min_val_size is None:
            min_val_size = 1

        data_length = len(self)
        if min_train_size + min_val_size > data_length:
            raise ValueError(
                f"The dataset is only of size {data_length}. To satisfy "
                "min_train_size and min_val_size, it must be at "
                f"least of size {min_train_size + min_val_size}")

        split_index = int(data_length * fraction)
        # Possibly override initial split to satisfy splits.
        split_index -= max(0, min_val_size - (data_length - split_index))
        split_index += max(0, min_train_size - split_index)
        left_slice = range(0, split_index)
        right_slice = range(split_index, data_length)

        left_raw_data = tuple(_slice_raw_data(self.raw_data, left_slice))
        right_raw_data = tuple(_slice_raw_data(self.raw_data, right_slice))
        train_dataset = Dataset(left_raw_data,
                                user_id=self.user_id,
                                metadata=self.metadata,
                                train_kwargs=self.train_kwargs,
                                eval_kwargs=self.eval_kwargs)
        val_dataset = Dataset(right_raw_data,
                              user_id=self.user_id,
                              metadata=self.metadata,
                              train_kwargs=self.train_kwargs,
                              eval_kwargs=self.eval_kwargs)
        return train_dataset, val_dataset

    def get_worker_partition(self) -> 'AbstractDataset':
        partition_range = get_ops().distributed.distribute_range(len(self))
        return Dataset(raw_data=tuple(
            _slice_raw_data(self.raw_data, partition_range)),
                       train_kwargs=self.train_kwargs,
                       eval_kwargs=self.eval_kwargs)


class TabularDataset(Dataset):
    """
    A dataset comprising tabular dataset: (features, labels).
    See :class:`~pfl.data.dataset.Dataset` for more information about
    parameters.
    """

    def __init__(self,
                 features: np.ndarray,
                 labels: np.ndarray,
                 metadata: Optional[Dict[str, Any]] = None,
                 val_indices: Optional[List[int]] = None,
                 train_kwargs: Optional[Dict[str, Any]] = None,
                 eval_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__((features, labels),
                         metadata=metadata,
                         val_indices=val_indices,
                         train_kwargs=train_kwargs,
                         eval_kwargs=eval_kwargs)
        self._features = features
        self._labels = labels

    @property
    def features(self) -> np.ndarray:
        return self._features

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def raw_data(self) -> Tuple[Any, ...]:
        return (self._features, self._labels)


class DatasetSplit(AbstractDataset):
    """
    Decorates the dataset split into predefined ``train_dataset`` and ``val_dataset``.
    This means that when using an instance of this class without splitting has
    the same behavior as ``train_dataset`` except ``iter`` is only implemented
    for the two datasets from the split.

    This class is useful when using algorithms that require a local train-val
    split, e.g. meta-learning, and you want to pre-define how the split should
    be done.
    """

    def __init__(self, train_dataset: AbstractDataset,
                 val_dataset: AbstractDataset):
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset

    @property
    def user_id(self) -> Optional[str]:
        return self._train_dataset.user_id

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._train_dataset.metadata

    @property
    def raw_data(self) -> Tuple[Any, ...]:
        """
        The raw data of the train dataset.
        """
        return self._train_dataset.raw_data

    @property
    def train_kwargs(self) -> Dict[str, Any]:
        """
        Additional training arguments of the train dataset.
        """
        return self._train_dataset.train_kwargs

    @property
    def eval_kwargs(self) -> Dict[str, Any]:
        """
        Additional evaluation arguments of the train dataset.
        """
        return self._train_dataset.eval_kwargs

    def __len__(self):
        return len(self._train_dataset)

    def iter(self, batch_size: Optional[int]):
        raise NotImplementedError(
            "Can't iterate DatasetSplit, need to split it first")

    def split(
        self,
        fraction: Optional[float] = None,
        min_train_size: Optional[int] = None,
        min_val_size: Optional[int] = None
    ) -> Tuple['AbstractDataset', 'AbstractDataset']:
        """
        Split the dataset by returning the train and val ``Dataset`` previously
        specified in the constructor.

        :param fraction:
            Has no effect since the split is determined when initializing.
        :param min_train_size:
            Has no effect since the split is determined when initializing.
        :param min_val_size:
            Has no effect since the split is determined when initializing.
        :returns:
            A tuple with the user datasets ``train_dataset`` and ``val_dataset``
            previously specified in the constructor.
        """
        input_assert_str = (
            "A `DatasetSplit` has already a pre-defined split. "
            "The only supported value for `{0}` is `None` (current value is `{1}`)."
        )
        assert fraction is None, input_assert_str.format('fraction', fraction)
        assert min_train_size is None, input_assert_str.format(
            'min_train_size', min_train_size)
        assert min_val_size is None, input_assert_str.format(
            'min_val_size', min_val_size)
        return self._train_dataset, self._val_dataset

    def get_worker_partition(self) -> 'AbstractDataset':
        partition_range = get_ops().distributed.distribute_range(len(self))
        partitions = [
            Dataset(raw_data=tuple(_slice_raw_data(d.raw_data,
                                                   partition_range)),
                    train_kwargs=self.train_kwargs,
                    eval_kwargs=self.eval_kwargs)
            for d in [self._train_dataset, self._val_dataset]
        ]
        return DatasetSplit(*partitions)
