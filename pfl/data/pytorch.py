# Copyright © 2023-2024 Apple Inc.

import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch

from pfl.data.dataset import AbstractDataset, Dataset
from pfl.data.federated_dataset import FederatedDataset
from pfl.internal.ops import pytorch_ops
from pfl.internal.ops.selector import get_framework_module as get_ops


class PyTorchTensorDataset(Dataset):
    """
    In-memory PyTorch tensors representing a user dataset.
    See :class:`~pfl.data.dataset.Dataset` for more information about
    the class and its parameters.

    :param tensors:
        A list or dictionary of tensors which can be accepted into your model's
        ``loss`` and ``metrics`` functions like this:

        .. code-block:: python

            model.loss(*tensors, **train_kwargs).backward()
            model.metrics(*tensors, **eval_kwargs)

        See :class:`~pfl.model.pytorch.PyTorchModel` for more
        information about ``loss`` and ``metrics`` functions.
    """

    def __init__(self,
                 tensors: Union[Tuple[torch.Tensor, ...], Dict[str,
                                                               torch.Tensor]],
                 user_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 val_indices: Optional[List[int]] = None,
                 train_kwargs: Optional[Dict[str, Any]] = None,
                 eval_kwargs: Optional[Dict[str, Any]] = None):
        self._tensor_keys = None
        if isinstance(tensors, Dict):
            self._tensor_keys = list(tensors.keys())
            tensors = tuple(tensors[key] for key in self._tensor_keys)

        super().__init__(tensors,
                         user_id=user_id,
                         metadata=metadata,
                         val_indices=val_indices,
                         train_kwargs=train_kwargs,
                         eval_kwargs=eval_kwargs)

    def iter(self, batch_size: Optional[int]):  # noqa: A003
        for batch in super().iter(batch_size):
            if self._tensor_keys is not None:
                batch = dict(zip(self._tensor_keys, batch))
            yield batch


class _ShardedDataset(torch.utils.data.Dataset):

    def __init__(self, original_dataset, index, num_shards):
        self.dataset = original_dataset
        self._index = index
        self._num_shards = num_shards

    def __len__(self):
        length = len(self.dataset) // self._num_shards
        if self._index < len(self.dataset) % self._num_shards:
            length += 1
        return length

    def __getitem__(self, index):
        return self.dataset[self._index + index * self._num_shards]


class PyTorchDataDataset(AbstractDataset):
    """
    Dataset for representing a user using ``torch.utils.data.Dataset``
    as source. Use this if the (user) dataset is big and can't fit
    into memory, e.g. a big central eval dataset or cross-silo FL.
    Otherwise, consider using faster in-memory dataset
    :class:`~pfl.data.dataset.Dataset` along with
    :class:`~pfl.data.pytorch.PyTorchFederatedDataset` for parallel
    preprocessing.

    :example:

        .. code-block:: python

            class MyDataset(torch.utils.data.Dataset):
                def __init__(self):
                    self.data = torch.randn(100, 2)
                    self.labels = torch.randint(0, 2, (100,))

                def __getitem__(self, index):
                    return self.data[index], self.labels[index]

                def __len__(self):
                    return len(self.labels)

            PyTorchDataDataset(raw_data=MyDataset())


    :param raw_data:
        The PyTorch dataset to represent a user.
    :param train_kwargs:
        Additional keyword arguments to propagate through pfl for the
        user. Will be unpacked in the PyTorch model's call to loss function.
    :param eval_kwargs:
        Additional keyword arguments to propagate through pfl for the
        user. Will be unpacked in the PyTorch model's call to ``metrics``
        function.
    """

    def __init__(self,
                 raw_data: torch.utils.data.Dataset,
                 user_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 train_kwargs: Optional[Dict[str, Any]] = None,
                 eval_kwargs: Optional[Dict[str, Any]] = None,
                 **dataloader_kwargs):
        assert 'batch_size' not in dataloader_kwargs, (
            "Batch size must instead be provided when calling `iter`")
        self._raw_data = raw_data
        self._user_id = user_id
        self._metadata = metadata or {}
        self._train_kwargs = train_kwargs or {}
        self._eval_kwargs = eval_kwargs or {}
        self._dataloader_kwargs = dataloader_kwargs

    @property
    def raw_data(self) -> torch.utils.data.Dataset:
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
        return len(self._raw_data)

    def iter(self, batch_size: Optional[int]):  # noqa: A003
        return iter(
            torch.utils.data.DataLoader(self._raw_data,
                                        batch_size=batch_size,
                                        **self._dataloader_kwargs))

    def split(
            self,
            fraction: Optional[float] = None,
            min_train_size: Optional[int] = None,
            min_val_size: Optional[int] = None) -> Tuple['Dataset', 'Dataset']:
        raise NotImplementedError("Not implemented for this class, use "
                                  "`pfl.data.Dataset` instead")

    def get_worker_partition(self) -> 'PyTorchDataDataset':
        assert_msg = ("Providing sampler not supported when dataset "
                      "is requested to be sharded.")
        assert 'sampler' not in self._dataloader_kwargs, assert_msg
        assert 'batch_sampler' not in self._dataloader_kwargs, assert_msg
        return PyTorchDataDataset(_ShardedDataset(
            self.raw_data,
            index=get_ops().distributed.global_rank,
            num_shards=get_ops().distributed.world_size),
                                  train_kwargs=self.train_kwargs,
                                  eval_kwargs=self.eval_kwargs,
                                  **self._dataloader_kwargs)


class PyTorchFederatedDataset(FederatedDataset):
    """
    Create a federated dataset from a PyTorch dataset and sampler.
    A `torch.utils.data.DataLoader` is created from the arguments and is
    used to load user datasets asynchronously.

    :param dataset:
        A PyTorch dataset instance. ``__getitem__``, which is the method
        you override to load a datapoint, should be constructed such
        that the index parameter of ``__getitem__`` is assumed to be a user
        ID and the returned tensor should not be for one datapoint, but the
        entire data of the user with this user ID. The first dimension of the
        tensor(s) returned should be the number of samples for that user.
        E.g., for user_id='user_123' who has 9 datapoints with input
        features of shape [256, 256, 3] and output labels of shape [17],
        dataset['user_123'] would consist of a list of two tensors of
        shapes [torch.Size([9, 256, 256, 3]), torch.Size([9, 17])].
    :param user_sampler:
        Sampling mechanism that samples a user id.
        The interface of the user sampling mechanism should be
        `callable() -> user_id`. This parameter is the same as
        `user_sampler` in constructor of
        :class:~`pfl.data.federated_dataset.FederatedDataset`.
        In most cases you want to use `MinimizeReuseUserSampler` because its
        behaviour mimics what usually happens in live federated learning with
        user devices.
    :param dataset_cls:
        The dataset class to wrap around tensors returned from
        ``torch.utils.data.DataLoader``. This is
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
        simulations. Leaving this ``None`` will not affect the final outcomes
        and metrics of PFL simulation, but the simulation will be slower if
        users have varying dataset sizes.
    :param dataloader_kwargs:
        Keyword arguments to add to initialization of ``DataLoader``.
        It is important to specify these parameters correctly to receive
        any kind of parallelization for data loading. For details, see
        ``torch.utils.data.DataLoader``.
    """

    def __init__(self,
                 dataset: 'torch.utils.data.Dataset',
                 user_sampler: Callable[[], Any],
                 dataset_cls: Optional[Type[PyTorchTensorDataset]] = None,
                 dataset_kwargs: Optional[Dict] = None,
                 user_id_to_weight: Optional[Dict[Any, int]] = None,
                 **dataloader_kwargs):

        assert (
            'batch_size' not in dataloader_kwargs
            or dataloader_kwargs['batch_size'] == 1
            or dataloader_kwargs['batch_size'] is None
        ), ('pfl requires batch_size=1 because you should load the '
            'full batch of the user dataset in your `Dataset.__getitem__`. '
            'Either do not specify batch_size in `dataloader_kwargs` or '
            'set it to 1.')

        self._dataset_cls = Dataset if dataset_cls is None else dataset_cls
        self._dataset_kwargs = dataset_kwargs or {}

        prefetch_factor = 0
        if dataloader_kwargs.get("num_workers", 0) > 0:
            # prefetch_factor is default to 2 in PyTorch data loader
            prefetch_factor = dataloader_kwargs.get("prefetch_factor") or 2

        if prefetch_factor > 0:
            # TODO: supports _SortedCohortSampler with prefetching
            user_id_to_weight = None

        super().__init__(self._tensors_to_pfl_dataset, user_sampler,
                         user_id_to_weight)

        self._pt_dataset = dataset
        self._dataloader_kwargs = dataloader_kwargs
        self.sampler = self._get_pt_sampler(self.sampler)

    def _tensors_to_pfl_dataset(self, tensors):
        # This is a hack, but `tensors` is now the dataset already loaded
        # by DataLoader instead of the user ID.

        def process_tensor(tensor):
            assert tensor.shape[0] == 1, (
                'The batch size for the PyTorch DataLoader must be 1. '
                '`Dataset.__getitem__ ` should return the entire data of the '
                'user.')
            tensor = tensor.to(pytorch_ops.get_default_device())
            # Squeeze first dimension since we use batch_size=1 in DataLoader
            # and therefore assume the tensor is of shape
            # `[1, num_samples, num_features]`.
            return torch.squeeze(tensor, dim=0)

        return self._dataset_cls(
            tuple([process_tensor(tensor) for tensor in tensors]),
            **self._dataset_kwargs)

    def _get_pt_sampler(self, sampler):
        # PyTorch asserts that sampler must inherit
        # from `torch.utils.data.Sampler`.
        class DataSampler(torch.utils.data.Sampler):

            def __init__(self, underlying_iterator):
                self._underlying_iterator = underlying_iterator

            def __iter__(self):
                return self

            def __next__(self):
                sampled = next(self._underlying_iterator)
                return sampled

        # We need to iterate through the data and the seeds in order, but
        # separately.
        sampler_1, sampler_2 = itertools.tee(sampler)
        underlying_data_iterator = (data for (data, _seed) in sampler_1)
        seed_iterator = (seed for (_data, seed) in sampler_2)

        dl_iter = iter(
            torch.utils.data.DataLoader(
                self._pt_dataset,
                sampler=DataSampler(underlying_data_iterator),
                **self._dataloader_kwargs))

        return zip(dl_iter, seed_iterator)
