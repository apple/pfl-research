# Copyright © 2023-2024 Apple Inc.

from collections import namedtuple
from typing import Any, Callable, Dict, List, Union

import torch

IGNORE_INDEX = -100

IndexedData = namedtuple('IndexedData', ['data', 'index'])


class GetItemDataset(torch.utils.data.Dataset):
    """ Wraps a dataset that has __getitem__. """

    def __init__(self, data: Union[Dict, List], return_index: bool = False):
        super().__init__()
        self._data = data
        self._return_index = return_index

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        if self._return_index:
            # Return both data and index. The index can be used in the below
            # `UserIDCollatorWrapper` to pass the `user_id` to
            # `PyTorchFederatedDataset`
            return IndexedData(data=self._data[i], index=i)
        return self._data[i]


class UserIDCollatorWrapper:
    """
    Wraps an existing collator to add user ID to the resulting collated data.
    This is useful for `PyTorchFederatedDataset` which does not have `user_id`
    by default.
    """

    def __init__(self, collator: Callable[[List], Dict[str, Any]]):
        self._collator = collator

    def __call__(self, indexed_data: IndexedData) -> Dict[str, Any]:
        assert isinstance(indexed_data, IndexedData), (
            "`UserIDCollatorWrapper` only supports `torch.utils.data.Dataset` "
            "that returns `IndexedData`")
        data = self._collator(indexed_data.data)
        data["user_id"] = indexed_data.index
        return data
