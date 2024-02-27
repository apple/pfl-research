# Copyright © 2023-2024 Apple Inc.

import logging
from typing import Dict, List, Union

import torch

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)


class GetItemDataset(torch.utils.data.Dataset):
    """ Wraps a dataset that has __getitem__. """

    def __init__(self, data: Union[Dict, List]):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
