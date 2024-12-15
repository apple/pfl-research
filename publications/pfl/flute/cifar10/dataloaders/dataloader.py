# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Original code at https://github.com/microsoft/msrflute/blob/main/experiments/classif_cnn/dataloaders/dataloader.py.

import numpy as np
import torch
from core.dataloader import BaseDataLoader
from experiments.cifar10.dataloaders.dataset import Dataset


class DataLoader(BaseDataLoader):

    def __init__(self, mode, num_workers=0, **kwargs):
        args = kwargs['args']
        self.batch_size = args['batch_size']

        dataset = Dataset(
            data=kwargs['data'],
            test_only=(mode != 'train'),
            user_idx=kwargs.get('user_idx', None),
        )

        super().__init__(
            dataset,
            batch_size=self.batch_size,
            shuffle=(mode == 'train'),
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        x, y = list(zip(*batch))
        x, y = np.array(x), np.array(y)
        return {'x': torch.tensor(x), 'y': torch.tensor(y)}
