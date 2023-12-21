# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import itertools
from unittest.mock import patch

import numpy as np

from pfl.data.partition import partition_by_dirichlet_class_distribution


class TestPartition:

    def test_partition_by_dirichlet_class_distribution(self):
        labels = np.repeat(np.arange(4), 10)
        len_sampler = lambda: 4

        with patch('numpy.random.dirichlet') as mock_dirichlet:
            # uniform.
            mock_dirichlet.side_effect = lambda alpha: np.ones(len(alpha)
                                                               ) / len(alpha)
            with patch('numpy.random.uniform') as mock_uniform:
                cdf_cycler = itertools.cycle([0.2, 0.4, 0.6, 0.8])
                mock_uniform.side_effect = lambda: next(cdf_cycler)
                partitions = partition_by_dirichlet_class_distribution(
                    labels, alpha=0.1, user_dataset_len_sampler=len_sampler)
                mock_dirichlet.assert_called_with(alpha=[0.1] * 4)
                # According to mock above, each partition should have one
                # label from each class.
                for i, partition in enumerate(partitions):
                    np.testing.assert_array_equal(
                        partition, [9 - i, 19 - i, 29 - i, 39 - i])
