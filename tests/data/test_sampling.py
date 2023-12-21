# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import itertools
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from pfl.data import CrossSiloUserSampler, DirichletDataSampler, MinimizeReuseDataSampler, MinimizeReuseUserSampler


class TestMinimizeReuseDataSampler(unittest.TestCase):

    def test(self):

        bound = int(np.sum(np.arange(10)))

        # no samples.
        sampler = MinimizeReuseDataSampler(bound)
        indices = sampler(0)
        self.assertEqual(indices, [])

        # single sample.
        sampler = MinimizeReuseDataSampler(bound)
        indices = sampler(1)
        self.assertEqual(indices, [0])

        # sample entire range.
        sampler = MinimizeReuseDataSampler(bound)
        all_indices = sampler(bound)
        self.assertEqual(all_indices, list(range(bound)))

        # The entire range sampled in multiple calls should equal to entire
        # range called once.
        sampler = MinimizeReuseDataSampler(bound)
        samples_accum = []
        for n in range(10):
            partial_samples = sampler(n)
            self.assertEqual(len(partial_samples), n)
            samples_accum += partial_samples
        self.assertEqual(all_indices, samples_accum)

        # Should start sampling from beginning when bound reached, i.e. infinite
        # data.
        sampler = MinimizeReuseDataSampler(bound)
        samples_accum = []
        for _ in range(10):
            partial_samples = sampler(bound)
            samples_accum += partial_samples
        self.assertEqual(all_indices * 10, samples_accum)


class TestDirichletDataSampler(unittest.TestCase):

    def test(self):

        num_classes = 5
        alpha = 0.1 * np.ones(num_classes)
        num_cycles = 4
        # labels = [0, 0, 0, 0, 1, 1, 1, 1, ..., 4, 4, 4, 4]
        labels = np.repeat(np.arange(num_classes), num_cycles)

        # no samples
        sampler = DirichletDataSampler(alpha, labels)
        indices = sampler(0)
        self.assertEqual(indices, [])

        with patch('numpy.random.dirichlet') as mock_dirichlet:
            mock_dirichlet.side_effect = [
                np.array([1, 0, 0, 0, 0]),
                np.array([0, 0, 0, 0, 1]),
                np.array([0, 0.5, 0.5, 0, 0]),
                np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            ]
            with patch('numpy.random.multinomial') as mock_multinomial:
                mock_multinomial.side_effect = lambda n, **kwargs: (n * kwargs[
                    'pvals']).astype(int)

                with patch('numpy.random.choice') as mock_choice:

                    def choice_side_effect(arr, **kwargs):
                        idx_cycler = itertools.cycle(range(num_cycles))
                        returned_values = []
                        for _ in range(kwargs['size']):
                            returned_values.append(arr[next(idx_cycler)])

                        return returned_values

                    # choice will cycle through the four indices of each class
                    # in order.
                    mock_choice.side_effect = choice_side_effect

                    # We sample 2 points from first class (p=[1, 0, 0, 0, 0]).
                    # The first two indices are taken by idx_cycler.
                    indices = sampler(2)
                    self.assertEqual(indices, [0, 1])

                    # We sample 8 points from last class (p=[0, 0, 0, 0, 1]).
                    indices = sampler(8)
                    self.assertEqual(indices, [16, 17, 18, 19, 16, 17, 18, 19])

                    # We sample 8 points, 4 each from classes 1 and 2 resp.
                    indices = sampler(8)
                    self.assertEqual(indices, [4, 5, 6, 7, 8, 9, 10, 11])

                    # We sample 100 points, 20 from each class. The indices
                    # corresponding to each class will be looped over 5 times.
                    indices = sampler(100)
                    self.assertEqual(
                        indices,
                        sum([
                            list(range(4 * i, 4 * i + 4)) * 5 for i in range(5)
                        ], []))


class TestMinimizeReuseUserSampler(unittest.TestCase):

    def test(self):
        user_ids = ['Filip', 'Rogier', 'Aine', 'Matt']

        # single sample.
        sampler = MinimizeReuseUserSampler(user_ids)
        user_id = sampler()
        self.assertEqual(user_id, user_ids[0])

        # Should start sampling from beginning when end of user_ids reached,
        # i.e. infinite data.
        sampler = MinimizeReuseUserSampler(user_ids)
        samples_accum = []
        for _ in range(10):
            samples_accum.append(sampler())
        self.assertEqual(samples_accum,
                         list(itertools.islice(itertools.cycle(user_ids), 10)))


class TestCrossSiloUserSampler:

    def setup_method(self):
        user_ids = list(range(10))
        silo_to_user_ids = {
            0: [0, 1, 2, 3],
            1: [4, 5, 6],
            2: [7, 8, 9],
            3: [10, 11]
        }
        user_ids_to_silo = {}
        for k, v in silo_to_user_ids.items():
            for user_id in v:
                user_ids_to_silo[user_id] = k
        return user_ids, silo_to_user_ids, user_ids_to_silo

    def test_uneven(self, mock_ops):
        # Test when number of silos is not even distributed among processes
        user_ids, silo_to_user_ids, user_ids_to_silo = self.setup_method()
        with patch('pfl.data.sampling.get_ops',
                   side_effect=lambda: MagicMock(distributed=MagicMock(
                       world_size=6, local_size=3))):
            sampler = CrossSiloUserSampler(user_ids=user_ids,
                                           silo_to_user_ids=silo_to_user_ids)
            silos_accum = []
            # Test that each process only samples from a fixed silo
            for _ in range(10):
                silos_accum.append(user_ids_to_silo[sampler()])
            # Node 1 has silos [0, 2] shared among 3 processes
            #   Process 0 has silo 0
            #   Process 1 has silo 2
            #   Process 2 has silo 0
            # Node 2 has silos [1, 3] shared among 3 processes
            #   Process 0 has silo 1
            #   Process 1 has silo 3
            #   Process 2 has silo 1
            assert silos_accum == [0, 2, 0, 1, 3, 1, 0, 2, 0, 1]

    def test_even(self, mock_ops):
        # Test when number of silos is even distributed among processes
        user_ids, silo_to_user_ids, user_ids_to_silo = self.setup_method()
        with patch('pfl.data.sampling.get_ops',
                   side_effect=lambda: MagicMock(distributed=MagicMock(
                       world_size=8, local_size=2))):
            sampler = CrossSiloUserSampler(user_ids=user_ids,
                                           silo_to_user_ids=silo_to_user_ids)
            silos_accum = []
            # Test that each process only samples from a fixed silo
            for _ in range(10):
                silos_accum.append(user_ids_to_silo[sampler()])
            # 4 nodes where each node gets a silo shared by 2 processes
            assert silos_accum == [0, 0, 1, 1, 2, 2, 3, 3, 0, 0]
