# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import numpy as np

from pfl.aggregate.weighting import WeightByDatapoints


class WeightByTokens(WeightByDatapoints):

    def _num_tokens(self, user_dataset):
        return np.sum(user_dataset.raw_data[0] != 0)

    def reweight_statistics(self, statistics, user_dataset):
        statistics.reweight(self._num_tokens(user_dataset))


class WeightBySqrtTokens(WeightByTokens):

    def reweight_statistics(self, statistics, user_dataset):
        statistics.reweight(np.sqrt(self._num_tokens(user_dataset)))


class WeightByCubeRootTokens(WeightByTokens):

    def reweight_statistics(self, statistics, user_dataset):
        statistics.reweight(self._num_tokens(user_dataset)**(1 / 3))


class WeightByLogTokens(WeightByTokens):

    def reweight_statistics(self, statistics, user_dataset):
        statistics.reweight(np.log(self._num_tokens(user_dataset)))


class WeightByTokensClipped(WeightByTokens):

    def __init__(self, weight_clip):
        self._weight_clip = weight_clip

    def reweight_statistics(self, statistics, user_dataset):
        statistics.reweight(
            min(self._num_tokens(user_dataset), self._weight_clip))
