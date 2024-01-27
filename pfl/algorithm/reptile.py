# Copyright © 2023-2024 Apple Inc.

from pfl.algorithm.base import PersonalizedNNAlgorithm
from pfl.algorithm.federated_averaging import FederatedAveraging


class Reptile(FederatedAveraging, PersonalizedNNAlgorithm):
    """
    Defines the `Reptile <https://arxiv.org/abs/1803.02999>`_ algorithm by
    providing the implementation as hooks into the training process.
    """
