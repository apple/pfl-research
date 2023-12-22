# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _NormBase
from ..numpy.layer import positional_encoding


class _FrozenBatchNorm(_NormBase, ABC):
    """
    A special batch normalization module that will freeze the statistics
    during training and only update the affine parameters.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        # turn of training so no batchnorm statistics is collected
        # and use pretrained statistics in training as well
        self.training = False

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        bn_training = (self.running_mean is None) and (self.running_var is
                                                       None)

        return F.batch_norm(
            input,
            # If buffers are not tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats else None,
            self.running_var
            if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps)


class FrozenBatchNorm1D(_FrozenBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(
                input.dim()))


class FrozenBatchNorm2D(_FrozenBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                input.dim()))


class FrozenBatchNorm3D(_FrozenBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(
                input.dim()))


class Transpose2D(nn.Module):
    """
    Transpose Tensorflow style image to PyTorch compatible
    """

    def forward(self, inputs: torch.Tensor):
        return inputs.permute((0, 3, 1, 2))


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size: int, max_sequence_length: int):
        super().__init__()
        pe = positional_encoding(max_sequence_length, embedding_size)
        self.register_buffer('pe', torch.as_tensor(pe))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.unsqueeze(self.pe[:x.size(1)], 0)
