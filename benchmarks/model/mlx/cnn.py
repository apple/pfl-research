# Copyright © 2023-2024 Apple Inc.

import types
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from pfl.metrics import Weighted


def maxpool2d(x, pool_size=2, stride=2):
    batch, height, width, channels = x.shape
    out_height = (height - pool_size) // stride + 1
    out_width = (width - pool_size) // stride + 1
    x = x.reshape(batch, out_height, stride, out_width, stride, channels)
    # TODO: this is avg pooling, max doesn't work right now:
    # https://github.com/ml-explore/mlx/issues/1234
    return mx.mean(x, axis=(2, 4))


class CNN(nn.Module):

    def __init__(self, input_shape: Tuple[int, ...], num_outputs: int):
        super().__init__()

        in_channels = input_shape[-1]
        maxpool_output_size = (input_shape[0] - 4) // 2
        flatten_size = maxpool_output_size * maxpool_output_size * 64

        self.conv1 = nn.Conv2d(in_channels,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        # Results in nans when using MaxPool2d:
        # https://github.com/ml-explore/mlx/issues/1234
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(flatten_size, 128)
        self.linear2 = nn.Linear(128, num_outputs)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def num_params(self):
        nparams = sum(x.size for k, x in tree_flatten(self.parameters()))
        return nparams

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = maxpool2d(x)
        x = self.dropout1(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = nn.relu(self.linear1(x))
        x = self.dropout2(x)
        x = self.linear2(x)
        return x

    def loss(self, x, y, is_eval=False):
        if is_eval:
            self.eval()
        else:
            self.train()
        return nn.losses.cross_entropy(self(x), y.squeeze(), reduction="mean")

    def metrics(self, x, y):
        self.eval()
        pred = self(x)
        y = y.squeeze()
        num_samples = len(y)
        correct = (mx.argmax(pred, axis=1) == y).sum().item()
        loss = mx.sum(nn.losses.cross_entropy(pred, y)).item()
        return {
            "loss": Weighted(loss, num_samples),
            "accuracy": Weighted(correct, num_samples)
        }


def simple_cnn(input_shape: Tuple[int, ...], num_outputs: int) -> nn.Module:
    """
    A simple CNN with 2 convolutional layers and one dense hidden layer.

    :param input_shape:
        The shape of the input images, e.g. (32,32,3).
    :param num_outputs:
        Size of output softmax layer.
    :return:
        A MLX CNN model.
    """
    return CNN(input_shape, num_outputs)
