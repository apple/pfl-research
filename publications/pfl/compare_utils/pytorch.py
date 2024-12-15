# Copyright © 2024 Apple Inc.
from typing import Tuple

import torch
import torch.nn as nn


class Transpose2D(nn.Module):
    """
    Transpose Tensorflow style image to PyTorch compatible
    """

    def forward(self, inputs: torch.Tensor):
        return inputs.permute((0, 3, 1, 2))


def simple_cnn(input_shape: Tuple[int, ...], num_outputs: int,
               transpose: bool) -> nn.Module:
    """
    A simple CNN with 2 convolutional layers and one dense hidden layer.

    :param input_shape:
        The shape of the input images, e.g. (32,32,3).
    :param num_outputs:
        Size of output softmax layer.
    :return:
        A PyTorch CNN model.
    """
    in_channels = input_shape[-1]
    maxpool_output_size = (input_shape[0] - 4) // 2
    flatten_size = maxpool_output_size * maxpool_output_size * 64

    model = nn.Sequential(*[
        *([Transpose2D()] if transpose else []),
        nn.Conv2d(in_channels, 32, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(flatten_size, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_outputs),
    ])

    # Apply Glorot (Xavier) uniform initialization to match TF2 model.
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)

    return model
