import functools
import types
from typing import Tuple

import numpy as np
import torch.nn as nn

from .metrics import image_classification_loss, image_classification_metrics


def dnn(input_shape: Tuple[int, ...], hidden_dims: Tuple[int, ...],
        num_outputs: int) -> nn.Module:
    """
    Define a feed-forward neural network in PyTorch.

    :param input_shape:
        The shape of the input data (excluding batch size). E.g. if the
        input is an image of dimensions (12,12,3), then it will be flattened
        into a 432-dimensional vector before propagated through the network.
    :param hidden_dims:
        A tuple describing the size of each hidden layer.
    :param num_outputs:
        Size of output softmax layer.
    :return:
        A PyTorch DNN model.
    """

    in_features = int(np.prod(input_shape))
    layers = [nn.Flatten()]
    for dim in hidden_dims:
        layers.extend([nn.Linear(in_features, dim), nn.ReLU()])
        in_features = dim
    layers.append(nn.Linear(in_features, num_outputs))
    model = nn.Sequential(*layers)
    model.loss = types.MethodType(image_classification_loss, model)
    model.metrics = types.MethodType(image_classification_metrics, model)
    return model


def simple_dnn(input_shape: Tuple[int, ...], num_outputs: int) -> nn.Module:
    """
    Define a feed-forward neural network with 2 hidden layers of size 200.
    This is the same architecture as used in
    McMahan et al. 2017 https://arxiv.org/pdf/1602.05629.pdf.
    See ``dnn`` for description about parameters.
    """
    return functools.partial(dnn, hidden_dims=[200,
                                               200])(input_shape,
                                                     num_outputs=num_outputs)
