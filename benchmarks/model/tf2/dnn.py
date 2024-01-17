# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import functools
from typing import Tuple

import numpy as np
import tensorflow as tf  # type: ignore


def dnn(input_shape: Tuple[int, ...], hidden_dims: Tuple[int, ...],
        num_outputs: int) -> tf.keras.models.Model:
    """
    Define a feed-forward neural network in TF2/Keras.

    :param input_shape:
        The shape of the input data (excluding batch size). E.g. if the
        input is an image of dimensions (12,12,3), then it will be flattened
        into a 432-dimensional vector before propagated through the network.
    :param hidden_dims:
        A tuple describing the size of each hidden layer.
    :param num_outputs:
        Size of output softmax layer.
    :return:
        A Keras model compatible with TF2.
    """

    return tf.keras.models.Sequential([
        tf.keras.layers.Reshape(input_shape=input_shape,
                                target_shape=(np.prod(input_shape), )),
        *[
            tf.keras.layers.Dense(dim, activation=tf.nn.relu)
            for dim in hidden_dims
        ],
        tf.keras.layers.Dense(num_outputs, activation=tf.nn.softmax),
    ])


def simple_dnn(input_shape: Tuple[int, ...],
               num_outputs: int) -> tf.keras.models.Model:
    """
    Define a feed-forward neural network with 2 hidden layers of size 200.
    This is the same architecture as used in
    McMahan et al. 2017 https://arxiv.org/pdf/1602.05629.pdf.
    See ``dnn`` for description about parameters.
    """
    return functools.partial(dnn, hidden_dims=[200,
                                               200])(input_shape,
                                                     num_outputs=num_outputs)
